"""
AmenityAnalysis operator: snaps OpenStreetMap points of interest to their
nearest road network node and aggregates them into a per-node,
per-category count matrix.
"""

import json
import os
import time
from typing import Optional

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import psutil
from cupy import RawKernel

from urbanLib.citygraph import CityGraph
from urbanLib.operators.snapping_metrics import compute_snapping_metrics

# Kernel with distance calculation for geometric precision
amenity_kernel_code = """
extern "C" __global__
void snap_amenities_kernel(
    const float* amen_x, const float* amen_y, const int* amen_types,
    const float* edges_x1, const float* edges_y1,
    const float* edges_x2, const float* edges_y2,
    const long long* edge_node1_ids,
    const long long* edge_node2_ids,
    int N, int M,
    long long* out_node_ids,
    int* out_types,
    float* out_dists_m
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float ax = amen_x[idx];
    float ay = amen_y[idx];
    int type = amen_types[idx];

    float best_dist2 = 1e30f;
    long long best_node = -1;
    float best_proj_x = 0.0f, best_proj_y = 0.0f;

    for (int j = 0; j < M; ++j) {
        float x1 = edges_x1[j], y1 = edges_y1[j];
        float x2 = edges_x2[j], y2 = edges_y2[j];

        float dx = x2 - x1;
        float dy = y2 - y1;
        float len2 = dx*dx + dy*dy;
        if (len2 < 1e-8f) continue;

        float t = ((ax - x1)*dx + (ay - y1)*dy) / len2;
        t = fmaxf(0.0f, fminf(1.0f, t));

        float px = x1 + t * dx;
        float py = y1 + t * dy;
        float dist = (ax - px)*(ax - px) + (ay - py)*(ay - py);

        if (dist < best_dist2) {
            best_dist2 = dist;
            best_proj_x = px;
            best_proj_y = py;
            float d1 = (px - x1)*(px - x1) + (py - y1)*(py - y1);
            float d2 = (px - x2)*(px - x2) + (py - y2)*(py - y2);
            best_node = (d1 <= d2) ? edge_node1_ids[j] : edge_node2_ids[j];
        }
    }

    float meters_per_deg_lat = 111320.0f;
    float meters_per_deg_lon = 111320.0f * cosf(ay * 0.01745329251f);
    float dlon = ax - best_proj_x;
    float dlat = ay - best_proj_y;
    float dist_m = sqrtf(
        (dlon * meters_per_deg_lon)*(dlon * meters_per_deg_lon) +
        (dlat * meters_per_deg_lat)*(dlat * meters_per_deg_lat)
    );

    out_node_ids[idx] = best_node;
    out_types[idx]    = type;
    out_dists_m[idx]  = dist_m;
}
"""


def snap_amenities_rapids(
    city_graph: CityGraph,
    amenity_file: str,
    output_file: str = "nodes_with_amenities.csv",
    batch_size: int = 4000,
    metrics_output: str = "amenity_metrics.json",
    time_cpu_baseline: Optional[float] = None,
    max_snap_distance_m: Optional[float] = None,
    output_dir: str = ".",
    output_mapping: str = "amenity_node_mapping.csv",
    save_metrics: bool = True,
):
    print("=== GPU AMENITIES (RAPIDS + BATCH) + METRICS ===")

    metrics = {}
    time_start_total = time.time()

    # ============ METRIC: Data loading ============
    time_start = time.time()
    edges_df = city_graph.edges_df
    amen_df = cudf.read_csv(amenity_file)
    metrics["time_data_loading"] = time.time() - time_start

    metrics["num_amenities"] = len(amen_df)
    metrics["num_edges"] = len(edges_df)
    metrics["batch_size"] = batch_size

    # Unique amenity types
    time_start = time.time()
    unique_types = sorted(amen_df["amenity"].unique().to_pandas().tolist())
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    amen_df["amenity_type"] = amen_df["amenity"].map(type_to_idx)
    metrics["time_type_mapping"] = time.time() - time_start
    metrics["num_amenity_types"] = len(unique_types)

    print(f"Amenities: {len(amen_df):,} | Types: {len(unique_types)}")

    # ============ METRIC: Initial memory ============
    process = psutil.Process()
    ram_before = process.memory_info().rss / 1024**2

    # ============ METRIC: GPU transfer ============
    time_start = time.time()
    e_x1 = edges_df["Node1_Longitude"].to_cupy().astype(cp.float32)
    e_y1 = edges_df["Node1_Latitude"].to_cupy().astype(cp.float32)
    e_x2 = edges_df["Node2_Longitude"].to_cupy().astype(cp.float32)
    e_y2 = edges_df["Node2_Latitude"].to_cupy().astype(cp.float32)
    n1_ids = edges_df["Node1_ID"].astype("int64").to_cupy()
    n2_ids = edges_df["Node2_ID"].astype("int64").to_cupy()
    metrics["time_gpu_transfer"] = time.time() - time_start

    # ============ METRIC: Kernel compilation ============
    time_start = time.time()
    kernel = RawKernel(amenity_kernel_code, "snap_amenities_kernel")
    metrics["time_kernel_compilation"] = time.time() - time_start

    # ============ METRIC: GPU memory ============
    mempool = cp.get_default_memory_pool()
    vram_before = mempool.used_bytes() / 1024**2

    # ============ BATCH PROCESSING ============
    time_start_snapping = time.time()
    results_node = []
    results_type = []
    results_dist = []
    N = len(amen_df)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        print(
            f"Batch {(start // batch_size) + 1}/{(N - 1) // batch_size + 1} → {end - start:,} amenities"
        )

        batch = amen_df.iloc[start:end]
        a_x = batch["Longitude"].to_cupy().astype(cp.float32)
        a_y = batch["Latitude"].to_cupy().astype(cp.float32)
        a_t = batch["amenity_type"].to_cupy().astype(cp.int32)

        out_nodes = cp.zeros(end - start, dtype=cp.int64)
        out_types = cp.zeros(end - start, dtype=cp.int32)
        out_dist = cp.zeros(end - start, dtype=cp.float32)

        blocks = (len(a_x) + 255) // 256
        kernel(
            (blocks,),
            (256,),
            (
                a_x,
                a_y,
                a_t,
                e_x1,
                e_y1,
                e_x2,
                e_y2,
                n1_ids,
                n2_ids,
                len(a_x),
                len(e_x1),
                out_nodes,
                out_types,
                out_dist,
            ),
        )

        results_node.append(out_nodes.get())
        results_type.append(out_types.get())
        results_dist.append(out_dist.get())

        del a_x, a_y, a_t, out_nodes, out_types, out_dist

    metrics["time_snapping"] = time.time() - time_start_snapping

    # ============ CONCATENATE AND COMPUTE STATS ============
    node_ids = np.concatenate(results_node)
    types = np.concatenate(results_type)
    dists_m = np.concatenate(results_dist)

    snap_stats, is_outlier = compute_snapping_metrics(
        node_ids, dists_m, max_snap_distance_m
    )
    metrics.update(snap_stats)

    print(
        f"Median: {metrics['snapping_distance_median_m']:.2f} m | "
        f"p99: {metrics['snapping_distance_p99_m']:.2f} m | "
        f"Max: {metrics['snapping_distance_max_m']:.2f} m"
    )
    print(
        f">500m: {metrics['n_over_500m']} | >1000m: {metrics['n_over_1000m']} | >5000m: {metrics['n_over_5000m']}"
    )
    if max_snap_distance_m is not None:
        print(f"Outliers removed (>{max_snap_distance_m} m): {metrics['n_outliers']:,}")
        print(
            f"Post-filter — Mean: {metrics['filtered_mean_m']:.2f} m | "
            f"Median: {metrics['filtered_median_m']:.2f} m | "
            f"p99: {metrics['filtered_p99_m']:.2f} m | "
            f"Max: {metrics['filtered_max_m']:.2f} m"
        )

    # ============ Per-amenity detail (NodeID, type, distance, outlier) ============
    type_names = np.array(unique_types)[types]
    mapping_df = pd.DataFrame(
        {
            "AmenityID": np.arange(len(node_ids)),
            "NodeID": node_ids,
            "amenity_type": type_names,
            "snap_distance_m": dists_m,
            "is_outlier": is_outlier,
        }
    )
    mapping_df = mapping_df[mapping_df["NodeID"] != -1]
    mapping_path = os.path.join(output_dir, output_mapping)
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Detailed mapping saved: {mapping_path}")

    # ============ METRIC: Aggregation and counting ============
    time_start = time.time()
    mask = (node_ids != -1) & (~is_outlier)
    count_df = cudf.DataFrame(
        {
            "NodeID": node_ids[mask],
            "amenity_type": types[mask],
            "count": np.ones(mask.sum(), dtype=np.int32),
        }
    )
    aggregated = count_df.groupby(["NodeID", "amenity_type"]).count().reset_index()
    metrics["time_aggregation"] = time.time() - time_start

    # ============ METRIC: Pivot and merge ============
    time_start = time.time()
    pivot = aggregated.pivot_table(
        index="NodeID", columns="amenity_type", values="count", fill_value=0
    ).astype(int)
    pivot.columns = [unique_types[int(col)] for col in pivot.columns]

    # All network nodes
    all_nodes = city_graph.unique_nodes()

    final = all_nodes.merge(pivot, on="NodeID", how="left").fillna(0)

    # Ensure every amenity type has a column, even if no node matched it
    for amenity_type in unique_types:
        if amenity_type not in final.columns:
            final[amenity_type] = 0
        final[amenity_type] = final[amenity_type].astype("int32")

    final_pd = final.to_pandas()
    metrics["time_pivot_merge"] = time.time() - time_start

    # ============ METRIC: Export ============
    time_start = time.time()
    final_pd["total_amenities"] = final_pd[unique_types].sum(axis=1)
    final_pd = final_pd.sort_values("total_amenities", ascending=False).drop(
        "total_amenities", axis=1
    )
    final_pd = final_pd.reset_index(drop=True)
    output_file = os.path.join(output_dir, output_file)
    metrics_output = os.path.join(output_dir, metrics_output)
    final_pd.to_csv(output_file, index=False)
    metrics["time_export"] = time.time() - time_start

    metrics["num_nodes"] = len(final_pd)

    # ============ FINAL METRICS ============
    metrics["time_total"] = time.time() - time_start_total

    # Final memory snapshot
    ram_after = process.memory_info().rss / 1024**2
    vram_after = mempool.used_bytes() / 1024**2
    metrics["ram_peak_mb"] = float(ram_after)
    metrics["ram_used_mb"] = float(ram_after - ram_before)
    metrics["vram_peak_mb"] = float(vram_after)
    metrics["vram_used_mb"] = float(vram_after - vram_before)

    # Speedup vs CPU baseline, if one was provided
    if time_cpu_baseline:
        metrics["time_cpu_baseline"] = time_cpu_baseline
        metrics["speedup"] = time_cpu_baseline / metrics["time_total"]

    # ============ SAVE METRICS ============
    if save_metrics:
        with open(metrics_output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_output}")

    # ============ REPORT ============
    print(f"\nDone -> {output_file}")
    print(f"   Total nodes: {len(final_pd):,}")
    print(f"   Amenity types: {len(unique_types)}")
    print("\nTop 5 nodes by amenity count:")
    for _, row in final_pd.head(5).iterrows():
        total = row[unique_types].sum()
        print(f"   - Node {row['NodeID']} -> {int(total)} amenities")

    # ============ METRICS REPORT ============
    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS - AMENITIES")
    print("=" * 70)

    print("\nEXECUTION TIMES:")
    print(f"   - Data loading:         {metrics['time_data_loading']:8.3f} s")
    print(f"   - Type mapping:         {metrics['time_type_mapping']:8.3f} s")
    print(f"   - GPU transfer:         {metrics['time_gpu_transfer']:8.3f} s")
    print(f"   - Kernel compilation:   {metrics['time_kernel_compilation']:8.3f} s")
    print(f"   - Snapping (GPU):       {metrics['time_snapping']:8.3f} s")
    print(f"   - Aggregation:          {metrics['time_aggregation']:8.3f} s")
    print(f"   - Pivot/Merge:          {metrics['time_pivot_merge']:8.3f} s")
    print(f"   - Export:               {metrics['time_export']:8.3f} s")
    print("   - --------------------------------")
    print(f"   - TOTAL:                {metrics['time_total']:8.3f} s")

    print("\nMEMORY USAGE:")
    print(f"   - RAM peak:             {metrics['ram_peak_mb']:8.1f} MB")
    print(f"   - RAM used:             {metrics['ram_used_mb']:8.1f} MB")
    print(f"   - VRAM peak:            {metrics['vram_peak_mb']:8.1f} MB")
    print(f"   - VRAM used:            {metrics['vram_used_mb']:8.1f} MB")

    print("\nGEOMETRIC PRECISION (Snapping):")
    print(f"   - Mean distance:        {metrics['snapping_distance_mean_m']:8.2f} m")
    print(f"   - Median distance:      {metrics['snapping_distance_median_m']:8.2f} m")
    print(f"   - p95:                  {metrics['snapping_distance_p95_m']:8.2f} m")
    print(f"   - p99:                  {metrics['snapping_distance_p99_m']:8.2f} m")
    print(f"   - Max distance:         {metrics['snapping_distance_max_m']:8.2f} m")
    print(f"   - Std deviation:        {metrics['snapping_distance_std_m']:8.2f} m")
    print(f"   - >500m:  {metrics['n_over_500m']:6} ({metrics['pct_over_500m']:.2f}%)")
    print(
        f"   - >1000m: {metrics['n_over_1000m']:6} ({metrics['pct_over_1000m']:.2f}%)"
    )
    print(
        f"   - >5000m: {metrics['n_over_5000m']:6} ({metrics['pct_over_5000m']:.2f}%)"
    )
    if "filter_threshold_m" in metrics:
        print(
            f"   - Filtered (>{metrics['filter_threshold_m']:.0f}m): "
            f"{metrics['n_outliers']} ({metrics['pct_outliers']:.2f}%)"
        )
        print(f"   - Post-filter mean:     {metrics['filtered_mean_m']:8.2f} m")
        print(f"   - Post-filter median:   {metrics['filtered_median_m']:8.2f} m")
        print(f"   - Post-filter p99:      {metrics['filtered_p99_m']:8.2f} m")
        print(f"   - Post-filter max:      {metrics['filtered_max_m']:8.2f} m")

    if time_cpu_baseline:
        print("\nSPEEDUP:")
        print(f"   - CPU baseline time:    {metrics['time_cpu_baseline']:8.3f} s")
        print(f"   - GPU time:             {metrics['time_total']:8.3f} s")
        print(f"   - Speedup:              {metrics['speedup']:8.2f}x")

    print(f"\nMetrics saved to: {metrics_output}")
    print("=" * 70 + "\n")

    return final_pd, metrics


if __name__ == "__main__":
    result, metrics = snap_amenities_rapids(
        city_graph=CityGraph("manhattan_streets.csv"),
        amenity_file="manhattan_amenities.csv",
        output_file="nodes_with_amenities.csv",
        batch_size=4000,
        metrics_output="amenity_metrics.json",
        time_cpu_baseline=None,  # Set this to your CPU baseline time
        max_snap_distance_m=500.0,
    )
