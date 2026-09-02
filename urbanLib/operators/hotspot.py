"""
EdgeNodeAssignment operator: snaps crime events to their nearest road
network node via point-segment projection, and produces both an aggregated
per-node crime count and a detailed crime-to-node mapping.
"""

import json
import os
import time
from collections import Counter
from typing import Optional

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from cupy import RawKernel

from urbanLib.citygraph import CityGraph
from urbanLib.operators.snapping_metrics import compute_snapping_metrics

snap_kernel_code = """
extern "C" __global__
void snap_crimes_kernel(
    const float* crimes_x, const float* crimes_y,
    const float* edges_x1, const float* edges_y1,
    const float* edges_x2, const float* edges_y2,
    const long long* edge_node1_ids,
    const long long* edge_node2_ids,
    int N, int M,
    long long* out_node_ids,
    float* out_dists_m
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float cx = crimes_x[idx];
    float cy = crimes_y[idx];

    float best_dist2 = 1e30f;
    long long best_node_id = -1;
    float best_proj_x = 0.0f;
    float best_proj_y = 0.0f;

    for (int j = 0; j < M; ++j) {
        float x1 = edges_x1[j];
        float y1 = edges_y1[j];
        float x2 = edges_x2[j];
        float y2 = edges_y2[j];

        float dx = x2 - x1;
        float dy = y2 - y1;
        float len2 = dx*dx + dy*dy;

        if (len2 < 1e-8f) continue;

        float t = ((cx - x1)*dx + (cy - y1)*dy) / len2;
        t = fmaxf(0.0f, fminf(1.0f, t));

        float proj_x = x1 + t * dx;
        float proj_y = y1 + t * dy;

        float dist = (cx - proj_x)*(cx - proj_x) + (cy - proj_y)*(cy - proj_y);

        if (dist < best_dist2) {
            best_dist2 = dist;
            best_proj_x = proj_x;
            best_proj_y = proj_y;

            float d1 = (proj_x - x1)*(proj_x - x1) + (proj_y - y1)*(proj_y - y1);
            float d2 = (proj_x - x2)*(proj_x - x2) + (proj_y - y2)*(proj_y - y2);

            best_node_id = (d1 <= d2) ? edge_node1_ids[j] : edge_node2_ids[j];
        }
    }

    float meters_per_deg_lat = 111320.0f;
    float meters_per_deg_lon = 111320.0f * cosf(cy * 0.01745329251f);
    float dlon = cx - best_proj_x;
    float dlat = cy - best_proj_y;
    float dist_m = sqrtf(
        (dlon * meters_per_deg_lon)*(dlon * meters_per_deg_lon) +
        (dlat * meters_per_deg_lat)*(dlat * meters_per_deg_lat)
    );

    out_node_ids[idx] = best_node_id;
    out_dists_m[idx]  = dist_m;
}
"""


def gpu_hotspot_complete(
    city_graph: CityGraph,
    crime_file: str,
    output_hotspots: str = "nodes_with_crimes.csv",
    output_mapping: str = "crime_node_mapping.csv",
    batch_size: int = 4000,
    max_snap_distance_m: Optional[float] = None,
    output_dir: str = ".",
    metrics_output: str = "hotspot_metrics.json",
    time_cpu_baseline: Optional[float] = None,
    save_metrics: bool = True,
):
    print("=== GPU-HOTSPOT - Hotspot generation + crime-node detail mapping ===")

    time_start_total = time.time()

    # Load data
    edges_df = city_graph.edges_df
    crimes_df = cudf.read_csv(crime_file)

    print(f"Edges: {len(edges_df):,} | Crimes: {len(crimes_df):,}")

    # Move edge geometry to GPU
    e_x1 = edges_df["Node1_Longitude"].to_cupy().astype(cp.float32)
    e_y1 = edges_df["Node1_Latitude"].to_cupy().astype(cp.float32)
    e_x2 = edges_df["Node2_Longitude"].to_cupy().astype(cp.float32)
    e_y2 = edges_df["Node2_Latitude"].to_cupy().astype(cp.float32)

    node1_ids = edges_df["Node1_ID"].astype("int64").to_cupy()
    node2_ids = edges_df["Node2_ID"].astype("int64").to_cupy()

    kernel = RawKernel(snap_kernel_code, "snap_crimes_kernel")

    # Process in batches
    time_start_snapping = time.time()
    results = []
    all_dists = []
    N = len(crimes_df)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        print(f"Processing {start:,} -> {end:,} ({end - start:,} crimes)...")

        batch = crimes_df.iloc[start:end]
        cx = batch["Longitude"].to_cupy().astype(cp.float32)
        cy = batch["Latitude"].to_cupy().astype(cp.float32)

        out_ids = cp.zeros(end - start, dtype=cp.int64)
        out_dist = cp.zeros(end - start, dtype=cp.float32)

        blocks = (len(cx) + 255) // 256
        kernel(
            (blocks,),
            (256,),
            (
                cx,
                cy,
                e_x1,
                e_y1,
                e_x2,
                e_y2,
                node1_ids,
                node2_ids,
                len(cx),
                len(e_x1),
                out_ids,
                out_dist,
            ),
        )

        results.append(out_ids.get())
        all_dists.append(out_dist.get())

        del cx, cy, out_ids, out_dist

    # Concatenate results
    all_node_ids = np.concatenate(results)
    all_dists_m = np.concatenate(all_dists)
    time_snapping = time.time() - time_start_snapping
    print("Snapping complete.")

    snap_metrics, is_outlier = compute_snapping_metrics(
        all_node_ids, all_dists_m, max_snap_distance_m
    )

    print(
        f"Mean: {snap_metrics['snapping_distance_mean_m']:.2f} m | "
        f"Median: {snap_metrics['snapping_distance_median_m']:.2f} m | "
        f"p95: {snap_metrics['snapping_distance_p95_m']:.2f} m | "
        f"p99: {snap_metrics['snapping_distance_p99_m']:.2f} m | "
        f"Max: {snap_metrics['snapping_distance_max_m']:.2f} m"
    )
    print(
        f">500m: {snap_metrics['n_over_500m']} ({snap_metrics['pct_over_500m']:.2f}%) | "
        f">1000m: {snap_metrics['n_over_1000m']} ({snap_metrics['pct_over_1000m']:.2f}%) | "
        f">5000m: {snap_metrics['n_over_5000m']} ({snap_metrics['pct_over_5000m']:.2f}%)"
    )

    if max_snap_distance_m is not None:
        print(
            f"Outliers removed (>{max_snap_distance_m} m): {snap_metrics['n_outliers']:,} "
            f"({snap_metrics['pct_outliers']:.2f}%)"
        )
        print(
            f"Post-filter — Mean: {snap_metrics['filtered_mean_m']:.2f} m | "
            f"Median: {snap_metrics['filtered_median_m']:.2f} m | "
            f"p99: {snap_metrics['filtered_p99_m']:.2f} m | "
            f"Max: {snap_metrics['filtered_max_m']:.2f} m"
        )

    # Build a detailed crime-to-node mapping

    # Look for an existing unique crime ID column
    possible_id_cols = [
        "CrimeID",
        "IncidentID",
        "CaseNumber",
        "ID",
        "Case_ID",
        "Report_No",
    ]
    crime_id_col = None

    for col in possible_id_cols:
        if col in crimes_df.columns:
            crime_id_col = col
            print(f"Using existing column as CrimeID: '{col}'")
            break

    # Otherwise, create temporary sequential IDs
    if crime_id_col is None:
        print("No crime ID column found -> creating temporary IDs (0,1,2,...)")
        crimes_df["TempCrimeID"] = range(len(crimes_df))
        crime_id_col = "TempCrimeID"

    mapping_df = pd.DataFrame(
        {
            "NodeID": all_node_ids,
            "CrimeID": crimes_df[crime_id_col].to_numpy(),
            "snap_distance_m": all_dists_m,
            "is_outlier": is_outlier,
        }
    )

    # Drop invalid assignments (-1)
    mapping_df = mapping_df[mapping_df["NodeID"] != -1]

    # Sort by NodeID so repeated assignments are easy to spot
    mapping_df = mapping_df.sort_values("NodeID")

    mapping_path = os.path.join(output_dir, output_mapping)
    mapping_df.to_csv(mapping_path, index=False)
    print(f"\nDetailed mapping saved: {mapping_path}")
    print(f"Total assignments: {len(mapping_df):,}")

    top_repeated = mapping_df["NodeID"].value_counts().head(5)
    print("\nTop nodes by assigned crimes:")
    print(top_repeated)

    # Build the aggregated hotspot file

    contribute = (all_node_ids != -1) & (~is_outlier)
    counts = Counter(all_node_ids[contribute])

    all_nodes = city_graph.unique_nodes().to_pandas()
    all_nodes["NodeID"] = all_nodes["NodeID"].astype("int64")

    all_nodes["Weight"] = all_nodes["NodeID"].map(counts).fillna(0).astype(int)
    final = all_nodes.sort_values("Weight", ascending=False)
    final = final[["NodeID", "Latitude", "Longitude", "Weight"]]

    hotspots_path = os.path.join(output_dir, output_hotspots)
    final.to_csv(hotspots_path, index=False)
    print(f"\nAggregated hotspots saved: {hotspots_path}")
    print(f"   Total nodes: {len(final):,}")
    print(f"   With crimes: {(final['Weight'] > 0).sum():,}")

    print("\nTop 5 hotspots:")
    for _, row in final.head(5).iterrows():
        print(f"   - {int(row['NodeID'])} -> {row['Weight']} crimes")

    snap_metrics["time_snapping"] = time_snapping
    snap_metrics["time_total"] = time.time() - time_start_total
    if time_cpu_baseline is not None:
        snap_metrics["time_cpu_baseline"] = time_cpu_baseline
        snap_metrics["speedup"] = time_cpu_baseline / snap_metrics["time_total"]
        print(
            f"\nSpeedup: {snap_metrics['speedup']:.2f}x "
            f"(CPU {time_cpu_baseline:.2f}s -> GPU {snap_metrics['time_total']:.2f}s)"
        )

    if save_metrics:
        metrics_path = os.path.join(output_dir, metrics_output)
        with open(metrics_path, "w") as f:
            json.dump(snap_metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")

    return final, snap_metrics


# =============================================================================
if __name__ == "__main__":
    gpu_hotspot_complete(
        city_graph=CityGraph("manhattan_streets.csv"),
        crime_file="manhattan_crimes_pre.csv",
        output_hotspots="nodes_with_crimes.csv",
        output_mapping="crime_node_mapping.csv",
        batch_size=4000,
        max_snap_distance_m=500.0,
    )
