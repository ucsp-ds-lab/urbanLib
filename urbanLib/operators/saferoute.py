"""
SafestShortestPathFinder operator: multi-objective shortest-path search
that balances travel distance against accumulated crime risk, computed via
a weighted Dijkstra over the road network.
"""

import json
import os
import time
from typing import Any, Literal, Optional, Union

import cudf
import cupy as cp
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import psutil
from shapely.geometry import LineString

from urbanLib.citygraph import CityGraph

# Named presets map to alpha ∈ [0, 1]: 0 = pure distance, 1 = pure safety.
# Users can also pass any float in [0, 1] directly.
Mode = Union[Literal["safe", "fast", "balanced"], float]
PRESETS: dict[str, float] = {"fast": 0.0, "balanced": 0.5, "safe": 1.0}


class SafeRouteGPU:
    def __init__(self):
        self.nodes_df = None
        self.edges_df = None
        self.danger_df = None
        self.node_id_to_idx = {}
        self.idx_to_node_id = {}
        self.max_danger = 1
        self.max_distance = None
        self.metrics = {}
        self.route_metrics = {}
        self._node_cache: dict[tuple[float, float], dict] = {}
        self.time_start_total = None

    def load_network(self, city_graph: CityGraph):
        print(f"[SafeRouteGPU] Loading road network from {city_graph.edges_file}...")
        time_start = time.time()

        df = city_graph.edges_df
        self.metrics["num_edges_raw"] = len(df)
        print(f"Edges loaded: {len(df):,}")

        # Map node IDs to local indices
        all_nodes_pd = (
            cudf.concat([df["Node1_ID"], df["Node2_ID"]]).unique().to_pandas()
        )
        self.node_id_to_idx = {int(nid): i for i, nid in enumerate(all_nodes_pd)}
        self.idx_to_node_id = {v: k for k, v in self.node_id_to_idx.items()}
        self.metrics["num_nodes"] = len(all_nodes_pd)
        print(f"Unique nodes: {len(all_nodes_pd):,}")

        # Nodes with coordinates
        all_nodes_df = city_graph.unique_nodes().rename(
            columns={"NodeID": "node_id", "Latitude": "lat", "Longitude": "lon"}
        )
        all_nodes_df = all_nodes_df.sort_values("node_id").reset_index(drop=True)

        local_idx_list = [
            self.node_id_to_idx[int(nid)] for nid in all_nodes_df["node_id"].to_pandas()
        ]
        self.nodes_df = all_nodes_df.assign(local_idx=local_idx_list)

        # Edges with GPU-computed distances
        time_gpu_start = time.time()
        src_local = df["Node1_ID"].map(self.node_id_to_idx).astype("int32")
        dst_local = df["Node2_ID"].map(self.node_id_to_idx).astype("int32")

        lat1 = df["Node1_Latitude"].values
        lon1 = df["Node1_Longitude"].values
        lat2 = df["Node2_Latitude"].values
        lon2 = df["Node2_Longitude"].values
        distances = self._haversine_gpu(lat1, lon1, lat2, lon2)
        self.metrics["time_haversine_gpu"] = time.time() - time_gpu_start

        self.edges_df = cudf.DataFrame(
            {"src": src_local, "dst": dst_local, "distance": distances}
        )
        self.metrics["num_edges_processed"] = len(self.edges_df)
        self.metrics["time_load_network"] = time.time() - time_start
        print(f"Edges processed: {len(self.edges_df):,}")

    def load_danger(self, danger_file: str):
        print(f"[SafeRouteGPU] Loading danger data from {danger_file}...")
        time_start = time.time()

        danger_df = cudf.read_csv(danger_file)

        danger_df["NodeID"] = danger_df["NodeID"].astype("int64")
        self.nodes_df["node_id"] = self.nodes_df["node_id"].astype("int64")

        merged = self.nodes_df.merge(
            danger_df[["NodeID", "Weight"]],
            left_on="node_id",
            right_on="NodeID",
            how="left",
        )
        merged["Weight"] = merged["Weight"].fillna(0).astype("int32")
        self.danger_df = merged

        self.max_danger = int(merged["Weight"].max())
        self.metrics["max_danger"] = self.max_danger
        self.metrics["nodes_with_danger"] = int((merged["Weight"] > 0).sum())
        self.metrics["time_load_danger"] = time.time() - time_start
        print(f"Max danger: {self.max_danger:,} crimes")
        self._build_cache()

    def _build_cache(self):
        self._edges_pd = self.edges_df.to_pandas()
        self._nodes_pd = self.nodes_df.to_pandas()
        self._danger_levels = self.danger_df.set_index("local_idx")[
            "Weight"
        ].to_pandas()
        self._danger_lookup = self._danger_levels.to_dict()
        self._node_lookup = self._nodes_pd.set_index("local_idx")[
            ["node_id", "lat", "lon"]
        ].to_dict("index")
        src_arr = self._edges_pd["src"].values.astype(int)
        dst_arr = self._edges_pd["dst"].values.astype(int)
        self._src_arr = src_arr
        self._dst_arr = dst_arr
        self._dist_arr = self._edges_pd["distance"].values
        danger_vec = self._danger_levels.reindex(
            range(len(self._nodes_pd)), fill_value=0
        ).values
        self._danger_src = danger_vec[src_arr]
        self._danger_dst = danger_vec[dst_arr]
        
        self.max_distance = float(self._dist_arr.max())
        norm_dist = self._dist_arr / self.max_distance
        avg_danger = (self._danger_src + self._danger_dst) / 2
        norm_danger = avg_danger / max(1, self.max_danger)

        self._G_base = nx.Graph()
        self._G_base.add_edges_from(
            (int(u), int(v), {"dist": float(d), "danger": float(g)})
            for u, v, d, g in zip(src_arr, dst_arr, norm_dist, norm_danger)
        )
        self._edge_dist = {
            (int(u), int(v)): float(d)
            for u, v, d in zip(src_arr, dst_arr, self._dist_arr)
        }
        self._graph_num_nodes = self._G_base.number_of_nodes()
        self._graph_num_edges = self._G_base.number_of_edges()
        print(
            f"[Cache] {self._graph_num_nodes:,} nodes | {self._graph_num_edges:,} edges"
        )

    _haversine_kernel = cp.ElementwiseKernel(
        "float64 lat1, float64 lon1, float64 lat2, float64 lon2",
        "float64 dist",
        """
        const double R    = 6371000.0;
        const double PI   = 3.14159265358979323846;
        const double deg  = PI / 180.0;
        double dphi    = (lat2 - lat1) * deg;
        double dlambda = (lon2 - lon1) * deg;
        double phi1    = lat1 * deg;
        double phi2    = lat2 * deg;
        double s_dphi  = sin(dphi * 0.5);
        double s_dl    = sin(dlambda * 0.5);
        double a = s_dphi*s_dphi + cos(phi1)*cos(phi2)*s_dl*s_dl;
        dist = 2.0 * R * atan2(sqrt(a), sqrt(1.0 - a));
        """,
        "haversine_fused",
    )

    def _haversine_gpu(self, lat1, lon1, lat2, lon2):
        return self._haversine_kernel(lat1, lon1, lat2, lon2)

    def _find_closest_node(self, lat: float, lon: float) -> dict[str, Any]:
        if (lat, lon) in self._node_cache:
            return self._node_cache[(lat, lon)]
        time_start = time.time()
        lats = self._nodes_pd["lat"].values
        lons = self._nodes_pd["lon"].values
        R = 6371000.0
        phi1 = np.radians(lats)
        phi2 = np.radians(lat)
        dphi = np.radians(lat - lats)
        dlambda = np.radians(lon - lons)
        a = (
            np.sin(dphi / 2) ** 2
            + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        )
        dists = 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        i = int(np.argmin(dists))
        row = self._nodes_pd.iloc[i]
        min_dist = dists[i]
        time_elapsed = time.time() - time_start
        print(
            f"Node: {int(row['node_id']):,} ({row['lat']:.4f}, {row['lon']:.4f}) | {min_dist:.0f}m | {time_elapsed:.3f}s"
        )
        result = {
            "node_id": int(row["node_id"]),
            "local_idx": int(row["local_idx"]),
            "distance": min_dist,
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "search_time": time_elapsed,
        }
        self._node_cache[(lat, lon)] = result
        return result

    def find_route(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        mode: Mode = "balanced",
        label: Optional[str] = None,
    ) -> dict[str, Any]:

        time_route_start = time.time()
        route_metrics = {"mode": mode}

        mode_str = mode if isinstance(mode, str) else f"alpha={mode:.2f}"
        print(f"\nMode: {mode_str.upper()}")

        # ============ METRIC: Nearest-node search ============
        start_info = self._find_closest_node(*start)
        end_info = self._find_closest_node(*end)
        route_metrics["time_find_start_node"] = start_info["search_time"]
        route_metrics["time_find_end_node"] = end_info["search_time"]
        route_metrics["start_node_distance_m"] = start_info["distance"]
        route_metrics["end_node_distance_m"] = end_info["distance"]

        if start_info["node_id"] == end_info["node_id"]:
            return {"error": "Start and end map to the same node."}

        start_idx, end_idx = start_info["local_idx"], end_info["local_idx"]

        # ============ METRIC: Weight calculation ============
        if isinstance(mode, str):
            alpha = PRESETS[mode]
        else:
            alpha = float(mode)
            if not 0.0 <= alpha <= 1.0:
                raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        safety_w = alpha
        dist_w = 1.0 - alpha
        route_metrics["alpha"] = alpha
        route_metrics["weight_safety"] = safety_w
        route_metrics["weight_distance"] = dist_w

        time_weight_start = time.time()
        weight_fn = lambda u, v, d: safety_w * d["danger"] + dist_w * d["dist"]
        route_metrics["time_weight_calculation"] = time.time() - time_weight_start
        print(f"Dijkstra {start_idx} → {end_idx} (alpha={alpha:.2f})")
        route_metrics["graph_nodes"] = self._graph_num_nodes
        route_metrics["graph_edges"] = self._graph_num_edges

        # ============ METRIC: Dijkstra ============
        dijkstra_start = time.time()
        try:
            total_cost, path = nx.single_source_dijkstra(
                self._G_base,
                start_idx,
                end_idx,
                weight=weight_fn,
            )
            dijkstra_time = time.time() - dijkstra_start
            route_metrics["time_dijkstra"] = dijkstra_time
        except nx.NetworkXNoPath:
            return {"error": "No path between start and end."}

        # ============ METRIC: Coordinate construction ============
        time_coords_start = time.time()
        route_coords = []
        real_distance = 0
        total_danger = 0

        for i, local_idx in enumerate(path):
            info = self._node_lookup.get(
                local_idx, {"node_id": -1, "lat": 0.0, "lon": 0.0}
            )
            node_danger = int(self._danger_lookup.get(local_idx, 0))
            total_danger += node_danger
            route_coords.append(
                {
                    "osm_id": int(info["node_id"]),
                    "local_idx": int(local_idx),
                    "lat": float(info["lat"]),
                    "lon": float(info["lon"]),
                    "danger": node_danger,
                }
            )
            if i > 0:
                u, v = path[i - 1], local_idx
                real_distance += self._edge_dist.get((u, v)) or self._edge_dist.get(
                    (v, u), 0.0
                )

        route_metrics["time_route_construction"] = time.time() - time_coords_start

        # ============ ROUTE METRICS ============
        route_metrics["route_nodes_count"] = len(path)
        route_metrics["route_distance_km"] = real_distance / 1000
        route_metrics["route_cost"] = total_cost
        route_metrics["route_total_danger"] = total_danger
        route_metrics["route_avg_danger_per_node"] = (
            total_danger / len(path) if len(path) > 0 else 0
        )
        route_metrics["time_total_route_finding"] = time.time() - time_route_start

        print(
            f"Route: {len(path)} nodes | Cost: {total_cost:.3f} | {real_distance / 1000:.1f}km | {dijkstra_time:.2f}s"
        )

        self.route_metrics[label if label is not None else mode_str] = route_metrics

        return {
            "mode": mode,
            "success": True,
            "path_local_idx": path,
            "total_cost": total_cost,
            "real_distance_km": real_distance / 1000,
            "nodes_count": len(path),
            "total_danger": total_danger,
            "start_node": start_info["node_id"],
            "end_node": end_info["node_id"],
            "dijkstra_time": dijkstra_time,
            "route_coords": route_coords,
            "metrics": route_metrics,
        }

    def save_route_json(self, result: dict[str, Any], filename: Optional[str] = None):
        if filename is None:
            filename = f"route_{result['mode']}.json"

        simple_route = {
            "mode": result["mode"],
            "alpha": result["metrics"]["alpha"],
            "success": result["success"],
            "nodes_count": result["nodes_count"],
            "distance_km": round(result["real_distance_km"], 2),
            "route_objective": round(result["total_cost"], 3),
            "total_danger": result["total_danger"],
            "danger_per_km": round(
                result["total_danger"] / max(result["real_distance_km"], 1e-6), 2
            ),
            "computation_time_s": round(result["dijkstra_time"], 3),
            "start_node": result["start_node"],
            "end_node": result["end_node"],
            "route": result["route_coords"],
        }

        with open(filename, "w") as f:
            json.dump(simple_route, f, indent=2)
        print(f"JSON saved: {filename}")

    def save_route_geojson(
        self, result: dict[str, Any], filename: Optional[str] = None
    ):
        if filename is None:
            filename = f"route_{result['mode']}.geojson"
        coords = [(point["lon"], point["lat"]) for point in result["route_coords"]]
        gdf = gpd.GeoDataFrame(
            {
                "mode": [result["mode"]],
                "alpha": [result["metrics"]["alpha"]],
                "distance_km": [result["real_distance_km"]],
                "route_objective": [result["total_cost"]],
                "nodes_count": [result["nodes_count"]],
                "total_danger": [result["total_danger"]],
            },
            geometry=[LineString(coords)],
            crs="EPSG:4326",
        )
        gdf.to_file(filename, driver="GeoJSON")
        print(f"GeoJSON saved: {filename}")

    def save_metrics(self, filename: str = "saferoute_metrics.json"):
        """Save all accumulated metrics."""
        process = psutil.Process()
        self.metrics["ram_peak_mb"] = process.memory_info().rss / 1024**2

        mempool = cp.get_default_memory_pool()
        self.metrics["vram_peak_mb"] = mempool.used_bytes() / 1024**2

        # Include route metrics, if any were recorded
        if hasattr(self, "route_metrics"):
            self.metrics["routes"] = self.route_metrics

        with open(filename, "w") as f:
            json.dump(self.metrics, f, indent=2)

        print(f"\nMetrics saved to: {filename}")
        self._print_metrics_summary()

    def _print_metrics_summary(self):
        """Print a metrics summary."""
        print("\n" + "=" * 70)
        print("PERFORMANCE METRICS - SAFE ROUTE")
        print("=" * 70)

        print("\nLOADING TIMES:")
        print(
            f"   - Load road network:    {self.metrics.get('time_load_network', 0):8.3f} s"
        )
        print(
            f"     - Haversine GPU:      {self.metrics.get('time_haversine_gpu', 0):8.3f} s"
        )
        print(
            f"   - Load danger data:     {self.metrics.get('time_load_danger', 0):8.3f} s"
        )

        if hasattr(self, "route_metrics"):
            print("\nROUTE SEARCH TIMES:")
            for mode, metrics in self.route_metrics.items():
                print(f"\n   Mode: {mode.upper()}")
                print(
                    f"   - Find start node:      {metrics.get('time_find_start_node', 0):8.3f} s"
                )
                print(
                    f"   - Find end node:        {metrics.get('time_find_end_node', 0):8.3f} s"
                )
                print(
                    f"   - Compute/assign weights: {metrics.get('time_weight_calculation', 0):8.3f} s"
                )
                print(
                    f"   - Dijkstra:               {metrics.get('time_dijkstra', 0):8.3f} s"
                )
                print(
                    f"   - Build route:          {metrics.get('time_route_construction', 0):8.3f} s"
                )
                print("   - --------------------------------")
                print(
                    f"   - TOTAL route:          {metrics.get('time_total_route_finding', 0):8.3f} s"
                )

        print("\nMEMORY USAGE:")
        print(
            f"   - RAM peak:             {self.metrics.get('ram_peak_mb', 0):8.1f} MB"
        )
        print(
            f"   - VRAM peak:            {self.metrics.get('vram_peak_mb', 0):8.1f} MB"
        )

        print("\nGRAPH DATA:")
        print(f"   - Total nodes:          {self.metrics.get('num_nodes', 0):8,}")
        print(
            f"   - Total edges:          {self.metrics.get('num_edges_processed', 0):8,}"
        )
        print(
            f"   - Nodes with danger:    {self.metrics.get('nodes_with_danger', 0):8,}"
        )
        print(
            f"   - Max danger:           {self.metrics.get('max_danger', 0):8,} crimes"
        )

        if hasattr(self, "route_metrics"):
            print("\nROUTE STATISTICS:")
            for mode, metrics in self.route_metrics.items():
                print(f"\n   Mode: {mode.upper()}")
                print(
                    f"   - Nodes in route:       {metrics.get('route_nodes_count', 0):8,}"
                )
                print(
                    f"   - Distance:             {metrics.get('route_distance_km', 0):8.2f} km"
                )
                print(f"   - Total cost:           {metrics.get('route_cost', 0):8.3f}")
                print(
                    f"   - Total danger:         {metrics.get('route_total_danger', 0):8.1f}"
                )
                print(
                    f"   - Avg danger/node:      {metrics.get('route_avg_danger_per_node', 0):8.2f}"
                )

        print("=" * 70 + "\n")

    def evaluate_batch(
        self,
        pairs: list[tuple[str, tuple[float, float], tuple[float, float]]],
        modes: Optional[list[Mode]] = None,
        output_csv: str = "routes_comparison.csv",
        output_dir: str = ".",
        save_routes: bool = True,
    ) -> pd.DataFrame:
        if modes is None:
            modes = ["fast", "balanced", "safe"]
        single_pair = len(pairs) == 1
        rows = []
        for pair_name, start, end in pairs:
            print(f"\n{'=' * 60}\nPair: {pair_name}")
            # Pre-resolve O-D nodes once; _find_closest_node caches by (lat, lon)
            self._find_closest_node(*start)
            self._find_closest_node(*end)
            for mode in modes:
                mode_label = f"alpha_{mode:.2f}" if isinstance(mode, float) else mode
                result = self.find_route(
                    start, end, mode, label=f"{pair_name}_{mode_label}"
                )
                if not result.get("success"):
                    print(f"  [{mode}] No route: {result.get('error')}")
                    continue
                if save_routes:
                    prefix = "" if single_pair else f"{pair_name}_"
                    self.save_route_json(
                        result, os.path.join(output_dir, f"{prefix}route_{mode}.json")
                    )
                    self.save_route_geojson(
                        result,
                        os.path.join(output_dir, f"{prefix}route_{mode}.geojson"),
                    )
                m = result["metrics"]
                dist_km = result["real_distance_km"]
                rows.append(
                    {
                        "pair_name": pair_name,
                        "mode": mode_label,
                        "alpha": m["alpha"],
                        "distance_km": round(dist_km, 3),
                        "total_danger": result["total_danger"],
                        "danger_per_km": round(
                            result["total_danger"] / max(dist_km, 1e-6), 2
                        ),
                        "avg_danger_node": round(m["route_avg_danger_per_node"], 2),
                        "nodes": result["nodes_count"],
                        "dijkstra_time_s": round(m["time_dijkstra"], 4),
                        "weight_time_s": round(m["time_weight_calculation"], 4),
                        "route_objective": round(result["total_cost"], 4),
                    }
                )
        df = pd.DataFrame(rows)
        csv_path = os.path.join(output_dir, output_csv)
        df.to_csv(csv_path, index=False)
        print(f"\nTable saved: {csv_path}")
        print(df.to_string(index=False))
        return df


def find(
    city_graph: CityGraph,
    danger_file: str,
    start: tuple[float, float],
    end: tuple[float, float],
    mode: Mode = "balanced",
    save_files: bool = True,
    metrics_output: str = "saferoute_metrics.json",
    time_cpu_baseline: Optional[float] = None,
    output_dir: str = ".",
):
    """
    Find a route and return complete performance metrics.

    Args:
        city_graph: Already-loaded road network (CityGraph)
        danger_file: CSV with per-node danger scores
        start: (lat, lon) origin coordinates
        end: (lat, lon) destination coordinates
        mode: Route mode ("safe", "fast", "balanced")
        save_files: Whether to save the route as JSON and GeoJSON
        metrics_output: File to save metrics to
        time_cpu_baseline: CPU baseline time, to compute speedup
    """
    sr = SafeRouteGPU()
    sr.load_network(city_graph)
    sr.load_danger(danger_file)

    result = sr.find_route(start, end, mode)

    if save_files and result.get("success"):
        sr.save_route_json(result, os.path.join(output_dir, f"route_{mode}.json"))
        sr.save_route_geojson(result, os.path.join(output_dir, f"route_{mode}.geojson"))
        print(f"\nFiles saved for mode '{mode}'")

    if time_cpu_baseline:
        sr.metrics["time_cpu_baseline"] = time_cpu_baseline
        mode_key = mode if isinstance(mode, str) else f"alpha={mode:.2f}"
        if mode_key in sr.route_metrics:
            sr.route_metrics[mode_key]["speedup"] = (
                time_cpu_baseline
                / sr.route_metrics[mode_key]["time_total_route_finding"]
            )

    sr.save_metrics(os.path.join(output_dir, metrics_output))

    return result, sr.metrics


if __name__ == "__main__":
    sr = SafeRouteGPU()
    sr.load_network(CityGraph("manhattan_streets.csv"))
    sr.load_danger("nodes_with_crimes.csv")

    pairs = [
        # (name,                      start (lat, lon),        end (lat, lon))
        ("Times_Sq_to_Grand_Central", (40.758, -73.985), (40.752, -73.977)),  # ~1.5 km
        ("Wall_St_to_Times_Sq", (40.707, -74.011), (40.758, -73.985)),  # ~5 km
        ("Battery_to_Harlem", (40.700, -74.016), (40.810, -73.945)),  # ~12 km
    ]

    df = sr.evaluate_batch(pairs, output_csv="routes_comparison.csv")
    sr.save_metrics("saferoute_metrics.json")
