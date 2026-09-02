"""
Sequential, single-threaded CPU baseline (no vectorization, no GPU) for the
point-segment snapping algorithm. Replicates the exact same projection math
as the CUDA kernel implementation, so that GPU-vs-CPU speedup measurements
reflect only "one thread" vs "thousands of parallel threads" -- not a
different algorithm.
"""

import math
import time

import pandas as pd


def snap_points_cpu(e_x1, e_y1, e_x2, e_y2, e_n1, e_n2, p_x, p_y, progress_every=2000):
    """Brute-force O(N*M) snapping in pure Python. Returns (node_ids, dists_m, elapsed_s)."""
    N = len(p_x)
    M = len(e_x1)
    node_ids = [-1] * N
    dists_m = [0.0] * N

    t0 = time.time()
    for i in range(N):
        px = p_x[i]
        py = p_y[i]

        best_dist2 = 1e30
        best_node = -1
        best_proj_x = 0.0
        best_proj_y = 0.0

        for j in range(M):
            x1 = e_x1[j]
            y1 = e_y1[j]
            x2 = e_x2[j]
            y2 = e_y2[j]

            dx = x2 - x1
            dy = y2 - y1
            len2 = dx * dx + dy * dy
            if len2 < 1e-8:
                continue

            t = ((px - x1) * dx + (py - y1) * dy) / len2
            if t < 0.0:
                t = 0.0
            elif t > 1.0:
                t = 1.0

            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist2 = (px - proj_x) ** 2 + (py - proj_y) ** 2

            if dist2 < best_dist2:
                best_dist2 = dist2
                best_proj_x = proj_x
                best_proj_y = proj_y
                d1 = (proj_x - x1) ** 2 + (proj_y - y1) ** 2
                d2 = (proj_x - x2) ** 2 + (proj_y - y2) ** 2
                best_node = e_n1[j] if d1 <= d2 else e_n2[j]

        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(py))
        dlon = px - best_proj_x
        dlat = py - best_proj_y
        dist_m = math.sqrt(
            (dlon * meters_per_deg_lon) ** 2 + (dlat * meters_per_deg_lat) ** 2
        )

        node_ids[i] = best_node
        dists_m[i] = dist_m

        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - (i + 1)) / rate if rate > 0 else float("nan")
            print(f"  CPU {i + 1:,}/{N:,} points ({rate:.1f} pts/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    return node_ids, dists_m, elapsed


def run_cpu_baseline(
    edges_file,
    points_file,
    lon_col="Longitude",
    lat_col="Latitude",
    progress_every=2000,
):
    """Load edges and points with pandas, then run the brute-force snapping baseline. Returns (node_ids, dists_m, elapsed_s)."""
    edges_df = pd.read_csv(edges_file)
    points_df = pd.read_csv(points_file)

    e_x1 = edges_df["Node1_Longitude"].tolist()
    e_y1 = edges_df["Node1_Latitude"].tolist()
    e_x2 = edges_df["Node2_Longitude"].tolist()
    e_y2 = edges_df["Node2_Latitude"].tolist()
    e_n1 = edges_df["Node1_ID"].tolist()
    e_n2 = edges_df["Node2_ID"].tolist()

    p_x = points_df[lon_col].tolist()
    p_y = points_df[lat_col].tolist()

    print(
        f"CPU baseline: {len(p_x):,} points x {len(e_x1):,} edges "
        f"= {len(p_x) * len(e_x1):,} comparisons"
    )

    return snap_points_cpu(
        e_x1, e_y1, e_x2, e_y2, e_n1, e_n2, p_x, p_y, progress_every=progress_every
    )
