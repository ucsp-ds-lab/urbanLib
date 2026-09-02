"""
Snapping distance statistics shared by the EdgeNodeAssignment and
AmenityAnalysis operators.
"""

import numpy as np


def compute_snapping_metrics(node_ids, dists_m, max_snap_distance_m=None):
    """
    Args:
        node_ids: node IDs assigned by the kernel (-1 = unassigned)
        dists_m: snapping distances in meters
        max_snap_distance_m: if given, also flags outliers (distance above
            the threshold) and adds post-filter statistics

    Returns:
        (metrics: dict, is_outlier: np.ndarray[bool])
    """
    valid = node_ids != -1
    d = dists_m[valid]
    n_valid = int(valid.sum())

    def stat(fn, arr):
        return float(fn(arr)) if len(arr) else None

    def pct(count):
        return float(count / n_valid * 100) if n_valid else 0.0

    n_over_500m = int((d > 500).sum())
    n_over_1000m = int((d > 1000).sum())
    n_over_5000m = int((d > 5000).sum())

    metrics = {
        "n_total": len(node_ids),
        "n_valid": n_valid,
        "snapping_distance_mean_m": stat(np.mean, d),
        "snapping_distance_median_m": stat(np.median, d),
        "snapping_distance_p95_m": stat(lambda a: np.percentile(a, 95), d),
        "snapping_distance_p99_m": stat(lambda a: np.percentile(a, 99), d),
        "snapping_distance_max_m": stat(np.max, d),
        "snapping_distance_std_m": stat(np.std, d),
        "n_over_500m": n_over_500m,
        "n_over_1000m": n_over_1000m,
        "n_over_5000m": n_over_5000m,
        "pct_over_500m": pct(n_over_500m),
        "pct_over_1000m": pct(n_over_1000m),
        "pct_over_5000m": pct(n_over_5000m),
    }

    is_outlier = np.zeros(len(node_ids), dtype=bool)
    if max_snap_distance_m is not None:
        is_outlier = dists_m > max_snap_distance_m
        n_out = int(is_outlier.sum())
        metrics["filter_threshold_m"] = max_snap_distance_m
        metrics["n_outliers"] = n_out
        metrics["pct_outliers"] = pct(n_out)

        d_f = dists_m[valid & ~is_outlier]
        metrics["filtered_mean_m"] = stat(np.mean, d_f)
        metrics["filtered_median_m"] = stat(np.median, d_f)
        metrics["filtered_p95_m"] = stat(lambda a: np.percentile(a, 95), d_f)
        metrics["filtered_p99_m"] = stat(lambda a: np.percentile(a, 99), d_f)
        metrics["filtered_max_m"] = stat(np.max, d_f)
        metrics["filtered_std_m"] = stat(np.std, d_f)

    return metrics, is_outlier
