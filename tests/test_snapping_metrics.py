"""
Tests for compute_snapping_metrics -- pure NumPy, no GPU required.

Pins down the exact keys and values these statistics must produce, since
they end up in the JSON metrics files used to report benchmark results.
"""
import numpy as np

from urbanLib.operators.snapping_metrics import compute_snapping_metrics


def test_basic_stats_without_filter():
    node_ids = np.array([1, 2, -1, 3])
    dists_m = np.array([10.0, 20.0, 999.0, 30.0])  # the -1 shouldn't count

    metrics, is_outlier = compute_snapping_metrics(node_ids, dists_m)

    assert metrics['n_total'] == 4
    assert metrics['n_valid'] == 3
    assert metrics['snapping_distance_mean_m'] == 20.0
    assert metrics['snapping_distance_median_m'] == 20.0
    assert metrics['snapping_distance_max_m'] == 30.0
    assert metrics['n_over_500m'] == 0
    assert 'filter_threshold_m' not in metrics
    assert not is_outlier.any()


def test_outlier_filtering():
    node_ids = np.array([1, 2, 3, 4])
    dists_m = np.array([10.0, 20.0, 600.0, 30.0])  # 600m exceeds the 500m threshold

    metrics, is_outlier = compute_snapping_metrics(node_ids, dists_m, max_snap_distance_m=500.0)

    assert metrics['n_outliers'] == 1
    assert metrics['pct_outliers'] == 25.0
    assert list(is_outlier) == [False, False, True, False]
    # the "filtered_*" stats exclude the outlier
    assert metrics['filtered_mean_m'] == 20.0
    assert metrics['filtered_max_m'] == 30.0


def test_no_valid_points_does_not_crash():
    node_ids = np.array([-1, -1, -1])
    dists_m = np.array([10.0, 20.0, 30.0])

    metrics, is_outlier = compute_snapping_metrics(node_ids, dists_m, max_snap_distance_m=500.0)

    assert metrics['n_valid'] == 0
    assert metrics['snapping_distance_mean_m'] is None
    assert metrics['snapping_distance_max_m'] is None
    assert metrics['pct_over_500m'] == 0.0
    assert metrics['pct_outliers'] == 0.0
    assert metrics['filtered_mean_m'] is None
    assert not is_outlier.any()
