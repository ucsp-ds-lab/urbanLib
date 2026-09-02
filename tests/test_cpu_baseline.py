"""
Tests for snap_points_cpu (urbanLib/operators/cpu_baseline.py).

This function is a pure-Python brute-force baseline, so it's the only piece
of the snapping algorithm that can be tested without a GPU. It also serves
as a reference for what the equivalent parallel implementation should
return for the same inputs.

Uses the `small_network` fixture from conftest.py.
"""
import pytest

from urbanLib.operators.cpu_baseline import snap_points_cpu

M_PER_DEG = 111320.0  # exact m/degree of longitude at the equator (lat=0), see conftest.py


def test_point_on_edge_a_closer_to_node1(small_network):
    # Point on edge A's line, 20% of the way from Node1 to Node2
    # -> closer to Node1, snap distance ~0 (it lies exactly on the line).
    n = small_network
    node_ids, dists_m, _ = snap_points_cpu(
        n["e_x1"], n["e_y1"], n["e_x2"], n["e_y2"], n["e_n1"], n["e_n2"],
        p_x=[0.0002], p_y=[0.0], progress_every=0,
    )
    assert node_ids == [1]
    assert dists_m[0] == pytest.approx(0.0, abs=1e-6)


def test_point_on_edge_a_closer_to_node2(small_network):
    n = small_network
    node_ids, dists_m, _ = snap_points_cpu(
        n["e_x1"], n["e_y1"], n["e_x2"], n["e_y2"], n["e_n1"], n["e_n2"],
        p_x=[0.0009], p_y=[0.0], progress_every=0,
    )
    assert node_ids == [2]
    assert dists_m[0] == pytest.approx(0.0, abs=1e-6)


def test_point_off_axis_midpoint_ties_to_node1(small_network):
    # Point off the line, projecting exactly to edge A's midpoint (t=0.5)
    # -> d1 == d2, and both the kernel and the CPU baseline break ties with
    # "<=", so the result is Node1.
    n = small_network
    lat_offset = 0.0005  # degrees of latitude of perpendicular distance
    node_ids, dists_m, _ = snap_points_cpu(
        n["e_x1"], n["e_y1"], n["e_x2"], n["e_y2"], n["e_n1"], n["e_n2"],
        p_x=[0.0005], p_y=[lat_offset], progress_every=0,
    )
    expected_dist = lat_offset * M_PER_DEG
    assert node_ids == [1]
    assert dists_m[0] == pytest.approx(expected_dist, rel=1e-6)


def test_point_near_node3_picks_edge_b_not_edge_a(small_network):
    # Point close to Node3: the brute-force loop must scan every edge and
    # keep the global best (edge B), not just the first one it evaluates.
    n = small_network
    node_ids, dists_m, _ = snap_points_cpu(
        n["e_x1"], n["e_y1"], n["e_x2"], n["e_y2"], n["e_n1"], n["e_n2"],
        p_x=[0.0019], p_y=[0.0], progress_every=0,
    )
    assert node_ids == [3]
    assert dists_m[0] == pytest.approx(0.0, abs=1e-6)


def test_degenerate_edge_is_skipped():
    # An edge with zero length (len2 < 1e-8) must not blow up on division
    # by len2, nor be picked as the best candidate.
    node_ids, dists_m, _ = snap_points_cpu(
        e_x1=[0.0, 0.0], e_y1=[0.0, 0.0],
        e_x2=[0.0, 0.001], e_y2=[0.0, 0.0],  # edge 0 is degenerate (same point)
        e_n1=[1, 1], e_n2=[1, 2],
        p_x=[0.0005], p_y=[0.0], progress_every=0,
    )
    assert node_ids == [1]  # resolved via the valid edge (edge 1), not the degenerate one
