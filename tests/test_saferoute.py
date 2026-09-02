"""
Tests for the SafestShortestPathFinder operator
(urbanLib/operators/saferoute.py).

Require cudf/cupy.
"""
import pandas as pd
import pytest

pytest.importorskip("cudf")

from urbanLib.citygraph import CityGraph
from urbanLib.operators.saferoute import SafeRouteGPU


def test_find_route_between_endpoints(tmp_path, streets_csv):
    danger = pd.DataFrame({"NodeID": [1, 2, 3], "Weight": [0, 5, 0]})
    danger_csv = tmp_path / "danger.csv"
    danger.to_csv(danger_csv, index=False)

    sr = SafeRouteGPU()
    sr.load_network(CityGraph(streets_csv))
    sr.load_danger(str(danger_csv))

    result = sr.find_route(start=(0.0, 0.0), end=(0.0, 0.002), mode="fast")

    assert result["success"] is True
    assert result["start_node"] == 1
    assert result["end_node"] == 3
    assert result["nodes_count"] == 3  # goes through Node1 -> Node2 -> Node3
