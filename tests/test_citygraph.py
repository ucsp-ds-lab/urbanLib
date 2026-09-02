"""
Tests for urbanLib/citygraph.py.

Require cudf.
"""
import pytest

pytest.importorskip("cudf")

from urbanLib.citygraph import CityGraph


def test_loads_edges_from_csv(streets_csv):
    graph = CityGraph(streets_csv)
    assert len(graph) == 2
    assert list(graph.edges_df.columns) == [
        "Node1_ID", "Node1_Latitude", "Node1_Longitude",
        "Node2_ID", "Node2_Latitude", "Node2_Longitude",
    ]


def test_repr_shows_path_and_edge_count(streets_csv):
    graph = CityGraph(streets_csv)
    r = repr(graph)
    assert streets_csv in r
    assert "2 edges" in r
