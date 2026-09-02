"""
Tests for the EdgeNodeAssignment operator (urbanLib/operators/hotspot.py).

Require cudf/cupy. Uses the same 3-node / 2-edge network 
and points as the CPU baseline tests, so the CUDA kernel's
output can be compared directly against the brute-force baseline (same
projection math, different execution model).
"""
import pandas as pd
import pytest

pytest.importorskip("cudf")

from urbanLib.citygraph import CityGraph
from urbanLib.operators.hotspot import gpu_hotspot_complete


def test_hotspot_matches_cpu_baseline_assignment(tmp_path, streets_csv):
    crimes = pd.DataFrame({
        "Latitude": [0.0, 0.0],
        "Longitude": [0.0002, 0.0019],  # same points as the CPU baseline tests: -> Node1, Node3
    })
    crimes_csv = tmp_path / "crimes.csv"
    crimes.to_csv(crimes_csv, index=False)

    city_graph = CityGraph(streets_csv)
    result_df, metrics = gpu_hotspot_complete(
        city_graph=city_graph,
        crime_file=str(crimes_csv),
        output_dir=str(tmp_path),
    )

    weights = dict(zip(result_df["NodeID"], result_df["Weight"]))
    assert weights[1] == 1
    assert weights[3] == 1
    assert weights.get(2, 0) == 0
    assert metrics["n_valid"] == 2
