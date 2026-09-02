"""
Tests for the AmenityAnalysis operator (urbanLib/operators/amenity.py).

Require cudf/cupy.
"""
import pandas as pd
import pytest

pytest.importorskip("cudf")

from urbanLib.citygraph import CityGraph
from urbanLib.operators.amenity import snap_amenities_rapids


def test_amenity_counts_per_node_and_type(tmp_path, streets_csv):
    amenities = pd.DataFrame({
        "Latitude": [0.0, 0.0],
        "Longitude": [0.0002, 0.0002],  # both snap to Node1
        "amenity": ["bar", "hospital"],
    })
    amenities_csv = tmp_path / "amenities.csv"
    amenities.to_csv(amenities_csv, index=False)

    city_graph = CityGraph(streets_csv)
    result_df, metrics = snap_amenities_rapids(
        city_graph=city_graph,
        amenity_file=str(amenities_csv),
        output_dir=str(tmp_path),
    )

    row = result_df[result_df["NodeID"] == 1].iloc[0]
    assert row["bar"] == 1
    assert row["hospital"] == 1
    assert metrics["num_amenity_types"] == 2
