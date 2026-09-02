"""
Smoke tests for visualize_hotspots and visualize_routes: render each map
end to end and check the output HTML for the expected content.
"""
import json

import pandas as pd

from urbanLib.visualization.maps import visualize_hotspots, visualize_routes


def test_visualize_hotspots_smoke(tmp_path):
    crimes = pd.DataFrame({
        "NodeID": [1, 2, 3],
        "Latitude": [40.70, 40.71, 40.72],
        "Longitude": [-74.00, -74.01, -74.02],
        "Weight": [10, 5, 0],
    })
    crimes_csv = tmp_path / "nodes_with_crimes.csv"
    crimes.to_csv(crimes_csv, index=False)

    output_html = tmp_path / "hotspots_map.html"
    visualize_hotspots(
        crimes_file=str(crimes_csv),
        city_center=(40.70, -74.00),
        output_file=str(output_html),
    )

    html = output_html.read_text()
    assert "HOTSPOT #1" in html
    assert "DANGER SCALE" in html
    assert "STATISTICS" in html
    # a node with Weight=0 should not be included on the map
    assert "HOTSPOT #3" not in html

    # Each marker's crime weight must be attached as a custom Leaflet
    # option, since that's what the cluster's icon_create_function sums to
    # label a cluster bubble (instead of just counting grouped nodes).
    assert '"crimeWeight": 10' in html
    assert '"crimeWeight": 5' in html
    # the cluster coloring/labeling function must reference that same key
    assert "markers[i].options.crimeWeight" in html


def test_visualize_routes_marks_correct_best_value(tmp_path):
    # safe: less danger/km but more distance; fast: the opposite.
    # the best value for "Dist. (km)" should be marked on FAST, not SAFE.
    routes = {
        "route_safe.json": {
            "mode": "safe", "distance_km": 20.0, "danger_per_km": 5.0,
            "nodes_count": 50, "computation_time_s": 0.05,
        },
        "route_fast.json": {
            "mode": "fast", "distance_km": 10.0, "danger_per_km": 50.0,
            "nodes_count": 30, "computation_time_s": 0.02,
        },
        "route_balanced.json": {
            "mode": "balanced", "distance_km": 15.0, "danger_per_km": 20.0,
            "nodes_count": 40, "computation_time_s": 0.03,
        },
    }
    for filename, data in routes.items():
        data = dict(data, route=[{"lat": 40.70, "lon": -74.00}, {"lat": 40.71, "lon": -74.01}])
        (tmp_path / filename).write_text(json.dumps(data))

    output_html = tmp_path / "saferoute_comparison.html"
    visualize_routes(
        route_dir=str(tmp_path),
        city_center=(40.70, -74.00),
        output_file=str(output_html),
    )

    html = output_html.read_text()
    assert "SAFEROUTE ROUTE COMPARISON" in html
    assert "ROUTE TYPES" in html
    # the real minimum of the Distance column is 10.00 km (FAST) -> it
    # should be marked with the "best value" asterisk
    assert "10.00 km*" in html
    assert "20.00 km*" not in html  # SAFE is not the best on distance
