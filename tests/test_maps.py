"""
Tests for urbanLib/visualization/maps.py.

test_amenity_category_names_are_escaped_in_html guards against untrusted
data ending up as raw HTML: OSM category names (amenity CSV columns) come
from external, uncontrolled data and must be escaped before being
interpolated into a Folium popup.
"""
import pandas as pd

from urbanLib.visualization.maps import visualize_amenities


def test_amenity_category_names_are_escaped_in_html(tmp_path):
    malicious_category = "<script>alert(1)</script>"
    df = pd.DataFrame({
        "NodeID": [1],
        "Latitude": [40.7],
        "Longitude": [-74.0],
        malicious_category: [3],
    })
    amenities_csv = tmp_path / "nodes_with_amenities.csv"
    df.to_csv(amenities_csv, index=False)

    output_html = tmp_path / "amenities_map.html"
    visualize_amenities(
        amenities_file=str(amenities_csv),
        city_center=(40.7, -74.0),
        output_file=str(output_html),
    )

    html = output_html.read_text()
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
