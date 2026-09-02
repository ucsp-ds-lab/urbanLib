"""
Shared fixtures for the urbanLib test suite.

The test road network is intentionally simple: 3 nodes on the equator
(lat=0), where 1 degree of longitude equals exactly 111320 m (cos(0)=1).
This makes it possible to compute the expected snapping distances by hand,
independently of the formula under test.

    Node 1 (0.0, 0.000) --- Edge A --- Node 2 (0.0, 0.001) --- Edge B --- Node 3 (0.0, 0.002)
"""
import csv

import pytest


@pytest.fixture
def small_network():
    """Raw coordinates for the 3-node / 2-edge network, without going through a CSV."""
    return {
        "e_x1": [0.000, 0.001],
        "e_y1": [0.0, 0.0],
        "e_x2": [0.001, 0.002],
        "e_y2": [0.0, 0.0],
        "e_n1": [1, 2],
        "e_n2": [2, 3],
    }


@pytest.fixture
def streets_csv(tmp_path, small_network):
    """The same 3-node / 2-edge network, written as a CSV in the format the
    operators expect (Node1_ID, Node1_Latitude, ...)."""
    path = tmp_path / "streets.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Node1_ID", "Node1_Latitude", "Node1_Longitude",
            "Node2_ID", "Node2_Latitude", "Node2_Longitude",
        ])
        n = small_network
        for i in range(len(n["e_n1"])):
            writer.writerow([
                n["e_n1"][i], n["e_y1"][i], n["e_x1"][i],
                n["e_n2"][i], n["e_y2"][i], n["e_x2"][i],
            ])
    return str(path)
