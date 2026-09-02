"""
CityGraph Core

Loads the road network once and keeps it in GPU memory (cuDF) as a shared
resource across operators (EdgeNodeAssignment, AmenityAnalysis,
SafestShortestPathFinder).
"""

import cudf


class CityGraph:
    def __init__(self, edges_file: str):
        self.edges_file = edges_file
        self.edges_df = cudf.read_csv(edges_file)

    def __len__(self):
        return len(self.edges_df)

    def __repr__(self):
        return f"CityGraph({self.edges_file!r}, {len(self)} edges)"

    def unique_nodes(self) -> cudf.DataFrame:
        """Deduplicated NodeID/Latitude/Longitude table, built from the
        Node1_*/Node2_* columns every edge carries for both its endpoints."""
        df = self.edges_df
        nodes1 = df[["Node1_ID", "Node1_Latitude", "Node1_Longitude"]].rename(
            columns={
                "Node1_ID": "NodeID",
                "Node1_Latitude": "Latitude",
                "Node1_Longitude": "Longitude",
            }
        )
        nodes2 = df[["Node2_ID", "Node2_Latitude", "Node2_Longitude"]].rename(
            columns={
                "Node2_ID": "NodeID",
                "Node2_Latitude": "Latitude",
                "Node2_Longitude": "Longitude",
            }
        )
        return cudf.concat([nodes1, nodes2]).drop_duplicates("NodeID")
