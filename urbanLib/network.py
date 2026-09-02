"""
Network Module
Functions to download and process street network graphs from OpenStreetMap.
"""

import osmnx as ox
import networkx as nx
import pandas as pd


def download_city_network(city_name, network_type="drive"):
    """
    Download a city's street network graph from OpenStreetMap.

    Args:
        city_name (str): City name (e.g. "Arequipa, Peru" or "Chicago, USA")
        network_type (str): Network type to download
            - "drive": Street network for vehicles (default)
            - "walk": Pedestrian network
            - "bike": Cycling network
            - "all": All networks

    Returns:
        networkx.MultiDiGraph: The city graph with nodes and edges

    Example:
        >>> G = download_city_network("Miraflores, Lima, Peru")
        >>> print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")
    """
    print(f"Downloading street network for: {city_name}")
    print(f"Network type: {network_type}")

    try:
        G = ox.graph_from_place(city_name, network_type=network_type)

        print(f"  Download complete")
        print(f"  Nodes: {len(G.nodes())}")
        print(f"  Edges: {len(G.edges())}")

        return G

    except Exception as e:
        print(f"  Error downloading city: {e}")
        raise


def export_to_csv(graph, output_path="city_streets.csv"):
    """
    Export a NetworkX graph to CSV format.

    Output format:
    Node1_ID, Node1_Latitude, Node1_Longitude, Node2_ID, Node2_Latitude, Node2_Longitude

    Args:
        graph (networkx.Graph): NetworkX graph (from osmnx or another source)
        output_path (str): Output CSV file path

    Returns:
        str: Path to the generated CSV file

    Example:
        >>> G = download_city_network("Arequipa, Peru")
        >>> export_to_csv(G, "arequipa_edges.csv")
    """
    print(f"\nExporting graph to CSV: {output_path}")

    edges_data = []
    missing_coords = 0

    for u, v, key in graph.edges(keys=True):
        node1 = graph.nodes[u]
        node2 = graph.nodes[v]

        # Only keep edges where both endpoints have coordinates (y=lat, x=lon)
        if "y" in node1 and "x" in node1 and "y" in node2 and "x" in node2:
            edges_data.append(
                {
                    "Node1_ID": str(u),
                    "Node1_Latitude": node1["y"],
                    "Node1_Longitude": node1["x"],
                    "Node2_ID": str(v),
                    "Node2_Latitude": node2["y"],
                    "Node2_Longitude": node2["x"],
                }
            )
        else:
            missing_coords += 1

    if missing_coords > 0:
        print(f"   {missing_coords} edges without coordinates (skipped)")

    df = pd.DataFrame(edges_data)
    df.to_csv(output_path, index=False)

    print(f"  CSV file created successfully")
    print(f"  Edges exported: {len(edges_data)}")
    print(f"  Path: {output_path}")

    return output_path


def get_city_streets(city_name, output_csv="city_streets.csv", network_type="drive"):
    """
    Download and prepare a city's street data in a single step.

    This is the main entry point most users will call: it downloads the
    graph and exports it directly to CSV.

    Args:
        city_name (str): City name (e.g. "Arequipa, Peru")
        output_csv (str): Output CSV file path
        network_type (str): Network type ("drive", "walk", "bike", "all")

    Returns:
        dict: Process information with the following keys:
            - 'graph': The downloaded NetworkX graph
            - 'csv_path': Path to the generated CSV
            - 'nodes_count': Number of nodes
            - 'edges_count': Number of edges
            - 'city_name': City name

    Example:
        >>> info = get_city_streets("Arequipa, Peru")
        >>> print(f"City: {info['city_name']}")
        >>> print(f"CSV generated at: {info['csv_path']}")
    """
    print("=" * 60)
    print("UrbanCrimeLib: City Data Preparation")
    print("=" * 60)

    graph = download_city_network(city_name, network_type)
    csv_path = export_to_csv(graph, output_csv)

    info = {
        "graph": graph,
        "csv_path": csv_path,
        "nodes_count": len(graph.nodes()),
        "edges_count": len(graph.edges()),
        "city_name": city_name,
    }

    print("\n" + "=" * 60)
    print("City data preparation complete")
    print("=" * 60)
    print(f"City: {city_name}")
    print(f"Nodes: {info['nodes_count']}")
    print(f"Edges: {info['edges_count']}")
    print(f"CSV saved to: {csv_path}")
    print("\nYou can now run GPU-Hotspot:")
    print(f"  urbanlib run --city <slug> --operator hotspot --crimes <crimes.csv>")
    print("=" * 60)

    return info


def visualize_network(graph, save_path=None, figsize=(12, 12)):
    """
    Visualize the city graph (requires matplotlib).

    Args:
        graph (networkx.Graph): Graph to visualize
        save_path (str, optional): If given, saves the image to this path
        figsize (tuple): Figure size

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    try:
        fig, ax = ox.plot_graph(
            graph,
            figsize=figsize,
            node_size=0,
            edge_linewidth=0.5,
            show=False,
            close=False,
        )

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Visualization saved to: {save_path}")

        return fig

    except Exception as e:
        print(f"Warning: could not visualize graph: {e}")
        return None


if __name__ == "__main__":
    import sys

    city = sys.argv[1] if len(sys.argv) > 1 else "Manhattan, New York, USA"
    get_city_streets(city)
