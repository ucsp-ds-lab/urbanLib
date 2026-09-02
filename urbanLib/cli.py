"""
urbanLib CLI

The road network for --city is fetched from OSM automatically the first
time it's needed, and reused from disk on later runs -- no separate step
required.

Usage:
  urbanlib run    --city "Manhattan, New York, USA" --operator hotspot --crimes data/crimes.csv
  urbanlib run    --city "Manhattan, New York, USA" --operator amenity --amenities data/amenities.csv
  urbanlib run    --city "Manhattan, New York, USA" --operator saferoute --start 40.758,-73.985 --end 40.700,-74.016
  urbanlib run    --city "Manhattan, New York, USA" --operator all --crimes data/crimes.csv --amenities data/amenities.csv
  urbanlib visualize --city "Manhattan, New York, USA" --operator hotspot,amenity,routes

  urbanlib fetch  --city "Manhattan, New York, USA"   # optional: force a re-fetch, or pick --network-type

  # Add --save-metrics to any `run` above to also write a *_metrics.json
  # (timings, RAM/VRAM usage, snapping stats) per operator. Off by default.
"""

import argparse
import os
import sys


def _output_dir(city: str) -> str:
    path = os.path.join("output", city)
    os.makedirs(path, exist_ok=True)
    return path


def _data_dir(city: str) -> str:
    return os.path.join("data", city)


def _streets_file(city: str) -> str:
    return os.path.join(_data_dir(city), "streets.csv")


def _city_slug(city_name: str) -> str:
    return city_name.split(",")[0].strip().lower().replace(" ", "_")


def _infer_city_center(slug: str):
    """Best-effort map center for a city: the average coordinate of its
    cached road network, or a fallback if nothing is cached yet."""
    path = _streets_file(slug)
    if os.path.exists(path):
        import csv
        lats, lons = [], []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                lats.append(float(row["Node1_Latitude"]))
                lons.append(float(row["Node1_Longitude"]))
        if lats:
            return (sum(lats) / len(lats), sum(lons) / len(lons))
    return (40.754, -73.984)  # Manhattan, used only if nothing is cached


def _ensure_streets_file(city_name: str, network_type: str = "drive"):
    """Return (slug, streets_path) for city_name, fetching the road network
    from OSM if it isn't already cached on disk."""
    slug = _city_slug(city_name)
    path = _streets_file(slug)
    if not os.path.exists(path):
        from urbanLib.network import get_city_streets

        os.makedirs(_data_dir(slug), exist_ok=True)
        try:
            get_city_streets(city_name, output_csv=path, network_type=network_type)
        except Exception as e:
            print(f"Error: could not fetch road network for '{city_name}': {e}")
            sys.exit(1)
    return slug, path


def _parse_modes(raw: str, presets):
    """Parse a comma-separated --modes value into a list of preset names
    and/or float alphas, exiting cleanly on an invalid token."""
    modes = []
    for token in (m.strip() for m in raw.split(",")):
        if token in presets:
            modes.append(token)
            continue
        try:
            alpha = float(token)
        except ValueError:
            alpha = None
        if alpha is None or not 0.0 <= alpha <= 1.0:
            print(
                f"Error: invalid --modes value '{token}' "
                f"(expected one of {', '.join(presets)}, or a float in [0, 1])"
            )
            sys.exit(1)
        modes.append(alpha)
    return modes


def cmd_fetch(args):
    from urbanLib.network import get_city_streets

    slug = _city_slug(args.city)
    os.makedirs(_data_dir(slug), exist_ok=True)
    get_city_streets(
        args.city, output_csv=_streets_file(slug), network_type=args.network_type
    )


def cmd_run(args):
    operators = [o.strip() for o in args.operator.split(",")]
    if "all" in operators:
        operators = ["hotspot", "amenity", "saferoute"]

    slug, streets = _ensure_streets_file(args.city, args.network_type)
    out_dir = _output_dir(slug)

    # Load the road network once and share it across whichever operators
    # run in this invocation.
    from urbanLib.citygraph import CityGraph

    city_graph = CityGraph(streets)

    if "hotspot" in operators:
        if not args.crimes:
            print("Error: --crimes required for hotspot operator")
            sys.exit(1)
        from urbanLib.operators.hotspot import gpu_hotspot_complete

        gpu_hotspot_complete(
            city_graph=city_graph,
            crime_file=args.crimes,
            max_snap_distance_m=args.max_snap_distance,
            output_dir=out_dir,
            save_metrics=args.save_metrics,
        )

    if "amenity" in operators:
        if not args.amenities:
            print("Error: --amenities required for amenity operator")
            sys.exit(1)
        from urbanLib.operators.amenity import snap_amenities_rapids

        snap_amenities_rapids(
            city_graph=city_graph,
            amenity_file=args.amenities,
            max_snap_distance_m=args.max_snap_distance,
            output_dir=out_dir,
            save_metrics=args.save_metrics,
        )

    if "saferoute" in operators:
        if not args.start or not args.end:
            print("Error: --start and --end required for saferoute operator")
            sys.exit(1)
        danger_file = os.path.join(out_dir, "nodes_with_crimes.csv")
        if not os.path.exists(danger_file):
            print(f"Error: {danger_file} not found. Run the hotspot operator first.")
            sys.exit(1)

        start = tuple(float(x) for x in args.start.split(","))
        end = tuple(float(x) for x in args.end.split(","))

        from urbanLib.operators.saferoute import SafeRouteGPU, PRESETS

        modes = _parse_modes(args.modes, PRESETS)

        sr = SafeRouteGPU()
        sr.load_network(city_graph)
        sr.load_danger(danger_file)
        pairs = [(f"pair_0", start, end)]
        sr.evaluate_batch(pairs, modes=modes, output_dir=out_dir)
        if args.save_metrics:
            sr.save_metrics(os.path.join(out_dir, "saferoute_metrics.json"))


def cmd_visualize(args):
    slug = _city_slug(args.city)
    out_dir = _output_dir(slug)
    operators = [o.strip() for o in args.operator.split(",")]
    if "all" in operators:
        operators = ["hotspot", "amenity", "routes"]

    # Default map center inferred from the city's cached road network;
    # overridden by --center if given.
    city_center = _infer_city_center(slug)
    if args.center:
        city_center = tuple(float(x) for x in args.center.split(","))

    from urbanLib.visualization.maps import (
        visualize_hotspots,
        visualize_amenities,
        visualize_routes,
    )

    if "hotspot" in operators:
        crimes_file = os.path.join(out_dir, "nodes_with_crimes.csv")
        visualize_hotspots(
            crimes_file=crimes_file,
            city_center=city_center,
            output_file=os.path.join(out_dir, "hotspots_map.html"),
        )

    if "amenity" in operators:
        amenities_file = os.path.join(out_dir, "nodes_with_amenities.csv")
        visualize_amenities(
            amenities_file=amenities_file,
            city_center=city_center,
            output_file=os.path.join(out_dir, "amenities_map.html"),
        )

    if "routes" in operators:
        visualize_routes(
            route_dir=out_dir,
            city_center=city_center,
            output_file=os.path.join(out_dir, "saferoute_comparison.html"),
        )


def main():
    parser = argparse.ArgumentParser(
        prog="urbanlib",
        description="UrbanLib — GPU-parallel urban analysis",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- fetch ---
    p_fetch = sub.add_parser("fetch", help="Download city road network from OSM")
    p_fetch.add_argument(
        "--city", required=True, help='OSM city name (e.g. "Manhattan, New York, USA")'
    )
    p_fetch.add_argument(
        "--network-type", default="drive", choices=["drive", "walk", "bike", "all"]
    )

    # --- run ---
    p_run = sub.add_parser("run", help="Run one or more analysis operators")
    p_run.add_argument(
        "--city",
        required=True,
        help='OSM city name (e.g. "Manhattan, New York, USA"), or an already-cached '
        "city slug. The road network is fetched automatically on first use.",
    )
    p_run.add_argument(
        "--network-type",
        default="drive",
        choices=["drive", "walk", "bike", "all"],
        help="Used only if the road network isn't already cached",
    )
    p_run.add_argument(
        "--operator",
        required=True,
        help="hotspot | amenity | saferoute | all (comma-separated)",
    )
    p_run.add_argument(
        "--crimes", default=None, help="Crimes CSV (required for hotspot)"
    )
    p_run.add_argument(
        "--amenities", default=None, help="Amenities CSV (required for amenity)"
    )
    p_run.add_argument(
        "--start", default=None, help="lat,lon start (required for saferoute)"
    )
    p_run.add_argument(
        "--end", default=None, help="lat,lon destination (required for saferoute)"
    )
    p_run.add_argument(
        "--modes",
        default="fast,balanced,safe",
        help="Route modes, comma-separated: fast|balanced|safe, or a float alpha in [0, 1]",
    )
    p_run.add_argument(
        "--max-snap-distance",
        type=float,
        default=None,
        help="Max snapping distance in meters (filters outliers)",
    )
    p_run.add_argument(
        "--save-metrics",
        action="store_true",
        help="Also save a *_metrics.json (timings, RAM/VRAM usage, snapping "
        "stats) for each operator run. Off by default.",
    )

    # --- visualize ---
    p_viz = sub.add_parser("visualize", help="Generate Folium maps from results")
    p_viz.add_argument(
        "--city", required=True, help="City name or slug (same as used with 'run')"
    )
    p_viz.add_argument(
        "--operator", default="all", help="hotspot | amenity | routes | all"
    )
    p_viz.add_argument("--center", default=None, help="lat,lon map center (optional)")

    args = parser.parse_args()

    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "visualize":
        cmd_visualize(args)


if __name__ == "__main__":
    main()
