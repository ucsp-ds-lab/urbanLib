import json
import os
import time

import folium
import pandas as pd
import psutil
from folium.plugins import MarkerCluster
from jinja2 import Environment, PackageLoader, select_autoescape

# autoescape=True means any value interpolated into a template is escaped
# by default unless explicitly marked safe. This protects against untrusted
# external data (e.g. OpenStreetMap category names) ending up as raw
# HTML/JS in a rendered popup.
_env = Environment(
    loader=PackageLoader("urbanLib.visualization", "templates"),
    autoescape=select_autoescape(["html"]),
)


def _render(template_name: str, macro_name: str, **kwargs) -> str:
    """Render one template macro and return the HTML as a plain string."""
    macro = getattr(_env.get_template(template_name).module, macro_name)
    return str(macro(**kwargs))


class VisualizationMetrics:
    def __init__(self):
        self.metrics = {}
        self.time_start_total = None
        self.process = psutil.Process()
        self.ram_before = self.process.memory_info().rss / 1024**2

    def start_timer(self):
        self.time_start_total = time.time()

    def save_metrics(self, filename: str = "visualization_metrics.json"):
        ram_after = self.process.memory_info().rss / 1024**2
        self.metrics["ram_peak_mb"] = float(ram_after)
        self.metrics["ram_used_mb"] = float(ram_after - self.ram_before)
        if self.time_start_total:
            self.metrics["time_total"] = time.time() - self.time_start_total
        with open(filename, "w") as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Visualization metrics saved: {filename}")
        self._print_summary()

    def _print_summary(self):
        print("\n" + "=" * 70)
        print("VISUALIZATION METRICS")
        print("=" * 70)
        print("\nGENERATION TIMES:")
        if "hotspots" in self.metrics:
            print(f"   Hotspot map:          {self.metrics['hotspots']['time']:8.3f} s")
        if "amenities" in self.metrics:
            print(
                f"   Amenities map:        {self.metrics['amenities']['time']:8.3f} s"
            )
        if "routes" in self.metrics:
            print(f"   Routes map:           {self.metrics['routes']['time']:8.3f} s")
        print(f"   TOTAL:                {self.metrics.get('time_total', 0):8.3f} s")
        print("\nMEMORY USAGE:")
        print(f"   RAM peak:             {self.metrics.get('ram_peak_mb', 0):8.1f} MB")
        print(f"   RAM used:             {self.metrics.get('ram_used_mb', 0):8.1f} MB")
        print("=" * 70 + "\n")


def _add_shared_styles(m: folium.Map):
    m.get_root().html.add_child(folium.Element(_render("_styles.html", "style")))


def _cluster_icon_js(priority_colors: list, value_key: str, css_class: str) -> str:
    """Build a MarkerCluster icon_create_function that:
    - colors a cluster bubble by the highest-priority color among its child
      markers, instead of Leaflet's default size-based coloring (which has
      no relation to our danger/density scale). Reads `fillColor`, a real
      Leaflet path option every CircleMarker already carries.
    - labels the bubble with the sum of `value_key` across its children
      (e.g. total crimes), instead of the raw count of grouped markers.
      `value_key` must be set on each marker's `.options` dict beforehand
      (Folium drops unknown constructor kwargs, so this is set after
      construction -- see the call sites in visualize_hotspots/amenities).

    priority_colors: fillColor hex values, from highest to lowest priority.
    """
    color_list_js = ", ".join(f"'{c}'" for c in priority_colors)
    return f"""
        function(cluster) {{
            var priority = [{color_list_js}];
            var markers = cluster.getAllChildMarkers();
            var best = priority.length;
            var total = 0;
            for (var i = 0; i < markers.length; i++) {{
                var idx = priority.indexOf(markers[i].options.fillColor);
                if (idx !== -1 && idx < best) {{ best = idx; }}
                total += markers[i].options.{value_key} || 0;
            }}
            var color = best < priority.length ? priority[best] : '#999999';
            return L.divIcon({{
                html: '<div class="{css_class}" style="background:' + color + '">'
                      + total.toLocaleString() + '</div>',
                className: 'ul-cluster-wrapper',
                iconSize: L.point(38, 38)
            }});
        }}
    """


def visualize_hotspots(
    crimes_file: str,
    city_center: tuple,
    output_file: str = "hotspots_map.html",
    viz_metrics: VisualizationMetrics = None,
):
    """Hotspots: fixed radius + colors by crime density."""
    print("BUILDING HOTSPOT MAP")
    time_start = time.time()

    crimes_df = pd.read_csv(crimes_file)
    dangerous_nodes = crimes_df[crimes_df["Weight"] > 0].copy()
    print(f"{len(dangerous_nodes):,} dangerous nodes found")

    dangerous_nodes = dangerous_nodes.sort_values("Weight", ascending=False)
    p90 = dangerous_nodes["Weight"].quantile(0.9)
    p75 = dangerous_nodes["Weight"].quantile(0.75)
    p50 = dangerous_nodes["Weight"].quantile(0.5)

    m = folium.Map(location=list(city_center), zoom_start=13, tiles="OpenStreetMap")
    _add_shared_styles(m)

    # Thousands of individual markers make the map slow to render and pan;
    # clustering groups nearby ones into a single number until you zoom in,
    # without losing per-node popups (unlike a heatmap).
    # Single-hue red ramp, validated with the dataviz skill's ordinal
    # checks (monotone lightness, single hue, light-end contrast >= 2:1).
    # Replaces the old yellow/orange/red/darkred scheme, which spanned an
    # 81 deg hue range (not a single-hue ramp) and had a yellow light end
    # at 1.05:1 contrast -- effectively invisible on a light basemap.
    cluster = MarkerCluster(
        name="Hotspots",
        icon_create_function=_cluster_icon_js(
            ["#67001f", "#b2182b", "#d6604d", "#e8875c"],
            "crimeWeight",
            "ul-cluster-badge",
        ),
    ).add_to(m)

    for _, row in dangerous_nodes.iterrows():
        weight = row["Weight"]
        if weight >= p90:
            fill_color = "#67001f"
        elif weight >= p75:
            fill_color = "#b2182b"
        elif weight >= p50:
            fill_color = "#d6604d"
        else:
            fill_color = "#e8875c"

        popup_html = _render(
            "hotspot_popup.html",
            "popup",
            node_id=int(row["NodeID"]),
            weight=int(weight),
            lat=row["Latitude"],
            lon=row["Longitude"],
        )
        marker = folium.CircleMarker(
            [row["Latitude"], row["Longitude"]],
            radius=7,
            popup=popup_html,
            tooltip=f"{weight} crimes",
            # A white outline (instead of a same-hue border) is what keeps
            # the marker visible regardless of the basemap color underneath
            # it -- a matching-tone border blends into busy/similarly
            # colored tiles.
            color="white",
            fillColor=fill_color,
            fillOpacity=0.9,
            weight=2,
            opacity=1,
        )
        marker.options["crimeWeight"] = int(weight)
        marker.add_to(cluster)

    top10 = dangerous_nodes.head(10)
    top_layer = folium.FeatureGroup(name="TOP 10 HOTSPOTS", show=True)
    for rank, (_, row) in enumerate(top10.iterrows(), 1):
        folium.CircleMarker(
            [row["Latitude"], row["Longitude"]],
            radius=10,
            popup=_render(
                "hotspot_popup.html", "top_popup", rank=rank, weight=int(row["Weight"])
            ),
            tooltip=f"#{rank} - {row['Weight']} crimes",
            color="black",
            fillColor="darkred",
            fillOpacity=0.9,
            weight=3,
        ).add_to(top_layer)
    m.add_child(top_layer)

    legend_html = _render(
        "legend.html",
        "render",
        title="DANGER SCALE",
        items=[
            {"color": "#67001f", "label": ">90%"},
            {"color": "#b2182b", "label": "75-90%"},
            {"color": "#d6604d", "label": "50-75%"},
            {"color": "#e8875c", "label": "<50%"},
        ],
    )
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)

    total_crimes = dangerous_nodes["Weight"].sum()
    stats_html = _render(
        "stat_panel.html",
        "render",
        title="STATISTICS",
        border_color="darkred",
        stats=[
            {"label": "Total crimes", "value": f"{total_crimes:,}"},
            {"label": "Active hotspots", "value": f"{len(dangerous_nodes):,}"},
        ],
    )
    m.get_root().html.add_child(folium.Element(stats_html))

    m.save(output_file)
    elapsed = time.time() - time_start
    file_size = os.path.getsize(output_file) / 1024
    print(f"{output_file} created ({elapsed:.2f}s, {file_size:.1f} KB)")

    if viz_metrics is not None:
        viz_metrics.metrics["hotspots"] = {
            "time": elapsed,
            "num_nodes": len(dangerous_nodes),
            "total_crimes": int(total_crimes),
            "file_size_kb": file_size,
        }
    return m


def visualize_amenities(
    amenities_file: str,
    city_center: tuple,
    output_file: str = "amenities_map.html",
    viz_metrics: VisualizationMetrics = None,
):
    """Amenities: colors by service density."""
    print("\nBUILDING AMENITIES MAP...")
    time_start = time.time()

    try:
        amenities_df = pd.read_csv(amenities_file)
        amenity_cols = [
            c
            for c in amenities_df.columns
            if c not in ["NodeID", "Latitude", "Longitude"]
        ]
        amenities_df["total_amenities"] = amenities_df[amenity_cols].sum(axis=1)
        active = amenities_df[amenities_df["total_amenities"] > 0].sort_values(
            "total_amenities", ascending=False
        )
        print(f"{len(active):,} nodes with services found")

        p90 = active["total_amenities"].quantile(0.9)
        p75 = active["total_amenities"].quantile(0.75)
        p50 = active["total_amenities"].quantile(0.5)

        m = folium.Map(location=list(city_center), zoom_start=13, tiles="OpenStreetMap")
        _add_shared_styles(m)

        # Single-hue green ramp, validated with the dataviz skill's
        # ordinal checks. The old lime/lightgreen/green/darkgreen scheme
        # had lime and lightgreen at almost identical lightness (0.866 vs
        # 0.868) -- two different density tiers reading as the same color.
        cluster = MarkerCluster(
            name="Amenities",
            icon_create_function=_cluster_icon_js(
                ["#00441b", "#1b7837", "#5aae61", "#78c26f"],
                "amenityTotal",
                "ul-cluster-badge",
            ),
        ).add_to(m)

        for _, row in active.iterrows():
            total = row["total_amenities"]
            if total >= p90:
                fill_color = "#00441b"
            elif total >= p75:
                fill_color = "#1b7837"
            elif total >= p50:
                fill_color = "#5aae61"
            else:
                fill_color = "#78c26f"

            # "category" comes from OpenStreetMap tags (external, untrusted
            # data) -- passed raw here since the template's autoescaping
            # handles the escaping.
            details = [
                {"category": c, "count": int(row[c])}
                for c in amenity_cols
                if row[c] > 0
            ]
            popup_html = _render(
                "amenity_popup.html", "popup", total=int(total), details=details
            )
            marker = folium.CircleMarker(
                [row["Latitude"], row["Longitude"]],
                radius=8,
                popup=popup_html,
                tooltip=f"{int(total)} services",
                color="white",
                fillColor=fill_color,
                fillOpacity=0.9,
                weight=2,
            )
            marker.options["amenityTotal"] = int(total)
            marker.add_to(cluster)

        top10 = active.head(10)
        top_layer = folium.FeatureGroup(name="TOP 10 ZONES", show=True)
        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            folium.CircleMarker(
                [row["Latitude"], row["Longitude"]],
                radius=12,
                popup=_render(
                    "amenity_popup.html",
                    "top_popup",
                    rank=rank,
                    total=int(row["total_amenities"]),
                ),
                tooltip=f"#{rank}",
                color="darkgreen",
                fillColor="gold",
                fillOpacity=0.9,
                weight=3,
            ).add_to(top_layer)
        m.add_child(top_layer)

        legend_html = _render(
            "legend.html",
            "render",
            title="SERVICES SCALE",
            items=[
                {"color": "#00441b", "label": ">90%"},
                {"color": "#1b7837", "label": "75-90%"},
                {"color": "#5aae61", "label": "50-75%"},
                {"color": "#78c26f", "label": "<50%"},
            ],
        )
        m.get_root().html.add_child(folium.Element(legend_html))

        total_services = active["total_amenities"].sum()
        stats_html = _render(
            "stat_panel.html",
            "render",
            title="SERVICES",
            border_color="darkgreen",
            stats=[
                {"label": "Total", "value": f"{total_services:,}"},
                {"label": "Zones", "value": f"{len(active):,}"},
            ],
        )
        m.get_root().html.add_child(folium.Element(stats_html))

        folium.LayerControl().add_to(m)
        m.save(output_file)
        elapsed = time.time() - time_start
        file_size = os.path.getsize(output_file) / 1024
        print(f"{output_file} created ({elapsed:.2f}s, {file_size:.1f} KB)")

        if viz_metrics is not None:
            viz_metrics.metrics["amenities"] = {
                "time": elapsed,
                "num_nodes": len(active),
                "total_services": int(total_services),
                "num_types": len(amenity_cols),
                "file_size_kb": file_size,
            }
        return m

    except Exception as e:
        print(f"Error amenities: {e}")
        return None


def visualize_routes(
    route_dir: str,
    city_center: tuple,
    output_file: str = "saferoute_comparison.html",
    viz_metrics: VisualizationMetrics = None,
    pair_prefix: str = "",
):
    """SafeRoute routes: draws the three modes on a comparison map."""
    print("\nBUILDING SAFEROUTE MAP...")
    time_start = time.time()

    p = f"{pair_prefix}_" if pair_prefix else ""
    route_configs = {
        f"{p}route_safe.json": {
            "color": "#388e3c",
            "label": "SAFE ROUTE",
            "weight": 8,
            "subtitle": "Priority: Safety",
        },
        f"{p}route_fast.json": {
            "color": "#1565c0",
            "label": "FAST ROUTE",
            "weight": 6,
            "subtitle": "Priority: Distance",
        },
        f"{p}route_balanced.json": {
            "color": "#e65100",
            "label": "BALANCED ROUTE",
            "weight": 7,
            "subtitle": "50/50 Safety / Distance",
        },
    }

    m = folium.Map(location=list(city_center), zoom_start=12, tiles="OpenStreetMap")
    _add_shared_styles(m)
    routes_found, total_nodes, route_stats, routes_present = 0, 0, [], []
    od_markers_added = False

    for filename, config in route_configs.items():
        path = os.path.join(route_dir, filename)
        try:
            with open(path) as f:
                data = json.load(f)
            coords = [(pt["lat"], pt["lon"]) for pt in data["route"]]
            popup_html = _render(
                "route_popup.html",
                "popup",
                label=config["label"],
                distance_km=data["distance_km"],
                danger_per_km=data.get("danger_per_km", "—"),
                total_danger=data.get("total_danger", "—"),
                computation_time_s=data["computation_time_s"],
                nodes_count=data["nodes_count"],
            )
            folium.PolyLine(
                coords,
                color=config["color"],
                weight=config["weight"],
                opacity=0.9,
                popup=popup_html,
            ).add_to(m)
            if not od_markers_added:
                folium.Marker(
                    coords[0],
                    popup="START",
                    icon=folium.Icon(color="green", icon="play", prefix="fa"),
                ).add_to(m)
                folium.Marker(
                    coords[-1],
                    popup="END",
                    icon=folium.Icon(color="red", icon="stop", prefix="fa"),
                ).add_to(m)
                od_markers_added = True
            routes_found += 1
            total_nodes += data["nodes_count"]
            routes_present.append(config)
            route_stats.append(
                {
                    "route": config["label"],
                    "color": config["color"],
                    "distance_km": round(data["distance_km"], 1),
                    "danger_per_km": round(
                        data.get(
                            "danger_per_km",
                            data.get("total_danger", 0)
                            / max(data["distance_km"], 1e-6),
                        ),
                        1,
                    ),
                    "nodes": data["nodes_count"],
                    "time_s": round(data["computation_time_s"], 3),
                }
            )
            print(
                f"{filename} loaded - {data['distance_km']:.1f} km | {data['nodes_count']} nodes"
            )
        except Exception as e:
            print(f"{filename} not found: {e}")

    if route_stats:
        best = {
            "distance_km": min(r["distance_km"] for r in route_stats),
            "danger_per_km": min(r["danger_per_km"] for r in route_stats),
            "nodes": min(r["nodes"] for r in route_stats),
            "time_s": min(r["time_s"] for r in route_stats),
        }
        table_html = _render("route_table.html", "render", rows=route_stats, best=best)
        m.get_root().html.add_child(folium.Element(table_html))

    legend_html = _render(
        "route_legend.html",
        "render",
        routes=[
            {"color": cfg["color"], "label": cfg["label"], "subtitle": cfg["subtitle"]}
            for cfg in route_configs.values()
        ],
    )
    m.get_root().html.add_child(folium.Element(legend_html))
    folium.LayerControl().add_to(m)
    m.save(output_file)

    elapsed = time.time() - time_start
    file_size = os.path.getsize(output_file) / 1024
    print(
        f"{output_file} created ({elapsed:.2f}s, {file_size:.1f} KB, {routes_found}/3 routes)"
    )

    if viz_metrics is not None:
        viz_metrics.metrics["routes"] = {
            "time": elapsed,
            "num_routes": routes_found,
            "total_nodes": total_nodes,
            "file_size_kb": file_size,
        }
    return m


def visualize_all(
    city_center: tuple,
    crimes_file: str = None,
    amenities_file: str = None,
    route_dir: str = ".",
    output_dir: str = ".",
    metrics_file: str = "visualization_metrics.json",
):
    """Generate all available maps and save metrics."""
    print("=== VISUALIZING OPERATORS ===\n")
    viz = VisualizationMetrics()
    viz.start_timer()

    if crimes_file:
        visualize_hotspots(
            crimes_file=crimes_file,
            city_center=city_center,
            output_file=os.path.join(output_dir, "hotspots_map.html"),
            viz_metrics=viz,
        )
    if amenities_file:
        visualize_amenities(
            amenities_file=amenities_file,
            city_center=city_center,
            output_file=os.path.join(output_dir, "amenities_map.html"),
            viz_metrics=viz,
        )
    visualize_routes(
        route_dir=route_dir,
        city_center=city_center,
        output_file=os.path.join(output_dir, "saferoute_comparison.html"),
        viz_metrics=viz,
    )

    viz.save_metrics(os.path.join(output_dir, metrics_file))


if __name__ == "__main__":
    visualize_all(
        city_center=(40.754, -73.984),
        crimes_file="nodes_with_crimes.csv",
        amenities_file="nodes_with_amenities.csv",
        route_dir=".",
        output_dir=".",
    )
