# saferoute_gpu.py - VERSIÓN FINAL COMPLETA
import cudf
import cupy as cp
import numpy as np
import networkx as nx
import json
import geopandas as gpd
from shapely.geometry import LineString
from typing import Tuple, Literal, Dict, Any, List
import time
import os

Mode = Literal["safe", "fast", "balanced"]

class SafeRouteGPU:
    def __init__(self):
        self.nodes_df = None
        self.edges_df = None
        self.danger_df = None
        self.node_id_to_idx = {}
        self.idx_to_node_id = {}
        self.max_danger = 1
        self.max_distance = 15000.0
        self.city_bb = {"min_lat": 41.6, "max_lat": 42.1, "min_lon": -87.9, "max_lon": -87.5}

    def load_network(self, edges_file: str):
        print(f"[SafeRouteGPU] Cargando red vial desde {edges_file}...")
        df = cudf.read_csv(edges_file)
        print(f"Edges cargados: {len(df):,}")

        # Mapeo nodos
        all_nodes_pd = cudf.concat([df['Node1_ID'], df['Node2_ID']]).unique().to_pandas()
        self.node_id_to_idx = {int(nid): i for i, nid in enumerate(all_nodes_pd)}
        self.idx_to_node_id = {v: k for k, v in self.node_id_to_idx.items()}
        print(f"Nodos únicos: {len(all_nodes_pd):,}")

        # Nodes con coordenadas
        nodes1 = df[['Node1_ID', 'Node1_Latitude', 'Node1_Longitude']].rename(
            columns={'Node1_ID': 'node_id', 'Node1_Latitude': 'lat', 'Node1_Longitude': 'lon'})
        nodes2 = df[['Node2_ID', 'Node2_Latitude', 'Node2_Longitude']].rename(
            columns={'Node2_ID': 'node_id', 'Node2_Latitude': 'lat', 'Node2_Longitude': 'lon'})
        
        all_nodes_df = cudf.concat([nodes1, nodes2]).drop_duplicates(subset=['node_id'])
        all_nodes_df = all_nodes_df.sort_values('node_id').reset_index(drop=True)
        
        local_idx_list = [self.node_id_to_idx[int(nid)] for nid in all_nodes_df['node_id'].to_pandas()]
        self.nodes_df = all_nodes_df.assign(local_idx=local_idx_list)

        # Edges con distancias GPU
        src_local = df['Node1_ID'].map(self.node_id_to_idx).astype('int32')
        dst_local = df['Node2_ID'].map(self.node_id_to_idx).astype('int32')

        lat1 = cp.asarray(df['Node1_Latitude'])
        lon1 = cp.asarray(df['Node1_Longitude'])
        lat2 = cp.asarray(df['Node2_Latitude'])
        lon2 = cp.asarray(df['Node2_Longitude'])
        distances = self._haversine_gpu(lat1, lon1, lat2, lon2)

        self.edges_df = cudf.DataFrame({
            'src': src_local,
            'dst': dst_local,
            'distance': distances
        })
        print(f"Edges procesados: {len(self.edges_df):,}")

    def load_danger(self, danger_file: str):
        print(f"[SafeRouteGPU] Cargando peligrosidad desde {danger_file}...")
        danger_df = cudf.read_csv(danger_file)
        
        danger_df['NodeID'] = danger_df['NodeID'].astype('int64')
        self.nodes_df['node_id'] = self.nodes_df['node_id'].astype('int64')
        
        merged = self.nodes_df.merge(
            danger_df[['NodeID', 'Weight']], 
            left_on='node_id', 
            right_on='NodeID', 
            how='left'
        )
        merged['Weight'] = merged['Weight'].fillna(0).astype('int32')
        self.danger_df = merged
        
        self.max_danger = int(merged['Weight'].max())
        print(f"Max peligro: {self.max_danger:,} crímenes")

    def _haversine_gpu(self, lat1, lon1, lat2, lon2):
        R = 6371000.0
        phi1 = cp.deg2rad(lat1)
        phi2 = cp.deg2rad(lat2)
        dphi = cp.deg2rad(lat2 - lat1)
        dlambda = cp.deg2rad(lon2 - lon1)
        a = (cp.sin(dphi/2)**2 + cp.cos(phi1) * cp.cos(phi2) * cp.sin(dlambda/2)**2)
        c = 2 * cp.arctan2(cp.sqrt(a), cp.sqrt(1-a))
        return R * c

    def _haversine_cpu(self, lat1, lon1, lat2, lon2):
        R = 6371000.0
        phi1, phi2 = np.radians([lat1, lat2])
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def _find_closest_node(self, lat: float, lon: float) -> Dict[str, Any]:
        nodes_pd = self.nodes_df.to_pandas()
        dists = np.array([
            self._haversine_cpu(row['lat'], row['lon'], lat, lon)
            for _, row in nodes_pd.iterrows()
        ])
        
        closest_idx = np.argmin(dists)
        min_dist = dists[closest_idx]
        closest_row = nodes_pd.iloc[closest_idx]
        
        closest_node = int(closest_row['node_id'])
        closest_lat, closest_lon = closest_row['lat'], closest_row['lon']
        
        print(f"Nodo: {closest_node:,} ({closest_lat:.4f}, {closest_lon:.4f}) | {min_dist:.0f}m")
        return {
            'node_id': closest_node,
            'local_idx': int(closest_row['local_idx']),
            'distance': min_dist,
            'lat': closest_lat,
            'lon': closest_lon
        }

    def find_route(self, start: Tuple[float, float], end: Tuple[float, float], 
                   mode: Mode = "balanced") -> Dict[str, Any]:
        
        print(f"\nModo: {mode.upper()}")
        start_info = self._find_closest_node(*start)
        end_info = self._find_closest_node(*end)
        
        if start_info['node_id'] == end_info['node_id']:
            return {"error": "Start y End son el mismo nodo!"}

        start_idx, end_idx = start_info['local_idx'], end_info['local_idx']

        # Calcular pesos
        weights = {"safe": (0.9, 0.1), "fast": (0.1, 0.9), "balanced": (0.5, 0.5)}
        safety_w, dist_w = weights[mode]

        print(f"Calculando {len(self.edges_df):,} pesos...")
        
        edges_pd = self.edges_df.to_pandas()
        danger_levels = self.danger_df.set_index('local_idx')['Weight'].to_pandas()
        
        danger_src = np.array([danger_levels.get(int(src), 0) for src in edges_pd['src']])
        danger_dst = np.array([danger_levels.get(int(dst), 0) for dst in edges_pd['dst']])
        distances = edges_pd['distance'].values
        
        avg_danger = (danger_src + danger_dst) / 2
        norm_danger = avg_danger / max(1, self.max_danger)
        norm_dist = distances / self.max_distance
        
        edge_weights = safety_w * norm_danger + dist_w * norm_dist

        # Dijkstra con NetworkX
        print(f"Dijkstra {start_idx} → {end_idx}")
        G_nx = nx.Graph()
        
        for src, dst, weight in zip(edges_pd['src'], edges_pd['dst'], edge_weights):
            G_nx.add_edge(int(src), int(dst), weight=float(weight))
        
        start_time = time.time()
        try:
            path = nx.shortest_path(G_nx, start_idx, end_idx, weight='weight')
            total_cost = nx.shortest_path_length(G_nx, start_idx, end_idx, weight='weight')
            dijkstra_time = time.time() - start_time
        except nx.NetworkXNoPath:
            return {"error": "No hay ruta entre start y end"}

        # Coordenadas de la ruta
        route_coords = []
        nodes_pd = self.nodes_df.to_pandas()
        real_distance = 0
        
        for i, local_idx in enumerate(path):
            node_row = nodes_pd[nodes_pd['local_idx'] == local_idx].iloc[0]
            route_coords.append({
                'osm_id': int(node_row['node_id']),
                'local_idx': int(local_idx),
                'lat': float(node_row['lat']),
                'lon': float(node_row['lon'])
            })
            
            if i > 0:
                prev_row = nodes_pd[nodes_pd['local_idx'] == path[i-1]].iloc[0]
                real_distance += self._haversine_cpu(
                    prev_row['lat'], prev_row['lon'], 
                    node_row['lat'], node_row['lon']
                )

        path_osm = [self.idx_to_node_id[i] for i in path]
        
        print(f"Ruta: {len(path)} nodos | Costo: {total_cost:.3f} | {real_distance/1000:.1f}km | {dijkstra_time:.2f}s")
        
        return {
            "mode": mode,
            "success": True,
            "path_osm_ids": path_osm,
            "path_local_idx": path,
            "total_cost": total_cost,
            "real_distance_km": real_distance / 1000,
            "nodes_count": len(path),
            "start_node": start_info['node_id'],
            "end_node": end_info['node_id'],
            "dijkstra_time": dijkstra_time,
            "route_coords": route_coords
        }

    def save_route_json(self, result: Dict[str, Any], filename: str = None):
        if filename is None:
            filename = f"route_{result['mode']}.json"
            
        simple_route = {
            "mode": result["mode"],
            "success": result["success"],
            "nodes_count": result["nodes_count"],
            "distance_km": round(result["real_distance_km"], 2),
            "safety_cost": round(result["total_cost"], 3),
            "computation_time_s": round(result["dijkstra_time"], 2),
            "start_node": result["start_node"],
            "end_node": result["end_node"],
            "route": result["route_coords"]
        }
        
        with open(filename, 'w') as f:
            json.dump(simple_route, f, indent=2)
        print(f"JSON guardado: {filename}")

    def save_route_geojson(self, result: Dict[str, Any], filename: str = None):
        try:
            if filename is None:
                filename = f"route_{result['mode']}.geojson"
                
            coords = [(point['lon'], point['lat']) for point in result["route_coords"]]
            line = LineString(coords)
            
            import geopandas as gpd
            gdf = gpd.GeoDataFrame({
                'mode': [result['mode']],
                'distance_km': [result['real_distance_km']],
                'safety_cost': [result['total_cost']],
                'nodes_count': [result['nodes_count']]
            }, geometry=[line], crs="EPSG:4326")
            
            gdf.to_file(filename, driver='GeoJSON')
            print(f"GeoJSON guardado: {filename}")
        except ImportError:
            print("geopandas no instalado. Solo JSON.")


def find(edges_file: str, danger_file: str, 
         start: Tuple[float, float], end: Tuple[float, float], 
         mode: Mode = "balanced", save_files: bool = True):
    sr = SafeRouteGPU()
    sr.load_network(edges_file)
    sr.load_danger(danger_file)
    
    result = sr.find_route(start, end, mode)
    
    if save_files and result.get("success"):
        sr.save_route_json(result)
        sr.save_route_geojson(result)
        print(f"\nArchivos guardados para modo '{mode}'")
    
    return result


if __name__ == "__main__":
    result = find(
        edges_file="chicago_streets.csv",
        danger_file="nodes_with_crimes_FULL.csv",
        start=(41.8781, -87.6298),  # The Loop
        end=(41.9485, -87.6553),    # Wrigley Field
        mode="safe",
        save_files=True
    )
    
    if result.get("success"):
        print("\n" + "="*60)
        print("RUTA SEGURA ENCONTRADA Y GUARDADA")
        print("="*60)
        print(f"Nodos: {result['nodes_count']}")
        print(f"Distancia: {result['real_distance_km']:.1f} km")
        print(f"Costo seguridad: {result['total_cost']:.3f}")
        print(f"Tiempo: {result['dijkstra_time']:.2f}s")
        print(f"Archivos: route_safe.json + route_safe.geojson")
    else:
        print(f"Error: {result.get('error', 'Desconocido')}")