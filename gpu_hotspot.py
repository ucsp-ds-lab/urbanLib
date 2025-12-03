# gpu_hotspot.py
# VERSIÓN Batch + Kernel CuPy + Todos los nodos (incluye Weight=0)

import cudf
import cupy as cp
from cupy import RawKernel
import pandas as pd
import numpy as np
from collections import Counter

snap_kernel_code = """
extern "C" __global__
void snap_crimes_kernel(
    const float* crimes_x, const float* crimes_y,
    const float* edges_x1, const float* edges_y1,
    const float* edges_x2, const float* edges_y2,
    const long long* edge_node1_ids,   // Node1_ID como int64
    const long long* edge_node2_ids,   // Node2_ID como int64
    int N, int M,
    long long* out_node_ids            // Output: NodeID real asignado
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    float cx = crimes_x[idx];
    float cy = crimes_y[idx];
    
    float best_dist = 1e30f;  // Valor grande (en vez de INFINITY)
    long long best_node_id = -1;
    
    for (int j = 0; j < M; ++j) {
        float x1 = edges_x1[j];
        float y1 = edges_y1[j];
        float x2 = edges_x2[j];
        float y2 = edges_y2[j];
        
        float dx = x2 - x1;
        float dy = y2 - y1;
        float len2 = dx*dx + dy*dy;
        
        if (len2 < 1e-8f) continue;
        
        float t = ((cx - x1)*dx + (cy - y1)*dy) / len2;
        t = fmaxf(0.0f, fminf(1.0f, t));
        
        float proj_x = x1 + t * dx;
        float proj_y = y1 + t * dy;
        
        float dist = (cx - proj_x)*(cx - proj_x) + (cy - proj_y)*(cy - proj_y);
        
        if (dist < best_dist) {
            best_dist = dist;
            
            // Elegir nodo más cercano al punto proyectado
            float d1 = (proj_x - x1)*(proj_x - x1) + (proj_y - y1)*(proj_y - y1);
            float d2 = (proj_x - x2)*(proj_x - x2) + (proj_y - y2)*(proj_y - y2);
            
            best_node_id = (d1 <= d2) ? edge_node1_ids[j] : edge_node2_ids[j];
        }
    }
    
    out_node_ids[idx] = best_node_id;
}
"""

def gpu_hotspot_complete(
    edges_file: str,
    crime_file: str,
    output_file: str = "nodes_with_crimes.csv",
    batch_size: int = 4000
):
    print("=== GPU-HOTSPOT FINAL - 100% FUNCIONAL ===")
    
    # Cargar datos
    edges_df = cudf.read_csv(edges_file)
    crimes_df = cudf.read_csv(crime_file)
    
    print(f"Edges: {len(edges_df):,} | Crímenes: {len(crimes_df):,}")
    
    # Preparar arrays en GPU (solo una vez)
    e_x1 = edges_df['Node1_Longitude'].to_cupy().astype(cp.float32)
    e_y1 = edges_df['Node1_Latitude'].to_cupy().astype(cp.float32)
    e_x2 = edges_df['Node2_Longitude'].to_cupy().astype(cp.float32)
    e_y2 = edges_df['Node2_Latitude'].to_cupy().astype(cp.float32)
    
    # Convertir NodeID a int64 (OSM IDs son grandes)
    node1_ids = edges_df['Node1_ID'].astype('int64').to_cupy()
    node2_ids = edges_df['Node2_ID'].astype('int64').to_cupy()
    
    # Compilar kernel
    kernel = RawKernel(snap_kernel_code, 'snap_crimes_kernel')
    
    # Procesar por batches
    results = []
    N = len(crimes_df)
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        print(f"Procesando {start:,} → {end:,} ({end-start:,} crímenes)...")
        
        batch = crimes_df.iloc[start:end]
        cx = batch['Longitude'].to_cupy().astype(cp.float32)
        cy = batch['Latitude'].to_cupy().astype(cp.float32)
        
        out_ids = cp.zeros(end - start, dtype=cp.int64)
        
        blocks = (len(cx) + 255) // 256
        kernel((blocks,), (256,), (
            cx, cy,
            e_x1, e_y1, e_x2, e_y2,
            node1_ids, node2_ids,
            len(cx), len(e_x1),
            out_ids
        ))
        
        results.append(out_ids.get())
        
        # Limpiar memoria
        del cx, cy, out_ids
    
    # Concatenar todos los resultados
    all_node_ids = np.concatenate(results)
    print("Snapping completado. Contando...")
    
    # Conteo final
    counts = Counter(all_node_ids)
    
    # Todos los nodos (incluyendo Weight=0)
    nodes1 = edges_df[['Node1_ID', 'Node1_Latitude', 'Node1_Longitude']].rename(
        columns={'Node1_ID': 'NodeID', 'Node1_Latitude': 'Latitude', 'Node1_Longitude': 'Longitude'})
    nodes2 = edges_df[['Node2_ID', 'Node2_Latitude', 'Node2_Longitude']].rename(
        columns={'Node2_ID': 'NodeID', 'Node2_Latitude': 'Latitude', 'Node2_Longitude': 'Longitude'})
    
    all_nodes = pd.concat([nodes1.to_pandas(), nodes2.to_pandas()]).drop_duplicates('NodeID')
    all_nodes['NodeID'] = all_nodes['NodeID'].astype('int64')
    
    all_nodes['Weight'] = all_nodes['NodeID'].map(counts).fillna(0).astype(int)
    final = all_nodes.sort_values('Weight', ascending=False)
    final = final[['NodeID', 'Latitude', 'Longitude', 'Weight']]
    
    final.to_csv(output_file, index=False)
    
    print(f"\n¡ÉXITO! → {output_file}")
    print(f"   Total nodos: {len(final):,}")
    print(f"   Con crímenes: {(final['Weight'] > 0).sum():,}")
    print(f"   Sin crímenes: {(final['Weight'] == 0).sum():,}")
    
    print("\nTop 10 hotspots:")
    for _, row in final.head(10).iterrows():
        print(f"   • {int(row['NodeID'])} ({row['Latitude']:.6f}, {row['Longitude']:.6f}) → {row['Weight']} crímenes")
    
    return final

# =============================================================================
if __name__ == "__main__":
    gpu_hotspot_complete(
        edges_file="chicago_streets.csv",
        crime_file="crime_locations.csv",
        output_file="nodes_with_crimes_FULL.csv",
        batch_size=4000  # Ajusta: 3000 para 6GB, 8000 para 12GB+
    )