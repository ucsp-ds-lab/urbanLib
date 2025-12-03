# gpu_amenity.py
import cudf
import cupy as cp
from cupy import RawKernel
import pandas as pd
import numpy as np

# Kernel idéntico al de crímenes, pero devuelve (NodeID, amenity_type)
amenity_kernel_code = """
extern "C" __global__
void snap_amenities_kernel(
    const float* amen_x, const float* amen_y, const int* amen_types,
    const float* edges_x1, const float* edges_y1,
    const float* edges_x2, const float* edges_y2,
    const long long* edge_node1_ids,
    const long long* edge_node2_ids,
    int N, int M,
    long long* out_node_ids,
    int* out_types
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    float ax = amen_x[idx];
    float ay = amen_y[idx];
    int type = amen_types[idx];
    
    float best_dist = 1e30f;
    long long best_node = -1;
    
    for (int j = 0; j < M; ++j) {
        float x1 = edges_x1[j], y1 = edges_y1[j];
        float x2 = edges_x2[j], y2 = edges_y2[j];
        
        float dx = x2 - x1;
        float dy = y2 - y1;
        float len2 = dx*dx + dy*dy;
        if (len2 < 1e-8f) continue;
        
        float t = ((ax - x1)*dx + (ay - y1)*dy) / len2;
        t = fmaxf(0.0f, fminf(1.0f, t));
        
        float px = x1 + t * dx;
        float py = y1 + t * dy;
        float dist = (ax - px)*(ax - px) + (ay - py)*(ay - py);
        
        if (dist < best_dist) {
            best_dist = dist;
            float d1 = (px - x1)*(px - x1) + (py - y1)*(py - y1);
            float d2 = (px - x2)*(px - x2) + (py - y2)*(py - y2);
            best_node = (d1 <= d2) ? edge_node1_ids[j] : edge_node2_ids[j];
        }
    }
    
    out_node_ids[idx] = best_node;
    out_types[idx] = type;
}
"""

def snap_amenities_rapids(
    edges_file: str,
    amenity_file: str,
    output_file: str = "nodes_with_amenities.csv",
    batch_size: int = 4000
):
    print("=== GPU-HOTSPOT: AMENITIES (RAPIDS + BATCH) ===")
    
    # Cargar
    edges_df = cudf.read_csv(edges_file)
    amen_df = cudf.read_csv(amenity_file)  # amenity,Latitude,Longitude
    
    # Tipos únicos
    unique_types = sorted(amen_df['amenity'].unique().to_pandas().tolist())
    type_to_idx = {t: i for i, t in enumerate(unique_types)}
    amen_df['amenity_type'] = amen_df['amenity'].map(type_to_idx)
    
    print(f"Amenities: {len(amen_df):,} | Tipos: {len(unique_types)}")
    
    # Preparar arrays GPU
    e_x1 = edges_df['Node1_Longitude'].to_cupy().astype(cp.float32)
    e_y1 = edges_df['Node1_Latitude'].to_cupy().astype(cp.float32)
    e_x2 = edges_df['Node2_Longitude'].to_cupy().astype(cp.float32)
    e_y2 = edges_df['Node2_Latitude'].to_cupy().astype(cp.float32)
    n1_ids = edges_df['Node1_ID'].astype('int64').to_cupy()
    n2_ids = edges_df['Node2_ID'].astype('int64').to_cupy()
    
    kernel = RawKernel(amenity_kernel_code, 'snap_amenities_kernel')
    
    # Procesar por batches
    results_node = []
    results_type = []
    N = len(amen_df)
    
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        print(f"Batch {(start//batch_size)+1}/{(N-1)//batch_size + 1} → {end-start:,} amenities")
        
        batch = amen_df.iloc[start:end]
        a_x = batch['Longitude'].to_cupy().astype(cp.float32)
        a_y = batch['Latitude'].to_cupy().astype(cp.float32)
        a_t = batch['amenity_type'].to_cupy().astype(cp.int32)
        
        out_nodes = cp.zeros(end-start, dtype=cp.int64)
        out_types = cp.zeros(end-start, dtype=cp.int32)
        
        blocks = (len(a_x) + 255) // 256
        kernel((blocks,), (256,), (
            a_x, a_y, a_t,
            e_x1, e_y1, e_x2, e_y2,
            n1_ids, n2_ids,
            len(a_x), len(e_x1),
            out_nodes, out_types
        ))
        
        results_node.append(out_nodes.get())
        results_type.append(out_types.get())
    
    # Reconstruir
    node_ids = np.concatenate(results_node)
    types = np.concatenate(results_type)
    
    # Conteo por (node, type)
    count_df = cudf.DataFrame({
        'NodeID': node_ids,
        'amenity_type': types,
        'count': cp.ones(len(node_ids), dtype=cp.int32).get()
    })
    aggregated = count_df.groupby(['NodeID', 'amenity_type']).count().reset_index()
    
    # Pivot
    pivot = aggregated.pivot_table(
        index='NodeID', columns='amenity_type', values='count', fill_value=0
    ).astype(int)
    pivot.columns = [unique_types[int(col)] for col in pivot.columns]
    
    # Todos los nodos (incluyendo los que no tienen amenities)
    nodes1 = edges_df[['Node1_ID', 'Node1_Latitude', 'Node1_Longitude']].rename(
        columns={'Node1_ID': 'NodeID', 'Node1_Latitude': 'Latitude', 'Node1_Longitude': 'Longitude'})
    nodes2 = edges_df[['Node2_ID', 'Node2_Latitude', 'Node2_Longitude']].rename(
        columns={'Node2_ID': 'NodeID', 'Node2_Latitude': 'Latitude', 'Node2_Longitude': 'Longitude'})
    all_nodes = cudf.concat([nodes1, nodes2]).drop_duplicates('NodeID')
    
    # Merge final
    final = all_nodes.merge(pivot, on='NodeID', how='left').fillna(0)
    
    # Asegurar que todas las columnas existan
    for amenity_type in unique_types:
        if amenity_type not in final.columns:
            final[amenity_type] = 0
        final[amenity_type] = final[amenity_type].astype('int32')

    final_pd = final.to_pandas()

    # Ordenar por total de amenities
    final_pd['total_amenities'] = final_pd[unique_types].sum(axis=1)
    final_pd = final_pd.sort_values('total_amenities', ascending=False).drop('total_amenities', axis=1)
    final_pd = final_pd.reset_index(drop=True)

    final_pd.to_csv(output_file, index=False)
    
    print(f"\nListo → {output_file}")
    print(f"   Nodos totales: {len(final_pd):,}")
    print(f"   Tipos de amenity: {len(unique_types)}")
    print("\nTop 5 nodos con más amenities:")
    for _, row in final_pd.head(5).iterrows():
        total = row[unique_types].sum()
        print(f"   • Node {row['NodeID']} → {int(total)} amenities")
    
    return final_pd

# Uso
if __name__ == "__main__":
    snap_amenities_rapids(
        edges_file="chicago_streets.csv",
        amenity_file="amenities.csv",
        output_file="nodes_with_amenities_full.csv"
    )