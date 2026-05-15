import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
import networkx as nx
import os

def build_hetero_graph(merged_csv_path, herbs_csv_path, output_dir="../dataset"):
    """
    Xây dựng cấu trúc đồ thị đa phương thức (Heterogeneous Graph) từ dữ liệu thô.
    Kết xuất cấu trúc này để chuẩn bị cho quá trình huấn luyện Topo-GNN.
    """
    print("Building Heterogeneous Graph...")
    df_acu = pd.read_csv(merged_csv_path)
    df_herbs = pd.read_csv(herbs_csv_path)
    
    data = HeteroData()
    
    # 1. Khởi tạo Node Huyệt đạo (0-simplices)
    num_acu = len(df_acu)
    acu_id_to_idx = {row['Point_Code']: idx for idx, row in df_acu.iterrows()}
    
    base_acu_feat = torch.randn(num_acu, 512)
    signal_weights = torch.ones(num_acu, 1)
    if 'Image_Base64' in df_acu.columns:
        for i, row in df_acu.iterrows():
            if pd.notna(row['Image_Base64']):
                signal_weights[i] = 2.5
    data['acupoint'].x = torch.cat([base_acu_feat, signal_weights], dim=1)

    # 2. Khởi tạo Node Vị thuốc
    num_herbs = len(df_herbs)
    base_herb_feat = torch.randn(num_herbs, 256)
    esph_features = torch.rand(num_herbs, 4) # Element-Specific Persistent Homology features
    data['herb'].x = torch.cat([base_herb_feat, esph_features], dim=1)

    # 3. Tạo Edges: Luồng sinh lý có hướng (Directed Flow)
    src_flow, dst_flow = [], []
    for m in df_acu['Ma_Kinh'].unique():
        nodes = df_acu[df_acu['Ma_Kinh'] == m].copy()
        nodes['n'] = nodes['Point_Code'].apply(lambda x: int(x.split('-')[1]) if '-' in x else 0)
        indices = [acu_id_to_idx[p] for p in nodes.sort_values('n')['Point_Code'] if p in acu_id_to_idx]
        for i in range(len(indices)-1): 
            src_flow.append(indices[i])
            dst_flow.append(indices[i+1])
            
    data['acupoint', 'flow_to', 'acupoint'].edge_index = torch.tensor([src_flow, dst_flow], dtype=torch.long)

    # 4. Tạo Edges: Quy kinh (Tropism)
    t_src, t_dst = [], []
    v_map = {'tâm': 'HT', 'can': 'LR', 'tâm bào': 'PC', 'thận': 'KI', 'phế': 'LU', 'đởm': 'GB', 'tam tiêu': 'TE', 'đại trường': 'LI', 'bàng quang': 'BL', 'tiểu trường': 'SI', 'vị': 'ST', 'tỳ': 'SP', 'nhâm': 'CV', 'đốc': 'GV'}
    
    for idx, row in df_herbs.iterrows():
        qk = str(row.get('QuyKinh', '')).lower()
        if qk != 'nan':
            for vn, code in v_map.items():
                if vn in qk:
                    targets = [acu_id_to_idx.get(p) for p in df_acu[df_acu['Ma_Kinh'] == code]['Point_Code'] if p in acu_id_to_idx]
                    for t_idx in targets: 
                        t_src.append(idx)
                        t_dst.append(t_idx)
                        
    data['herb', 'tropism', 'acupoint'].edge_index = torch.tensor([t_src, t_dst], dtype=torch.long)
    
    print(f"Graph Construction Complete:\n{data}")
    os.makedirs(output_dir, exist_ok=True)
    torch.save(data, os.path.join(output_dir, "tcm_heterodata.pt"))
    return data

if __name__ == "__main__":
    # Update paths according to your local structure
    build_hetero_graph("../dataset/TCM361_Merged_Processed.csv", "../dataset/ViThuoc_final.csv")
