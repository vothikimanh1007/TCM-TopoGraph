import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
import torch_geometric.transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score
import os

# --- 1. MODEL ARCHITECTURE ---
class BaseGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.dropout = torch.nn.Dropout(p=0.3)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.dropout(x)
        return self.conv2(x, edge_index)

class HeteroLinkPredictor(torch.nn.Module):
    def forward(self, x_herb, x_acupoint, edge_label_index):
        return (x_herb[edge_label_index[0]] * x_acupoint[edge_label_index[1]]).sum(dim=-1)

class TopoGNN_Model(torch.nn.Module):
    def __init__(self, metadata, hidden_channels=128, out_channels=64):
        super().__init__()
        self.encoder = to_hetero(BaseGNN(hidden_channels, out_channels), metadata, aggr='sum')
        self.decoder = HeteroLinkPredictor()

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.decoder(z_dict['herb'], z_dict['acupoint'], edge_label_index)

# --- 2. TOPOLOGICAL LOSS FUNCTION ---
def topological_penalty_loss(z_dict, lambda_reg=0.1):
    """
    Hàm phạt Tô-pô (Topological Regularization). Phạt các nút nếu không gian nhúng
    (embedding space) bị co sập (collapse), giúp duy trì chuỗi đồng điều Path Homology.
    """
    acu_norm = torch.norm(z_dict['acupoint'], p=2, dim=1).mean()
    herb_norm = torch.norm(z_dict['herb'], p=2, dim=1).mean()
    return lambda_reg * (1.0 / (acu_norm + herb_norm + 1e-6))

# --- 3. TRAINING PIPELINE ---
def train(data_path="../dataset/tcm_heterodata.pt", output_model="../models/TopoGNN_Weights.pth"):
    print("Loading graph data...")
    data = torch.load(data_path)
    data = T.ToUndirected()(data) # Bi-directional for message passing
    
    transform = T.RandomLinkSplit(
        num_val=0.15, num_test=0.15, disjoint_train_ratio=0.3,
        neg_sampling_ratio=1.5, add_negative_train_samples=False,
        edge_types=[('herb', 'tropism', 'acupoint')],
        rev_edge_types=[('acupoint', 'rev_tropism', 'herb')]
    )
    train_data, val_data, test_data = transform(data)
    
    model = TopoGNN_Model(metadata=data.metadata())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    print("Starting training loop...")
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        
        edge_label_index = train_data['herb', 'tropism', 'acupoint'].edge_label_index
        edge_label = train_data['herb', 'tropism', 'acupoint'].edge_label
        
        pred = model(train_data.x_dict, train_data.edge_index_dict, edge_label_index)
        bce_loss = criterion(pred, edge_label)
        
        z_dict = model.encoder(train_data.x_dict, train_data.edge_index_dict)
        topo_loss = topological_penalty_loss(z_dict, lambda_reg=0.1)
        
        loss = bce_loss + topo_loss
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_idx = val_data['herb', 'tropism', 'acupoint'].edge_label_index
                val_lbl = val_data['herb', 'tropism', 'acupoint'].edge_label
                val_pred = model(val_data.x_dict, val_data.edge_index_dict, val_idx)
                
                v_lbl, v_prd = val_lbl.numpy(), val_pred.numpy()
                auc = roc_auc_score(v_lbl, v_prd)
                ap = average_precision_score(v_lbl, v_prd)
                print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val AUC: {auc:.4f} | Val AP: {ap:.4f}")
                
    os.makedirs(os.path.dirname(output_model), exist_ok=True)
    torch.save(model.state_dict(), output_model)
    print(f"Model saved to {output_model}")

if __name__ == "__main__":
    train()
