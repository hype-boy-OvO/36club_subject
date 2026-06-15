import torch
import torch.nn as nn

class Graph_Encoder(nn.Module):
    def __init__(self, main_model_dim=768, num_nodes=256, node_dim=64, rank_dim=16):
        super().__init__()

        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.rank_dim = rank_dim

        self.compressor = nn.Sequential(
            nn.Linear(main_model_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        # [Batch, 256] -> [Batch, 256 * 64]
        self.node_projector = nn.Linear(256, num_nodes * node_dim)


        self.edge_u = nn.Linear(256, num_nodes * rank_dim)
        self.edge_v = nn.Linear(256, num_nodes * rank_dim)

    def forward(self, feat):
        if feat.dim() == 3:
            feat = feat[:, 0, :]
        else:
            feat = feat[0,:]
            
        batch_size = feat.size(0)
        
        compressed_feat = self.compressor(feat)
        
        node_update = self.node_projector(compressed_feat).view(batch_size, self.num_nodes, self.node_dim)
        node_update = torch.tanh(node_update)
        
        u = self.edge_u(compressed_feat).view(batch_size, self.num_nodes, self.rank_dim)
        v = self.edge_v(compressed_feat).view(batch_size, self.num_nodes, self.rank_dim)
        
        # [Batch, 256, 32] x [Batch, 32, 256] -> [Batch, 256, 256] 바운더리 생성
        edge_update = torch.bmm(u, v.transpose(1, 2))
        edge_update = torch.sigmoid(edge_update)
        
        return node_update, edge_update