import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_encoder import Graph_Encoder

class Graph_Memory(nn.Module):
    def __init__(self, main_model_dim=768 , num_nodes=256, node_dim=64, threshold=0.5, a=0.9):
        super().__init__()
        self.register_buffer("nodes", torch.randn(num_nodes, node_dim))
        self.edges = nn.Parameter(torch.eye(num_nodes))
        self.graph_encoder = Graph_Encoder()
        self.wq = nn.Linear(main_model_dim,node_dim)

        self.threshold = threshold
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.a = a

        self.mha_layer = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=4,    # 헤드 개수  
            batch_first=True
        )
    

    def get_activated_nodes(self, new_nodes):

        # new_nodes: [Batch, Num_Nodes, Node_Dim]
        batch_size = new_nodes.size(0)
    
        strong_nodes = F.relu(torch.mean(new_nodes, dim=-1))
    
        _, top8_indices = torch.topk(strong_nodes, k=8, dim=1)
    
        strong_edges = F.relu(self.edges - self.threshold)
    
        activated_history = [top8_indices]
    
        curr_indices = top8_indices
        for i in range(3):
            # 엣지 맵에서 현재 노드들의 연결 정보 가져오기
            # curr_indices[Batch, 8]를 인덱스로 사용하여 [Batch, 8, Num_Nodes] 추출
            connectivity = strong_edges[curr_indices] 
        
            # 각 배치/노드별로 가장 강하게 연결된 다음 노드 찾기
            _, next_indices = torch.max(connectivity, dim=2) # [Batch, 8]
        
            curr_indices = next_indices
            activated_history.append(curr_indices)
    
        # 모든 단계의 노드를 하나로 합침: [Batch, 32]
        all_activated = torch.cat(activated_history, dim=1)
    
        return all_activated


    def read_memory(self, query, new_nodes):
        all_activated = self.get_activated_nodes(new_nodes)
        activated_nodes = self.nodes[all_activated]
        query = self.wq(query)
        attn_output, _ = self.mha_layer(query=query, key=activated_nodes, value=activated_nodes)
        return attn_output
    
    def write_memory(self, new_nodes, new_edges):
        with torch.no_grad():
            self.nodes.copy_(self.nodes * self.a + new_nodes.mean(dim=0) * (1 - self.a))
            self.edges.copy_(self.edges * self.a + new_edges.mean(dim=0) * (1 - self.a))

    def forward(self,main_model_output):
        new_nodes, new_edges = self.graph_encoder(main_model_output)

        if main_model_output.dim() == 3:
            main_model_output = main_model_output[:, 0, :].unsqueeze(1)
        else:
            main_model_output = main_model_output[0, :].unsqueeze(1)

        output = self.read_memory(query=main_model_output, new_nodes=new_nodes)
        return output, new_nodes, new_edges


