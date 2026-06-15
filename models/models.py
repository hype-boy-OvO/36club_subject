import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class Main_Model(nn.Module):
    def __init__(self, model_name="kykim/bert-kor-base", device=None):
        super().__init__()
        self.device = device if device else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, input_ids):
        outputs = self.model(input_ids)
        output = outputs.last_hidden_state

        if output.dim() == 3:
            output = output[:, 0, :]
        return output
        
class Prediction_Model(nn.Module):
    def __init__(self, main_model_dim=768, node_dim=64):
        super().__init__()
        self.predict_model = nn.Sequential(
            nn.Linear(main_model_dim + node_dim, (main_model_dim + node_dim) // 2),
            nn.LayerNorm((main_model_dim + node_dim) // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear((main_model_dim + node_dim) // 2, 1),
            nn.Softplus() # 오차 예측값은 항상 >= 0
        )
    
    def forward(self,input_ids, z):
        z = z.expand(-1, input_ids.size(1), -1)
        data = torch.cat([input_ids,z], dim=-1)
        return self.predict_model(data)
    
class Final_Model(nn.Module):
    def __init__(self, main_model_dim=768, node_dim=64, vocab_size=42000):
        super().__init__()
        self.final_model = nn.Sequential(
            nn.Linear(main_model_dim + node_dim, (main_model_dim + node_dim)//2),
            nn.GELU(),
            nn.Linear((main_model_dim + node_dim)//2, vocab_size)
        )

    def forward(self, input_ids, z):
        z = z.expand(-1, input_ids.size(1), -1)
        data = torch.cat([input_ids,z], dim=-1)
        return self.final_model(data)