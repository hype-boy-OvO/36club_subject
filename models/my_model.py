from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import memory
from .models import Prediction_Model, Final_Model, Main_Model

class My_Model(Main_Model):
    def __init__(self, model_name="kykim/bert-kor-base", device=None):
        super().__init__(model_name=model_name)

        self.graph_memory = memory.Graph_Memory()
        self.prediction_model = Prediction_Model()
        self.final_model = Final_Model()

    def loss_func(self, logits, labels, predicted_loss, vocab_size=42000, b=1.0, l=0.1):
        batch_size = predicted_loss.size(0)
        
        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        token_loss = criterion(logits.view(-1, vocab_size), labels.view(-1))  # 차원: (batch_size * seq_len,)
        
        token_loss = token_loss.view(batch_size, -1)  # 차원: (batch_size, seq_len)
        
        # Data Collator가 만든 -100(마스킹 안 된 곳)을 제외하고 문장별 평균 loss 구하기
        mask = (labels != -100).float()  # 마스킹된 타겟 토큰 위치는 1, 아니면 0 (batch_size, seq_len)
        
        actual_loss = (token_loss * mask)   # 문장별 실제 MLM 손실의 합 (batch_size,)
        
        # 최종 문장 레벨의 실제 손실 (차원: batch_size,)
        
        # 4. predicted_loss의 차원을 (batch_size,)로 통일 (만약 prediction_model 아웃풋이 (batch_size, 1)일 경우 대비)
        predicted_loss = predicted_loss.squeeze(-1)

        # 5. 설계하신 메타 러닝 손실 계산 수행 (모두 batch_size 차원으로 정렬됨)
        meta_loss = torch.abs(actual_loss.detach() - predicted_loss)
        scale_factor = 1 + b * meta_loss

        total_loss = scale_factor * actual_loss + l * (meta_loss ** 2)
        
        # 배치 전체의 평균값 반환
        return total_loss.mean()


    def forward(self, input_ids, labels=None, attention_mask=None):

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            output = outputs.last_hidden_state
        
        # 그래프 메모리 레이어 통과
        z, new_nodes, new_edges = self.graph_memory(output)

        # 각 서브 모델 예측
        final_ouput = self.final_model(output, z)      # 예측된 MLM Logits
        predicted_loss = self.prediction_model(output, z)  # 예측된 Loss 값
        
        # 손실 함수 계산 (labels가 있을 때만 계산하도록 방어 코드 추가 가능)
        total_loss = None
        if labels is not None:
            total_loss = self.loss_func(
                logits=final_ouput, 
                labels=labels, 
                predicted_loss=predicted_loss,
                vocab_size=final_ouput.size(-1) # 선언한 vocab_size를 유동적으로 반영
            )

        return final_ouput, total_loss, new_nodes, new_edges