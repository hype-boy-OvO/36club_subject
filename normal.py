import torch
import torch.nn as nn
from transformers import AutoModel
import torch.optim as optim
import pre_process
import os
import openpyxl

class FrozenBertMLMModel(nn.Module):
    def __init__(self, model_name="kykim/bert-kor-base"):
        super().__init__()
        # 1. BERT 백본 모델 로드
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 2. BERT의 모든 가중치 고정 (Freeze)
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # 3. 빈칸을 예측할 간단한 선형 레이어 (Classifier Head) 정의
        # BERT의 hidden_size(768) -> 전체 토큰 사전 크기(vocab_size)로 변환
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        
    def forward(self, input_ids, attention_mask):
        # BERT 본체는 그래디언트 계산을 하지 않도록 context 분리 (메모리 절약)
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # 은닉 상태 추출: [batch_size, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state
        
        # 선형 레이어를 통과시켜 각 토큰 위치마다 단어 예측 로짓 생성
        # [batch_size, seq_len, vocab_size]
        logits = self.classifier(sequence_output)
        return logits
    
# 1. 하이퍼파라미터 및 장치(GPU/CPU) 설정
excel_path = r"datasets\2025_2026_Trend_And_General_Sentences.xlsx" 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 데이터로더 준비 (if_train=1을 주어 data_collator가 적용되도록 함)
normal_sent = pre_process.get_dataloader(excel_path=excel_path,target_category="일반 일상 문장" ,if_train=1)
trendy_sent = pre_process.get_dataloader(excel_path=excel_path,target_category="유행어/신조어")
it_sent = pre_process.get_dataloader(excel_path=excel_path,target_category="IT 신기술")
bio_sent = pre_process.get_dataloader(excel_path=excel_path,target_category="생물/바이오 분야")

model = FrozenBertMLMModel("kykim/bert-kor-base").to(device)

# ignore_index=-100 설정을 통해 마스킹되지 않은(정답 레이블이 -100인) 토큰은 로스 계산에서 제외합니다.
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# 오직 새로 추가한 'classifier(예측 헤드)'의 파라미터만 업데이트하도록 지정합니다.
optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3)

file_name = r"history\loss_history.xlsx"
if os.path.exists(file_name):
    wb = openpyxl.load_workbook(file_name)
    ws = wb.active

def study(dataloader, epochs=70,kind=None,save=True):
    next_row = 2
    total_loss = 0
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
        
            optimizer.zero_grad()

            logits = model(input_ids, attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 1 == 0 and save:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{step}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                
                if(save):
                   ws[f"{kind}{next_row}"] = total_loss/1
                   total_loss = 0
                   next_row += 1

            elif step % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{step}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                
        
    if(save):
        wb.save(file_name)

study(dataloader=normal_sent, save=False, epochs=50)
"""
study(dataloader=trendy_sent,kind="G")
study(dataloader=it_sent,kind="I")
study(dataloader=bio_sent,kind="K")

study(dataloader=trendy_sent,kind="H")
study(dataloader=it_sent,kind="J")
study(dataloader=bio_sent,kind="L")
"""

data1 = ["나 아는사람 강다니엘 닮은 이모가 다시보게되는게 다시 그때처럼 안닮게 엄마보면 느껴지는걸수도 있는거임? 엄마도?"]
data2 = ["나랏말싸미 듕귁에 달아문자와로 서르 사맛디 아니할쎄 이런 전차로 어린 백셩이 니르고져 홇베이셔도 마참네 제 뜨들 시러펴디 몯핧 노미하니아 내 이랄 윙하야 어엿비너겨 새로 스믈 여듫 짜랄 맹가노니사람마다 해여 수비니겨 날로 쑤메 뻔한킈 하고져 할따라미니라"]

gang = pre_process.get_custom_dataloader(data1)
hoon = pre_process.get_custom_dataloader(data2)

study(dataloader=gang ,kind="Q")
study(dataloader=hoon,kind="S")

study(dataloader=gang ,kind="R")
study(dataloader=hoon,kind="T")