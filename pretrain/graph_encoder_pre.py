import torch
import torch.nn as nn
import torch.optim as optim
import models
import memory
import pre_process
from transformers import DefaultDataCollator, Trainer, TrainingArguments


model_name = "kykim/bert-kor-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.Main_Model(model_name)
encoder_model = memory.Graph_Encoder()

model.to(device)
encoder_model.to(device)

def loss_func(output, nodes, edges, v_fixed):
    output_reduced = torch.matmul(output, v_fixed)
    output_reduced = torch.tanh(output_reduced)
    output_reduced = output_reduced.view(output_reduced.size(0), 256, 2)
    node_loss = (torch.mean(output_reduced, dim=-1) - torch.mean(nodes, dim=-1))**2
    node_loss = torch.mean(node_loss)

    target_edges = torch.matmul(output_reduced, output_reduced.transpose(-1, -2))
    target_edges = torch.sigmoid(target_edges)
    edge_loss = (target_edges - edges)**2
    edge_loss = torch.mean(edge_loss)
    return node_loss+edge_loss

optimizer = optim.AdamW(encoder_model.parameters(), lr=1e-4, weight_decay=1e-4)


excel_path = r"datasets\2025_2026_Trend_And_General_Sentences.xlsx"
target_category = "일반 일상 문장"
dataloader = pre_process.get_dataloader(excel_path=excel_path, target_category=target_category)

with torch.no_grad():
    all_outputs = []

    for batch in dataloader:
        # 데이터를 GPU/CPU로 이동
        input_ids = batch['input_ids'].to(device)
        
        # 모델에 통과시켜 원래의 768차원 아웃풋 추출 
        # (상황에 따라 중간 hidden_states를 쓰거나 모델 출력을 활용)
        outputs = model(input_ids=input_ids)
        flat_output = outputs.view(-1, 768)
        all_outputs.append(flat_output.cpu())

    all_entire_data = torch.cat(all_outputs, dim=0)
    all_entire_data = all_entire_data.to(device)
    _, _, v_fixed = torch.pca_lowrank(all_entire_data, q=256*2, center=True)

epochs = 50
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()

        with torch.no_grad():
            output = model(input_ids=batch['input_ids'])

        nodes, edges = encoder_model(output)

        loss = loss_func(output, nodes, edges, v_fixed)
        
        loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Step [{step}/{len(dataloader)}] | Loss: {loss.item():.4f}")

torch.save(
    encoder_model.state_dict(),
    "./pretrained_graph_encoder.pt"
)