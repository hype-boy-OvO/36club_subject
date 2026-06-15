import torch
import torch.optim as optim
import torch.nn as nn
import models
import pre_process
import pandas as pd


model_name = "kykim/bert-kor-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.My_Model(model_name)
model.to(device)
excel_path = r"datasets\2025_2026_Trend_And_General_Sentences.xlsx"

trendy_sent = pre_process.get_dataloader(excel_path=excel_path,target_category="유행어/신조어")
it_sent = pre_process.get_dataloader(excel_path=excel_path,target_category="IT 신기술")
bio_sent = pre_process.get_dataloader(excel_path=excel_path,target_category="생물/바이오 분야")

model.load_state_dict(
    torch.load("./pretrained_model.pt")
)

graph_encoder_params = list(
    model.graph_memory.graph_encoder.parameters()
)

other_graph_memory_params = [
    p
    for name, p in model.graph_memory.named_parameters()
    if not name.startswith("graph_encoder.")
]

optimizer = optim.AdamW([
    {
        "params": graph_encoder_params,
        "lr": 3e-5
    },
    {
        "params": other_graph_memory_params,
        "lr": 1e-4
    },
    {
        "params": model.prediction_model.parameters(),
        "lr": 3e-5
    },
    {
        "params": model.final_model.parameters(),
        "lr": 3e-5
    }
], weight_decay=1e-4)

def study(dataloader, epochs=20,kind=None):
    loss_history = []
    total_loss = 0
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
        
            optimizer.zero_grad()

            ouput, loss, new_nodes, new_edges = model(**batch)
        
            loss.backward()
            optimizer.step()

            model.graph_memory.write_memory(new_nodes, new_edges)

            if step % 1 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{step}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                
                loss_history.append({
                    "loss": loss.item()
                    })
        

    pd.DataFrame(loss_history).to_csv(
        f"{kind}_loss_history.csv",
        index=True,
        encoding="utf-8-sig"
        )
    
study(dataloader=trendy_sent,kind="trend_1")
study(dataloader=it_sent,kind="it_1")
study(dataloader=bio_sent,kind="bio_1")

study(dataloader=trendy_sent,kind="trend_2")
study(dataloader=it_sent,kind="it_2")
study(dataloader=bio_sent,kind="bio_2")





