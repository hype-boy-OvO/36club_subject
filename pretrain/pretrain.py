import torch
import torch.optim as optim
import torch.nn as nn
import models
import pre_process


model_name = "kykim/bert-kor-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.My_Model(model_name)
model.to(device)
excel_path = r"datasets\2025_2026_Trend_And_General_Sentences.xlsx"

normal_sent = pre_process.get_dataloader(excel_path=excel_path,target_category="일반 일상 문장", if_train=1)

model.graph_memory.graph_encoder.load_state_dict(
    torch.load("./pretrained_graph_encoder.pt")
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
        "lr": 1e-4
    },
    {
        "params": model.final_model.parameters(),
        "lr": 1e-4
    }
], weight_decay=1e-4)

epochs = 50
for epoch in range(epochs):
    for step, batch in enumerate(normal_sent):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()

        ouput, loss, new_nodes, new_edges = model(**batch)
        
        loss.backward()
        optimizer.step()

        model.graph_memory.write_memory(new_nodes, new_edges)

        if step % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | Step [{step}/{len(normal_sent)}] | Loss: {loss.item():.4f}")


torch.save(
    model.state_dict(),
    "pretrained_model.pt"
)






