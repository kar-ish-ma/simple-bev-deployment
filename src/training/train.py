import torch
import torch.nn as nn
import torch.optim as optim
import os

from src.models.model import SimpleBEV
from src.utils.bev_utils import generate_dummy_bev

device = "cuda" if torch.cuda.is_available() else "cpu"

# Ensure outputs folder exists
os.makedirs("../outputs", exist_ok=True)

model = SimpleBEV().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

print("Training started...")

for epoch in range(5):
    bev = generate_dummy_bev()
    bev = torch.tensor(bev).unsqueeze(0).to(device)
    label = torch.tensor([[1.0]]).to(device)

    optimizer.zero_grad()
    output = model(bev)
    loss = criterion(output, label)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "../outputs/bev_model.pth")
print("Model saved at outputs/bev_model.pth")