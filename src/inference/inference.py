import torch
import yaml
import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.models.model import SimpleBEV

config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")

default_config = {
    "model": {"input_size": 256, "num_classes": 1},
    "paths": {
        "weights": "outputs/bev_model.pth",
        "onnx": "outputs/bev_model.onnx",
        "int8": "outputs/bev_model_int8.onnx"
    }
}

if not os.path.exists(config_path) or os.path.getsize(config_path) == 0:
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(default_config, f)

with open(config_path, "r") as f:
    config = yaml.safe_load(f) or default_config

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)

weights_path = os.path.join(PROJECT_ROOT, config["paths"]["weights"])

if not os.path.exists(weights_path):
    print("Weights not found. Creating fresh model weights...")
    model = SimpleBEV(config["model"]["num_classes"]).to(device)
    torch.save(model.state_dict(), weights_path)

model = SimpleBEV(config["model"]["num_classes"]).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

input_size = config["model"]["input_size"]
x = torch.randn(1, 3, input_size, input_size).to(device)

with torch.no_grad():
    output = model(x)

print("Inference successful.")
print("Output:", output)