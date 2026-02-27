import torch
import yaml
import os
import sys

if sys.platform.startswith("win"):
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.models.model import SimpleBEV

config_dir = os.path.join(PROJECT_ROOT, "configs")
os.makedirs(config_dir, exist_ok=True)
config_path = os.path.join(config_dir, "config.yaml")

default_config = {
    "model": {"input_size": 256, "num_classes": 1},
    "paths": {
        "weights": "outputs/bev_model.pth",
        "onnx": "outputs/bev_model.onnx",
        "int8": "outputs/bev_model_int8.onnx"
    }
}

if not os.path.exists(config_path) or os.path.getsize(config_path) == 0:
    with open(config_path, "w") as f:
        yaml.dump(default_config, f)

with open(config_path, "r") as f:
    config = yaml.safe_load(f) or default_config

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(os.path.join(PROJECT_ROOT, "outputs"), exist_ok=True)

weights_path = os.path.join(PROJECT_ROOT, config["paths"]["weights"])
onnx_path = os.path.join(PROJECT_ROOT, config["paths"]["onnx"])

if not os.path.exists(weights_path):
    print("Weights not found. Creating fresh model weights...")
    model = SimpleBEV(config["model"]["num_classes"]).to(device)
    torch.save(model.state_dict(), weights_path)

model = SimpleBEV(config["model"]["num_classes"]).to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

input_size = config["model"]["input_size"]
dummy_input = torch.randn(1, 3, input_size, input_size).to(device)

print("Exporting ONNX model...")
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=18,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
)
print(f"ONNX export complete → {onnx_path}")