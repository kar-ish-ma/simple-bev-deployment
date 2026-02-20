import torch
import numpy as np
import time
import os
import onnxruntime as ort

from src.models.model import SimpleBEV

# --------------------------------------------------
# Resolve Project Root
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------
# Paths
# --------------------------------------------------
weights_path = os.path.join(PROJECT_ROOT, "outputs", "bev_model.pth")
onnx_path = os.path.join(PROJECT_ROOT, "outputs", "bev_model.onnx")
int8_path = os.path.join(PROJECT_ROOT, "outputs", "bev_model_int8.onnx")

# --------------------------------------------------
# Load PyTorch Model
# --------------------------------------------------
model = SimpleBEV().to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# --------------------------------------------------
# Load ONNX Model
# --------------------------------------------------
if not os.path.exists(onnx_path):
    raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

session = ort.InferenceSession(onnx_path)
input_name = session.get_inputs()[0].name

# --------------------------------------------------
# Generate Same Input
# --------------------------------------------------
input_tensor = torch.randn(1, 3, 256, 256).to(device)
input_numpy = input_tensor.cpu().numpy().astype(np.float32)

# --------------------------------------------------
# 1️⃣ Output Comparison
# --------------------------------------------------
with torch.no_grad():
    torch_output = model(input_tensor).cpu().numpy()

onnx_output = session.run(None, {input_name: input_numpy})[0]

print("=== Output Comparison ===")
print("Max Absolute Difference:",
      np.max(np.abs(torch_output - onnx_output)))
print()

# --------------------------------------------------
# 2️⃣ Speed Benchmark
# --------------------------------------------------
runs = 200

# Torch timing
start = time.time()
for _ in range(runs):
    with torch.no_grad():
        model(input_tensor)
torch_time = (time.time() - start) / runs

# ONNX timing
start = time.time()
for _ in range(runs):
    session.run(None, {input_name: input_numpy})
onnx_time = (time.time() - start) / runs

print("=== Speed Comparison ===")
print(f"PyTorch Avg Inference Time: {torch_time:.6f} sec")
print(f"ONNX Avg Inference Time:    {onnx_time:.6f} sec")
print()

# --------------------------------------------------
# 3️⃣ Model Size Comparison
# --------------------------------------------------
torch_size = os.path.getsize(weights_path) / (1024 * 1024)
onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)

print("=== Model Size Comparison ===")
print(f"PyTorch Model Size: {torch_size:.2f} MB")
print(f"ONNX Model Size:    {onnx_size:.2f} MB")
print()

# --------------------------------------------------
# 4️⃣ INT8 Comparison (if exists)
# --------------------------------------------------
if os.path.exists(int8_path):
    int8_session = ort.InferenceSession(int8_path)

    start = time.time()
    for _ in range(runs):
        int8_session.run(None, {input_name: input_numpy})
    int8_time = (time.time() - start) / runs

    int8_size = os.path.getsize(int8_path) / (1024 * 1024)

    print("=== INT8 Comparison ===")
    print(f"INT8 Avg Inference Time: {int8_time:.6f} sec")
    print(f"INT8 Model Size: {int8_size:.2f} MB")
else:
    print("INT8 model not found. Skipping INT8 comparison.")