import torch
import numpy as np
import time
import os
import sys
import onnxruntime as ort

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.models.model import SimpleBEV

device = "cuda" if torch.cuda.is_available() else "cpu"

weights_path = os.path.join(PROJECT_ROOT, "outputs", "bev_model.pth")
onnx_path   = os.path.join(PROJECT_ROOT, "outputs", "bev_model.onnx")
int8_path   = os.path.join(PROJECT_ROOT, "outputs", "bev_model_int8.onnx")

# ── Load PyTorch model ─────────────────────────────
model = SimpleBEV().to(device)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# ── Load ONNX FP32 session ─────────────────────────
if not os.path.exists(onnx_path):
    raise FileNotFoundError(f"ONNX model not found at {onnx_path}")

opts = ort.SessionOptions()
opts.intra_op_num_threads = 4

session = ort.InferenceSession(onnx_path, sess_options=opts)
input_name = session.get_inputs()[0].name

# ── Shared input ────────────────────────────────────
input_tensor = torch.randn(1, 3, 256, 256).to(device)
input_numpy  = input_tensor.cpu().numpy().astype(np.float32)

# ── PyTorch output (reference) ─────────────────────
with torch.no_grad():
    torch_output = model(input_tensor).cpu().numpy()

# ── ONNX FP32 output ───────────────────────────────
onnx_output = session.run(None, {input_name: input_numpy})[0]

# 1. Output Comparison
diff = torch_output - onnx_output
mse  = np.mean(diff ** 2)
rmse = np.sqrt(mse)

print("=== Output Comparison: PyTorch vs ONNX FP32 ===")
print(f"MSE:  {mse:.10f}  (near-zero means faithful export)")
print(f"RMSE: {rmse:.10f}")
print()

# 2. Speed Benchmark
runs = 200

start = time.time()
for _ in range(runs):
    with torch.no_grad():
        model(input_tensor)
torch_time = (time.time() - start) / runs

start = time.time()
for _ in range(runs):
    session.run(None, {input_name: input_numpy})
onnx_time = (time.time() - start) / runs

print("=== Speed Comparison ===")
print(f"PyTorch Avg Inference Time : {torch_time:.6f} sec")
print(f"ONNX FP32 Avg Inference Time: {onnx_time:.6f} sec")
print(f"Speedup (ONNX vs PyTorch)  : {torch_time/onnx_time:.2f}x")
print()

# 3. Model Size
torch_size = os.path.getsize(weights_path) / (1024 * 1024)
onnx_size  = os.path.getsize(onnx_path)  / (1024 * 1024)

print("=== Model Size Comparison ===")
print(f"PyTorch Model Size : {torch_size:.2f} MB")
print(f"ONNX FP32 Size     : {onnx_size:.2f} MB")
print()

# 4. INT8 Comparison
if os.path.exists(int8_path):
    int8_session = ort.InferenceSession(int8_path, sess_options=opts)

    int8_output = int8_session.run(None, {input_name: input_numpy})[0]

    int8_diff = torch_output - int8_output
    int8_mse  = np.mean(int8_diff ** 2)
    int8_rmse = np.sqrt(int8_mse)

    start = time.time()
    for _ in range(runs):
        int8_session.run(None, {input_name: input_numpy})
    int8_time = (time.time() - start) / runs

    int8_size = os.path.getsize(int8_path) / (1024 * 1024)

    print("=== INT8 Comparison ===")
    print(f"INT8 MSE vs PyTorch  : {int8_mse:.10f}")
    print(f"INT8 RMSE vs PyTorch : {int8_rmse:.10f}")
    print(f"INT8 Avg Inference   : {int8_time:.6f} sec")
    print(f"INT8 Model Size      : {int8_size:.2f} MB")
    print(f"Speedup (INT8 vs PyTorch): {torch_time/int8_time:.2f}x")
    print()

    print("=== Summary Table ===")
    print(f"{'Format':<15} {'MSE vs PT':<20} {'Inference (s)':<18} {'Size (MB)':<10}")
    print(f"{'PyTorch':<15} {'0 (reference)':<20} {torch_time:<18.6f} {torch_size:<10.2f}")
    print(f"{'ONNX FP32':<15} {mse:<20.10f} {onnx_time:<18.6f} {onnx_size:<10.2f}")
    print(f"{'ONNX INT8':<15} {int8_mse:<20.10f} {int8_time:<18.6f} {int8_size:<10.2f}")
else:
    print("INT8 model not found. Skipping INT8 comparison.")