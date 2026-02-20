# Simple BEV Deployment Pipeline

A modular, end-to-end Bird’s Eye View (BEV) deep learning pipeline built with PyTorch and optimized for deployment scenarios including edge systems and RISC-V boards.

This project demonstrates:

- Structured model training
- ONNX export (opset 18)
- Static INT8 quantization
- Model benchmarking (FP32 vs ONNX vs INT8)
- Fully automated execution via Bash
- Clean, production-ready Python package structure

---

## Project Architecture

```
simple-bev-deployment/
├─ src/
│  ├─ models/
│  │  └─ model.py
│  ├─ training/
│  │  └─ train.py
│  ├─ inference/
│  │  ├─ inference.py
│  │  └─ compare_model.py
│  ├─ utils/
│  │  ├─ export_onnx.py
│  │  └─ quantize_onnx.py
│  └─ run_pipeline.py
├─ configs/
│  └─ config.yaml
├─ outputs/
├─ requirements.txt
├─ run.sh
├─ Dockerfile
└─ README.md
```

---

## Features

- Modular Python package structure
- Automatic config generation if missing
- Automatic weight creation if missing
- ONNX export with latest supported opset
- Static INT8 quantization for edge deployment
- Performance and size benchmarking
- Single-command full automation
- Clean Git-compatible structure

---

## System Requirements

- Python 3.9+
- pip
- Git
- Optional: CUDA-enabled GPU

---

## Quick Start (Fully Automated)

From the project root:

```
bash run.sh
```

This will automatically:

1. Create a virtual environment (`bev/`)
2. Activate the environment
3. Upgrade pip
4. Install all dependencies
5. Train the model
6. Export ONNX model
7. Quantize to INT8
8. Run inference

No manual setup required.

---

## Manual Setup (Step-by-Step)

### 1. Create Virtual Environment

```
python -m venv bev
```

### 2. Activate Environment

Windows (PowerShell):

```
.\bev\Scripts\activate
```

Linux / macOS:

```
source bev/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

---

## Running Individual Components

All commands must be executed from the project root.

### Train Model

```
python -m src.training.train
```

### Export ONNX

```
python -m src.utils.export_onnx
```

### Quantize to INT8

```
python -m src.utils.quantize_onnx
```

### Run Inference

```
python -m src.inference.inference
```

### Compare FP32 vs ONNX vs INT8

```
python -m src.inference.compare_model
```

---

## Generated Outputs

After execution:

```
outputs/
├─ bev_model.pth
├─ bev_model.onnx
└─ bev_model_int8.onnx
```

---

## Configuration

Located at:

```
configs/config.yaml
```

If missing or empty, it is automatically created with default values:

```yaml
model:
  input_size: 256
  num_classes: 1

paths:
  weights: outputs/bev_model.pth
  onnx: outputs/bev_model.onnx
  int8: outputs/bev_model_int8.onnx
```

---

## Benchmarking

The comparison module evaluates:

- Numerical output difference
- Average inference time
- Model size comparison

Example metrics printed:

- Max Absolute Difference
- PyTorch Avg Inference Time
- ONNX Avg Inference Time
- INT8 Avg Inference Time
- Model size in MB

---

## Edge / RISC-V Deployment Notes

- ONNX export uses opset 18.
- INT8 quantization reduces model size and improves inference latency.
- Designed to integrate with ONNX Runtime.
- Suitable for edge environments and RISC-V systems with compatible runtimes.
- Further hardware-specific optimizations may be applied depending on the target device.

---

## Clean Rebuild

To reset the project:

Linux/macOS:
```
rm -rf bev outputs
bash run.sh
```

Windows PowerShell:
```
rmdir bev -Recurse -Force
rmdir outputs -Recurse -Force
bash run.sh
```

---

## Docker (Optional)

If Dockerfile is configured:

```
docker build -t simple-bev .
docker run simple-bev
```

---

## Clone and Run

```
git clone https://github.com/kar-ish-ma/simple-bev-deployment.git
cd simple-bev-deployment
bash run.sh
```

---

## Resume-Ready Project Summary

Designed and implemented a modular BEV deep learning pipeline featuring automated training, ONNX export, INT8 quantization, and benchmarking. Optimized for edge deployment and RISC-V environments using ONNX Runtime and structured Python packaging.

---

## License

This project is intended for educational and research purposes.
