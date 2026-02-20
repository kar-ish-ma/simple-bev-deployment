#!/bin/bash

set -e  # stop on error

echo "======================================"
echo " BEV FULL AUTOMATION PIPELINE START "
echo "======================================"

PROJECT_ROOT=$(pwd)

# -----------------------------
# 1️⃣ Create Virtual Environment (if not exists)
# -----------------------------
if [ ! -d "bev" ]; then
    echo "Creating virtual environment..."
    python3 -m venv bev
fi

# -----------------------------
# 2️⃣ Activate Virtual Environment
# -----------------------------
echo "Activating virtual environment..."

if [ -f "bev/bin/activate" ]; then
    source bev/bin/activate
elif [ -f "bev/Scripts/activate" ]; then
    source bev/Scripts/activate
else
    echo "Virtual environment activation failed."
    exit 1
fi

# -----------------------------
# 3️⃣ Upgrade pip
# -----------------------------
echo "Upgrading pip..."
python -m pip install --upgrade pip

# -----------------------------
# 4️⃣ Install Requirements
# -----------------------------
echo "Installing requirements..."
pip install -r requirements.txt

# -----------------------------
# 5️⃣ Run Full Pipeline
# -----------------------------
echo "Running training..."
python -m src.training.train

echo "Exporting ONNX..."
python -m src.utils.export_onnx

echo "Quantizing INT8..."
python -m src.utils.quantize_onnx

echo "Running inference..."
python -m src.inference.inference

echo "======================================"
echo " PIPELINE COMPLETED SUCCESSFULLY "
echo "======================================"
