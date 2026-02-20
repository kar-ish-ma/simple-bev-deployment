import os
import sys


# Force UTF-8 output (fix Windows ONNX exporter crash)
if sys.platform.startswith("win"):
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.stdout.reconfigure(encoding="utf-8")
from src.models.model import SimpleBEV
import yaml
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)

# --------------------------------------------------
# Resolve Project Root
# --------------------------------------------------
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

# --------------------------------------------------
# Load Config
# --------------------------------------------------
config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

fp32_model = os.path.join(PROJECT_ROOT, config["paths"]["onnx"])
int8_model = os.path.join(PROJECT_ROOT, config["paths"]["int8"])

# --------------------------------------------------
# Calibration Data Reader
# --------------------------------------------------
class BEVCalibrationDataReader(CalibrationDataReader):
    def __init__(self, model_path, num_samples=50):
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.data = [
            {self.input_name: np.random.rand(1, 3, 256, 256).astype(np.float32)}
            for _ in range(num_samples)
        ]
        self.iterator = iter(self.data)

    def get_next(self):
        return next(self.iterator, None)

# --------------------------------------------------
# Run Quantization
# --------------------------------------------------
print("Starting INT8 quantization...")

data_reader = BEVCalibrationDataReader(fp32_model)

quantize_static(
    model_input=fp32_model,
    model_output=int8_model,
    calibration_data_reader=data_reader,
    quant_format=QuantFormat.QOperator,
    weight_type=QuantType.QInt8,
    activation_type=QuantType.QInt8,
)

print(f"INT8 model saved → {int8_model}")