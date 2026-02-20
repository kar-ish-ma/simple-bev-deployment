import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("../outputs/bev_model.onnx")
input_name = session.get_inputs()[0].name

test_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
output = session.run(None, {input_name: test_input})

print("ONNX inference successful.")
print("Output:", output)