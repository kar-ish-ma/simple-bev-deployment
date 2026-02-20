import subprocess
import sys
sys.stdout.reconfigure(encoding="utf-8")

steps = [
    "src.training.train",
    "src.utils.export_onnx",
    "src.utils.quantize_onnx",
    "src.inference.inference",
    "src.inference.compare_model"
]

print("\n🚀 Starting Full BEV Pipeline\n")

for step in steps:
    print(f"\n▶ Running: {step}")
    result = subprocess.run(
        [sys.executable, "-m", step],
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        print("❌ Error detected:")
        print(result.stderr)
        sys.exit(1)

print("\n✅ Full pipeline completed successfully.\n")