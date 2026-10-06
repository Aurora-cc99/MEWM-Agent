"""MEFlowNet diagnostic utilities for verifying model outputs."""
import sys
import os

print("=== MELLM Pipeline Diagnostic ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print()

print("Testing imports...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__} - CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    print(f"✗ PyTorch: {e}")

try:
    import cv2
    print(f"✓ OpenCV {cv2.__version__}")
except Exception as e:
    print(f"✗ OpenCV: {e}")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except Exception as e:
    print(f"✗ NumPy: {e}")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"✓ Transformers")
except Exception as e:
    print(f"✗ Transformers: {e}")

print()
print("Testing local modules...")

try:
    from config.parser import parse_args
    print("✓ config.parser")
except Exception as e:
    print(f"✗ config.parser: {e}")

try:
    from model.meflownet import MEFlowNet
    print("✓ model.meflownet")
except Exception as e:
    print(f"✗ model.meflownet: {e}")

try:
    from utils.utils import load_ckpt, coords_grid, bilinear_sampler
    print("✓ utils.utils")
except Exception as e:
    print(f"✗ utils.utils: {e}")

try:
    from flow_vis import flow_to_image
    print("✓ flow_vis")
except Exception as e:
    print(f"✗ flow_vis: {e}")

try:
    from inference_tools import InferenceWrapper
    print("✓ inference_tools")
except Exception as e:
    print(f"✗ inference_tools: {e}")

try:
    from flow_feature import get_prompt
    print("✓ flow_feature")
except Exception as e:
    print(f"✗ flow_feature: {e}")

print()
print("Checking files and directories...")

files_to_check = [
    "config/meflownet.json",
    "../ckpt/meflownet.pth",
    "../ckpt/LLM/config.json",
    "data/test/test.jpg"
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path)
        print(f"✓ {file_path} ({size} bytes)")
    else:
        print(f"✗ {file_path} - NOT FOUND")

print()
print("=== Diagnostic Complete ===")
