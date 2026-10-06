"""Import smoke-test for the MEFlowNet pipeline."""
import sys
import traceback

imports_to_test = [
    ("sys", "import sys"),
    ("argparse", "import argparse"),
    ("os", "import os"),
    ("cv2", "import cv2"),
    ("math", "import math"),
    ("numpy as np", "import numpy as np"),
    ("json", "import json"),
    ("torch", "import torch"),
    ("torch.nn as nn", "import torch.nn as nn"),
    ("torch.nn.functional as F", "import torch.nn.functional as F"),
    ("torch.utils.data as data", "import torch.utils.data as data"),
    ("config.parser", "from config.parser import parse_args"),
    ("model.meflownet", "from model.meflownet import MEFlowNet"),
    ("utils.utils", "from utils.utils import load_ckpt, coords_grid, bilinear_sampler"),
    ("scipy.interpolate", "from scipy.interpolate import griddata"),
    ("flow_vis", "from flow_vis import flow_to_image"),
    ("inference_tools", "from inference_tools import InferenceWrapper"),
    ("flow_feature", "from flow_feature import get_prompt"),
    ("transformers", "from transformers import AutoModelForCausalLM, AutoTokenizer"),
]

print("Testing imports...")
for name, import_statement in imports_to_test:
    try:
        exec(import_statement)
        print(f"✓ {name}")
    except Exception as e:
        print(f"✗ {name}: {e}")
        traceback.print_exc()
