"""MEFlowNet optical-flow pipeline entry point."""
import sys
import argparse
import os
import cv2
import math
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from config.parser import parse_args
from model.meflownet import MEFlowNet

from utils.utils import load_ckpt, coords_grid, bilinear_sampler
from scipy.interpolate import griddata

from flow_vis import flow_to_image
from inference_tools import InferenceWrapper
import argparse

from flow_feature import get_prompt

from transformers import AutoModelForCausalLM, AutoTokenizer
import json


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    onset_path = "data/test/onset.jpg"
    apex_path = "data/test/apex.jpg"

    with open("config/meflownet.json", "r") as f:
        args = json.load(f)
    args = argparse.Namespace(**args)

    model = MEFlowNet(args)
    ckpt_file_path = os.path.join("..", "ckpt", "meflownet.pth")
    load_ckpt(model, ckpt_file_path)
    model = model.to(device)
    model.eval()

    wrapped_model = InferenceWrapper(
        model,
        scale=args.scale,
        train_size=args.image_size,
        pad_to_train_size=False,
        tiling=False,
    )
    prompt = get_prompt(wrapped_model, onset_path, apex_path, device)
    print(prompt)

    model_name = os.path.join("..", "ckpt", "LLM")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**model_inputs, max_new_tokens=16384)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(content)
