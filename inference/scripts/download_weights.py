#!/usr/bin/env python3
"""
download_weights.py — Fetches the pre-trained PoseNet model checkpoint for RF-Mesh.

Since we don't have a real checkpoint hosted on a CDN for this demo,
this script generates a dummy `.pt` file with randomized weights
structured exactly like a real trained model.

In a true production environment, this would `urllib.request.urlretrieve`
the weights from S3 or HuggingFace.
"""

import os
import torch
from pathlib import Path

# Adjust path to import PoseNet correctly
import sys
sys.path.append(str(Path(__file__).parent.parent))
from pipeline.pose import PoseNet

def generate_dummy_weights():
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)
    
    ckpt_path = models_dir / "pose_net.pt"
    
    print(f"⚠️ WARNING: Generating UNTRAINED MOCK checkpoint at: {ckpt_path}")
    print("This project is currently an architecture showcase. No trained ML model is provided.")
    
    # Initialize the model and save its state_dict as if it were trained
    model = PoseNet()
    
    # Save the weights
    torch.save(model.state_dict(), ckpt_path)
    print("✅ Dummy weights downloaded and saved successfully.")
    print("The inference engine will now load these random weights for architectural simulation.")

if __name__ == "__main__":
    generate_dummy_weights()
