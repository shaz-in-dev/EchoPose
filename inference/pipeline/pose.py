"""
pipeline/pose.py — CSI → 17-keypoint skeleton estimation

Architecture:
  Input:  [nodes, subcarriers, doppler_bins] float32 tensor
  Output: 17 × {x, y, z, confidence} = 68 floats

The model is a lightweight 3-layer 1D-CNN + MLP.
In production, load a pre-trained .pt checkpoint.
For demo / simulation, the model generates plausible random poses
until a real checkpoint is provided.

COCO-17 keypoints:
  0  nose        5  l_shoulder  10 r_wrist   15 l_ankle
  1  l_eye        6  r_shoulder  11 l_hip     16 r_ankle
  2  r_eye        7  l_elbow     12 r_hip
  3  l_ear        8  r_elbow     13 l_knee
  4  r_ear        9  l_wrist     14 r_knee
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict
import os
import hashlib
import logging

logger = logging.getLogger("rf_inference.pose")

try:
    import onnxruntime as ort
    has_ort = True
except ImportError:
    has_ort = False

from optimized_inference import OptimizedInference

EXPECTED_NODES  = int(os.getenv("EXPECTED_NODES", "3"))
NUM_KEYPOINTS   = 17
MAX_PEOPLE      = 3  # Multi-person support
FEATURE_SHAPE   = (EXPECTED_NODES, 64, 16)   # (nodes, subcarriers, doppler_bins)
MODEL_CKPT      = Path(__file__).parent.parent / "models" / "pose_net.pt"
ONNX_CKPT       = Path(__file__).parent.parent / "models" / "pose_net.onnx"

from pipeline.pose_net_v2 import PoseNetV2
# Ensure we map the V2 architecture to the Estimator
class PoseNet(PoseNetV2):
    pass

class PoseEstimator:
    """Wraps PoseNet with checkpoint loading and inference."""

    def __init__(self):
        self.use_onnx = False
        self.use_optimized = False
        self.onnx_sess = None
        self.optimized = None
        self.model = None
        self.sim_tick = 0
        # Default to non-simulation; only the PyTorch fallback path below can
        # set this True. Must be set before any early return (ONNX/optimized
        # backends return before reaching that path) so `estimator.is_simulation`
        # is always a valid attribute, never an AttributeError.
        self.is_simulation = False

        # 0. Try OptimizedInference first (TensorRT / CoreML / INT8 Quantized)
        try:
            opt = OptimizedInference()
            if opt.session is not None:
                self.optimized = opt
                self.use_optimized = True
                logger.info("Using OptimizedInference backend: %s", opt.backend)
                return
        except Exception as e:
            logger.debug("OptimizedInference unavailable: %s", e)

        # 1. Try ONNX (standard providers)
        if ONNX_CKPT.exists() and has_ort:
            expected_hash = os.getenv("EXPECTED_ONNX_HASH")
            if expected_hash:
                h = hashlib.sha256()
                with open(ONNX_CKPT, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                file_hash = h.hexdigest()
                if file_hash != expected_hash:
                    raise ValueError(f"CRITICAL SECURITY: ONNX model SHA256 mismatch! Expected {expected_hash}, got {file_hash}")
                    
            logger.info("Found ONNX checkpoint at %s. Using ONNX Runtime.", ONNX_CKPT)
            # Set providers based on environment
            providers = ['CPUExecutionProvider']
            env_dev = os.getenv("INFERENCE_DEVICE", "auto").lower()
            if env_dev == "auto" or env_dev == "cuda":
                 providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # Note: CoreML is currently broken on Windows, focusing on CUDA/CPU
            
            try:
                self.onnx_sess = ort.InferenceSession(str(ONNX_CKPT), providers=providers)
                self.use_onnx = True
                return
            except Exception as e:
                logger.warning("Failed to load ONNX: %s. Falling back to PyTorch.", e)

        # 2. Fallback to native PyTorch
        env_dev = os.getenv("INFERENCE_DEVICE", "auto").lower()
        if env_dev == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(env_dev)

        try:
            self.model = PoseNet().to(self.device).eval()
        except RuntimeError as e:
            logger.warning("Failed to use device %s, falling back to CPU: %s", self.device, e)
            self.device = torch.device("cpu")
            self.model = PoseNet().to(self.device).eval()

        if MODEL_CKPT.exists():
            state = torch.load(MODEL_CKPT, map_location=self.device)
            self.model.load_state_dict(state)
            logger.info("Loaded PyTorch checkpoint from %s", MODEL_CKPT)
        else:
            logger.info("No checkpoint found at %s — simulation mode.", MODEL_CKPT)
        self.is_simulation = not MODEL_CKPT.exists() and not self.use_onnx
        self.sim_tick = 0

        # In production, simulation mode means the server would silently stream
        # a scripted fake skeleton to real users with only a buried JSON flag as
        # the tell. Require an explicit opt-in instead of letting that happen by
        # accident (e.g. a missing/renamed checkpoint on a production deploy).
        _is_production = os.getenv("ECHOPOSE_ENV", "development").lower() == "production"
        _allow_sim = os.getenv("ALLOW_SIMULATION_MODE", "false").lower() == "true"
        if self.is_simulation and _is_production and not _allow_sim:
            raise RuntimeError(
                f"No trained model checkpoint found at {MODEL_CKPT}, but ECHOPOSE_ENV=production. "
                "Refusing to start in simulation mode — it would serve a scripted fake skeleton "
                "to real users. Provide a real checkpoint, or set ALLOW_SIMULATION_MODE=true to "
                "explicitly acknowledge you want simulated pose data in production."
            )

    @torch.no_grad()
    def predict(self, features: np.ndarray, per_person_features: list = None) -> List[List[Dict]]:
        """
        Args:
            features: ndarray [nodes, subcarriers, doppler_bins]
            per_person_features: optional list of per-person feature tensors from disambiguation
        Returns:
            list of people, each containing 17 keypoints: [[{x, y, z, conf}, ...], ...]
        """
        if self.is_simulation:
            import math
            self.sim_tick += 1
            t = self.sim_tick
            s = math.sin
            walk = t * 0.04
            # Synthetic pose to prove pipeline works 
            base_pose = [
                {'x': 0.5, 'y': 0.15, 'z': 0.5},
                {'x': 0.48, 'y': 0.13, 'z': 0.5},
                {'x': 0.52, 'y': 0.13, 'z': 0.5},
                {'x': 0.45, 'y': 0.14, 'z': 0.5},
                {'x': 0.55, 'y': 0.14, 'z': 0.5},
                {'x': 0.4, 'y': 0.28, 'z': 0.5},
                {'x': 0.6, 'y': 0.28, 'z': 0.5},
                {'x': 0.35 + s(walk)*0.04, 'y': 0.42, 'z': 0.5 + s(walk)*0.05},
                {'x': 0.65 - s(walk)*0.04, 'y': 0.42, 'z': 0.5 - s(walk)*0.05},
                {'x': 0.3 + s(walk)*0.07, 'y': 0.56, 'z': 0.5 + s(walk)*0.08},
                {'x': 0.7 - s(walk)*0.07, 'y': 0.56, 'z': 0.5 - s(walk)*0.08},
                {'x': 0.44, 'y': 0.58, 'z': 0.5},
                {'x': 0.56, 'y': 0.58, 'z': 0.5},
                {'x': 0.42 + s(walk+1)*0.07, 'y': 0.73, 'z': 0.5 + s(walk+1)*0.07},
                {'x': 0.58 - s(walk+1)*0.07, 'y': 0.73, 'z': 0.5 - s(walk+1)*0.07},
                {'x': 0.42 + s(walk+2)*0.10, 'y': 0.88, 'z': 0.5 + s(walk+2)*0.10},
                {'x': 0.58 - s(walk+2)*0.10, 'y': 0.88, 'z': 0.5 - s(walk+2)*0.10},
            ]
            for p in base_pose:
                p['confidence'] = 0.9
            return [base_pose]  # [one_person_kps] — list of 17 kp dicts

        # Batch dimension setup
        x_np = np.expand_dims(features.astype(np.float32), axis=0) # [1, N, S, D]

        if self.use_optimized:
            raw = self.optimized.infer(x_np)
            if raw is not None:
                raw = raw.squeeze(0)  # [MAX_PEOPLE, 17, 4]
            else:
                # Fall through to PyTorch if optimized returns None
                x = torch.tensor(x_np).to(self.device)
                raw = self.model(x).squeeze(0).cpu().numpy()
        elif self.use_onnx:
            input_name = self.onnx_sess.get_inputs()[0].name
            raw = self.onnx_sess.run(None, {input_name: x_np})[0]
            raw = raw.squeeze(0)  # [MAX_PEOPLE, 17, 4]
        else:
            x = torch.tensor(x_np).to(self.device)
            raw = self.model(x).squeeze(0).cpu().numpy()  # [MAX_PEOPLE, 17, 4]

        if raw.ndim == 2:  # single-person output [17, 4]
            raw = raw[np.newaxis, ...]  # → [1, 17, 4]
        elif raw.ndim != 3:
            logger.warning(f"Unexpected pose output shape {raw.shape}, skipping frame")
            return []

        results = []
        num_people = min(MAX_PEOPLE, raw.shape[0]) if hasattr(raw, 'shape') else MAX_PEOPLE
        for person_idx in range(num_people):
            person_raw = raw[person_idx]
            keypoints = []
            for kp in person_raw:
                if len(kp) < 4:
                    continue
                keypoints.append({
                    "x":          float(np.clip(kp[0], -10, 10)),
                    "y":          float(np.clip(kp[1], -10, 10)),
                    "z":          float(np.clip(kp[2], -10, 10)),
                    "confidence": float(np.clip(kp[3], 0, 1)),
                })
            results.append(keypoints)
        return results
