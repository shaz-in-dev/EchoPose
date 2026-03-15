"""
pipeline/fusion.py — multi-node CSI bundle → unified feature tensor

Receives a SyncedBundle (JSON from the Rust aggregator) and combines
amplitude data from all nodes into a single [nodes, subcarriers, ...]
matrix ready for the denoiser.
"""

from typing import Any, Dict
from pipeline.advanced_denoise import AdvancedDenoiser
from pipeline.robust_processing import RobustCSIProcessor
import os
import numpy as np

EXPECTED_NODES = int(os.getenv("EXPECTED_NODES", "3"))

class FusionPipeline:
    def __init__(self):
        self.denoiser = AdvancedDenoiser(
            num_nodes=EXPECTED_NODES, 
            num_sub=64, 
            sample_hz=20.0,
            stages=['wiener', 'wavelet', 'spectral']
        )
        self.robustness = RobustCSIProcessor(expected_nodes=EXPECTED_NODES)

    def process_bundle(self, bundle: Dict[str, Any]) -> np.ndarray:
        """
        Push each node's amplitudes into the advanced denoiser.
        Extract Doppler features and pass them through adversarial bounds.
        """
        active_nodes = []
        for frame in bundle.get("frames", []):
            node_id    = int(frame["node_id"])
            amplitudes = frame["amplitudes"]
            self.denoiser.push(node_id, amplitudes)
            if node_id not in active_nodes:
                active_nodes.append(node_id)

        # 1. World-class multi-stage denoising
        features, confidence = self.denoiser.compute_features()
        
        # 2. Adversarial hardening (NLOS, Interference, Failures)
        hardened_features, metrics = self.robustness.process_bundle(features, active_nodes)
        
        return hardened_features
