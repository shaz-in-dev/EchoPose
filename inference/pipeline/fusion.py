"""
pipeline/fusion.py — multi-node CSI bundle → unified feature tensor

Receives a SyncedBundle (JSON from the Rust aggregator) and combines
amplitude data from all nodes into a single [nodes, subcarriers, ...]
matrix ready for the denoiser.
"""

from typing import Any, Dict, List, Tuple
from pipeline.advanced_denoise import AdvancedDenoiser
from pipeline.robust_processing import RobustCSIProcessor
from research.disambiguation import MultiPersonDisambiguation
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
        self.disambiguator = MultiPersonDisambiguation(max_people=3)

    def process_bundle(self, bundle: Dict[str, Any]) -> Tuple[np.ndarray, List]:
        """
        Push each node's amplitudes into the advanced denoiser.
        Extract Doppler features and pass them through adversarial bounds.
        """
        active_nodes = []
        for frame in bundle.get("frames") or []:
            nid = frame.get("node_id")
            amps = frame.get("amplitudes")
            if nid is None or amps is None:
                continue
            node_id = int(nid)
            self.denoiser.push(node_id, amps)
            if node_id not in active_nodes:
                active_nodes.append(node_id)

        # 1. Multi-stage denoising (Wiener, Wavelet, Spectral Subtraction)
        features, confidence = self.denoiser.compute_features()
        
        # 2. Adversarial hardening (NLOS, Interference, Failures)
        hardened_features, metrics = self.robustness.process_bundle(features, active_nodes)
        
        # 3. Multi-person disambiguation via DBSCAN clustering
        # Extract mean Doppler spectrum across subcarriers for clustering
        doppler_spectrum = np.mean(hardened_features, axis=1)  # [nodes, doppler_bins]
        # Collapse subcarrier dimension before disambiguation
        doppler_view = np.mean(hardened_features, axis=1)  # [nodes, doppler_bins]
        per_person_tensors = self.disambiguator.disentangle_csi_signatures(doppler_view, doppler_spectrum)
        
        return hardened_features, per_person_tensors
