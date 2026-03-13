"""
pipeline/fusion.py — multi-node CSI bundle → unified feature tensor

Receives a SyncedBundle (JSON from the Rust aggregator) and combines
amplitude data from all nodes into a single [nodes, subcarriers, ...]
matrix ready for the denoiser.
"""

from typing import Any, Dict
from pipeline.denoise import CsiDenoiser
import os

EXPECTED_NODES = int(os.getenv("EXPECTED_NODES", "3"))

class FusionPipeline:
    def __init__(self):
        self.denoiser = CsiDenoiser(num_nodes=EXPECTED_NODES, num_sub=64, sample_hz=20.0)

    def process_bundle(self, bundle: Dict[str, Any]):
        """
        Push each node's amplitudes into the denoiser, then compute
        and return the Doppler feature tensor.
        """
        for frame in bundle.get("frames", []):
            node_id    = int(frame["node_id"])
            amplitudes = frame["amplitudes"]
            self.denoiser.push(node_id, amplitudes)

        return self.denoiser.compute_features()
