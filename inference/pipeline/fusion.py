"""
pipeline/fusion.py — multi-node CSI bundle → unified feature tensor

Receives a SyncedBundle (JSON from the Rust aggregator) and combines
amplitude data from all nodes into a single [nodes, subcarriers, ...]
matrix ready for the denoiser.
"""

from typing import Any, Dict
from pipeline.denoise import CsiDenoiser

denoiser = CsiDenoiser(num_nodes=3, num_sub=64, sample_hz=20.0)


def process_bundle(bundle: Dict[str, Any]):
    """
    Push each node's amplitudes into the denoiser, then compute
    and return the Doppler feature tensor.

    Args:
        bundle: {window_us: int, frames: [{node_id, amplitudes, phases}, ...]}
    Returns:
        np.ndarray [3, 64, 16] Doppler features
    """
    for frame in bundle.get("frames", []):
        node_id    = int(frame["node_id"])
        amplitudes = frame["amplitudes"]
        denoiser.push(node_id, amplitudes)

    return denoiser.compute_features()
