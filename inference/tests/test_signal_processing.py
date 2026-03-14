"""
tests/test_signal_processing.py — Comprehensive Unit Testing (Feature 5)

Ensures our advanced math pipeline (wavelet, wiener) functions perfectly
without regressions.
"""

import pytest
import numpy as np
from pipeline.advanced_denoise import AdvancedDenoiser
from pipeline.robust_processing import RobustCSIProcessor

@pytest.fixture
def denoiser():
    return AdvancedDenoiser(num_nodes=3, num_sub=64, sample_hz=20.0)

@pytest.fixture
def processor():
    return RobustCSIProcessor(expected_nodes=3, num_sub=64)

def test_wavelet_denoising_preserves_shape(denoiser):
    sig = np.random.normal(0, 1, 100) + np.sin(np.linspace(0, 10, 100))
    clean = denoiser._apply_wavelet(sig)
    assert len(clean) == len(sig)
    # Variance should decrease strictly after soft thresholding
    assert np.var(clean) < np.var(sig)

def test_wiener_filter_handles_small_arrays(denoiser):
    sig = np.array([1.0, 2.0, 3.0])
    res = denoiser._apply_wiener(sig)
    assert np.allclose(sig, res)  # Passes directly if too small

def test_nlos_detection(processor):
    # Simulated Line of Sight (low variance)
    los_csi = np.ones((10, 64)) * 5.0
    los_csi += np.random.normal(0, 0.1, (10, 64))
    assert not processor.detect_nlos(los_csi)
    
    # Simulated NLOS (extremely high variance across subcarriers)
    nlos_csi = np.random.exponential(scale=10.0, size=(10, 64))
    assert processor.detect_nlos(nlos_csi)

def test_interference_mitigation(processor):
    spectrum = np.ones(64)
    # Insert massive impulsive noise spike
    spectrum[15] = 1000.0
    cleaned = processor.handle_interference(spectrum)
    assert cleaned[15] < 1000.0
    assert cleaned[15] == np.median(spectrum)

def test_node_health_degradation(processor):
    features = np.ones((3, 64, 16))
    active_nodes = [0, 2] # Node 1 is suddenly missing
    
    features = processor.adapt_to_missing_nodes(features, active_nodes)
    assert processor.node_health[1] == 0.8  # Degraded
    assert processor.node_health[0] == 1.0  # Healthy
