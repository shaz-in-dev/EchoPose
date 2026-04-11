"""tests/test_frequency_transfer.py — FrequencyDomainTransferLearning coverage."""

import pytest
import numpy as np
from research.frequency_transfer import FrequencyDomainTransferLearning


@pytest.fixture
def ft():
    return FrequencyDomainTransferLearning()


# ── identity transform ────────────────────────────────────────

def test_same_frequency_identity(ft):
    csi = np.random.randn(64)
    result = ft.normalize_to_frequency_invariant_space(csi, 5.2e9)
    np.testing.assert_allclose(result, csi, atol=1e-6)


# ── different frequencies ──────────────────────────────────────

def test_24ghz_transform(ft):
    csi = np.random.randn(64)
    result = ft.normalize_to_frequency_invariant_space(csi, 2.4e9)
    assert result.shape == csi.shape


def test_6ghz_transform(ft):
    csi = np.random.randn(64)
    result = ft.normalize_to_frequency_invariant_space(csi, 6.0e9)
    assert result.shape == csi.shape


# ── multi-dimensional ─────────────────────────────────────────

def test_2d_input(ft):
    csi = np.random.randn(10, 64)
    result = ft.normalize_to_frequency_invariant_space(csi, 2.4e9)
    assert result.shape == (10, 64)


def test_3d_input(ft):
    csi = np.random.randn(3, 10, 64)
    result = ft.normalize_to_frequency_invariant_space(csi, 2.4e9)
    assert result.shape == (3, 10, 64)


# ── output properties ──────────────────────────────────────────

def test_output_finite(ft):
    csi = np.random.randn(64)
    result = ft.normalize_to_frequency_invariant_space(csi, 2.4e9)
    assert np.all(np.isfinite(result))


def test_zeros_input(ft):
    csi = np.zeros(64)
    result = ft.normalize_to_frequency_invariant_space(csi, 2.4e9)
    np.testing.assert_allclose(result, 0.0)


# ── source frequency ──────────────────────────────────────────

def test_default_source_frequency(ft):
    assert ft.source_frequency_hz == 5.2e9
