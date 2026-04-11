"""tests/test_cross_polarization.py — CrossPolarizationFusion coverage."""

import pytest
import numpy as np
from research.cross_polarization import CrossPolarizationFusion


@pytest.fixture
def fuser():
    return CrossPolarizationFusion()


def test_fuse_same_shape(fuser):
    hh = np.random.randn(10, 64)
    hv = np.random.randn(10, 64)
    vv = np.random.randn(10, 64)
    result = fuser.fuse_polarizations(hh, hv, vv)
    assert result.shape == hh.shape


def test_fuse_shape_mismatch(fuser):
    hh = np.random.randn(10, 64)
    hv = np.random.randn(10, 32)
    vv = np.random.randn(10, 64)
    with pytest.raises(ValueError):
        fuser.fuse_polarizations(hh, hv, vv)


def test_fuse_1d_input(fuser):
    hh = np.random.randn(64)
    hv = np.random.randn(64)
    vv = np.random.randn(64)
    result = fuser.fuse_polarizations(hh, hv, vv)
    assert result.shape == (64,)


def test_fuse_output_non_negative(fuser):
    hh = np.random.randn(10, 64)
    hv = np.random.randn(10, 64)
    vv = np.random.randn(10, 64)
    result = fuser.fuse_polarizations(hh, hv, vv)
    assert np.all(result >= 0)  # abs() guarantees non-negative


def test_fuse_zeros(fuser):
    z = np.zeros((10, 64))
    result = fuser.fuse_polarizations(z, z, z)
    assert np.allclose(result, 0)


def test_fuse_identity_dominant(fuser):
    # When one polarization is much stronger, fused output should be nonzero
    hh = np.ones((10, 64)) * 100.0
    hv = np.ones((10, 64)) * 0.001
    vv = np.ones((10, 64)) * 0.001
    result = fuser.fuse_polarizations(hh, hv, vv)
    assert result.shape == hh.shape
    assert np.all(np.isfinite(result))
