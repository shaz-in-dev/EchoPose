"""
tests/test_extras.py — Additional coverage for temporal filter, pose sim, security, logger
"""

import pytest
import numpy as np
import time

from pipeline.temporal_filter_v2 import TemporalPoseFilterV2
from pipeline.pose import PoseEstimator
from security import RateLimiter, IncomingCSIBundle
from custom_logger import StructuredLogger


# ── Temporal Filter ────────────────────────────────────────────────

def _make_skeleton(x_offset=0.0):
    return [
        {"x": 0.5 + x_offset, "y": 0.15 + i * 0.05, "z": 0.5, "confidence": 0.9}
        for i in range(17)
    ]


def test_temporal_filter_smooths_jitter():
    filt = TemporalPoseFilterV2(max_people=1)
    skels = [_make_skeleton()]
    for _ in range(5):
        result = filt.filter(skels)
    assert len(result) == 1
    assert len(result[0]) == 17
    assert all("x" in kp and "confidence" in kp for kp in result[0])


def test_temporal_filter_handles_empty():
    filt = TemporalPoseFilterV2(max_people=2)
    result = filt.filter([])
    assert result == []


# ── Pose Estimator (simulation mode) ──────────────────────────────

def test_pose_estimator_simulation_output():
    est = PoseEstimator()
    assert est.is_simulation  # No checkpoint in tests
    features = np.random.randn(3, 64, 16).astype(np.float32)
    skels = est.predict(features)
    assert isinstance(skels, list)
    assert len(skels) >= 1
    assert len(skels[0]) == 17
    for kp in skels[0]:
        assert 0.0 <= kp["x"] <= 1.0
        assert 0.0 <= kp["y"] <= 1.0
        assert "confidence" in kp


def test_pose_estimator_accepts_per_person():
    est = PoseEstimator()
    features = np.random.randn(3, 64, 16).astype(np.float32)
    per_person = [np.random.randn(3, 64, 16)]
    skels = est.predict(features, per_person_features=per_person)
    assert len(skels) >= 1


# ── Security ──────────────────────────────────────────────────────

def test_rate_limiter_allows_normal_traffic():
    rl = RateLimiter(requests_per_second=10)
    for _ in range(10):
        assert rl.check_rate_limit("127.0.0.1")


def test_rate_limiter_blocks_excess():
    from fastapi import HTTPException
    rl = RateLimiter(requests_per_second=3)
    for _ in range(3):
        rl.check_rate_limit("10.0.0.1")
    with pytest.raises(HTTPException) as exc:
        rl.check_rate_limit("10.0.0.1")
    assert exc.value.status_code == 429


def test_incoming_csi_bundle_valid():
    bundle = IncomingCSIBundle(
        window_us=50000,
        frames=[
            {"node_id": 0, "amplitudes": [1.0] * 64},
            {"node_id": 1, "amplitudes": [2.0] * 64},
        ],
    )
    assert bundle.window_us == 50000


def test_incoming_csi_bundle_rejects_empty():
    with pytest.raises(Exception):
        IncomingCSIBundle(window_us=50000, frames=[])


# ── StructuredLogger ──────────────────────────────────────────────

def test_structured_logger_writes_and_closes(tmp_path):
    logger = StructuredLogger(log_dir=str(tmp_path))
    logger.log_inference(12.5, 0.85, [], {"node_0": 1.0})
    logger.close()
    content = (tmp_path / "structured_inference.jsonl").read_text()
    assert "rf_inference" in content
    assert "12.5" in content
