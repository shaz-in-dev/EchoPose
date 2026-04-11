"""tests/test_acoustic_detector.py — AcousticEventDetector coverage."""

import pytest
import time
import numpy as np
from unittest.mock import patch
from pipeline.tactical.acoustic_detector import AcousticEventDetector


@pytest.fixture
def detector():
    d = AcousticEventDetector(fs=20.0)
    d.set_node_positions([(0, 0, 0), (10, 0, 0), (5, 10, 0)])
    return d


def _fill_buffer(det, node_id, n_frames, amplitude=0.01):
    for _ in range(n_frames):
        det.push(node_id, np.random.randn(64) * amplitude)


# ── basic detection ────────────────────────────────────────────

def test_no_event_on_quiet(detector):
    for nid in range(3):
        _fill_buffer(detector, nid, 60, amplitude=0.01)
    result = detector.detect()
    assert result["event_detected"] is False


def test_event_on_impulse(detector):
    for nid in range(3):
        _fill_buffer(detector, nid, 60, amplitude=0.01)
    # Inject a massive spike on all 3 nodes
    for nid in range(3):
        detector.push(nid, np.ones(64) * 100.0)
    with patch("time.time", return_value=time.time() + 10):
        detector._last_event_ts = 0.0
        result = detector.detect()
    assert result["event_detected"] is True
    assert result["event_type"] == "GUNSHOT"


def test_cooldown_prevents_double_detection(detector):
    for nid in range(3):
        _fill_buffer(detector, nid, 60, amplitude=0.01)
        detector.push(nid, np.ones(64) * 100.0)
    detector._last_event_ts = 0
    result1 = detector.detect()
    # Second detect within cooldown should be blocked
    result2 = detector.detect()
    assert result2.get("cooldown", False) or not result2["event_detected"]


# ── classification ────────────────────────────────────────────

def test_classify_gunshot(detector):
    detections = {0: 100, 1: 101, 2: 102}
    assert detector._classify_event(detections) == "GUNSHOT"


def test_classify_loud_impulse(detector):
    detections = {0: 100, 1: 101}
    assert detector._classify_event(detections) == "LOUD_IMPULSE"


def test_classify_unknown(detector):
    detections = {0: 100}
    assert detector._classify_event(detections) == "UNKNOWN_IMPULSE"


# ── triangulation ──────────────────────────────────────────────

def test_triangulate_needs_3_nodes(detector):
    result = detector._triangulate({0: 10, 1: 12})
    assert result is None


def test_triangulate_returns_position(detector):
    result = detector._triangulate({0: 10, 1: 10, 2: 10})
    assert result is not None
    assert "x" in result
    assert "y" in result


# ── push / buffer ──────────────────────────────────────────────

def test_push_creates_buffer(detector):
    detector.push(0, np.zeros(64))
    assert 0 in detector._node_bufs
    assert len(detector._node_bufs[0]) == 1


def test_buffer_cap(detector):
    for _ in range(200):
        detector.push(0, np.zeros(64))
    assert len(detector._node_bufs[0]) <= detector._max


# ── event log ──────────────────────────────────────────────────

def test_event_log_empty(detector):
    assert detector.event_log == []


def test_set_node_positions(detector):
    assert len(detector._node_positions) == 3
