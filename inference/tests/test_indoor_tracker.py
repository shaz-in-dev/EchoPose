"""tests/test_indoor_tracker.py — IndoorTracker coverage."""

import pytest
import numpy as np
from pipeline.tactical.indoor_tracker import IndoorTracker


@pytest.fixture
def tracker():
    t = IndoorTracker(fs=20.0)
    t.set_nodes([(0, 0, 0), (10, 0, 0), (5, 10, 0), (10, 10, 0)])
    return t


# ── update ────────────────────────────────────────────────────

def test_update_returns_position(tracker):
    rssi = {0: -40.0, 1: -50.0, 2: -45.0, 3: -48.0}
    result = tracker.update(rssi)
    assert "x" in result
    assert "y" in result
    assert "z" in result
    assert "accuracy_m" in result


def test_needs_3_nodes(tracker):
    result = tracker.update({0: -40.0, 1: -50.0})
    assert result["status"] == "need_3_nodes"


def test_velocity_after_updates(tracker):
    rssi1 = {0: -40.0, 1: -50.0, 2: -45.0}
    tracker.update(rssi1)
    rssi2 = {0: -42.0, 1: -48.0, 2: -47.0}
    result = tracker.update(rssi2)
    assert "velocity_ms" in result
    assert isinstance(result["velocity_ms"], float)


def test_phase_refinement(tracker):
    rssi = {0: -40.0, 1: -50.0, 2: -45.0}
    phase = {0: np.linspace(0, 2 * np.pi, 64),
             1: np.linspace(0, 4 * np.pi, 64),
             2: np.linspace(0, 3 * np.pi, 64)}
    result = tracker.update(rssi, node_csi_phase=phase)
    assert "x" in result


# ── track ──────────────────────────────────────────────────────

def test_get_track_empty(tracker):
    assert tracker.get_track() == []


def test_get_track_after_updates(tracker):
    for i in range(5):
        rssi = {0: -40.0 - i, 1: -50.0 + i, 2: -45.0}
        tracker.update(rssi)
    track = tracker.get_track()
    assert len(track) == 5
    assert "x" in track[0]


# ── helpers ────────────────────────────────────────────────────

def test_rssi_to_distance(tracker):
    d = tracker._rssi_to_distance(-30.0)
    assert abs(d - 1.0) < 0.01  # -30 at 1m reference


def test_rssi_farther_is_larger(tracker):
    d_near = tracker._rssi_to_distance(-35.0)
    d_far = tracker._rssi_to_distance(-55.0)
    assert d_far > d_near


def test_phase_to_distance(tracker):
    phase = np.linspace(0, 2 * np.pi, 64)
    d = tracker._phase_to_distance(phase)
    assert d >= 0


def test_heading_initial(tracker):
    assert tracker._heading() == 0.0


def test_smoothing(tracker):
    pos = np.array([5.0, 5.0, 0.0])
    tracker._track.append(np.array([4.0, 4.0, 0.0]))
    smoothed = tracker._smooth(pos)
    # Should be between 4 and 5
    assert 4.0 < smoothed[0] < 5.0


def test_accuracy_with_more_nodes(tracker):
    a3 = tracker._accuracy_estimate({0: 1.0, 1: 2.0, 2: 3.0})
    a4 = tracker._accuracy_estimate({0: 1, 1: 2, 2: 3, 3: 4})
    assert a4 < a3
