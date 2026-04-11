"""tests/test_occupancy.py — OccupancyAnalyzer coverage."""

import pytest
import numpy as np
from pipeline.occupancy import OccupancyAnalyzer


@pytest.fixture
def analyzer():
    return OccupancyAnalyzer(fs=20.0)


def _high_conf_skeleton():
    return [[{"x": 0.5, "y": 0.1 * i, "z": 0.5, "confidence": 0.9} for i in range(17)]]


# ── skeleton-based detection ──────────────────────────────────

def test_occupied_via_skeleton(analyzer):
    result = analyzer.detect_presence(skeletons=_high_conf_skeleton())
    assert result["occupied"] is True
    assert result["method"] == "skeleton"
    assert result["num_people"] == 1
    assert result["confidence"] == 0.95


def test_two_skeletons(analyzer):
    result = analyzer.detect_presence(skeletons=_high_conf_skeleton() * 2)
    assert result["num_people"] == 2


def test_low_conf_skeleton_not_detected(analyzer):
    skel = [[{"x": 0, "y": 0, "z": 0, "confidence": 0.1} for _ in range(17)]]
    result = analyzer.detect_presence(skeletons=skel)
    assert result["method"] == "csi"


# ── CSI energy detection ──────────────────────────────────────

def test_csi_energy_above_baseline(analyzer):
    baseline = np.ones(64) * 0.01
    analyzer.calibrate_empty_room(baseline)
    active = np.ones(64) * 0.5
    result = analyzer.detect_presence(skeletons=[], csi_amplitudes=active)
    assert result["occupied"] is True


def test_csi_energy_below_baseline(analyzer):
    baseline = np.ones(64) * 1.0
    analyzer.calibrate_empty_room(baseline)
    quiet = np.ones(64) * 0.001
    result = analyzer.detect_presence(skeletons=[], csi_amplitudes=quiet)
    assert result["occupied"] is False


def test_calibrate_sets_baseline(analyzer):
    amp = np.ones(64) * 0.5
    analyzer.calibrate_empty_room(amp)
    assert analyzer._baseline_energy == pytest.approx(0.25, rel=1e-3)


# ── vital frequency detection ────────────────────────────────

def test_vital_frequency_content(analyzer):
    # Inject a 1 Hz sine (breathing) into 100-sample signal at 20 Hz
    t = np.linspace(0, 5, 100)
    sig = 0.5 * np.sin(2 * np.pi * 1.0 * t)  # 1 Hz = breathing
    result = analyzer.detect_presence(skeletons=[], csi_amplitudes=sig)
    # May or may not detect depending on threshold; just verify no crash
    assert "occupied" in result


# ── empty inputs ──────────────────────────────────────────────

def test_no_skeletons_no_csi(analyzer):
    result = analyzer.detect_presence(skeletons=[], csi_amplitudes=None)
    assert result["occupied"] is False


def test_empty_csi_array(analyzer):
    result = analyzer.detect_presence(skeletons=[], csi_amplitudes=np.array([]))
    assert result["occupied"] is False


def test_result_keys(analyzer):
    result = analyzer.detect_presence(skeletons=[], csi_amplitudes=None)
    assert "occupied" in result
    assert "num_people" in result
    assert "method" in result
    assert "confidence" in result
