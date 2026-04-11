"""tests/test_emotion.py — EmotionDetector coverage."""

import pytest
import numpy as np
from pipeline.emotion import EmotionDetector


@pytest.fixture
def detector():
    return EmotionDetector()


def _skeleton_with_tilt(tilt=0.0):
    skel = [{"x": 0.5, "y": 0.1 * i, "z": 0.5} for i in range(17)]
    skel[5]["y"] = 0.28
    skel[6]["y"] = 0.28 + tilt
    return skel


# ── baseline tests ────────────────────────────────────────────

def test_initial_baselines(detector):
    assert detector._hr_baseline == 65.0
    assert detector._rr_baseline == 15.0


def test_update_baselines(detector):
    for _ in range(200):
        detector.update_baselines(80.0, 20.0)
    assert detector._hr_baseline > 65.0
    assert detector._rr_baseline > 15.0


# ── stress estimation ──────────────────────────────────────────

def test_calm_at_baseline(detector):
    result = detector.estimate_stress(hr=65.0, rr=15.0)
    assert result["stress_level"] == "CALM"
    assert result["stress_score"] < 30


def test_high_stress_elevated_vitals(detector):
    result = detector.estimate_stress(hr=130.0, rr=30.0)
    assert result["stress_level"] == "HIGH_STRESS"
    assert result["stress_score"] >= 60


def test_moderate_stress(detector):
    result = detector.estimate_stress(hr=100.0, rr=25.0)
    assert result["stress_level"] in ("CALM", "MODERATE", "HIGH_STRESS")
    assert result["stress_score"] >= 0


def test_none_hr_rr_fallback(detector):
    result = detector.estimate_stress(hr=None, rr=None)
    assert result["stress_level"] == "CALM"


def test_postural_stress_shoulder_tilt(detector):
    skel = _skeleton_with_tilt(tilt=0.15)
    result = detector.estimate_stress(hr=65.0, rr=15.0, skeleton=skel)
    assert result["stress_score"] > 0


def test_postural_no_tilt(detector):
    skel = _skeleton_with_tilt(tilt=0.0)
    result = detector.estimate_stress(hr=65.0, rr=15.0, skeleton=skel)
    # No postural stress contribution
    assert result["stress_score"] < 1


def test_result_keys(detector):
    result = detector.estimate_stress(hr=70.0, rr=16.0)
    assert "stress_level" in result
    assert "stress_score" in result
    assert "hr_elevation_pct" in result
    assert "rr_elevation_pct" in result


def test_stress_score_clamped(detector):
    result = detector.estimate_stress(hr=300.0, rr=100.0, skeleton=_skeleton_with_tilt(0.5))
    assert 0 <= result["stress_score"] <= 100
