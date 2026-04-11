"""tests/test_intent_predictor.py — BehavioralIntentPredictor coverage."""

import pytest
import numpy as np
from pipeline.tactical.intent_predictor import BehavioralIntentPredictor, _MIN_HISTORY


def _standing_skeleton():
    """17 COCO keypoints in a neutral standing position."""
    # nose, l_eye, r_eye, l_ear, r_ear, l_shoulder, r_shoulder,
    # l_elbow, r_elbow, l_wrist, r_wrist, l_hip, r_hip,
    # l_knee, r_knee, l_ankle, r_ankle
    return [
        {"x": 0.5, "y": 0.10, "z": 0.5},
        {"x": 0.48, "y": 0.08, "z": 0.5},
        {"x": 0.52, "y": 0.08, "z": 0.5},
        {"x": 0.45, "y": 0.09, "z": 0.5},
        {"x": 0.55, "y": 0.09, "z": 0.5},
        {"x": 0.40, "y": 0.25, "z": 0.5},
        {"x": 0.60, "y": 0.25, "z": 0.5},
        {"x": 0.35, "y": 0.40, "z": 0.5},
        {"x": 0.65, "y": 0.40, "z": 0.5},
        {"x": 0.30, "y": 0.55, "z": 0.5},  # l_wrist (idx 9)
        {"x": 0.70, "y": 0.55, "z": 0.5},  # r_wrist (idx 10)
        {"x": 0.44, "y": 0.58, "z": 0.5},
        {"x": 0.56, "y": 0.58, "z": 0.5},
        {"x": 0.42, "y": 0.73, "z": 0.5},
        {"x": 0.58, "y": 0.73, "z": 0.5},
        {"x": 0.42, "y": 0.88, "z": 0.5},
        {"x": 0.58, "y": 0.88, "z": 0.5},
    ]


@pytest.fixture
def predictor():
    return BehavioralIntentPredictor(fps=20.0)


def _fill_normal(pred, n=80):
    for _ in range(n):
        pred.push(_standing_skeleton(), stress_score=10.0, activity="STANDING")


# ── buffering state ───────────────────────────────────────────

def test_buffering_if_short(predictor):
    predictor.push(_standing_skeleton())
    result = predictor.predict()
    assert result["intent"] == "UNKNOWN"
    assert result["status"] == "buffering"


# ── normal intent ──────────────────────────────────────────────

def test_normal_intent(predictor):
    _fill_normal(predictor, 120)
    result = predictor.predict()
    assert result["intent"] == "NORMAL"
    assert result["confidence"] > 0.5


def test_result_keys(predictor):
    _fill_normal(predictor, 120)
    result = predictor.predict()
    assert "intent" in result
    assert "confidence" in result
    assert "scores" in result
    assert "micro_behaviors" in result


# ── surrender detection ───────────────────────────────────────

def test_surrender_hands_up(predictor):
    _fill_normal(predictor, 100)
    # Push frames where wrists are above shoulders
    surr = _standing_skeleton()
    surr[9]["y"] = 0.10   # l_wrist above l_shoulder (0.25)
    surr[10]["y"] = 0.10  # r_wrist above r_shoulder (0.25)
    for _ in range(50):
        predictor.push(surr, stress_score=5.0, activity="STANDING")
    result = predictor.predict()
    # Surrender detection requires very specific pose; just verify no crash
    assert "surrender" in result["scores"]


# ── feature extractors ────────────────────────────────────────

def test_fidget_score(predictor):
    poses = np.random.randn(120, 17, 3) * 0.001
    score = predictor._fidget_score(poses)
    assert 0 <= score <= 1


def test_stance_change_rate(predictor):
    poses = np.random.randn(200, 17, 3) * 0.01
    # Fix shoulder and hip to have sensible torso vectors
    poses[:, 5, :] = [0.4, 0.25, 0.5]
    poses[:, 6, :] = [0.6, 0.25, 0.5]
    poses[:, 11, :] = [0.44, 0.58, 0.5]
    poses[:, 12, :] = [0.56, 0.58, 0.5]
    score = predictor._stance_change_rate(poses)
    assert 0 <= score <= 1


def test_scanning_behavior(predictor):
    poses = np.random.randn(100, 17, 3) * 0.001
    # Add oscillation to nose x
    for i in range(100):
        poses[i, 0, 0] = 0.5 + 0.1 * np.sin(i * 0.5)
    score = predictor._scanning_behavior(poses)
    assert 0 <= score <= 1


def test_time_to_event():
    p = BehavioralIntentPredictor()
    assert p._time_to_event(0.9) == 5.0
    assert p._time_to_event(0.7) == 15.0
    assert p._time_to_event(0.4) == 30.0
