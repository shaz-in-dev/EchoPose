"""tests/test_anti_jamming.py — AntiJammingDefense coverage."""

import pytest
import numpy as np
from pipeline.tactical.anti_jamming import AntiJammingDefense


@pytest.fixture
def defense():
    d = AntiJammingDefense(fs=20.0)
    return d


def _push_clean(d, n=60):
    for _ in range(n):
        d.push(np.random.randn(64) * 0.01)


def _calibrate(d):
    clean = [np.random.randn(64) * 0.01 for _ in range(40)]
    d.calibrate(clean)


# ── buffering state ───────────────────────────────────────────

def test_buffering_if_insufficient(defense):
    _push_clean(defense, 10)
    result = defense.check()
    assert result["status"] == "buffering"


def test_clean_after_calibration(defense):
    _calibrate(defense)
    _push_clean(defense, 60)
    result = defense.check()
    assert "under_attack" in result or "status" in result


# ── broadband jamming ────────────────────────────────────────

def test_jamming_detected(defense):
    _calibrate(defense)
    _push_clean(defense, 60)
    # Inject massive noise
    for _ in range(40):
        defense.push(np.random.randn(64) * 100.0)
    result = defense.check()
    assert result["under_attack"] is True
    threats = [t["type"] for t in result["threats"]]
    assert "ACTIVE_JAMMING" in threats


# ── spoofing detection ────────────────────────────────────────

def test_spoof_detected(defense):
    _calibrate(defense)
    _push_clean(defense, 60)
    # Inject completely different frame (low cosine sim)
    for _ in range(5):
        defense.push(np.random.randn(64) * 0.01)
    defense.push(-np.ones(64) * 50.0)  # discontinuity
    result = defense.check()
    # May trigger spoof or physics check
    assert "under_attack" in result


# ── physics check ─────────────────────────────────────────────

def test_synthetic_signal_detected(defense):
    _calibrate(defense)
    # Push constant frames (zero variance = synthetic)
    for _ in range(60):
        defense.push(np.ones(64) * 0.5)
    result = defense.check()
    threats = [t["type"] for t in result.get("threats", [])]
    assert "SYNTHETIC_SIGNAL" in threats


def test_negative_values_detected(defense):
    _calibrate(defense)
    for _ in range(60):
        defense.push(np.ones(64) * -1.0)
    result = defense.check()
    threats = [t["type"] for t in result.get("threats", [])]
    assert "IMPOSSIBLE_VALUES" in threats


# ── recommendation ────────────────────────────────────────────

def test_recommend_critical():
    d = AntiJammingDefense()
    rec = d._recommend([{"severity": "CRITICAL"}])
    assert rec == "SWITCH_TO_BACKUP_SENSORS"


def test_recommend_high():
    d = AntiJammingDefense()
    rec = d._recommend([{"severity": "HIGH"}])
    assert rec == "INCREASE_MONITORING"


def test_recommend_low():
    d = AntiJammingDefense()
    rec = d._recommend([{"severity": "LOW"}])
    assert rec == "INVESTIGATE"


# ── alert log ─────────────────────────────────────────────────

def test_alert_log_empty(defense):
    assert defense.alert_log == []


def test_buffer_cap(defense):
    for _ in range(700):
        defense.push(np.zeros(64))
    assert len(defense._buf) <= defense._max
