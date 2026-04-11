"""tests/test_sensor_fusion.py — MultiDomainFusion coverage."""

import pytest
import numpy as np
from pipeline.tactical.sensor_fusion import MultiDomainFusion


@pytest.fixture
def fusion():
    return MultiDomainFusion(association_dist=2.0)


# ── ingest & COP ──────────────────────────────────────────────

def test_ingest_creates_track(fusion):
    fusion.ingest("wifi_csi", [{"x": 5, "y": 5, "z": 0, "confidence": 0.9}])
    cop = fusion.get_cop()
    assert cop["total_tracks"] == 1


def test_multi_modality_fuses(fusion):
    fusion.ingest("wifi_csi", [{"x": 5, "y": 5, "z": 0, "confidence": 0.9}])
    fusion.ingest("radar", [{"x": 5.5, "y": 5.5, "z": 0, "confidence": 0.8}])
    cop = fusion.get_cop()
    assert cop["total_tracks"] == 1
    assert len(cop["active_modalities"]) == 2


def test_distant_detections_separate(fusion):
    fusion.ingest("wifi_csi", [{"x": 0, "y": 0, "z": 0, "confidence": 0.9}])
    fusion.ingest("radar", [{"x": 50, "y": 50, "z": 0, "confidence": 0.8}])
    cop = fusion.get_cop()
    assert cop["total_tracks"] == 2


def test_multiple_detections_one_modality(fusion):
    fusion.ingest("wifi_csi", [
        {"x": 0, "y": 0, "z": 0, "confidence": 0.9},
        {"x": 20, "y": 20, "z": 0, "confidence": 0.8},
    ])
    cop = fusion.get_cop()
    assert cop["total_tracks"] == 2


# ── track properties ──────────────────────────────────────────

def test_track_has_fields(fusion):
    fusion.ingest("wifi_csi", [{"x": 1, "y": 2, "z": 3, "confidence": 0.9}])
    cop = fusion.get_cop()
    t = cop["tracks"][0]
    assert "track_id" in t
    assert "position" in t
    assert "velocity_ms" in t
    assert "confidence" in t
    assert "classification" in t


def test_classification_upgrade(fusion):
    fusion.ingest("wifi_csi", [{"x": 5, "y": 5, "z": 0, "confidence": 0.9}])
    fusion.ingest("thermal", [{"x": 5, "y": 5, "z": 0, "confidence": 0.8,
                               "classification": "COMBATANT"}])
    cop = fusion.get_cop()
    assert cop["tracks"][0]["classification"] == "COMBATANT"


# ── track count ───────────────────────────────────────────────

def test_track_count(fusion):
    assert fusion.track_count == 0
    fusion.ingest("wifi_csi", [{"x": 0, "y": 0, "z": 0, "confidence": 0.9}])
    assert fusion.track_count == 1


# ── get specific track ────────────────────────────────────────

def test_get_nonexistent_track(fusion):
    assert fusion.get_track("fake-id") is None


def test_get_existing_track(fusion):
    fusion.ingest("wifi_csi", [{"x": 1, "y": 2, "z": 0, "confidence": 0.9}])
    cop = fusion.get_cop()
    tid = cop["tracks"][0]["track_id"]
    t = fusion.get_track(tid)
    assert t is not None
    assert t["track_id"] == tid


# ── confidence ────────────────────────────────────────────────

def test_confidence_caps_at_099(fusion):
    for mod in ["wifi_csi", "radar", "thermal", "acoustic", "visual", "seismic"]:
        fusion.ingest(mod, [{"x": 5, "y": 5, "z": 0, "confidence": 1.0}])
    cop = fusion.get_cop()
    assert cop["tracks"][0]["confidence"] <= 0.99


# ── cop structure ──────────────────────────────────────────────

def test_cop_has_timestamp(fusion):
    cop = fusion.get_cop()
    assert "timestamp" in cop
    assert "total_tracks" in cop
    assert "active_modalities" in cop
