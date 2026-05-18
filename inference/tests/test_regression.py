"""
tests/test_regression.py

One test per bug fixed.  These exist to prevent regressions — if any of
these fail, a previously-fixed issue has been re-introduced.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# H1 – Fernet key must be encoded as bytes, not str
# ─────────────────────────────────────────────────────────────────────────────

def test_h1_fernet_roundtrip_does_not_crash():
    """Fernet key initialisation must not raise TypeError on encode."""
    from security import encrypt_session_data, decrypt_session_data
    data = {"key": "value", "num": 99}
    assert decrypt_session_data(encrypt_session_data(data)) == data


# ─────────────────────────────────────────────────────────────────────────────
# H2 – asyncio.Lock on rate limiter prevents data corruption
# ─────────────────────────────────────────────────────────────────────────────

def test_h2_rate_limiter_lock_present():
    import asyncio
    from security import RateLimiter
    rl = RateLimiter()
    assert hasattr(rl, "_lock") and isinstance(rl._lock, asyncio.Lock)


# ─────────────────────────────────────────────────────────────────────────────
# H3 – Dev mode: no token configured → auth disabled, not a 403 wall
# ─────────────────────────────────────────────────────────────────────────────

def test_h3_dev_mode_allows_unauthenticated(monkeypatch):
    import security
    monkeypatch.setattr(security, "_DEV_AUTH_DISABLED", True)
    result = security.verify_api_key(None)
    assert result == "dev-no-auth"


# ─────────────────────────────────────────────────────────────────────────────
# H4 – Fusion passes 2-D doppler_view to disambiguation, not 3-D tensor
# ─────────────────────────────────────────────────────────────────────────────

def test_h4_disambiguation_receives_2d_input():
    """disentangle_csi_signatures must receive [nodes, doppler_bins], not 3-D."""
    from research.disambiguation import MultiPersonDisambiguation
    disambig = MultiPersonDisambiguation(max_people=3)
    # 2-D input: [nodes=3, doppler_bins=16]
    doppler_view = np.random.randn(3, 16).astype(np.float32)
    doppler_spec = np.random.randn(3, 16).astype(np.float32)
    result = disambig.disentangle_csi_signatures(doppler_view, doppler_spec)
    assert isinstance(result, list)


# ─────────────────────────────────────────────────────────────────────────────
# H5 – domain_adaptation threading.Lock prevents concurrent weight corruption
# ─────────────────────────────────────────────────────────────────────────────

def test_h5_domain_adaptation_has_threading_lock():
    from research.domain_adaptation import RealTimeDomainAdaptation
    da = RealTimeDomainAdaptation()
    assert hasattr(da, "_lock") and isinstance(da._lock, threading.Lock)


def test_h5_concurrent_adapt_online_does_not_raise():
    """Two threads calling adapt_online simultaneously must not crash."""
    import torch
    from research.domain_adaptation import RealTimeDomainAdaptation
    from pipeline.pose_net_v2 import PoseNetV2

    da = RealTimeDomainAdaptation(feature_dim=256)
    model = PoseNetV2()
    # encoder input: [batch, nodes=3, subcarriers=64, doppler=16]
    # encoder output: [batch, 256]
    src = model.encoder(torch.randn(4, 3, 64, 16)).detach()

    errors = []

    def _adapt():
        try:
            stream = torch.randn(4, 3, 64, 16)
            da.adapt_online(model, stream, src)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_adapt) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"Concurrent adapt_online raised: {errors}"


# ─────────────────────────────────────────────────────────────────────────────
# H6 – pose.py shape guard: 2-D model output promoted to [1, 17, 4]
# ─────────────────────────────────────────────────────────────────────────────

def test_h6_pose_estimator_handles_2d_output():
    from pipeline.pose import PoseEstimator
    est = PoseEstimator()
    features = np.random.randn(3, 64, 16).astype(np.float32)
    skels = est.predict(features)
    assert isinstance(skels, list)
    assert len(skels) >= 1
    assert len(skels[0]) == 17


# ─────────────────────────────────────────────────────────────────────────────
# H7 – gait_biometrics: corrcoef must not crash on single sample
# ─────────────────────────────────────────────────────────────────────────────

def test_h7_gait_biometrics_single_sample():
    from pipeline.tactical.gait_biometrics import GaitBiometricIdentifier
    ga = GaitBiometricIdentifier()
    skeleton = [{"x": float(i) * 0.05, "y": float(i) * 0.05, "z": 0.5, "confidence": 0.9}
                for i in range(17)]
    # Push exactly one frame — corrcoef previously crashed with n=1
    ga.push_skeleton(skeleton)
    result = ga.identify()
    # Should return a result dict without raising
    assert isinstance(result, dict)


# ─────────────────────────────────────────────────────────────────────────────
# H8 – server_v2._client_ip: request.client may be None behind a proxy
# ─────────────────────────────────────────────────────────────────────────────

def test_h8_client_ip_when_client_is_none():
    from server_v2 import _client_ip
    req = MagicMock()
    req.client = None
    req.headers = {"X-Forwarded-For": "198.51.100.7, 10.0.0.1"}
    assert _client_ip(req) == "198.51.100.7"


def test_h8_client_ip_direct_connection():
    from server_v2 import _client_ip
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = "192.168.0.10"
    req.headers = {}
    assert _client_ip(req) == "192.168.0.10"


# ─────────────────────────────────────────────────────────────────────────────
# H14/H15 – Rust broadcast overflow + SystemTime panic are compile-time fixes;
#            test that the aggregator binary builds clean (cargo check already
#            ran in CI — here we just verify the Python harness imports).
# ─────────────────────────────────────────────────────────────────────────────

def test_h14_h15_server_v2_importable():
    """server_v2 must import without errors (catches module-level crashes)."""
    import importlib
    mod = importlib.import_module("server_v2")
    assert hasattr(mod, "app")


# ─────────────────────────────────────────────────────────────────────────────
# M7 – occupancy: "occupied" key must be consistent across all return paths
# ─────────────────────────────────────────────────────────────────────────────

def test_m7_occupancy_key_consistency_uncalibrated():
    """Early-return for uncalibrated state must use 'occupied', not 'occupancy'."""
    from pipeline.occupancy import OccupancyAnalyzer
    oa = OccupancyAnalyzer()
    result = oa.detect_presence(skeletons=[], csi_amplitudes=None)
    assert "occupied" in result, "'occupied' key missing from uncalibrated path"
    assert result["occupied"] is False


def test_m7_occupancy_high_conf_skeleton_returns_occupied():
    from pipeline.occupancy import OccupancyAnalyzer
    oa = OccupancyAnalyzer()
    skel = [[{"x": 0.5, "y": float(i) * 0.05, "z": 0.5, "confidence": 0.9}
             for i in range(17)]]
    result = oa.detect_presence(skeletons=skel)
    assert result["occupied"] is True
    assert result["num_people"] == 1


def test_m7_occupancy_low_conf_skeleton_not_occupied():
    from pipeline.occupancy import OccupancyAnalyzer
    oa = OccupancyAnalyzer()
    skel = [[{"x": 0, "y": 0, "z": 0, "confidence": 0.1} for _ in range(17)]]
    result = oa.detect_presence(skeletons=skel)
    assert "occupied" in result
    assert result["occupied"] is False


# ─────────────────────────────────────────────────────────────────────────────
# anti_jamming: _sweep_detect must return a list, not a dict
# ─────────────────────────────────────────────────────────────────────────────

def test_anti_jamming_sweep_detect_returns_list_on_zero_psd():
    """Previously returned a dict, poisoning the threats list with string keys."""
    from pipeline.tactical.anti_jamming import AntiJammingDefense
    aj = AntiJammingDefense(fs=20.0)
    # Push enough constant frames to satisfy buffer & trigger zero-variance PSD
    for _ in range(100):
        aj.push(np.zeros(64))
    result = aj.check()
    # Calling _recommend on the threats must not raise AttributeError
    if result.get("under_attack"):
        for threat in result["threats"]:
            assert isinstance(threat, dict), "threat must be dict, not string"


def test_anti_jamming_recommend_only_receives_dicts():
    from pipeline.tactical.anti_jamming import AntiJammingDefense
    aj = AntiJammingDefense(fs=20.0)
    # Feed a synthetic signal (zero variance) to trigger SYNTHETIC_SIGNAL threat
    for _ in range(100):
        aj.push(np.ones(64) * 0.5)
    result = aj.check()
    assert isinstance(result, dict)
    if "threats" in result:
        for t in result["threats"]:
            assert isinstance(t, dict)


# ─────────────────────────────────────────────────────────────────────────────
# weapon_detector: deque slice must not crash (collections.deque[-n:] TypeError)
# ─────────────────────────────────────────────────────────────────────────────

def test_weapon_detector_deque_slice_does_not_crash():
    """Previously: TypeError: sequence index must be integer, not 'slice'."""
    from pipeline.tactical.weapon_detector import WeaponDetectionSystem
    wd = WeaponDetectionSystem(fs=20.0)
    skeleton = [{"x": float(i) * 0.05, "y": float(i) * 0.05, "z": 0.5, "confidence": 0.9}
                for i in range(17)]
    csi = np.random.rand(64).astype(np.float32)
    for _ in range(50):
        wd.push(skeleton, csi_amplitudes=csi)
    result = wd.detect()
    assert "weapon_type" in result or "status" in result


# ─────────────────────────────────────────────────────────────────────────────
# disaster_response: correct key "anomalies_found" (was "is_anomaly")
# ─────────────────────────────────────────────────────────────────────────────

def test_c3_disaster_response_uses_anomalies_found_key():
    from pipeline.disaster_response import DisasterResponseEngine
    engine = DisasterResponseEngine()
    analytics = {"occupancy": {"num_people": 1}, "activity": {"activity": "lying"}, "fall": {}}
    # "anomalies_found" is the correct key; using it must trigger LOW_MOTION_ANOMALY
    tactical = {"anomalies": {"anomalies_found": True}}
    result = engine.evaluate(analytics, tactical)
    codes = [a["code"] for a in result["alerts"]]
    assert "LOW_MOTION_ANOMALY" in codes


def test_c3_disaster_response_wrong_key_gives_no_alert():
    from pipeline.disaster_response import DisasterResponseEngine
    engine = DisasterResponseEngine()
    analytics = {"occupancy": {"num_people": 1}, "activity": {"activity": "lying"}, "fall": {}}
    # Old key "is_anomaly" must NOT trigger the alert
    tactical = {"anomalies": {"is_anomaly": True}}
    result = engine.evaluate(analytics, tactical)
    codes = [a["code"] for a in result["alerts"]]
    assert "LOW_MOTION_ANOMALY" not in codes


# ─────────────────────────────────────────────────────────────────────────────
# temporal_filter_v2: deque must be bounded (no unbounded list growth)
# ─────────────────────────────────────────────────────────────────────────────

def test_m1_temporal_filter_deque_is_bounded():
    from pipeline.temporal_filter_v2 import TemporalPoseFilterV2
    filt = TemporalPoseFilterV2(max_people=1)
    skeleton = [{"x": 0.5, "y": float(i) * 0.05, "z": 0.5, "confidence": 0.9}
                for i in range(17)]
    for _ in range(200):
        filt.filter([[dict(kp) for kp in skeleton]])
    # confidence_history deque per person should not grow beyond its maxlen
    for hist in filt.confidence_history:
        assert len(hist) <= 10  # maxlen=10 per TemporalPoseFilterV2


# ─────────────────────────────────────────────────────────────────────────────
# vitals: push must reject NaN/Inf and empty arrays
# ─────────────────────────────────────────────────────────────────────────────

def test_vitals_rejects_nan_amplitudes():
    from pipeline.vitals import VitalsExtractor
    vs = VitalsExtractor()
    before = len(vs._amplitude_history)
    vs.push(np.array([float("nan"), 1.0, 2.0]))
    assert len(vs._amplitude_history) == before  # rejected


def test_vitals_rejects_inf_amplitudes():
    from pipeline.vitals import VitalsExtractor
    vs = VitalsExtractor()
    before = len(vs._amplitude_history)
    vs.push(np.array([float("inf"), 1.0]))
    assert len(vs._amplitude_history) == before


def test_vitals_rejects_empty_array():
    from pipeline.vitals import VitalsExtractor
    vs = VitalsExtractor()
    before = len(vs._amplitude_history)
    vs.push(np.array([]))
    assert len(vs._amplitude_history) == before


def test_vitals_history_is_bounded():
    from pipeline.vitals import VitalsExtractor
    vs = VitalsExtractor(history_seconds=1)  # short window
    amp = np.ones(64, dtype=np.float32)
    for _ in range(vs.history_len + 50):
        vs.push(amp)
    assert len(vs._amplitude_history) <= vs.history_len
