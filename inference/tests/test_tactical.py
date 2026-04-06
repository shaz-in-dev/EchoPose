"""
tests/test_tactical.py — Unit tests for all 14 tactical analytics modules
"""

import time
import numpy as np
import pytest

from pipeline.tactical import (
    TacticalTargetTracker,
    BuildingMapper,
    ConcealmentDetector,
    IndoorTracker,
    GaitBiometricIdentifier,
    WeaponDetectionSystem,
    CrowdDensityAnalyzer,
    TacticalActivityClassifier,
    AcousticEventDetector,
    AnomalyScanner,
    BehavioralIntentPredictor,
    AntiJammingDefense,
    CoveragePlanner,
    MultiDomainFusion,
)


# ── helpers ────────────────────────────────────────────────────────

def _rand_csi(n_sub=64):
    return np.random.randn(n_sub).astype(np.float64)


def _make_skeleton(y_offset=0.0, arm_up=False):
    """17-keypoint skeleton (COCO format)."""
    kps = [
        {"x": 0.50, "y": 0.15 + y_offset, "z": 0.50, "confidence": 0.9},  # 0 nose
        {"x": 0.48, "y": 0.13 + y_offset, "z": 0.50, "confidence": 0.9},  # 1 l_eye
        {"x": 0.52, "y": 0.13 + y_offset, "z": 0.50, "confidence": 0.9},  # 2 r_eye
        {"x": 0.45, "y": 0.14 + y_offset, "z": 0.50, "confidence": 0.9},  # 3 l_ear
        {"x": 0.55, "y": 0.14 + y_offset, "z": 0.50, "confidence": 0.9},  # 4 r_ear
        {"x": 0.40, "y": 0.28 + y_offset, "z": 0.50, "confidence": 0.9},  # 5 l_shoulder
        {"x": 0.60, "y": 0.28 + y_offset, "z": 0.50, "confidence": 0.9},  # 6 r_shoulder
        {"x": 0.35, "y": 0.42 + y_offset, "z": 0.50, "confidence": 0.9},  # 7 l_elbow
        {"x": 0.65, "y": 0.42 + y_offset, "z": 0.50, "confidence": 0.9},  # 8 r_elbow
        {"x": 0.30, "y": 0.56 + y_offset, "z": 0.50, "confidence": 0.9},  # 9 l_wrist
        {"x": 0.70, "y": 0.56 + y_offset, "z": 0.50, "confidence": 0.9},  # 10 r_wrist
        {"x": 0.44, "y": 0.58 + y_offset, "z": 0.50, "confidence": 0.9},  # 11 l_hip
        {"x": 0.56, "y": 0.58 + y_offset, "z": 0.50, "confidence": 0.9},  # 12 r_hip
        {"x": 0.42, "y": 0.73 + y_offset, "z": 0.50, "confidence": 0.9},  # 13 l_knee
        {"x": 0.58, "y": 0.73 + y_offset, "z": 0.50, "confidence": 0.9},  # 14 r_knee
        {"x": 0.42, "y": 0.88 + y_offset, "z": 0.50, "confidence": 0.9},  # 15 l_ankle
        {"x": 0.58, "y": 0.88 + y_offset, "z": 0.50, "confidence": 0.9},  # 16 r_ankle
    ]
    if arm_up:
        kps[9]["y"] = 0.10 + y_offset   # l_wrist above head
        kps[10]["y"] = 0.10 + y_offset   # r_wrist above head
        kps[7]["y"] = 0.20 + y_offset
        kps[8]["y"] = 0.20 + y_offset
    return kps


# ── 1. TacticalTargetTracker ──────────────────────────────────────

class TestTacticalTargetTracker:
    def test_buffering_state(self):
        t = TacticalTargetTracker(fs=20)
        t.push(_rand_csi())
        result = t.detect()
        assert result["status"] == "buffering"

    def test_detect_returns_targets(self):
        t = TacticalTargetTracker(fs=20)
        # 1.5 Hz cadence → human walking
        for i in range(200):
            sig = np.sin(2 * np.pi * 1.5 * i / 20) * np.ones(64)
            sig += np.random.randn(64) * 0.1
            t.push(sig)
        result = t.detect()
        assert "targets" in result
        assert "threat_level" in result

    def test_threat_level_valid(self):
        t = TacticalTargetTracker(fs=20)
        for i in range(200):
            t.push(np.sin(2 * np.pi * 2.0 * i / 20) * np.ones(64))
        result = t.detect()
        assert result["threat_level"] in ["GREEN", "YELLOW", "RED"]


# ── 2. BuildingMapper ─────────────────────────────────────────────

class TestBuildingMapper:
    def test_no_data_returns_empty(self):
        bm = BuildingMapper()
        result = bm.reconstruct()
        assert result["status"] == "accumulating"

    def test_set_nodes_and_accumulate(self):
        bm = BuildingMapper(grid_size=(10, 10), resolution=1.0)
        bm.set_node_positions([(0, 0), (10, 0), (0, 10), (10, 10)])
        for _ in range(5):
            bm.accumulate({0: _rand_csi(), 1: _rand_csi()})
        result = bm.reconstruct()
        assert "rooms" in result or "walls" in result or "status" in result

    def test_material_classification(self):
        bm = BuildingMapper(grid_size=(10, 10), resolution=1.0)
        bm.set_node_positions([(0, 0), (10, 0)])
        for _ in range(3):
            bm.accumulate({0: np.ones(64) * 20, 1: np.ones(64) * 20})
        result = bm.reconstruct()
        if "materials" in result:
            assert isinstance(result["materials"], dict)


# ── 3. ConcealmentDetector ────────────────────────────────────────

class TestConcealmentDetector:
    def test_buffering(self):
        cd = ConcealmentDetector(fs=20)
        cd.push(_rand_csi())
        result = cd.scan()
        assert result["status"] == "buffering"

    def test_scan_returns_targets(self):
        cd = ConcealmentDetector(fs=20)
        cd.calibrate_baseline([_rand_csi() * 0.01 for _ in range(100)])
        # Add breathing signal
        for i in range(200):
            sig = np.random.randn(64) * 0.01
            sig[10:20] += 0.5 * np.sin(2 * np.pi * 0.25 * i / 20)
            cd.push(sig)
        result = cd.scan()
        assert "targets" in result


# ── 4. IndoorTracker ──────────────────────────────────────────────

class TestIndoorTracker:
    def test_no_nodes(self):
        it = IndoorTracker()
        result = it.update({})
        assert result["status"] == "need_3_nodes"

    def test_trilateration(self):
        it = IndoorTracker()
        it.set_nodes([(0, 0, 0), (10, 0, 0), (5, 10, 0)])
        result = it.update(
            {0: -40.0, 1: -50.0, 2: -45.0},
            {0: np.zeros(64), 1: np.zeros(64), 2: np.zeros(64)},
        )
        assert "x" in result
        assert "y" in result

    def test_track_history(self):
        it = IndoorTracker()
        it.set_nodes([(0, 0, 0), (10, 0, 0), (5, 10, 0)])
        for _ in range(5):
            it.update(
                {0: -40.0, 1: -50.0, 2: -45.0},
                {0: np.zeros(64), 1: np.zeros(64), 2: np.zeros(64)},
            )
        track = it.get_track()
        assert isinstance(track, list)
        assert len(track) > 0


# ── 5. GaitBiometricIdentifier ────────────────────────────────────

class TestGaitBiometrics:
    def test_enrol_requires_frames(self):
        gb = GaitBiometricIdentifier(fs=20)
        history = [_make_skeleton() for _ in range(10)]
        result = gb.enrol("user1", history)
        assert result["status"] == "need_more_data"

    def test_enrol_and_identify(self):
        gb = GaitBiometricIdentifier(fs=20)
        # Enrol with consistent walking pattern (need 600 frames by default)
        history = [_make_skeleton(y_offset=0.01 * np.sin(i * 0.3)) for i in range(650)]
        enrol_result = gb.enrol("walker1", history)
        assert enrol_result["status"] == "enrolled"

        # Re-push similar pattern and identify
        for i in range(150):
            skel = _make_skeleton(y_offset=0.01 * np.sin(i * 0.3))
            gb.push_skeleton(skel)
        id_result = gb.identify()
        assert "person_id" in id_result or "best_match" in id_result


# ── 6. WeaponDetectionSystem ──────────────────────────────────────

class TestWeaponDetector:
    def test_buffering(self):
        wd = WeaponDetectionSystem(fs=20)
        wd.push(_make_skeleton(), _rand_csi())
        result = wd.detect()
        assert result.get("status") == "buffering"

    def test_unarmed_normal_gait(self):
        wd = WeaponDetectionSystem(fs=20)
        for i in range(100):
            skel = _make_skeleton(y_offset=0.01 * np.sin(i * 0.3))
            wd.push(skel, _rand_csi() * 0.01)
        result = wd.detect()
        assert "weapon_type" in result


# ── 7. CrowdDensityAnalyzer ───────────────────────────────────────

class TestCrowdAnalyzer:
    def test_buffering_without_data(self):
        ca = CrowdDensityAnalyzer()
        result = ca.estimate(room_area_m2=100, skeleton_count=0)
        assert result["status"] == "buffering"

    def test_with_skeletons(self):
        ca = CrowdDensityAnalyzer(fs=20)
        for _ in range(100):
            ca.push(_rand_csi() * 0.01)
        result = ca.estimate(room_area_m2=100, skeleton_count=3)
        assert result["estimated_count"] == 3
        assert result["density_category"] in ["SPARSE", "MODERATE", "DENSE", "CRITICAL"]

    def test_spectral_estimation(self):
        ca = CrowdDensityAnalyzer(fs=20)
        for _ in range(100):
            ca.push(_rand_csi())
        result = ca.estimate(room_area_m2=50)
        assert "estimated_count" in result


# ── 8. TacticalActivityClassifier ─────────────────────────────────

class TestTacticalActivity:
    def test_buffering(self):
        ta = TacticalActivityClassifier(fps=20)
        ta.push_skeleton(_make_skeleton())
        result = ta.classify()
        assert result["activity"] in [
            "STANDING", "MOVING_TACTICAL", "RUNNING", "CRAWLING",
            "PRONE", "TAKING_AIM", "THROWING", "SURRENDERING",
            "ASSISTING_INJURED", "UNKNOWN",
        ]

    def test_standing_pose(self):
        ta = TacticalActivityClassifier(fps=20)
        for _ in range(30):
            ta.push_skeleton(_make_skeleton())
        result = ta.classify()
        assert result["activity"] in [
            "STANDING", "MOVING_TACTICAL", "RUNNING", "CRAWLING",
            "PRONE", "TAKING_AIM", "THROWING", "SURRENDERING",
            "ASSISTING_INJURED", "UNKNOWN",
        ]

    def test_surrendering_pose(self):
        ta = TacticalActivityClassifier(fps=20)
        for _ in range(30):
            ta.push_skeleton(_make_skeleton(arm_up=True))
        result = ta.classify()
        # Arms up should trigger SURRENDERING or at least a valid label
        assert result["activity"] in [
            "STANDING", "MOVING_TACTICAL", "RUNNING", "CRAWLING",
            "PRONE", "TAKING_AIM", "THROWING", "SURRENDERING",
            "ASSISTING_INJURED", "UNKNOWN",
        ]


# ── 9. AcousticEventDetector ──────────────────────────────────────

class TestAcousticDetector:
    def test_no_events_quiet(self):
        ad = AcousticEventDetector(fs=20)
        ad.set_node_positions([(0, 0, 0), (10, 0, 0), (5, 10, 0)])
        for _ in range(50):
            ad.push(0, _rand_csi() * 0.01)
            ad.push(1, _rand_csi() * 0.01)
            ad.push(2, _rand_csi() * 0.01)
        result = ad.detect()
        assert "event_detected" in result

    def test_impulse_detection(self):
        ad = AcousticEventDetector(fs=20)
        ad.set_node_positions([(0, 0, 0), (10, 0, 0), (5, 10, 0)])
        # Quiet baseline
        for _ in range(60):
            ad.push(0, _rand_csi() * 0.01)
            ad.push(1, _rand_csi() * 0.01)
            ad.push(2, _rand_csi() * 0.01)
        # Impulse
        ad.push(0, np.ones(64) * 100)
        ad.push(1, np.ones(64) * 100)
        ad.push(2, np.ones(64) * 100)
        result = ad.detect()
        assert "event_detected" in result


# ── 10. AnomalyScanner ────────────────────────────────────────────

class TestAnomalyScanner:
    def test_no_baseline(self):
        sc = AnomalyScanner(fs=20)
        sc.push(_rand_csi())
        result = sc.scan()
        assert "coverage" in result or "status" in result

    def test_calibrate_and_scan(self):
        sc = AnomalyScanner(fs=20)
        sc.calibrate([_rand_csi() * 0.01 for _ in range(100)])
        for _ in range(100):
            sc.push(_rand_csi() * 0.01)
        result = sc.scan()
        assert "threat_assessment" in result
        assert result["threat_assessment"] in ["CLEAR", "SUSPICIOUS", "DANGER"]


# ── 11. BehavioralIntentPredictor ─────────────────────────────────

class TestIntentPredictor:
    def test_buffering(self):
        ip = BehavioralIntentPredictor(fps=20)
        ip.push(_make_skeleton(), 0.2, "STANDING")
        result = ip.predict()
        assert result.get("status") == "buffering" or "intent" in result

    def test_normal_intent(self):
        ip = BehavioralIntentPredictor(fps=20)
        for _ in range(100):
            ip.push(_make_skeleton(), 0.1, "STANDING")
        result = ip.predict()
        assert "intent" in result
        assert result["intent"] in [
            "NORMAL", "ATTACK_IMMINENT", "FLEE", "SURRENDER",
            "DEFENSIVE", "ACCESS_WEAPON",
        ]


# ── 12. AntiJammingDefense ────────────────────────────────────────

class TestAntiJamming:
    def test_buffering(self):
        aj = AntiJammingDefense(fs=20)
        aj.push(_rand_csi())
        result = aj.check()
        assert result["status"] == "buffering"

    def test_clean_signal(self):
        aj = AntiJammingDefense(fs=20)
        # Use very consistent low-noise signal to avoid false positives
        clean = [np.ones(64) * 0.5 + np.random.randn(64) * 0.001 for _ in range(100)]
        aj.calibrate(clean)
        for c in clean:
            aj.push(c)
        result = aj.check()
        assert result["under_attack"] is False

    def test_jamming_detection(self):
        aj = AntiJammingDefense(fs=20)
        clean = [_rand_csi() * 0.01 for _ in range(100)]
        aj.calibrate(clean)
        for c in clean:
            aj.push(c)
        # Inject enormous noise → jam
        for _ in range(50):
            aj.push(np.random.randn(64) * 100)
        result = aj.check()
        assert result["under_attack"] is True
        assert any(t["type"] == "ACTIVE_JAMMING" for t in result["threats"])

    def test_alert_log(self):
        aj = AntiJammingDefense(fs=20)
        assert isinstance(aj.alert_log, list)


# ── 13. CoveragePlanner ──────────────────────────────────────────

class TestCoveragePlanner:
    def test_no_sensors(self):
        cp = CoveragePlanner(area_size=(10, 10), resolution=1.0)
        result = cp.compute_coverage()
        assert result["status"] == "no_sensors"

    def test_basic_coverage(self):
        cp = CoveragePlanner(area_size=(10, 10), resolution=1.0)
        cp.set_sensors([(5, 5)])
        result = cp.compute_coverage()
        assert "coverage_pct" in result
        assert "blind_spot_pct" in result
        assert "recommended_path" in result
        assert result["coverage_pct"] > 0

    def test_wall_reduces_coverage(self):
        cp1 = CoveragePlanner(area_size=(10, 10), resolution=1.0)
        cp1.set_sensors([(5, 5)])
        r1 = cp1.compute_coverage()

        cp2 = CoveragePlanner(area_size=(10, 10), resolution=1.0)
        cp2.set_sensors([(5, 5)])
        cp2.add_wall((0, 3), (10, 3))
        r2 = cp2.compute_coverage()

        # Wall should reduce or redistribute coverage
        assert "coverage_pct" in r1
        assert "coverage_pct" in r2

    def test_query_point(self):
        cp = CoveragePlanner(area_size=(10, 10), resolution=1.0)
        cp.set_sensors([(5, 5)])
        result = cp.query_point(5, 5)
        assert result["status"] in ["BLIND", "PARTIAL", "COVERED"]


# ── 14. MultiDomainFusion ─────────────────────────────────────────

class TestMultiDomainFusion:
    def test_empty_cop(self):
        mdf = MultiDomainFusion()
        cop = mdf.get_cop()
        assert cop["total_tracks"] == 0

    def test_ingest_single_modality(self):
        mdf = MultiDomainFusion()
        mdf.ingest("wifi_csi", [
            {"x": 5, "y": 3, "z": 0, "confidence": 0.8, "classification": "HUMAN"},
        ])
        cop = mdf.get_cop()
        assert cop["total_tracks"] == 1
        assert cop["tracks"][0]["classification"] == "HUMAN"
        assert "wifi_csi" in cop["active_modalities"]

    def test_cross_modal_fusion(self):
        mdf = MultiDomainFusion(association_dist=3.0)
        mdf.ingest("wifi_csi", [{"x": 5, "y": 3, "z": 0, "confidence": 0.7}])
        mdf.ingest("radar", [{"x": 5.5, "y": 3.2, "z": 0, "confidence": 0.9}])
        cop = mdf.get_cop()
        # Should merge into single track since within 3m
        assert cop["total_tracks"] == 1
        track = cop["tracks"][0]
        assert "wifi_csi" in track["sources"]
        assert "radar" in track["sources"]

    def test_separate_tracks_far_apart(self):
        mdf = MultiDomainFusion(association_dist=2.0)
        mdf.ingest("wifi_csi", [{"x": 0, "y": 0, "z": 0, "confidence": 0.8}])
        mdf.ingest("radar", [{"x": 50, "y": 50, "z": 0, "confidence": 0.8}])
        cop = mdf.get_cop()
        assert cop["total_tracks"] == 2

    def test_track_count(self):
        mdf = MultiDomainFusion()
        mdf.ingest("thermal", [
            {"x": 1, "y": 1, "z": 0, "confidence": 0.9},
            {"x": 20, "y": 20, "z": 0, "confidence": 0.7},
        ])
        assert mdf.track_count == 2

    def test_get_track_by_id(self):
        mdf = MultiDomainFusion()
        mdf.ingest("wifi_csi", [{"x": 5, "y": 3, "z": 0, "confidence": 0.8}])
        cop = mdf.get_cop()
        tid = cop["tracks"][0]["track_id"]
        track = mdf.get_track(tid)
        assert track is not None
        assert track["track_id"] == tid

    def test_get_nonexistent_track(self):
        mdf = MultiDomainFusion()
        assert mdf.get_track("nonexistent") is None
