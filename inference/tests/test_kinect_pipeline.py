"""Tests for the Kinect data-collection pipeline (inference/kinect/)."""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pytest

from kinect.pose_source import BodyFrame, Joint
from kinect.mock_kinect import MockKinectSource, _STAND_TEMPLATE
from kinect.joint_mapping import JointMapper, KINECT_TO_COCO
from kinect.transform import CoordTransform
from kinect.sync import SyncCorrelator
from kinect.recorder import AlignedRecorder


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_source():
    src = MockKinectSource(fps=30, emit_spike=False, seed=7)
    src.open()
    yield src
    src.close()


@pytest.fixture
def kinect_body(mock_source):
    return mock_source.read_one().first_body()


# ── MockKinectSource ──────────────────────────────────────────────────────────

class TestMockKinectSource:
    def test_emits_25_joints(self, mock_source):
        frame = mock_source.read_one()
        assert frame is not None
        assert len(frame.bodies) == 1
        assert len(frame.bodies[0]) == 25

    def test_frame_index_increments(self, mock_source):
        f1 = mock_source.read_one()
        f2 = mock_source.read_one()
        assert f2.frame_index == f1.frame_index + 1

    def test_closed_source_returns_none(self):
        src = MockKinectSource()
        src.open()
        src.close()
        assert src.read_one() is None

    def test_two_people(self):
        src = MockKinectSource(n_people=2)
        src.open()
        frame = src.read_one()
        src.close()
        assert frame.person_count == 2
        # Second person offset sideways, not overlapping
        x0 = frame.bodies[0][0].x
        x1 = frame.bodies[1][0].x
        assert abs(x1 - x0) > 0.5

    def test_joints_near_template(self, mock_source):
        """Noise is small — joints should be close to the standing template."""
        body = mock_source.read_one().first_body()
        head = body[3]
        tx, ty, tz = _STAND_TEMPLATE[3]
        # Walking animation moves z up to ±0.3, noise ±~0.05
        assert abs(head.x - tx) < 0.2
        assert abs(head.y - ty) < 0.5
        assert abs(head.z - tz) < 0.5

    def test_context_manager(self):
        with MockKinectSource() as src:
            assert src.read_one() is not None
        assert src.read_one() is None

    def test_reproducible_with_seed(self):
        with MockKinectSource(seed=1) as a, MockKinectSource(seed=1) as b:
            ja = a.read_one().first_body()[0]
            jb = b.read_one().first_body()[0]
        assert ja.x == pytest.approx(jb.x)
        assert ja.y == pytest.approx(jb.y)


# ── JointMapper ───────────────────────────────────────────────────────────────

class TestJointMapper:
    def test_mapping_table_covers_all_17(self):
        assert len(KINECT_TO_COCO) == 17
        for primary, fallback in KINECT_TO_COCO:
            assert 0 <= primary < 25
            assert fallback is None or 0 <= fallback < 25

    def test_outputs_17_joints(self, kinect_body):
        coco = JointMapper().map(kinect_body)
        assert len(coco) == 17

    def test_rejects_short_input(self):
        with pytest.raises(ValueError, match="25 Kinect joints"):
            JointMapper().map([Joint(0, 0, 0, 1.0)] * 10)

    def test_shoulders_map_correctly(self, kinect_body):
        coco = JointMapper().map(kinect_body)
        # COCO 5 = left_shoulder ← Kinect 4 = ShoulderLeft
        assert coco[5].x == pytest.approx(kinect_body[4].x)
        # COCO 6 = right_shoulder ← Kinect 8 = ShoulderRight
        assert coco[6].x == pytest.approx(kinect_body[8].x)

    def test_eyes_ears_derived_from_head_low_confidence(self, kinect_body):
        coco = JointMapper().map(kinect_body)
        head = kinect_body[3]
        for idx in (1, 2, 3, 4):  # eyes + ears
            assert coco[idx].x == pytest.approx(head.x)
            assert coco[idx].confidence <= 0.2

    def test_fallback_used_when_primary_untracked(self):
        body = [Joint(float(i), float(i), 2.0, 1.0) for i in range(25)]
        # Kill WristLeft (6) → COCO 9 left_wrist falls back to ElbowLeft (5)
        body[6] = Joint(99.0, 99.0, 99.0, 0.0)
        coco = JointMapper().map(body)
        assert coco[9].x == pytest.approx(5.0)        # ElbowLeft position
        assert coco[9].confidence == pytest.approx(0.7)  # downgraded

    def test_array_shape_and_dtype(self, kinect_body):
        arr = JointMapper().map_to_array(kinect_body)
        assert arr.shape == (17, 4)
        assert arr.dtype == np.float32

    def test_mapping_is_deterministic(self, kinect_body):
        m = JointMapper()
        a = m.map_to_array(kinect_body)
        b = m.map_to_array(kinect_body)
        np.testing.assert_array_equal(a, b)


# ── CoordTransform ────────────────────────────────────────────────────────────

class TestCoordTransform:
    def test_identity_is_noop(self):
        j = Joint(1.0, 2.0, 3.0, 0.9)
        out = CoordTransform.identity().apply(j)
        assert (out.x, out.y, out.z) == pytest.approx((1.0, 2.0, 3.0))
        assert out.confidence == 0.9

    def test_translation(self):
        t = CoordTransform(translation=(10.0, 0.0, -5.0))
        out = t.apply(Joint(1.0, 2.0, 3.0, 1.0))
        assert (out.x, out.y, out.z) == pytest.approx((11.0, 2.0, -2.0))

    def test_yaw_90_degrees(self):
        # 90° yaw: +x → -z, +z → +x  (rotation about Y axis)
        t = CoordTransform(yaw_deg=90.0)
        out = t.apply(Joint(1.0, 0.0, 0.0, 1.0))
        assert out.x == pytest.approx(0.0, abs=1e-9)
        assert out.z == pytest.approx(-1.0, abs=1e-9)
        out2 = t.apply(Joint(0.0, 0.0, 1.0, 1.0))
        assert out2.x == pytest.approx(1.0, abs=1e-9)

    def test_yaw_preserves_height(self):
        t = CoordTransform(yaw_deg=45.0)
        out = t.apply(Joint(1.0, 1.7, 1.0, 1.0))
        assert out.y == pytest.approx(1.7)

    def test_rotation_preserves_distances(self):
        t = CoordTransform(yaw_deg=33.0)
        a = t.apply(Joint(1.0, 0.5, 2.0, 1.0))
        b = t.apply(Joint(-1.0, 1.5, 0.5, 1.0))
        orig = math.dist((1.0, 0.5, 2.0), (-1.0, 1.5, 0.5))
        new  = math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
        assert new == pytest.approx(orig)

    def test_apply_array_matches_apply(self):
        t = CoordTransform(translation=(1.0, 2.0, 3.0), yaw_deg=30.0)
        arr = np.array([[0.5, 1.0, 2.0, 0.8]], dtype=np.float32)
        out_arr = t.apply_array(arr)
        out_j = t.apply(Joint(0.5, 1.0, 2.0, 0.8))
        assert out_arr[0, 0] == pytest.approx(out_j.x, abs=1e-5)
        assert out_arr[0, 1] == pytest.approx(out_j.y, abs=1e-5)
        assert out_arr[0, 2] == pytest.approx(out_j.z, abs=1e-5)
        assert out_arr[0, 3] == pytest.approx(0.8)  # confidence untouched

    def test_apply_array_does_not_mutate_input(self):
        t = CoordTransform(translation=(1.0, 0.0, 0.0))
        arr = np.zeros((17, 4), dtype=np.float32)
        before = arr.copy()
        t.apply_array(arr)
        np.testing.assert_array_equal(arr, before)

    def test_save_load_roundtrip(self, tmp_path):
        t = CoordTransform(translation=(1.5, 0.8, -2.0), yaw_deg=72.5)
        p = tmp_path / "transform.json"
        t.save(p)
        loaded = CoordTransform.load(p)
        j = Joint(0.3, 1.1, 2.2, 1.0)
        a, b = t.apply(j), loaded.apply(j)
        assert (a.x, a.y, a.z) == pytest.approx((b.x, b.y, b.z))


# ── SyncCorrelator ────────────────────────────────────────────────────────────

class TestSyncCorrelator:
    @staticmethod
    def _spike(n, fps, at_s, noise, rng):
        ts = np.arange(n) / fps
        val = np.exp(-0.5 * ((ts - at_s) / 0.15) ** 2) + rng.normal(0, noise, n)
        return ts, val

    @pytest.mark.parametrize("true_offset", [0.25, 0.75, 1.5, -0.5])
    def test_recovers_known_offset(self, true_offset):
        rng = np.random.default_rng(42)
        c = SyncCorrelator(max_lag_s=3.0, min_samples=20)
        kin_ts, kin_v = self._spike(300, 30.0, 4.0, 0.1, rng)
        csi_ts, csi_v = self._spike(200, 20.0, 4.0 - true_offset, 0.1, rng)
        for t, v in zip(kin_ts, kin_v):
            c.push_kinect(t, v)
        for t, v in zip(csi_ts, csi_v):
            c.push_csi(t, v)
        offset, conf = c.estimate_offset()
        assert offset is not None
        assert offset == pytest.approx(true_offset, abs=0.1)
        assert conf > 0.2

    def test_insufficient_data_returns_none(self):
        c = SyncCorrelator(min_samples=30)
        for i in range(5):
            c.push_kinect(i * 0.033, 0.1)
            c.push_csi(i * 0.05, 0.1)
        offset, conf = c.estimate_offset()
        assert offset is None
        assert conf == 0.0

    def test_no_overlap_returns_none(self):
        c = SyncCorrelator(min_samples=20)
        for i in range(50):
            c.push_kinect(i * 0.033, 0.1)         # 0 – 1.6 s
            c.push_csi(100 + i * 0.05, 0.1)       # 100 – 102.5 s
        offset, _ = c.estimate_offset()
        assert offset is None

    def test_clear_resets_buffers(self):
        c = SyncCorrelator(min_samples=10)
        rng = np.random.default_rng(0)
        ts, v = self._spike(100, 30.0, 1.5, 0.05, rng)
        for t, x in zip(ts, v):
            c.push_kinect(t, x)
            c.push_csi(t, x)
        c.clear()
        offset, _ = c.estimate_offset()
        assert offset is None

    def test_resample_onto_grid(self):
        src_ts = np.array([0.0, 1.0, 2.0])
        src_v  = np.array([0.0, 10.0, 20.0])
        grid   = np.array([-1.0, 0.5, 1.5, 3.0])
        out = SyncCorrelator.resample_onto_grid(src_ts, src_v, grid, fill_value=-1.0)
        assert out[0] == pytest.approx(-1.0)   # out of range → fill
        assert out[1] == pytest.approx(5.0)    # interp
        assert out[2] == pytest.approx(15.0)
        assert out[3] == pytest.approx(-1.0)


# ── AlignedRecorder ───────────────────────────────────────────────────────────

class TestAlignedRecorder:
    def _frame(self, ts=None):
        with MockKinectSource(seed=3) as src:
            frame = src.read_one()
        if ts is not None:
            frame.timestamp_s = ts
        return frame

    def test_write_read_roundtrip(self, tmp_path):
        out = tmp_path / "session.npz"
        rng = np.random.default_rng(0)
        meta = {"room": "test_room", "subject": "test_subject"}
        with AlignedRecorder(out, metadata=meta) as rec:
            for i in range(10):
                csi = rng.normal(1.0, 0.1, (3, 64, 16)).astype(np.float32)
                assert rec.add_window(csi, self._frame(ts=100.0 + i * 0.05))
            assert rec.window_count == 10

        data = AlignedRecorder.load(out)
        assert data["features"].shape == (10, 3, 64, 16)
        assert data["poses"].shape == (10, 17, 4)
        assert data["timestamps"].shape == (10,)
        assert data["metadata"]["room"] == "test_room"
        assert data["metadata"]["n_windows"] == 10
        np.testing.assert_allclose(
            data["timestamps"], 100.0 + np.arange(10) * 0.05)

    def test_rejects_wrong_csi_shape(self, tmp_path):
        rec = AlignedRecorder(tmp_path / "x.npz")
        rec.open()
        with pytest.raises(ValueError, match=r"\(3, 64, 16\)"):
            rec.add_window(np.zeros((64, 16)), self._frame())

    def test_skips_frame_without_bodies(self, tmp_path):
        rec = AlignedRecorder(tmp_path / "x.npz")
        rec.open()
        empty = BodyFrame(timestamp_s=1.0, bodies=[], source_id="test")
        assert rec.add_window(np.zeros((3, 64, 16), dtype=np.float32), empty) is False
        assert rec.window_count == 0

    def test_respects_max_windows(self, tmp_path):
        rec = AlignedRecorder(tmp_path / "x.npz", max_windows=2)
        rec.open()
        csi = np.zeros((3, 64, 16), dtype=np.float32)
        assert rec.add_window(csi, self._frame())
        assert rec.add_window(csi, self._frame())
        assert rec.add_window(csi, self._frame()) is False
        assert rec.window_count == 2

    def test_empty_session_writes_nothing(self, tmp_path):
        out = tmp_path / "empty.npz"
        with AlignedRecorder(out):
            pass
        assert not out.exists()

    def test_transform_applied_to_poses(self, tmp_path):
        out_a = tmp_path / "a.npz"
        out_b = tmp_path / "b.npz"
        csi = np.zeros((3, 64, 16), dtype=np.float32)
        frame = self._frame()

        with AlignedRecorder(out_a) as rec:
            rec.add_window(csi, frame)
        shift = CoordTransform(translation=(5.0, 0.0, 0.0))
        with AlignedRecorder(out_b, transform=shift) as rec:
            rec.add_window(csi, frame)

        a = AlignedRecorder.load(out_a)["poses"]
        b = AlignedRecorder.load(out_b)["poses"]
        np.testing.assert_allclose(b[0, :, 0], a[0, :, 0] + 5.0, atol=1e-5)
        np.testing.assert_allclose(b[0, :, 3], a[0, :, 3])  # confidence unchanged

    def test_loaded_data_matches_training_schema(self, tmp_path):
        """The .npz must be consumable by train_with_splits.py conventions."""
        out = tmp_path / "room__subject__20260101.npz"
        csi = np.ones((3, 64, 16), dtype=np.float32)
        with AlignedRecorder(out, metadata={"room": "r1", "subject": "s1"}) as rec:
            for _ in range(5):
                rec.add_window(csi, self._frame())
        data = AlignedRecorder.load(out)
        # Training expects float32 features (N,3,64,16) and poses (N,17,4)
        assert data["features"].dtype == np.float32
        assert data["poses"].dtype == np.float32
        assert json.dumps(data["metadata"])  # JSON-serialisable
