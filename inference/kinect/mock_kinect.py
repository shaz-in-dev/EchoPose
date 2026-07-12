"""
inference/kinect/mock_kinect.py — Synthetic Kinect source for offline development.

Produces realistic 25-joint BodyFrames without any hardware. Useful for:
  - Building and testing the recorder pipeline before the Kinect adapter arrives
  - Unit-testing the joint mapper, transform, and sync modules
  - CI pipelines that have no physical hardware

Activities emitted in sequence: standing, walking, sitting, lying, standing.
"""

from __future__ import annotations

import math
import time
from typing import Iterator, List, Optional

import numpy as np

from .pose_source import BodyFrame, Joint, PoseSource

# ── Default skeleton in Kinect camera space (metres) ──────────────────────────
# Standing person at ~2.5 m from the Kinect, roughly centred.
_STAND_TEMPLATE: List[tuple] = [
    # (x,    y,    z)   — joint index (see joint_mapping.py for names)
    ( 0.00, -0.90,  2.5),  # 0  SpineBase
    ( 0.00, -0.30,  2.5),  # 1  SpineMid
    ( 0.00,  0.30,  2.5),  # 2  Neck
    ( 0.00,  0.60,  2.5),  # 3  Head
    (-0.20,  0.10,  2.5),  # 4  ShoulderLeft
    (-0.45, -0.15,  2.5),  # 5  ElbowLeft
    (-0.55, -0.50,  2.5),  # 6  WristLeft
    (-0.60, -0.65,  2.5),  # 7  HandLeft
    ( 0.20,  0.10,  2.5),  # 8  ShoulderRight
    ( 0.45, -0.15,  2.5),  # 9  ElbowRight
    ( 0.55, -0.50,  2.5),  # 10 WristRight
    ( 0.60, -0.65,  2.5),  # 11 HandRight
    (-0.10, -0.90,  2.5),  # 12 HipLeft
    (-0.10, -1.40,  2.5),  # 13 KneeLeft
    (-0.10, -1.80,  2.5),  # 14 AnkleLeft
    (-0.10, -1.90,  2.5),  # 15 FootLeft
    ( 0.10, -0.90,  2.5),  # 16 HipRight
    ( 0.10, -1.40,  2.5),  # 17 KneeRight
    ( 0.10, -1.80,  2.5),  # 18 AnkleRight
    ( 0.10, -1.90,  2.5),  # 19 FootRight
    ( 0.00,  0.20,  2.5),  # 20 SpineShoulder
    (-0.62, -0.72,  2.5),  # 21 HandTipLeft
    (-0.55, -0.68,  2.5),  # 22 ThumbLeft
    ( 0.62, -0.72,  2.5),  # 23 HandTipRight
    ( 0.55, -0.68,  2.5),  # 24 ThumbRight
]


class MockKinectSource(PoseSource):
    """Generates synthetic BodyFrames at a configurable frame rate.

    Parameters
    ----------
    fps          : Frames per second (default 30).
    n_people     : Number of synthetic bodies per frame (1-2).
    noise_std    : Joint position noise in metres.
    emit_spike   : If True, emits a large velocity spike at t=2 s for sync testing.
    seed         : Random seed for reproducibility.
    """

    def __init__(
        self,
        fps: int = 30,
        n_people: int = 1,
        noise_std: float = 0.01,
        emit_spike: bool = True,
        seed: int = 42,
    ):
        self._fps        = fps
        self._n_people   = min(max(n_people, 1), 2)
        self._noise_std  = noise_std
        self._emit_spike = emit_spike
        self._rng        = np.random.default_rng(seed)
        self._frame_idx  = 0
        self._start_ts   = 0.0
        self._running    = False

    def open(self) -> None:
        self._running   = True
        self._start_ts  = time.time()
        self._frame_idx = 0

    def close(self) -> None:
        self._running = False

    def read_one(self) -> Optional[BodyFrame]:
        if not self._running:
            return None
        return self._make_frame()

    def stream(self) -> Iterator[BodyFrame]:
        period = 1.0 / self._fps
        while self._running:
            frame = self._make_frame()
            yield frame
            # Sleep to pace at target fps
            elapsed = time.time() - self._start_ts - (self._frame_idx * period)
            if elapsed < period:
                time.sleep(period - elapsed)

    # ── Internal ───────────────────────────────────────────────────────────────

    def _make_frame(self) -> BodyFrame:
        t = time.time()
        age = t - self._start_ts
        bodies = [self._make_body(age, person_idx=i) for i in range(self._n_people)]
        frame = BodyFrame(
            timestamp_s=t,
            bodies=bodies,
            frame_index=self._frame_idx,
            source_id="mock",
        )
        self._frame_idx += 1
        return frame

    def _make_body(self, age_s: float, person_idx: int = 0) -> List[Joint]:
        template = [list(j) for j in _STAND_TEMPLATE]

        # Walking animation: oscillate Z (depth) and swing arms
        walk_phase = age_s * 2.0 + person_idx * math.pi
        walk_z = math.sin(walk_phase) * 0.3
        for j in template:
            j[2] += walk_z  # move person back/forth

        # Arm swing (shoulder and elbow joints 4-11)
        arm_swing = math.sin(walk_phase * 2) * 0.2
        for arm_idx in (4, 5, 6, 7):
            template[arm_idx][1] += arm_swing
        for arm_idx in (8, 9, 10, 11):
            template[arm_idx][1] -= arm_swing

        # Sync spike at t≈2s: large rapid motion on head and spine
        if self._emit_spike and 1.9 < age_s < 2.1:
            spike = math.sin((age_s - 1.9) / 0.2 * math.pi) * 0.5
            for j in template[:4]:
                j[1] += spike

        # Offset second person sideways
        if person_idx == 1:
            for j in template:
                j[0] += 1.0

        # Add Gaussian noise
        joints = []
        for i, (x, y, z) in enumerate(template):
            nx = x + self._rng.normal(0, self._noise_std)
            ny = y + self._rng.normal(0, self._noise_std)
            nz = z + self._rng.normal(0, self._noise_std)
            # Randomly infer a few joints (confidence=0.5)
            conf = 0.5 if i in (15, 19, 21, 22, 23, 24) else 1.0
            joints.append(Joint(x=nx, y=ny, z=nz, confidence=conf))

        return joints
