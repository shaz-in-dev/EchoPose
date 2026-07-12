"""
inference/kinect/pykinect_source.py — Real Kinect v2 source (Windows + PyKinect2).

REQUIREMENTS (one-time setup):
  1. Install Kinect for Windows SDK 2.0 from Microsoft.
  2. Connect the Kinect v2 via the USB 3.0 Adapter for Windows.
  3. pip install pykinect2

This file only imports PyKinect2 when actually instantiated, so the rest of
the kinect/ package remains importable on Linux / CI without the SDK.

Usage::
    from kinect.pykinect_source import PyKinect2Source
    source = PyKinect2Source()
    source.open()
    for frame in source.stream():
        # frame.bodies[0] → list of 25 Joint in Kinect camera space
        process(frame)
    source.close()
"""

from __future__ import annotations

import time
from typing import Iterator, List, Optional

from .pose_source import BodyFrame, Joint, PoseSource

# Kinect joint state constants (from Kinect for Windows SDK)
_TRACKING   = 2
_INFERRED   = 1
_NOT_TRACKED = 0


class PyKinect2Source(PoseSource):
    """PoseSource backed by a real Kinect v2 via PyKinect2.

    Parameters
    ----------
    timeout_s : Max seconds to wait for a new BodyFrame before yielding None.
    """

    def __init__(self, timeout_s: float = 0.1):
        self._timeout = timeout_s
        self._runtime = None
        self._frame_idx = 0

    def open(self) -> None:
        try:
            from pykinect2 import PyKinectV2, PyKinectRuntime
            self._runtime = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Body)
        except ImportError as exc:
            raise ImportError(
                "PyKinect2 not installed or Kinect SDK not found. "
                "Install the Kinect for Windows SDK 2.0 and run: pip install pykinect2"
            ) from exc

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    def read_one(self) -> Optional[BodyFrame]:
        if self._runtime is None:
            return None
        deadline = time.time() + self._timeout
        while time.time() < deadline:
            if self._runtime.has_new_body_frame():
                return self._read_frame()
            time.sleep(0.005)
        return None

    def stream(self) -> Iterator[BodyFrame]:
        while self._runtime is not None:
            frame = self.read_one()
            if frame is not None:
                yield frame

    def _read_frame(self) -> BodyFrame:
        from pykinect2 import PyKinectV2
        bodies_data = self._runtime.get_last_body_frame()
        ts = time.time()
        bodies = []

        if bodies_data is not None:
            for i in range(bodies_data.body_count):
                body = bodies_data.bodies[i]
                if not body.is_tracked:
                    continue
                joints = self._extract_joints(body, PyKinectV2)
                bodies.append(joints)

        frame = BodyFrame(
            timestamp_s=ts,
            bodies=bodies,
            frame_index=self._frame_idx,
            source_id="kinect_v2",
        )
        self._frame_idx += 1
        return frame

    @staticmethod
    def _extract_joints(body, PyKinectV2) -> List[Joint]:
        joints = []
        for j_idx in range(25):  # JointType_Count = 25
            j  = body.joints[j_idx]
            js = body.joint_states[j_idx]
            pos = j.Position
            # Kinect SDK reports x,y,z in metres, camera space
            conf = {_TRACKING: 1.0, _INFERRED: 0.5, _NOT_TRACKED: 0.0}.get(int(js), 0.0)
            joints.append(Joint(
                x=float(pos.x),
                y=float(pos.y),
                z=float(pos.z),
                confidence=conf,
            ))
        return joints
