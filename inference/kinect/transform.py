"""
inference/kinect/transform.py — Kinect camera-space → world/room-space transform.

The Kinect reports joints in its own camera coordinate system (metres, right-hand).
This module applies a rigid transform (rotation + translation) to convert those
coordinates into a room/world coordinate system defined by the user.

Setup (do once per room):
  1. Place the Kinect at a known position + orientation.
  2. Measure: Kinect position in room coords (x_k, y_k, z_k) in metres.
  3. Measure: Kinect yaw angle (rotation around vertical axis) in degrees.
  4. Construct CoordTransform(translation, yaw_deg) and bake it to a JSON file.
  5. Load that JSON at the start of every session.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np

from .pose_source import Joint


class CoordTransform:
    """Rigid 3-D transform: Kinect camera space → room world space.

    Only yaw rotation is modelled here (Kinect usually mounted level).
    For tilted mounts, extend with pitch/roll.

    Parameters
    ----------
    translation : (x, y, z) in metres — Kinect origin in room coords.
    yaw_deg     : Kinect yaw in room frame, degrees. 0 = facing +X axis.
    """

    def __init__(
        self,
        translation: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        yaw_deg: float = 0.0,
    ):
        self.translation = np.array(translation, dtype=np.float64)
        self._yaw_deg = yaw_deg
        self._R = self._yaw_matrix(math.radians(yaw_deg))

    @staticmethod
    def _yaw_matrix(yaw_rad: float) -> np.ndarray:
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        return np.array([
            [ c, 0, s],
            [ 0, 1, 0],
            [-s, 0, c],
        ], dtype=np.float64)

    def apply(self, joint: Joint) -> Joint:
        """Transform a single Joint into room coords."""
        p = np.array([joint.x, joint.y, joint.z], dtype=np.float64)
        p_room = self._R @ p + self.translation
        return Joint(
            x=float(p_room[0]),
            y=float(p_room[1]),
            z=float(p_room[2]),
            confidence=joint.confidence,
        )

    def apply_body(self, joints: List[Joint]) -> List[Joint]:
        """Transform all joints in a body."""
        return [self.apply(j) for j in joints]

    def apply_array(self, arr: np.ndarray) -> np.ndarray:
        """Transform (N, 4) array [x, y, z, conf] in-place copy."""
        result = arr.copy()
        result[:, :3] = (self._R @ arr[:, :3].T).T + self.translation
        return result

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save transform parameters to JSON."""
        Path(path).write_text(json.dumps({
            "translation": self.translation.tolist(),
            "yaw_deg": self._yaw_deg,
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "CoordTransform":
        """Load from a JSON file written by save()."""
        data = json.loads(Path(path).read_text())
        return cls(
            translation=tuple(data["translation"]),
            yaw_deg=data["yaw_deg"],
        )

    @classmethod
    def identity(cls) -> "CoordTransform":
        """No-op transform — useful for testing or when Kinect IS the world origin."""
        return cls()
