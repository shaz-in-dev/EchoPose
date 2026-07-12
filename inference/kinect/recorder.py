"""
inference/kinect/recorder.py — Aligned CSI + pose .npz dataset writer.

Produces .npz files matching the schema expected by the training pipeline:
  features  : float32 (N, 3, 64, 16)  — CSI amplitude per node/subcarrier/doppler
  poses     : float32 (N, 17, 4)      — COCO-17 joints [x, y, z, conf] in world coords
  timestamps: float64 (N,)            — Unix timestamps for each window
  metadata  : dict (stored as JSON in npz "metadata" key)

Usage::
    rec = AlignedRecorder("data/sessions/session_001.npz", metadata={...})
    rec.open()
    ...
    rec.add_window(csi_window, pose_frame)
    ...
    rec.close()   # writes the .npz
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .joint_mapping import JointMapper
from .pose_source import BodyFrame
from .transform import CoordTransform


class AlignedRecorder:
    """Buffers paired (CSI, pose) windows and writes them as a .npz dataset.

    Parameters
    ----------
    output_path : Where to write the .npz file.
    transform   : CoordTransform to apply to each pose. Identity if None.
    metadata    : Session info dict (room, subject, hardware config, etc.).
    max_windows : Safety cap on buffered windows (prevents OOM on long sessions).
    """

    def __init__(
        self,
        output_path: str | Path,
        transform:   Optional[CoordTransform] = None,
        metadata:    Optional[Dict[str, Any]] = None,
        max_windows: int = 100_000,
    ):
        self._path       = Path(output_path)
        self._transform  = transform or CoordTransform.identity()
        self._metadata   = metadata or {}
        self._max_windows = max_windows
        self._mapper     = JointMapper()

        self._features:   List[np.ndarray] = []  # each (3, 64, 16)
        self._poses:      List[np.ndarray] = []  # each (17, 4)
        self._timestamps: List[float]      = []

        self._open_ts: Optional[float] = None

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._open_ts = time.time()
        self._metadata.setdefault("session_start", self._open_ts)

    def close(self) -> None:
        """Flush all buffered windows to disk."""
        if not self._features:
            return
        self._save()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ── Window ingestion ──────────────────────────────────────────────────────

    def add_window(
        self,
        csi_window:  np.ndarray,
        body_frame:  BodyFrame,
        timestamp_s: Optional[float] = None,
    ) -> bool:
        """Add one aligned window.

        Parameters
        ----------
        csi_window  : shape (3, 64, 16) — CSI amplitude from 3 nodes.
        body_frame  : BodyFrame with 25-joint Kinect bodies.
        timestamp_s : Window timestamp; defaults to body_frame.timestamp_s.

        Returns True if accepted, False if buffer is full or frame has no bodies.
        """
        if len(self._features) >= self._max_windows:
            return False
        if not body_frame.bodies:
            return False

        # Validate CSI shape
        arr = np.asarray(csi_window, dtype=np.float32)
        if arr.shape != (3, 64, 16):
            raise ValueError(
                f"csi_window must have shape (3, 64, 16), got {arr.shape}"
            )

        # Map Kinect-25 → COCO-17 for the first tracked body
        kinect_joints = body_frame.first_body()
        if len(kinect_joints) < 25:
            return False

        coco_arr = self._mapper.map_to_array(kinect_joints)  # (17, 4)
        # Apply coord transform to x,y,z in-place
        coco_arr = self._transform.apply_array(coco_arr)

        ts = timestamp_s if timestamp_s is not None else body_frame.timestamp_s

        self._features.append(arr)
        self._poses.append(coco_arr)
        self._timestamps.append(ts)
        return True

    @property
    def window_count(self) -> int:
        return len(self._features)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        features  = np.stack(self._features,  axis=0)  # (N, 3, 64, 16)
        poses     = np.stack(self._poses,     axis=0)  # (N, 17, 4)
        timestamps = np.array(self._timestamps, dtype=np.float64)

        self._metadata["n_windows"]    = len(features)
        self._metadata["session_end"]  = time.time()
        self._metadata["csi_shape"]    = list(features.shape)
        self._metadata["pose_shape"]   = list(poses.shape)

        np.savez_compressed(
            self._path,
            features=features,
            poses=poses,
            timestamps=timestamps,
            metadata=np.array(json.dumps(self._metadata)),
        )

    @staticmethod
    def load(path: str | Path):
        """Load an .npz dataset produced by AlignedRecorder.

        Returns dict with keys: features, poses, timestamps, metadata (as dict).
        """
        raw = np.load(path, allow_pickle=True)
        meta = json.loads(str(raw["metadata"]))
        return {
            "features":   raw["features"],
            "poses":      raw["poses"],
            "timestamps": raw["timestamps"],
            "metadata":   meta,
        }
