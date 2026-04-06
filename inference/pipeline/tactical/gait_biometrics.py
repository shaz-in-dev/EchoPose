"""
pipeline/tactical/gait_biometrics.py — Person identification via gait (Feature 5)

Each person has a unique walking style — stride length, cadence,
arm swing asymmetry, and joint-angle patterns.  This module builds
biometric templates from 30 s of observation and matches live gait
signatures against a stored database.

Accuracy: 92-98 % per published WiFi-gait-ID literature.
"""

import hashlib
import time
import numpy as np
from scipy.signal import welch
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("rf_inference.tactical.gait_biometrics")

SAMPLE_RATE = 20.0
_ENROLL_FRAMES = 600   # 30 s × 20 Hz
_MATCH_FRAMES = 100    # 5 s for live matching
_FEATURE_DIM = 12

# COCO-17 indices
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_WRIST, _R_WRIST = 9, 10


class GaitBiometricIdentifier:
    """Identifies individuals from their unique gait signature."""

    def __init__(self, fs: float = SAMPLE_RATE):
        self.fs = fs
        self._database: Dict[str, np.ndarray] = {}   # id → feature vector
        self._meta: Dict[str, Dict] = {}              # id → metadata
        self._live_buf: list[np.ndarray] = []
        self._max_buf = _ENROLL_FRAMES

    # ── enrolment ─────────────────────────────────────────────────

    def enrol(self, person_id: str, skeleton_history: List[List[Dict]],
              metadata: Optional[Dict] = None) -> Dict:
        """
        Build biometric template from skeleton history.

        skeleton_history: list of frames, each frame is 17 keypoint dicts.
        """
        if len(skeleton_history) < _ENROLL_FRAMES:
            return {"status": "need_more_data",
                    "frames": len(skeleton_history),
                    "required": _ENROLL_FRAMES}

        poses = self._to_array(skeleton_history[:_ENROLL_FRAMES])
        features = self._extract_features(poses)
        self._database[person_id] = features
        self._meta[person_id] = metadata or {}
        logger.info(f"Enrolled person '{person_id}' ({_FEATURE_DIM}-dim template).")

        return {"status": "enrolled", "person_id": person_id,
                "feature_dim": _FEATURE_DIM}

    # ── live identification ───────────────────────────────────────

    def push_skeleton(self, skeleton: List[Dict]) -> None:
        arr = np.array([[kp.get("x", 0), kp.get("y", 0), kp.get("z", 0)]
                        for kp in skeleton], dtype=np.float64)
        self._live_buf.append(arr)
        if len(self._live_buf) > self._max_buf:
            self._live_buf = self._live_buf[-self._max_buf:]

    def identify(self) -> Dict:
        """Match current live gait against enrolled templates."""
        if len(self._live_buf) < _MATCH_FRAMES:
            return {"matched": False, "status": "buffering",
                    "frames": len(self._live_buf)}
        if not self._database:
            return {"matched": False, "status": "database_empty"}

        poses = np.array(self._live_buf[-_MATCH_FRAMES:])
        query = self._extract_features(poses)

        best_id, best_score = None, -1.0
        scores = {}
        for pid, template in self._database.items():
            sim = self._cosine_sim(query, template)
            scores[pid] = round(float(sim), 3)
            if sim > best_score:
                best_score = sim
                best_id = pid

        matched = best_score > 0.82
        result: Dict = {
            "matched": matched,
            "person_id": best_id if matched else None,
            "confidence": round(float(best_score), 3),
            "scores": scores,
        }
        if matched and best_id in self._meta:
            result["metadata"] = self._meta[best_id]
        return result

    @property
    def enrolled_count(self) -> int:
        return len(self._database)

    # ── feature extraction ────────────────────────────────────────

    def _extract_features(self, poses: np.ndarray) -> np.ndarray:
        """Extract a fixed-length gait descriptor from [T, 17, 3] poses."""
        # 1. stride length (hip displacement per cycle)
        hip_c = (poses[:, _L_HIP] + poses[:, _R_HIP]) / 2.0
        dx = np.diff(hip_c[:, 0])
        dz = np.diff(hip_c[:, 2]) if hip_c.shape[1] > 2 else np.zeros_like(dx)
        step_dists = np.sqrt(dx ** 2 + dz ** 2)
        stride_len = float(np.mean(step_dists))
        stride_var = float(np.std(step_dists))

        # 2. cadence (dominant frequency of vertical hip oscillation)
        vert = hip_c[:, 1] - np.mean(hip_c[:, 1])
        cadence = self._dominant_freq(vert, 0.5, 4.0)

        # 3. arm swing asymmetry
        lw = poses[:, _L_WRIST, 1]
        rw = poses[:, _R_WRIST, 1]
        arm_asym = float(np.mean(np.abs(np.std(lw) - np.std(rw))))

        # 4. knee angle statistics (left & right)
        l_knee_angles = [self._angle(poses[t, _L_HIP], poses[t, _L_KNEE],
                                     poses[t, _L_ANKLE]) for t in range(len(poses))]
        r_knee_angles = [self._angle(poses[t, _R_HIP], poses[t, _R_KNEE],
                                     poses[t, _R_ANKLE]) for t in range(len(poses))]
        lk_mean, lk_std = float(np.mean(l_knee_angles)), float(np.std(l_knee_angles))
        rk_mean, rk_std = float(np.mean(r_knee_angles)), float(np.std(r_knee_angles))

        # 5. shoulder tilt variance
        sh_diff = poses[:, _L_SHOULDER, 1] - poses[:, _R_SHOULDER, 1]
        sh_var = float(np.std(sh_diff))

        # 6. vertical bounce amplitude
        bounce = float(np.std(hip_c[:, 1]))

        # 7. gait symmetry (L-R hip correlation)
        sym = float(np.corrcoef(poses[:, _L_HIP, 1], poses[:, _R_HIP, 1])[0, 1])

        return np.array([stride_len, stride_var, cadence, arm_asym,
                         lk_mean, lk_std, rk_mean, rk_std,
                         sh_var, bounce, sym, cadence * stride_len],
                        dtype=np.float64)

    # ── helpers ───────────────────────────────────────────────────

    def _to_array(self, frames: List[List[Dict]]) -> np.ndarray:
        return np.array([
            [[kp.get("x", 0), kp.get("y", 0), kp.get("z", 0)] for kp in frame]
            for frame in frames
        ], dtype=np.float64)

    def _dominant_freq(self, sig: np.ndarray, lo: float, hi: float) -> float:
        freqs, psd = welch(sig, fs=self.fs, nperseg=min(128, len(sig)))
        mask = (freqs >= lo) & (freqs <= hi)
        if not np.any(mask) or np.max(psd[mask]) == 0:
            return 0.0
        return float(freqs[mask][np.argmax(psd[mask])])

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a[:3] - b[:3]
        bc = c[:3] - b[:3]
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        d = np.dot(a, b)
        n = np.linalg.norm(a) * np.linalg.norm(b)
        return float(d / n) if n > 1e-12 else 0.0
