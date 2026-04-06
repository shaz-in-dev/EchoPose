"""
pipeline/activity.py — Activity classification & gait analysis from skeleton + CSI

Features:
  6. Gait Analysis & Speed Estimation
  7. Activity Type Classification (Standing/Walking/Running/Sitting/Lying)
  13. Repetitive Motion & Exercise Counting
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("rf_inference.activity")

ACTIVITY_LABELS = ["standing", "walking", "running", "sitting", "lying"]

# COCO-17 keypoint indices
_NOSE = 0
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16


class ActivityClassifier:
    """Classifies human activity from skeleton sequences and CSI Doppler."""

    def __init__(self, fps: float = 20.0):
        self.fps = fps
        self._pose_history: list[np.ndarray] = []
        self._max_history = int(fps * 10)  # 10-second window

    def push_skeleton(self, skeleton: List[Dict]) -> None:
        """Append a single-frame skeleton (list of 17 keypoint dicts) to history."""
        arr = np.array([[kp.get("x", 0), kp.get("y", 0), kp.get("z", 0)] for kp in skeleton], dtype=np.float64)
        self._pose_history.append(arr)
        if len(self._pose_history) > self._max_history:
            self._pose_history = self._pose_history[-self._max_history:]

    # ── Feature 6: Gait Analysis ──────────────────────────────────

    def analyze_gait(self) -> Dict:
        """
        Compute gait parameters from hip keypoint trajectories.

        Returns walking speed, stride length, cadence, and symmetry.
        """
        if len(self._pose_history) < int(self.fps * 2):
            return {"status": "buffering"}

        poses = np.array(self._pose_history)  # [T, 17, 3]

        hip_left = poses[:, _L_HIP]
        hip_right = poses[:, _R_HIP]
        hip_centre = (hip_left + hip_right) / 2.0

        # Vertical oscillation → step detection
        height = hip_centre[:, 1]
        height_detrended = height - np.mean(height)

        peaks, _ = sp_signal.find_peaks(-height_detrended, distance=int(self.fps * 0.4))
        step_count = len(peaks)
        duration_s = len(poses) / self.fps

        cadence = (step_count / duration_s) * 60.0 if duration_s > 0 else 0.0

        # Horizontal displacement → stride length & speed
        dx = np.diff(hip_centre[:, 0])
        dz = np.diff(hip_centre[:, 2]) if hip_centre.shape[1] > 2 else np.zeros_like(dx)
        total_dist = float(np.sum(np.sqrt(dx ** 2 + dz ** 2)))
        speed = total_dist / duration_s if duration_s > 0 else 0.0
        stride_length = total_dist / max(step_count, 1)

        # Symmetry: compare left-right hip vertical phase
        sym = self._gait_symmetry(hip_left[:, 1], hip_right[:, 1])

        return {
            "walking_speed_ms": round(speed, 3),
            "stride_length_m": round(stride_length, 3),
            "cadence_steps_min": round(cadence, 1),
            "gait_symmetry": round(sym, 2),
            "step_count": step_count,
        }

    # ── Feature 7: Activity Classification ────────────────────────

    def classify_activity(self, csi_doppler: Optional[np.ndarray] = None) -> Dict:
        """
        Rule-based activity classifier using skeleton geometry + Doppler.

        Returns one of: standing, walking, running, sitting, lying.
        """
        if len(self._pose_history) < int(self.fps):
            return {"activity": "unknown", "confidence": 0.0}

        poses = np.array(self._pose_history[-int(self.fps * 3):])  # last 3 s
        T = poses.shape[0]

        # Feature: vertical extent (head-to-ankle ratio)
        head_y = poses[:, _NOSE, 1]
        ankle_y = (poses[:, _L_ANKLE, 1] + poses[:, _R_ANKLE, 1]) / 2.0
        vert_extent = np.mean(np.abs(head_y - ankle_y))

        # Feature: centre-of-mass motion energy
        com = np.mean(poses[:, [_L_HIP, _R_HIP, _L_SHOULDER, _R_SHOULDER], :], axis=1)
        motion = np.sqrt(np.sum(np.diff(com, axis=0) ** 2, axis=1))
        motion_energy = float(np.mean(motion))

        # Feature: hip-knee angle (bent = sitting)
        knee_angle = self._mean_angle(
            poses[-1, _L_HIP], poses[-1, _L_KNEE], poses[-1, _L_ANKLE]
        )

        # Doppler energy (if available)
        doppler_energy = float(np.mean(np.abs(csi_doppler))) if csi_doppler is not None else 0.0

        # Decision thresholds
        if vert_extent < 0.15:
            activity, conf = "lying", 0.85
        elif knee_angle < 110 and motion_energy < 0.02:
            activity, conf = "sitting", 0.80
        elif motion_energy < 0.005:
            activity, conf = "standing", 0.80
        elif motion_energy > 0.04 or doppler_energy > 0.5:
            activity, conf = "running", 0.75
        else:
            activity, conf = "walking", 0.78

        return {"activity": activity, "confidence": round(conf, 2)}

    # ── Feature 13: Exercise Counting ─────────────────────────────

    def count_exercise_reps(self, exercise_type: str = "auto") -> Dict:
        """
        Count repetitions of common exercises from joint angle cycles.

        Supports: pushup, squat, jumping_jack, situp, auto-detect.
        """
        if len(self._pose_history) < int(self.fps * 3):
            return {"exercise_type": exercise_type, "reps": 0, "confidence": 0.0}

        poses = np.array(self._pose_history)

        if exercise_type == "auto":
            exercise_type = self._detect_exercise_type(poses)

        if exercise_type == "pushup":
            angles = np.array([
                self._angle(p[_L_SHOULDER], p[_L_ELBOW], p[_L_WRIST]) for p in poses
            ])
        elif exercise_type == "squat":
            angles = np.array([
                self._angle(p[_L_HIP], p[_L_KNEE], p[_L_ANKLE]) for p in poses
            ])
        elif exercise_type == "situp":
            angles = np.array([
                self._angle(p[_L_SHOULDER], p[_L_HIP], p[_L_KNEE]) for p in poses
            ])
        elif exercise_type == "jumping_jack":
            angles = np.array([
                self._angle(p[_L_HIP], p[_L_SHOULDER], p[_L_WRIST]) for p in poses
            ])
        else:
            return {"exercise_type": exercise_type, "reps": 0, "confidence": 0.0}

        # Count full cycles via zero-crossing of derivative
        angles_smooth = np.convolve(angles, np.ones(5) / 5, mode="same")
        diffs = np.diff(angles_smooth)
        sign_changes = np.diff(np.sign(diffs))
        reps = int(np.sum(sign_changes != 0) // 2)

        return {"exercise_type": exercise_type, "reps": reps, "confidence": 0.82}

    # ── helpers ───────────────────────────────────────────────────

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba = a[:3] - b[:3]
        bc = c[:3] - b[:3]
        cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1))))

    _mean_angle = lambda self, a, b, c: ActivityClassifier._angle(a, b, c)

    @staticmethod
    def _gait_symmetry(left_y: np.ndarray, right_y: np.ndarray) -> float:
        corr = np.corrcoef(left_y, right_y)[0, 1]
        return float(np.clip(abs(corr), 0, 1))

    def _detect_exercise_type(self, poses: np.ndarray) -> str:
        vert = np.std(poses[:, _NOSE, 1])
        elbow_var = np.std([self._angle(p[_L_SHOULDER], p[_L_ELBOW], p[_L_WRIST]) for p in poses])
        knee_var = np.std([self._angle(p[_L_HIP], p[_L_KNEE], p[_L_ANKLE]) for p in poses])

        if elbow_var > knee_var and vert < 0.05:
            return "pushup"
        if knee_var > 15:
            return "squat"
        if vert > 0.1:
            return "jumping_jack"
        return "situp"
