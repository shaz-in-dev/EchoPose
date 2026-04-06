"""
pipeline/tactical/tactical_activity.py — Combat activity classification (Feature 8)

Extends the civilian activity classifier with tactical-specific postures:
  STANDING, MOVING_TACTICAL, RUNNING, CRAWLING, PRONE,
  TAKING_AIM, THROWING, SURRENDERING, ASSISTING_INJURED.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("rf_inference.tactical.activity")

SAMPLE_RATE = 20.0

# COCO-17
_NOSE = 0
_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12
_L_KNEE, _R_KNEE = 13, 14
_L_ANKLE, _R_ANKLE = 15, 16

TACTICAL_LABELS = [
    "STANDING", "MOVING_TACTICAL", "RUNNING", "CRAWLING",
    "PRONE", "TAKING_AIM", "THROWING", "SURRENDERING",
    "ASSISTING_INJURED", "UNKNOWN",
]


class TacticalActivityClassifier:
    """Classify military-relevant postures and activities."""

    def __init__(self, fps: float = SAMPLE_RATE):
        self.fps = fps
        self._buf: list[np.ndarray] = []
        self._max = int(fps * 5)

    def push_skeleton(self, skeleton: List[Dict]) -> None:
        arr = np.array([[kp.get("x", 0), kp.get("y", 0), kp.get("z", 0)]
                        for kp in skeleton], dtype=np.float64)
        self._buf.append(arr)
        if len(self._buf) > self._max:
            self._buf = self._buf[-self._max:]

    def classify(self, csi_doppler: Optional[np.ndarray] = None) -> Dict:
        """Return tactical activity label and confidence."""
        if len(self._buf) < int(self.fps):
            return {"activity": "UNKNOWN", "confidence": 0.0, "status": "buffering"}

        poses = np.array(self._buf[-int(self.fps * 3):])
        latest = poses[-1]

        torso_angle = self._torso_angle(latest)
        speed = self._movement_speed(poses)
        arm_pos = self._arm_position(latest, poses)
        vert_extent = self._vertical_extent(latest)

        activity, conf = self._decide(torso_angle, speed, arm_pos, vert_extent)

        return {
            "activity": activity,
            "confidence": round(conf, 2),
            "torso_angle_deg": round(torso_angle, 1),
            "speed_ms": round(speed, 3),
            "vertical_extent": round(vert_extent, 3),
        }

    # ── feature extractors ────────────────────────────────────────

    def _torso_angle(self, pose: np.ndarray) -> float:
        """Angle of torso from vertical (0 = upright, 90 = horizontal)."""
        mid_shoulder = (pose[_L_SHOULDER] + pose[_R_SHOULDER]) / 2.0
        mid_hip = (pose[_L_HIP] + pose[_R_HIP]) / 2.0
        torso_vec = mid_shoulder - mid_hip
        vertical = np.array([0, -1, 0], dtype=np.float64)
        cos_a = np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

    def _movement_speed(self, poses: np.ndarray) -> float:
        """Centre-of-mass horizontal speed (m/s)."""
        com = np.mean(poses[:, [_L_HIP, _R_HIP], :], axis=1)
        d = np.diff(com, axis=0)
        step_dist = np.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2)
        return float(np.mean(step_dist) * self.fps)

    def _arm_position(self, latest: np.ndarray, history: np.ndarray) -> Dict:
        """Classify arm configuration: raised, extended-forward, at-sides."""
        lw = latest[_L_WRIST]
        rw = latest[_R_WRIST]
        ls = latest[_L_SHOULDER]
        rs = latest[_R_SHOULDER]
        lh = latest[_L_HIP]
        nose = latest[_NOSE]

        both_up = lw[1] < ls[1] and rw[1] < rs[1]
        forward_ext = (np.linalg.norm(lw - ls) > 0.25 or
                       np.linalg.norm(rw - rs) > 0.25)

        # Aim detection: one arm extended forward, other close to torso
        l_ext = float(np.linalg.norm(lw - ls))
        r_ext = float(np.linalg.norm(rw - rs))
        aiming = (l_ext > 0.25 and r_ext < 0.15) or (r_ext > 0.25 and l_ext < 0.15)

        # Throwing: rapid upward wrist velocity
        if len(history) >= 5:
            recent_wrists = history[-5:, _R_WRIST, 1]
            throw_vel = float(np.min(np.diff(recent_wrists)) * self.fps)
        else:
            throw_vel = 0.0

        return {
            "both_up": both_up,
            "forward_ext": forward_ext,
            "aiming": aiming,
            "throw_velocity": throw_vel,
        }

    def _vertical_extent(self, pose: np.ndarray) -> float:
        """Head-to-ankle vertical range (low = prone/crawling)."""
        head_y = pose[_NOSE, 1]
        ankle_y = (pose[_L_ANKLE, 1] + pose[_R_ANKLE, 1]) / 2.0
        return float(abs(head_y - ankle_y))

    # ── decision logic ────────────────────────────────────────────

    def _decide(self, torso_ang: float, speed: float,
                arm: Dict, vert: float) -> tuple:

        # Surrendering: both hands up, minimal movement
        if arm["both_up"] and speed < 0.1:
            return ("SURRENDERING", 0.90)

        # Throwing: rapid upward arm velocity
        if arm["throw_velocity"] < -1.5:
            return ("THROWING", 0.78)

        # Taking aim: one arm extended forward steadily
        if arm["aiming"] and speed < 0.3:
            return ("TAKING_AIM", 0.82)

        # Prone: very low vertical extent
        if vert < 0.10:
            return ("PRONE", 0.88)

        # Crawling: low torso angle + slow movement
        if torso_ang > 60 and speed < 0.5 and vert < 0.25:
            return ("CRAWLING", 0.84)

        # Running
        if speed > 1.5:
            return ("RUNNING", 0.85)

        # Moving tactical: crouched + moderate speed
        if torso_ang > 30 and 0.3 < speed < 1.5:
            return ("MOVING_TACTICAL", 0.80)

        # Assisting injured: leaning over + minimal self-movement
        if torso_ang > 45 and speed < 0.2 and not arm["both_up"]:
            return ("ASSISTING_INJURED", 0.70)

        # Standing (default)
        if speed < 0.15:
            return ("STANDING", 0.82)

        return ("UNKNOWN", 0.50)
