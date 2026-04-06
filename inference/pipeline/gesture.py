"""
pipeline/gesture.py — Hand gesture recognition from skeleton (Feature 10)

Recognises coarse hand gestures from COCO-17 skeleton wrist/elbow
trajectories: wave, point, raise, swipe_left, swipe_right, idle.
"""

import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger("rf_inference.gesture")

_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10

GESTURE_LABELS = ["idle", "wave", "point", "raise", "swipe_left", "swipe_right"]


class GestureRecognizer:
    """Recognise coarse hand/arm gestures from skeleton time-series."""

    def __init__(self, fps: float = 20.0):
        self.fps = fps
        self._wrist_history_l: list[np.ndarray] = []
        self._wrist_history_r: list[np.ndarray] = []
        self._max_history = int(fps * 3)

    def push_skeleton(self, skeleton: List[Dict]) -> None:
        kps = np.array([[kp.get("x", 0), kp.get("y", 0), kp.get("z", 0)] for kp in skeleton])
        self._wrist_history_l.append(kps[_L_WRIST])
        self._wrist_history_r.append(kps[_R_WRIST])
        for buf in (self._wrist_history_l, self._wrist_history_r):
            if len(buf) > self._max_history:
                del buf[0]

    def recognize(self) -> Dict:
        """Return the most likely gesture for each hand."""
        min_frames = int(self.fps * 0.5)
        if len(self._wrist_history_r) < min_frames:
            return {"left_hand": "idle", "right_hand": "idle", "confidence": 0.0}

        left = self._classify_hand(np.array(self._wrist_history_l))
        right = self._classify_hand(np.array(self._wrist_history_r))

        return {
            "left_hand": left["gesture"],
            "right_hand": right["gesture"],
            "confidence": round(max(left["confidence"], right["confidence"]), 2),
        }

    def _classify_hand(self, traj: np.ndarray) -> Dict:
        """Rule-based gesture classification from wrist trajectory."""
        if len(traj) < 5:
            return {"gesture": "idle", "confidence": 0.5}

        dx = np.diff(traj[:, 0])
        dy = np.diff(traj[:, 1])
        speed = np.sqrt(dx ** 2 + dy ** 2)
        mean_speed = float(np.mean(speed))

        # Direction sign changes → oscillation (wave)
        sign_changes_x = int(np.sum(np.diff(np.sign(dx)) != 0))
        sign_changes_y = int(np.sum(np.diff(np.sign(dy)) != 0))

        # Net horizontal displacement
        net_dx = float(traj[-1, 0] - traj[0, 0])
        net_dy = float(traj[-1, 1] - traj[0, 1])

        # Classify
        if mean_speed < 0.005:
            return {"gesture": "idle", "confidence": 0.85}

        if sign_changes_x >= 4 and mean_speed > 0.01:
            return {"gesture": "wave", "confidence": 0.80}

        if net_dy < -0.15 and mean_speed > 0.01:
            return {"gesture": "raise", "confidence": 0.78}

        if abs(net_dx) > 0.2 and sign_changes_x < 2:
            gesture = "swipe_right" if net_dx > 0 else "swipe_left"
            return {"gesture": gesture, "confidence": 0.75}

        if mean_speed > 0.008 and abs(net_dx) > 0.1:
            return {"gesture": "point", "confidence": 0.70}

        return {"gesture": "idle", "confidence": 0.60}
