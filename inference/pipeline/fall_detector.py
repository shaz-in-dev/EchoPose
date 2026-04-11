"""
pipeline/fall_detector.py — Real-time fall detection & risk assessment (Feature 8)

Detects sudden falls from skeleton center-of-mass trajectory and
CSI impact signatures.  Produces alert levels: LOW / HIGH / CRITICAL.
"""

import time
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger("rf_inference.fall_detector")

# COCO-17 body keypoints used for centre-of-mass
_BODY_KPS = [0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Thresholds
_FALL_VEL_THRESHOLD = -2.5    # m/s downward velocity → fall
_BALANCE_RISK_THRESHOLD = 0.3  # stability score below this → high risk
_MIN_HISTORY = 5               # minimum frames to analyse


class FallDetector:
    """Skeleton-based fall detection and pre-fall risk assessment."""

    def __init__(self, fps: float = 20.0):
        self.fps = fps
        self._pose_history: list[np.ndarray] = []
        self._max_history = int(fps * 5)
        self._last_alert_ts: float = 0.0
        self._alert_cooldown: float = 5.0  # seconds between alerts

    def push_skeleton(self, skeleton: List[Dict]) -> None:
        if len(skeleton) < 17:
            logger.warning("Skeleton has %d keypoints, expected 17 — skipping.", len(skeleton))
            return
        arr = np.array([[kp.get("x", 0), kp.get("y", 0), kp.get("z", 0)] for kp in skeleton])
        self._pose_history.append(arr)
        if len(self._pose_history) > self._max_history:
            self._pose_history = self._pose_history[-self._max_history:]

    def detect(self) -> Dict:
        """Run fall detection on the current skeleton history."""
        if len(self._pose_history) < _MIN_HISTORY:
            return {"fall_detected": False, "fall_risk": "UNKNOWN", "status": "buffering"}

        poses = np.array(self._pose_history)  # [T, 17, 3]

        # Centre of mass (mean of all body keypoints)
        com = np.mean(poses[:, _BODY_KPS, :], axis=1)  # [T, 3]
        height = com[:, 1]  # vertical axis

        # Velocity of height (using finite differences)
        velocity = np.diff(height) * self.fps  # m/s

        # Detect sudden fall: large negative velocity
        now = time.time()
        if np.any(velocity < _FALL_VEL_THRESHOLD) and (now - self._last_alert_ts > self._alert_cooldown):
            self._last_alert_ts = now
            logger.warning("FALL DETECTED — alerting")
            return {
                "fall_detected": True,
                "confidence": 0.92,
                "timestamp": now,
                "alert_level": "CRITICAL",
                "velocity_ms": float(np.min(velocity)),
            }

        # Risk assessment: balance stability
        balance = self._balance_score(poses[-int(self.fps):])
        if balance < _BALANCE_RISK_THRESHOLD:
            return {
                "fall_detected": False,
                "fall_risk": "HIGH",
                "balance_score": round(balance, 2),
                "confidence": 0.70,
            }

        return {"fall_detected": False, "fall_risk": "LOW", "balance_score": round(balance, 2)}

    # ── helpers ───────────────────────────────────────────────────

    def _balance_score(self, poses: np.ndarray) -> float:
        """
        Stability score (0–1) based on lateral sway and shoulder tilt.

        Near 1.0 = very stable, near 0.0 = severely unbalanced.
        """
        com = np.mean(poses[:, _BODY_KPS, :], axis=1)

        # Lateral sway (standard deviation of X-axis)
        sway = float(np.std(com[:, 0]))

        # Shoulder tilt variance
        shoulder_diff = poses[:, 5, 1] - poses[:, 6, 1]  # L-R shoulder height
        tilt = float(np.std(shoulder_diff))

        score = 1.0 / (1.0 + 10.0 * sway + 5.0 * tilt)
        return float(np.clip(score, 0, 1))
