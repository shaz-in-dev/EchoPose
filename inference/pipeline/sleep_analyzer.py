"""
pipeline/sleep_analyzer.py — Sleep stage classification (Feature 9)

Classifies sleep into: Awake, N1 (Light), N2 (Medium), N3 (Deep), REM
using skeleton immobility, heart-rate variability, breathing regularity,
and CSI-derived head micro-motions (proxy for eye movement).
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("rf_inference.sleep")


class SleepAnalyzer:
    """Sleep-stage classifier from combined vitals + skeleton data."""

    def __init__(self, fps: float = 20.0, window_minutes: int = 2):
        self.fps = fps
        self._hr_history: list[float] = []
        self._rr_history: list[float] = []
        self._motion_history: list[float] = []
        self._max_samples = int(window_minutes * 60)  # one entry per second
        self._last_coords = None  # M4: explicit init so push_motion guard is reliable

    def push_vitals(self, hr: Optional[float], rr: Optional[float]) -> None:
        if hr is not None:
            self._hr_history.append(hr)
        if rr is not None:
            self._rr_history.append(rr)
        for buf in (self._hr_history, self._rr_history):
            if len(buf) > self._max_samples:
                del buf[: len(buf) - self._max_samples]

    def push_motion(self, skeleton: List[Dict]) -> None:
        """Compute frame-level motion energy and store."""
        coords = np.array([[kp.get("x", 0), kp.get("y", 0), kp.get("z", 0)] for kp in skeleton])
        # M4: guard _last_coords against None (first frame or reset)
        if self._last_coords is not None:
            energy = float(np.mean(np.abs(coords.flatten() - self._last_coords.flatten())))
        else:
            energy = 0.0
        self._last_coords = coords
        self._motion_history.append(energy)
        if len(self._motion_history) > self._max_samples:
            self._motion_history = self._motion_history[-self._max_samples:]

    def classify(self) -> Dict:
        """Return current sleep stage estimate and confidence."""
        if len(self._hr_history) < 30 or len(self._motion_history) < 30:
            return {"sleep_stage": "UNKNOWN", "confidence": 0.0, "status": "buffering"}

        motion_level = float(np.mean(self._motion_history[-30:]))
        hr_arr = np.array(self._hr_history[-60:])
        rr_arr = np.array(self._rr_history[-60:]) if len(self._rr_history) >= 10 else np.array([15.0])

        hrv = float(np.std(np.diff(hr_arr))) if len(hr_arr) > 2 else 0.0
        mean_rr = float(np.mean(rr_arr))
        rr_regularity = 1.0 / (1.0 + float(np.std(rr_arr)))

        # Head micro-motion (proxy for REM eye movements) — last 10 s
        head_motion = float(np.mean(self._motion_history[-int(self.fps * 10):])) if len(self._motion_history) >= int(self.fps * 2) else 0.0

        # Decision logic
        if motion_level > 0.05:
            stage, conf = "AWAKE", 0.88
        elif head_motion > 0.02 and motion_level < 0.01:
            stage, conf = "REM", 0.72
        elif hrv > 0.3 and motion_level < 0.02:
            stage, conf = "N1", 0.70
        elif motion_level < 0.005 and mean_rr < 15 and rr_regularity > 0.7:
            stage, conf = "N3", 0.75
        else:
            stage, conf = "N2", 0.68

        return {
            "sleep_stage": stage,
            "confidence": round(conf, 2),
            "metrics": {
                "motion_level": round(motion_level, 4),
                "hrv": round(hrv, 3),
                "rr_mean": round(mean_rr, 1),
                "rr_regularity": round(rr_regularity, 3),
            },
        }
