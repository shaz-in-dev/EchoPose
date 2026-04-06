"""
pipeline/emotion.py — Stress / emotion level estimation (Feature 12)

Combines heart rate elevation, breathing rate, and postural cues
to produce a 0–100 stress score and categorical level.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("rf_inference.emotion")

_L_SHOULDER = 5
_R_SHOULDER = 6


class EmotionDetector:
    """Estimates stress level from vitals and posture."""

    def __init__(self):
        # Adaptive baselines (updated over a 30-minute EMA)
        self._hr_baseline: float = 65.0
        self._rr_baseline: float = 15.0
        self._ema_alpha: float = 0.005  # slow adaptation

    def update_baselines(self, hr: float, rr: float) -> None:
        """Update resting baselines using exponential moving average."""
        self._hr_baseline += self._ema_alpha * (hr - self._hr_baseline)
        self._rr_baseline += self._ema_alpha * (rr - self._rr_baseline)

    def estimate_stress(
        self,
        hr: Optional[float],
        rr: Optional[float],
        skeleton: Optional[List[Dict]] = None,
    ) -> Dict:
        """
        Produce a stress score (0–100) and level (CALM / MODERATE / HIGH_STRESS).
        """
        hr = hr or self._hr_baseline
        rr = rr or self._rr_baseline

        hr_elevation = (hr - self._hr_baseline) / max(self._hr_baseline, 1)
        rr_elevation = (rr - self._rr_baseline) / max(self._rr_baseline, 1)

        postural_stress = 0.0
        if skeleton and len(skeleton) >= 17:
            shoulder_tilt = abs(skeleton[_L_SHOULDER].get("y", 0) - skeleton[_R_SHOULDER].get("y", 0))
            postural_stress = 50.0 if shoulder_tilt > 0.08 else 0.0

        stress_score = (
            0.4 * float(np.clip(hr_elevation * 100, 0, 100))
            + 0.3 * float(np.clip(rr_elevation * 100, 0, 100))
            + 0.3 * postural_stress
        )
        stress_score = float(np.clip(stress_score, 0, 100))

        if stress_score < 30:
            level = "CALM"
        elif stress_score < 60:
            level = "MODERATE"
        else:
            level = "HIGH_STRESS"

        return {
            "stress_level": level,
            "stress_score": round(stress_score, 1),
            "hr_elevation_pct": round(hr_elevation * 100, 1),
            "rr_elevation_pct": round(rr_elevation * 100, 1),
        }
