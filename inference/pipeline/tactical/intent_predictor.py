"""
pipeline/tactical/intent_predictor.py — Behavioral intent prediction (Feature 11)

Analyses 10–30 s of gait, activity, and physiological history to
predict near-term intent:
  ATTACK, FLEE, DEFEND, SURRENDER, ACCESS_WEAPON, NORMAL.

Combines micro-behavior cues (fidgeting, stance changes, scanning)
with physiological stress indicators.
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("rf_inference.tactical.intent")

SAMPLE_RATE = 20.0
_MIN_HISTORY = 60   # frames (~3 s at 20 Hz)

# Pre-attack micro-behavior feature weights
_W_FIDGET = 0.20
_W_STANCE = 0.20
_W_STRESS = 0.25
_W_ASYM = 0.15
_W_SCAN = 0.20

INTENT_LABELS = [
    "NORMAL", "ATTACK_IMMINENT", "FLEE", "SURRENDER",
    "DEFENSIVE", "ACCESS_WEAPON",
]


class BehavioralIntentPredictor:
    """Predict near-term behavioral intent from multi-modal cues."""

    def __init__(self, fps: float = SAMPLE_RATE):
        self.fps = fps
        self._skel_buf: list[np.ndarray] = []
        self._stress_buf: list[float] = []
        self._activity_buf: list[str] = []
        self._max = int(fps * 30)

    def push(self, skeleton: List[Dict],
             stress_score: float = 0.0,
             activity: str = "STANDING") -> None:
        arr = np.array([[kp.get("x", 0), kp.get("y", 0), kp.get("z", 0)]
                        for kp in skeleton], dtype=np.float64)
        self._skel_buf.append(arr)
        self._stress_buf.append(stress_score)
        self._activity_buf.append(activity)
        for b in (self._skel_buf, self._stress_buf, self._activity_buf):
            if len(b) > self._max:
                del b[: len(b) - self._max]

    def predict(self) -> Dict:
        """Return predicted intent and time-to-event estimate."""
        if len(self._skel_buf) < _MIN_HISTORY:
            return {"intent": "UNKNOWN", "confidence": 0.0, "status": "buffering"}

        poses = np.array(self._skel_buf)

        fidget = self._fidget_score(poses)
        stance_change = self._stance_change_rate(poses)
        stress = self._recent_stress()
        asym = self._asymmetry_trend(poses)
        scan = self._scanning_behavior(poses)

        attack_score = (
            _W_FIDGET * fidget
            + _W_STANCE * stance_change
            + _W_STRESS * stress
            + _W_ASYM * asym
            + _W_SCAN * scan
        )
        attack_score = float(np.clip(attack_score, 0, 1))

        flee_score = self._flee_score(poses)
        surrender_score = self._surrender_score(poses)

        intent, conf = self._decide(attack_score, flee_score, surrender_score)

        tte = self._time_to_event(attack_score) if intent == "ATTACK_IMMINENT" else None

        return {
            "intent": intent,
            "confidence": round(conf, 2),
            "time_to_event_s": tte,
            "scores": {
                "attack": round(attack_score, 3),
                "flee": round(flee_score, 3),
                "surrender": round(surrender_score, 3),
            },
            "micro_behaviors": {
                "fidget": round(fidget, 3),
                "stance_change": round(stance_change, 3),
                "stress": round(stress, 3),
                "asymmetry_trend": round(asym, 3),
                "scanning": round(scan, 3),
            },
        }

    # ── feature extractors ────────────────────────────────────────

    def _fidget_score(self, poses: np.ndarray) -> float:
        """High-frequency hand/wrist jitter → checking gear / weapon."""
        wrists = poses[-int(self.fps * 5):, [9, 10], :]
        d = np.diff(wrists, axis=0)
        speed = np.sqrt(np.sum(d ** 2, axis=2))
        jitter = float(np.mean(np.abs(np.diff(speed, axis=0))))
        return float(np.clip(jitter / 0.02, 0, 1))

    def _stance_change_rate(self, poses: np.ndarray) -> float:
        """Sudden torso-angle changes over recent window."""
        recent = poses[-int(self.fps * 10):]
        mid_sh = (recent[:, 5] + recent[:, 6]) / 2.0
        mid_hip = (recent[:, 11] + recent[:, 12]) / 2.0
        torso = mid_sh - mid_hip
        angles = np.array([
            np.degrees(np.arccos(np.clip(
                np.dot(v, [0, -1, 0]) / (np.linalg.norm(v) + 1e-9), -1, 1)))
            for v in torso
        ])
        changes = np.abs(np.diff(angles))
        big_changes = np.sum(changes > 10)
        return float(np.clip(big_changes / 10.0, 0, 1))

    def _recent_stress(self) -> float:
        """Normalised mean stress over recent window."""
        recent = self._stress_buf[-int(self.fps * 10):]
        if not recent:
            return 0.0
        return float(np.clip(np.mean(recent) / 100.0, 0, 1))

    def _asymmetry_trend(self, poses: np.ndarray) -> float:
        """Increasing arm asymmetry → reaching for weapon."""
        recent = poses[-int(self.fps * 10):]
        lw = np.std(recent[:, 9, 1])
        rw = np.std(recent[:, 10, 1])
        asym = abs(lw - rw) / (lw + rw + 1e-9)
        return float(np.clip(asym / 0.5, 0, 1))

    def _scanning_behavior(self, poses: np.ndarray) -> float:
        """Head turning (nose X oscillation) → scanning environment."""
        recent = poses[-int(self.fps * 5):]
        nose_x = recent[:, 0, 0]
        sign_changes = int(np.sum(np.diff(np.sign(np.diff(nose_x))) != 0))
        return float(np.clip(sign_changes / 10.0, 0, 1))

    def _flee_score(self, poses: np.ndarray) -> float:
        """Backwards movement + weight shift toward exit."""
        com = np.mean(poses[-int(self.fps * 5):, [11, 12], :], axis=1)
        vel = np.diff(com, axis=0)
        backward = float(np.mean(vel[:, 2]))
        speed = float(np.mean(np.sqrt(np.sum(vel ** 2, axis=1))))
        if backward > 0.01 and speed > 0.02:
            return float(np.clip(speed / 0.1, 0, 1))
        return 0.0

    def _surrender_score(self, poses: np.ndarray) -> float:
        """Both hands above shoulders + no movement."""
        latest = poses[-1]
        lw_y, rw_y = latest[9, 1], latest[10, 1]
        ls_y, rs_y = latest[5, 1], latest[6, 1]
        both_up = lw_y < ls_y and rw_y < rs_y
        com = np.mean(poses[-int(self.fps * 2):, [11, 12], :], axis=1)
        still = float(np.std(com)) < 0.005
        return 0.9 if both_up and still else 0.0

    # ── decision ──────────────────────────────────────────────────

    def _decide(self, attack: float, flee: float, surr: float) -> tuple:
        scores = {"ATTACK_IMMINENT": attack, "FLEE": flee, "SURRENDER": surr}
        best = max(scores, key=scores.get)
        val = scores[best]
        if val < 0.3:
            return ("NORMAL", max(0.7, 1 - val))
        return (best, round(val, 2))

    def _time_to_event(self, score: float) -> Optional[float]:
        if score > 0.8:
            return 5.0
        if score > 0.6:
            return 15.0
        return 30.0
