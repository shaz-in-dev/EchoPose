"""
pipeline/tactical/threat_tracker.py — Through-wall target detection (Feature 1)

Detects and classifies human movement through solid barriers using
WiFi CSI micro-Doppler analysis.  Extracts cadence, classifies target
type (human / animal / mechanical), and estimates approach posture.

References:
  - WiFi micro-Doppler for through-wall detection (IEEE TGRS, 2019)
  - Gait-based human detection from CSI (ACM MobiCom, 2020)

IMPORTANT: Outputs are analytical only.
"""

import time
import numpy as np
from scipy.signal import welch, spectrogram, find_peaks
from typing import Dict, List
import logging

logger = logging.getLogger("rf_inference.tactical.threat")

SAMPLE_RATE = 20.0

# Micro-Doppler bands (Hz)
_CRAWL_BAND = (0.3, 1.0)
_WALK_BAND = (1.5, 2.5)
_RUN_BAND = (2.5, 4.0)
_ANIMAL_BAND = (4.0, 8.0)

_MIN_MOTION = 0.005
_ASYMMETRY_THRESHOLD = 0.35


class TacticalTargetTracker:
    """Through-wall target detection via CSI micro-Doppler."""

    def __init__(self, fs: float = SAMPLE_RATE, history_s: int = 10):
        self.fs = fs
        self._max = int(fs * history_s)
        self._buf: list[np.ndarray] = []
        self._track_id = 0

    def push(self, csi_amplitudes: np.ndarray) -> None:
        self._buf.append(np.asarray(csi_amplitudes, dtype=np.float64))
        if len(self._buf) > self._max:
            self._buf = self._buf[-self._max:]

    def detect(self) -> Dict:
        """Run through-wall detection on current buffer."""
        if len(self._buf) < int(self.fs * 3):
            return {"status": "buffering", "frames": len(self._buf)}

        history = np.array(self._buf)
        motion = self._motion_energy(history)
        cadence = self._extract_cadence(history)
        asym = self._gait_asymmetry(history)
        targets = self._classify(cadence, asym, motion, history)
        threat = self._assess(targets)

        return {
            "targets_detected": len(targets),
            "targets": targets,
            "threat_level": threat,
            "motion_energy": round(motion, 4),
            "timestamp": time.time(),
        }

    # ── micro-Doppler ─────────────────────────────────────────────

    def _motion_energy(self, h: np.ndarray) -> float:
        d = np.diff(h, axis=0)
        return float(np.mean(np.sqrt(np.sum(d ** 2, axis=1))))

    def _extract_cadence(self, h: np.ndarray) -> float:
        sig = np.mean(h, axis=1)
        sig -= np.mean(sig)
        if len(sig) < 20:
            return 0.0
        freqs, psd = welch(sig, fs=self.fs, nperseg=min(128, len(sig)))
        mask = (freqs >= 0.3) & (freqs <= 4.0)
        if not np.any(mask) or np.max(psd[mask]) == 0:
            return 0.0
        return float(freqs[mask][np.argmax(psd[mask])])

    def _gait_asymmetry(self, h: np.ndarray) -> float:
        if h.shape[1] < 4:
            return 0.0
        n = h.shape[1]
        lv = np.var(np.mean(h[:, :n // 2], axis=1))
        rv = np.var(np.mean(h[:, n // 2:], axis=1))
        total = lv + rv
        return float(abs(lv - rv) / total) if total > 1e-12 else 0.0

    def _estimate_count(self, h: np.ndarray) -> int:
        sig = np.mean(h, axis=1) - np.mean(np.mean(h, axis=1))
        if len(sig) < 20:
            return 1
        freqs, psd = welch(sig, fs=self.fs, nperseg=min(64, len(sig)))
        mask = (freqs >= 0.3) & (freqs <= 4.0)
        if not np.any(mask):
            return 1
        p = psd[mask]
        max_p = np.max(p)
        if max_p < 1e-12:
            return 1
        peaks, _ = find_peaks(p, height=max_p * 0.3, distance=3)
        return max(1, len(peaks))

    def _classify(self, cadence: float, asym: float,
                  motion: float, h: np.ndarray) -> List[Dict]:
        if motion < _MIN_MOTION or cadence == 0.0:
            return []

        if _ANIMAL_BAND[0] <= cadence <= _ANIMAL_BAND[1]:
            ttype = "ANIMAL"
        elif _CRAWL_BAND[0] <= cadence <= _CRAWL_BAND[1]:
            ttype = "HUMAN_CRAWLING"
        elif _WALK_BAND[0] <= cadence <= _WALK_BAND[1]:
            ttype = "HUMAN_WALKING"
        elif _RUN_BAND[0] <= cadence <= _RUN_BAND[1]:
            ttype = "HUMAN_RUNNING"
        else:
            ttype = "UNKNOWN"

        n = self._estimate_count(h)
        targets = []
        for _ in range(n):
            tid = self._track_id
            self._track_id += 1
            targets.append({
                "track_id": tid,
                "type": ttype,
                "cadence_hz": round(cadence, 2),
                "asymmetry": round(asym, 3),
                "encumbered": asym > _ASYMMETRY_THRESHOLD,
                "confidence": round(min(0.6 + 0.3 * (motion / 0.1), 0.95), 2),
            })
        return targets

    def _assess(self, targets: List[Dict]) -> str:
        if not targets:
            return "GREEN"
        for t in targets:
            if t["type"] == "HUMAN_RUNNING" and t.get("encumbered"):
                return "RED"
            if t["type"] in ("HUMAN_RUNNING", "HUMAN_CRAWLING"):
                return "YELLOW"
        return "YELLOW"
