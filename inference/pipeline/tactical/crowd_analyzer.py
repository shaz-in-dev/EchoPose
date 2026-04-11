"""
pipeline/tactical/crowd_analyzer.py — Crowd counting & density (Feature 7)

Estimates the number of people in a monitored area from multi-node
CSI using spectral clustering (≤20 targets) and statistical density
estimation (20+ targets).
"""

import numpy as np
from scipy.signal import welch, find_peaks
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("rf_inference.tactical.crowd")

SAMPLE_RATE = 20.0
_MAX_TRACKED = 20
_DENSITY_CATEGORIES = {0.5: "SPARSE", 1.0: "MODERATE", 2.0: "DENSE", 999: "CRITICAL"}


class CrowdDensityAnalyzer:
    """Count people and estimate crowd density from multi-node CSI."""

    def __init__(self, fs: float = SAMPLE_RATE):
        self.fs = fs
        self._buf: list[np.ndarray] = []
        self._max = int(fs * 10)

    def push(self, csi_amplitudes: np.ndarray) -> None:
        self._buf.append(np.asarray(csi_amplitudes, dtype=np.float64))
        if len(self._buf) > self._max:
            self._buf = self._buf[-self._max:]

    def estimate(self, room_area_m2: float = 50.0,
                 skeleton_count: int = 0) -> Dict:
        """
        Return estimated count, density, and category.

        skeleton_count: number of skeletons already tracked by the pose
                        estimator (high-confidence count for small groups).
        """
        if len(self._buf) < int(self.fs * 2):
            return {"status": "buffering"}

        history = np.array(self._buf)

        # Method 1: direct skeleton count (most reliable for ≤ pose-estimator max)
        skel_est = skeleton_count

        # Method 2: spectral peak counting (works up to ~20)
        spectral_est = self._spectral_count(history)

        # Method 3: CSI variance model (statistical, for large crowds)
        variance_est = self._variance_crowd_model(history, room_area_m2)

        # Fuse estimates
        if skel_est > 0 and skel_est <= 5:
            best = skel_est
            method = "skeleton"
            conf = 0.95
        elif spectral_est <= _MAX_TRACKED:
            best = max(spectral_est, skel_est)
            method = "spectral"
            conf = 0.82
        else:
            best = variance_est
            method = "statistical"
            conf = 0.70

        density = best / max(room_area_m2, 1.0)
        category = self._categorize(density)

        return {
            "estimated_count": best,
            "density_per_m2": round(density, 3),
            "density_category": category,
            "confidence": round(conf, 2),
            "method": method,
            "sub_estimates": {
                "skeleton": skel_est,
                "spectral": spectral_est,
                "statistical": variance_est,
            },
        }

    # ── counting methods ──────────────────────────────────────────

    def _spectral_count(self, h: np.ndarray) -> int:
        """Count distinct motion-frequency peaks in the Doppler spectrum."""
        sig = np.mean(h, axis=1) if h.ndim > 1 else h
        sig = sig - np.mean(sig)
        if len(sig) < 20:
            return 0
        freqs, psd = welch(sig, fs=self.fs, nperseg=min(128, len(sig)))
        mask = (freqs >= 0.3) & (freqs <= 8.0)
        if not np.any(mask):
            return 0
        p = psd[mask]
        if len(p) == 0 or not np.all(np.isfinite(p)):
            return 0
        max_p = np.max(p)
        if max_p < 1e-12:
            return 0
        peaks, props = find_peaks(p, height=max_p * 0.15, distance=3)
        return len(peaks)

    def _variance_crowd_model(self, h: np.ndarray, area: float) -> int:
        """
        Statistical crowd sizing via CSI variance scaling.

        More people → more multipath → higher subcarrier variance.
        Calibrated with empirical constants from indoor WiFi sensing papers.
        """
        time_var = float(np.mean(np.var(h, axis=0)))
        # Empirical: each person adds ~0.002 to mean subcarrier variance
        person_per_var = 0.002
        raw_est = time_var / person_per_var
        # Clamp to reasonable range
        return int(np.clip(raw_est, 0, area * 4))

    @staticmethod
    def _categorize(density: float) -> str:
        for thresh, cat in _DENSITY_CATEGORIES.items():
            if density < thresh:
                return cat
        return "CRITICAL"
