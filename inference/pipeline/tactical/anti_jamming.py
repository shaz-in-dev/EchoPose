"""
pipeline/tactical/anti_jamming.py — Counter-surveillance & anti-jam (Feature 12)

Detects attempts to jam, spoof, or degrade the WiFi sensing system by
monitoring CSI statistics for anomalies that violate physical
constraints.

Detection methods:
  1. Broadband noise floor elevation (active jamming)
  2. Statistical discontinuity (spoofed CSI)
  3. Physics-violation checks (impossible Doppler/range)
  4. Frequency-sweep detection (active RF sweep)
"""

import time
import numpy as np
from scipy.signal import welch
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("rf_inference.tactical.anti_jamming")

SAMPLE_RATE = 20.0
_NOISE_Z_THRESHOLD = 4.0       # Z-score for broadband noise
_DISCONTINUITY_THRESHOLD = 0.5 # cosine-sim drop between adjacent frames
_MIN_FRAMES = 40


class AntiJammingDefense:
    """Detect RF jamming, spoofing, and interference attacks on the CSI stream."""

    def __init__(self, fs: float = SAMPLE_RATE):
        self.fs = fs
        self._baseline_noise: Optional[float] = None
        self._baseline_spectrum: Optional[np.ndarray] = None
        self._buf: list[np.ndarray] = []
        self._max = int(fs * 30)
        self._alerts: list[Dict] = []

    def calibrate(self, clean_frames: List[np.ndarray]) -> None:
        """Record baseline noise floor & spectral shape from a clean environment."""
        arr = np.array(clean_frames)
        self._baseline_noise = float(np.std(arr))
        mean_sig = np.mean(arr, axis=(0,))
        if mean_sig.ndim > 0 and len(mean_sig) >= 10:
            _, psd = welch(mean_sig, fs=self.fs, nperseg=min(64, len(mean_sig)))
            self._baseline_spectrum = psd
        logger.info(f"Anti-jam baseline: noise_std={self._baseline_noise:.4f}")

    def push(self, csi_amplitudes: np.ndarray) -> None:
        self._buf.append(np.asarray(csi_amplitudes, dtype=np.float64))
        if len(self._buf) > self._max:
            self._buf = self._buf[-self._max:]

    def check(self) -> Dict:
        """Run all anti-jamming checks on the latest buffer."""
        if len(self._buf) < _MIN_FRAMES:
            return {"status": "buffering"}

        threats: List[Dict] = []
        threats.extend(self._broadband_jam())
        threats.extend(self._spoof_detect())
        threats.extend(self._physics_check())
        threats.extend(self._sweep_detect())

        under_attack = len(threats) > 0
        if under_attack:
            alert = {
                "under_attack": True,
                "threats": threats,
                "recommendation": self._recommend(threats),
                "timestamp": time.time(),
            }
            self._alerts.append(alert)
            logger.warning(f"Anti-jam alert: {len(threats)} threats detected")
            return alert

        return {"under_attack": False, "threats": [], "status": "clean"}

    @property
    def alert_log(self) -> List[Dict]:
        return list(self._alerts)

    # ── detection methods ─────────────────────────────────────────

    def _broadband_jam(self) -> List[Dict]:
        """Detect broadband noise elevation (active jamming)."""
        results = []
        recent = np.array(self._buf[-int(self.fs * 2):])
        noise = float(np.std(recent))

        if self._baseline_noise is not None:
            z = (noise - self._baseline_noise) / (self._baseline_noise + 1e-12)
            if z > _NOISE_Z_THRESHOLD:
                results.append({
                    "type": "ACTIVE_JAMMING",
                    "severity": "CRITICAL" if z > 8 else "HIGH",
                    "noise_z_score": round(z, 2),
                    "confidence": round(min(z / 12.0, 0.95), 2),
                })
        return results

    def _spoof_detect(self) -> List[Dict]:
        """Detect spoofed CSI via frame-to-frame discontinuity."""
        results = []
        if len(self._buf) < 10:
            return results

        recent = self._buf[-20:]
        for i in range(1, len(recent)):
            a = recent[i - 1].flatten()
            b = recent[i].flatten()
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a < 1e-12 or norm_b < 1e-12:
                continue
            sim = float(np.dot(a, b) / (norm_a * norm_b))
            if sim < _DISCONTINUITY_THRESHOLD:
                results.append({
                    "type": "CSI_SPOOFING",
                    "severity": "HIGH",
                    "frame_similarity": round(sim, 3),
                    "confidence": round(1 - sim, 2),
                })
                break  # one detection per check cycle
        return results

    def _physics_check(self) -> List[Dict]:
        """Check for physically impossible CSI values."""
        results = []
        recent = np.array(self._buf[-int(self.fs):])

        # Check for unnaturally constant signal (zero variance = synthetic)
        sub_var = np.var(recent, axis=0)
        zero_var_pct = float(np.mean(sub_var < 1e-10))
        if zero_var_pct > 0.5:
            results.append({
                "type": "SYNTHETIC_SIGNAL",
                "severity": "HIGH",
                "zero_var_pct": round(zero_var_pct, 3),
                "confidence": round(zero_var_pct, 2),
            })

        # Check for negative amplitudes (impossible for CSI magnitude)
        neg_pct = float(np.mean(recent < 0))
        if neg_pct > 0.1:
            results.append({
                "type": "IMPOSSIBLE_VALUES",
                "severity": "MEDIUM",
                "negative_pct": round(neg_pct, 3),
                "confidence": 0.90,
            })

        return results

    def _sweep_detect(self) -> List[Dict]:
        """Detect systematic frequency sweeps (active RF recon)."""
        results = []
        if len(self._buf) < int(self.fs * 3):
            return results

        recent = np.array(self._buf[-int(self.fs * 3):])
        sig = np.mean(recent, axis=1)
        sig -= np.mean(sig)
        if len(sig) < 20:
            return results

        freqs, psd = welch(sig, fs=self.fs, nperseg=min(64, len(sig)))

        # A sweep manifests as an unusually flat high-energy spectrum
        if len(psd) > 5:
            if np.all(psd < 1e-10):
                return results  # flat zero PSD — no signal, nothing to flag
            geometric_mean = np.exp(np.mean(np.log(psd + 1e-10)))
            arithmetic_mean = np.mean(psd) + 1e-12
            spectral_flatness = geometric_mean / arithmetic_mean
            flatness = spectral_flatness
            if flatness > 0.8 and float(np.mean(psd)) > 0.01:
                results.append({
                    "type": "FREQUENCY_SWEEP",
                    "severity": "HIGH",
                    "spectral_flatness": round(flatness, 3),
                    "confidence": round(min(flatness, 0.85), 2),
                })
        return results

    def _recommend(self, threats: List[Dict]) -> str:
        severities = [t.get("severity", "LOW") for t in threats]
        if "CRITICAL" in severities:
            return "SWITCH_TO_BACKUP_SENSORS"
        if "HIGH" in severities:
            return "INCREASE_MONITORING"
        return "INVESTIGATE"
