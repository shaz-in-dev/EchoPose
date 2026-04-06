"""
pipeline/tactical/anomaly_scanner.py — Anomalous-object / IED detection (Feature 10)

Detects anomalous objects by analysing high-frequency CSI components for
electromagnetic interference (EMI) patterns inconsistent with normal
building contents.  Targets phone-circuit triggers, timer oscillators,
and metallic density anomalies.

Outputs are probabilistic — all results require human verification.
"""

import numpy as np
from scipy.signal import welch, find_peaks
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("rf_inference.tactical.anomaly_scanner")

SAMPLE_RATE = 20.0

# Known EMI signature frequencies (Hz in CSI domain)
_PHONE_CLOCK_BANDS = [(0.8, 1.2), (1.8, 2.2)]   # GSM clock harmonics
_TIMER_BANDS = [(0.5, 0.6), (1.0, 1.1)]          # Common timer crystal harmonics
_DENSITY_THRESHOLD = 2.5                           # Z-score for metallic anomaly


class AnomalyScanner:
    """Detect anomalous electromagnetic / density signatures in CSI."""

    def __init__(self, fs: float = SAMPLE_RATE):
        self.fs = fs
        self._baseline: Optional[np.ndarray] = None
        self._buf: list[np.ndarray] = []
        self._max = int(fs * 15)

    def calibrate(self, empty_frames: List[np.ndarray]) -> None:
        """Record baseline CSI for an area known to be clear."""
        self._baseline = np.array(empty_frames)
        logger.info(f"Anomaly scanner calibrated with {len(empty_frames)} frames.")

    def push(self, csi_amplitudes: np.ndarray) -> None:
        self._buf.append(np.asarray(csi_amplitudes, dtype=np.float64))
        if len(self._buf) > self._max:
            self._buf = self._buf[-self._max:]

    def scan(self) -> Dict:
        """Scan for anomalous EMI / density signatures."""
        if len(self._buf) < int(self.fs * 3):
            return {"status": "buffering", "frames": len(self._buf)}

        history = np.array(self._buf)
        anomalies = []

        # 1. EMI pattern scan
        emi = self._emi_scan(history)
        anomalies.extend(emi)

        # 2. Density anomaly (metallic object)
        density = self._density_scan(history)
        anomalies.extend(density)

        # 3. Spectral anomaly (unusual persistent tones)
        spectral = self._spectral_anomaly(history)
        anomalies.extend(spectral)

        threat = "CLEAR"
        if any(a["severity"] == "HIGH" for a in anomalies):
            threat = "SUSPICIOUS"
        if any(a["severity"] == "CRITICAL" for a in anomalies):
            threat = "DANGER"

        return {
            "anomalies_found": len(anomalies),
            "anomalies": anomalies,
            "threat_assessment": threat,
            "scan_coverage": self._coverage(),
        }

    # ── scan methods ──────────────────────────────────────────────

    def _emi_scan(self, h: np.ndarray) -> List[Dict]:
        """Look for phone-circuit or timer-circuit EMI in CSI."""
        results = []
        sig = np.mean(h, axis=1) - np.mean(np.mean(h, axis=1))
        if len(sig) < 20:
            return results

        freqs, psd = welch(sig, fs=self.fs, nperseg=min(128, len(sig)))

        for label, bands in [("PHONE_CIRCUIT", _PHONE_CLOCK_BANDS),
                             ("TIMER_CIRCUIT", _TIMER_BANDS)]:
            for lo, hi in bands:
                mask = (freqs >= lo) & (freqs <= hi)
                if not np.any(mask):
                    continue
                band_power = float(np.max(psd[mask]))
                noise = float(np.mean(psd[~mask & (freqs > 0)])) + 1e-12
                snr = band_power / noise
                if snr > 4.0:
                    results.append({
                        "type": label,
                        "band_hz": [lo, hi],
                        "snr": round(snr, 2),
                        "severity": "HIGH" if snr > 8.0 else "MEDIUM",
                        "confidence": round(min(snr / 12.0, 0.90), 2),
                    })
        return results

    def _density_scan(self, h: np.ndarray) -> List[Dict]:
        """Detect abnormal high-density (metallic) reflections."""
        results = []
        if self._baseline is None:
            return results

        base_mean = np.mean(self._baseline, axis=0)
        live_mean = np.mean(h[-int(self.fs * 2):], axis=0)
        diff = live_mean - base_mean
        z = diff / (np.std(self._baseline, axis=0) + 1e-12)

        n_sub = len(z)
        chunk = max(n_sub // 8, 1)
        for i in range(0, n_sub, chunk):
            seg = z[i: i + chunk]
            max_z = float(np.max(np.abs(seg)))
            if max_z > _DENSITY_THRESHOLD:
                results.append({
                    "type": "METALLIC_ANOMALY",
                    "zone": f"sub_{i}-{i + chunk}",
                    "z_score": round(max_z, 2),
                    "severity": "CRITICAL" if max_z > 5.0 else "HIGH",
                    "confidence": round(min(max_z / 8.0, 0.85), 2),
                })
        return results

    def _spectral_anomaly(self, h: np.ndarray) -> List[Dict]:
        """Detect persistent narrowband tones not present in baseline."""
        results = []
        sig = np.mean(h, axis=1) - np.mean(np.mean(h, axis=1))
        if len(sig) < 40:
            return results

        freqs, psd = welch(sig, fs=self.fs, nperseg=min(128, len(sig)))
        peaks, props = find_peaks(psd, height=np.max(psd) * 0.4, distance=5)

        if self._baseline is not None and len(self._baseline) >= 40:
            base_sig = np.mean(self._baseline, axis=1) - np.mean(np.mean(self._baseline, axis=1))
            _, base_psd = welch(base_sig, fs=self.fs, nperseg=min(128, len(base_sig)))
        else:
            base_psd = np.zeros_like(psd)

        for pk in peaks:
            if pk >= len(base_psd):
                continue
            excess = psd[pk] / (base_psd[pk] + 1e-12)
            if excess > 5.0:
                results.append({
                    "type": "SPECTRAL_TONE",
                    "frequency_hz": round(float(freqs[pk]), 3),
                    "excess_ratio": round(float(excess), 1),
                    "severity": "MEDIUM",
                    "confidence": round(min(excess / 15.0, 0.80), 2),
                })
        return results

    def _coverage(self) -> str:
        if len(self._buf) >= int(self.fs * 10):
            return "FULL"
        if len(self._buf) >= int(self.fs * 5):
            return "PARTIAL"
        return "MINIMAL"
