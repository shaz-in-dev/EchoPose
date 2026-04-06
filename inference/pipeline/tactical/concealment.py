"""
pipeline/tactical/concealment.py — Concealment & hidden-target detection (Feature 3)

Detects people behind camouflage, inside vehicles, or in concealment
by comparing live CSI to an empty-room baseline and searching for
biological micro-motion signatures (heartbeat, breathing, involuntary
movements) that cannot be fully suppressed.
"""

import numpy as np
from scipy.signal import welch, butter, filtfilt
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("rf_inference.tactical.concealment")

SAMPLE_RATE = 20.0
_HR_BAND = (0.8, 3.0)     # Heart-rate micro-Doppler band (Hz)
_BR_BAND = (0.15, 0.5)    # Breathing band (Hz)
_MIN_FRAMES = 100          # ~5 s at 20 Hz
_DETECT_THRESHOLD = 3.0    # SNR above noise floor


class ConcealmentDetector:
    """Detect concealed humans from residual biological signatures in CSI."""

    def __init__(self, fs: float = SAMPLE_RATE):
        self.fs = fs
        self._baseline: Optional[np.ndarray] = None
        self._buf: list[np.ndarray] = []
        self._max = int(fs * 30)

    def calibrate_baseline(self, empty_frames: List[np.ndarray]) -> None:
        """Record CSI when the area is known to be empty."""
        self._baseline = np.mean(np.array(empty_frames), axis=0)
        logger.info("Concealment baseline calibrated.")

    def push(self, csi_amplitudes: np.ndarray) -> None:
        self._buf.append(np.asarray(csi_amplitudes, dtype=np.float64))
        if len(self._buf) > self._max:
            self._buf = self._buf[-self._max:]

    def scan(self) -> Dict:
        """Scan for concealed targets in the current buffer."""
        if len(self._buf) < _MIN_FRAMES:
            return {"status": "buffering", "frames": len(self._buf)}

        history = np.array(self._buf[-_MIN_FRAMES:])

        # Subtract baseline if available
        if self._baseline is not None:
            residual = history - self._baseline
        else:
            residual = history - np.mean(history, axis=0)

        targets = self._search_bio_signatures(residual)

        return {
            "concealed_targets": len(targets),
            "targets": targets,
            "scan_quality": self._scan_quality(residual),
        }

    # ── core algorithms ───────────────────────────────────────────

    def _search_bio_signatures(self, residual: np.ndarray) -> List[Dict]:
        """Search residual CSI for heartbeat/breathing micro-Doppler."""
        targets = []
        n_sub = residual.shape[1]
        chunk = max(n_sub // 4, 1)

        for zone in range(0, n_sub, chunk):
            end = min(zone + chunk, n_sub)
            sig = np.mean(residual[:, zone:end], axis=1)
            sig -= np.mean(sig)

            hr_snr = self._band_snr(sig, _HR_BAND)
            br_snr = self._band_snr(sig, _BR_BAND)

            if hr_snr > _DETECT_THRESHOLD or br_snr > _DETECT_THRESHOLD:
                micro_motion = self._micro_motion_energy(sig)
                targets.append({
                    "zone": f"sub_{zone}-{end}",
                    "heartbeat_snr": round(hr_snr, 2),
                    "breathing_snr": round(br_snr, 2),
                    "micro_motion": round(micro_motion, 4),
                    "confidence": round(min(max(hr_snr, br_snr) / 10.0, 0.95), 2),
                    "type": "CONCEALED_HUMAN",
                })

        return targets

    def _band_snr(self, sig: np.ndarray, band: tuple) -> float:
        """SNR of a specific frequency band vs. the rest of the spectrum."""
        if len(sig) < 20:
            return 0.0
        freqs, psd = welch(sig, fs=self.fs, nperseg=min(128, len(sig)))
        in_band = (freqs >= band[0]) & (freqs <= band[1])
        out_band = ~in_band & (freqs > 0)
        sp = float(np.mean(psd[in_band])) if np.any(in_band) else 0.0
        np_ = float(np.mean(psd[out_band])) if np.any(out_band) else 1e-12
        return sp / np_

    def _micro_motion_energy(self, sig: np.ndarray) -> float:
        """RMS of high-pass-filtered residual (> 0.5 Hz)."""
        nyq = self.fs / 2.0
        b, a = butter(3, 0.5 / nyq, btype="high")
        if len(sig) < 3 * max(len(b), len(a)):
            return 0.0
        hp = filtfilt(b, a, sig)
        return float(np.sqrt(np.mean(hp ** 2)))

    def _scan_quality(self, residual: np.ndarray) -> str:
        noise = float(np.std(residual))
        if noise < 0.01:
            return "HIGH"
        elif noise < 0.05:
            return "MEDIUM"
        return "LOW"
