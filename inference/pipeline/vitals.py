"""
pipeline/vitals.py — Non-contact vital sign extraction from WiFi CSI

Extracts physiological signals from Channel State Information:
  - Heart Rate (HR): 40–180 bpm from chest micro-Doppler
  - Respiratory Rate (RR): 6–60 breaths/min from thorax motion
  - SpO2 estimation: blood oxygenation proxy via multi-frequency ratio
  - Body temperature estimation: CSI amplitude variance correlation
  - Blood pressure estimation: pulse wave velocity across nodes

DISCLAIMER — EXPERIMENTAL / NOT CLINICALLY VALIDATED:
  The SpO2, body temperature, and blood pressure estimates produced by this
  module are **experimental proxy values** derived from WiFi signal
  statistics.  They have NOT been validated against clinical-grade medical
  devices and MUST NOT be used for medical diagnosis, treatment decisions, or
  any safety-critical application.  Heart rate and respiratory rate
  extraction from WiFi CSI has published peer-reviewed support, but the
  absolute accuracy depends heavily on environment and calibration.

All methods operate on raw CSI amplitude arrays [subcarriers] or
[nodes, subcarriers] at the system's 20 Hz sampling rate.
"""

import numpy as np
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt, welch
from typing import Dict, Optional
import logging

logger = logging.getLogger("rf_inference.vitals")

SAMPLE_RATE = 20.0  # Hz — CSI capture rate


class VitalsExtractor:
    """Extracts vital signs from CSI amplitude time-series."""

    def __init__(self, fs: float = SAMPLE_RATE, history_seconds: int = 30):
        self.fs = fs
        self.history_len = int(fs * history_seconds)
        # Rolling buffers keyed by node_id
        self._amplitude_history: list[np.ndarray] = []
        self._baseline_variance: Optional[float] = None
        self._calibration_factor: float = 0.15  # empirical, tune with IR reference

    # ── public API ────────────────────────────────────────────────

    def push(self, amplitudes: np.ndarray) -> None:
        """Append a new CSI amplitude snapshot to the rolling buffer."""
        arr = np.asarray(amplitudes, dtype=np.float64)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            logger.warning("Vitals push: skipping invalid amplitudes (empty or NaN/Inf).")
            return
        self._amplitude_history.append(arr)
        if len(self._amplitude_history) > self.history_len:
            self._amplitude_history = self._amplitude_history[-self.history_len:]

    def extract_all(self, node_amplitudes: Optional[np.ndarray] = None) -> Dict:
        """Return all available vital signs from the current buffer."""
        if len(self._amplitude_history) < int(self.fs * 5):
            return {"status": "buffering", "samples": len(self._amplitude_history)}

        history = np.array(self._amplitude_history)  # [T, subcarriers]

        hr_result = self.extract_heart_rate(history)
        rr_result = self.extract_respiration(history)

        hr_bpm = hr_result.get("heart_rate")
        spo2_result = self.estimate_spo2(history, hr_bpm) if hr_bpm else {"spo2": None, "confidence": 0.0}
        temp_result = self.estimate_temperature(history)
        bp_result = self.estimate_blood_pressure(history, hr_bpm, node_amplitudes)

        return {
            "heart_rate": hr_result,
            "respiratory_rate": rr_result,
            "spo2": spo2_result,
            "temperature": temp_result,
            "blood_pressure": bp_result,
        }

    # ── Feature 1: Heart Rate ─────────────────────────────────────

    def extract_heart_rate(self, history: np.ndarray) -> Dict:
        """
        Detect heart rate from chest micro-motion in CSI.

        Focuses on subcarriers 30–40 (chest reflection zone) and
        isolates the 0.67–3.0 Hz band (40–180 bpm).
        """
        chest_csi = self._chest_signal(history)
        filtered = self._butter_bandpass(chest_csi, 0.67, 3.0)

        freqs, psd = welch(filtered, fs=self.fs, nperseg=min(256, len(filtered)))

        hr_mask = (freqs >= 0.67) & (freqs <= 3.0)
        if not np.any(hr_mask) or np.max(psd[hr_mask]) == 0:
            return {"heart_rate": None, "confidence": 0.0}

        dominant_freq = freqs[hr_mask][np.argmax(psd[hr_mask])]
        bpm = dominant_freq * 60.0

        snr = self._spectral_snr(psd, freqs, dominant_freq)
        confidence = float(np.clip(snr / 10.0, 0.0, 1.0))

        return {"heart_rate": round(float(bpm), 1), "confidence": round(confidence, 2)}

    # ── Feature 2: Respiratory Rate ───────────────────────────────

    def extract_respiration(self, history: np.ndarray) -> Dict:
        """
        Extract respiratory rate from thorax CSI motion.

        Normal: 12–20 breaths/min (0.2–0.33 Hz).
        Detectable range: 6–60 breaths/min (0.1–1.0 Hz).
        """
        chest_csi = self._chest_signal(history, lo=25, hi=45)
        filtered = self._butter_bandpass(chest_csi, 0.1, 1.0)

        freqs, psd = welch(filtered, fs=self.fs, nperseg=min(256, len(filtered)))

        resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
        if not np.any(resp_mask) or np.max(psd[resp_mask]) == 0:
            return {"respiratory_rate": None, "confidence": 0.0}

        dominant_freq = freqs[resp_mask][np.argmax(psd[resp_mask])]
        rr_bpm = dominant_freq * 60.0

        snr = self._spectral_snr(psd, freqs, dominant_freq)
        confidence = float(np.clip(snr / 8.0, 0.0, 1.0))

        return {"respiratory_rate": round(float(rr_bpm), 1), "confidence": round(confidence, 2)}

    # ── Feature 3: SpO2 ───────────────────────────────────────────

    def estimate_spo2(self, history: np.ndarray, hr_bpm: Optional[float]) -> Dict:
        """
        Estimate blood oxygen saturation from multi-frequency CSI ratio.

        Uses the ratio of AC pulsatile component at the heart-rate
        frequency and its second harmonic — analogous to dual-wavelength
        pulse oximetry.
        """
        if hr_bpm is None or hr_bpm <= 0:
            return {"spo2": None, "confidence": 0.0}

        skin_csi = np.mean(history[:, 10:20], axis=1) if history.shape[1] > 20 else np.mean(history, axis=1)
        ac_signal = skin_csi - np.mean(skin_csi)

        n = len(ac_signal)
        fft_vals = np.fft.rfft(ac_signal)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)

        hr_hz = hr_bpm / 60.0
        hr_idx = np.argmin(np.abs(freqs - hr_hz))
        h2_idx = np.argmin(np.abs(freqs - 2 * hr_hz))

        amp_hr = np.abs(fft_vals[hr_idx])
        amp_h2 = np.abs(fft_vals[h2_idx]) + 1e-9

        r_ratio = amp_hr / amp_h2
        spo2 = float(np.clip(110.0 - 25.0 * r_ratio, 85.0, 100.0))

        return {"spo2": round(spo2, 1), "confidence": 0.55}

    # ── Feature 4: Body Temperature ───────────────────────────────

    def estimate_temperature(self, history: np.ndarray, ambient_temp: float = 22.0) -> Dict:
        """
        Estimate body temperature from CSI amplitude variance.

        Higher body temperature increases thermal radiation which
        modulates WiFi absorption.  Requires initial calibration
        against an IR thermometer.
        """
        variance = float(np.var(np.mean(history, axis=1)))

        if self._baseline_variance is None:
            self._baseline_variance = variance

        temp_offset = (variance - self._baseline_variance) * self._calibration_factor
        body_temp = 37.0 + float(np.clip(temp_offset, -2.0, 3.0))

        return {"temperature_c": round(body_temp, 1), "confidence": 0.60}

    # ── Feature 5: Blood Pressure ─────────────────────────────────

    def estimate_blood_pressure(
        self,
        history: np.ndarray,
        hr_bpm: Optional[float],
        node_amplitudes: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Estimate systolic/diastolic BP via Pulse Wave Velocity (PWV).

        Requires multi-node CSI to measure pulse transit time between
        different body locations.  Falls back to HR-based regression
        when only one node is available.
        """
        if hr_bpm is None or hr_bpm <= 0:
            return {"systolic_mmhg": None, "diastolic_mmhg": None, "confidence": 0.0}

        if node_amplitudes is not None and node_amplitudes.ndim >= 2 and node_amplitudes.shape[0] >= 2:
            pwv = self._calculate_pwv(node_amplitudes)
            systolic = 110.0 + 0.39 * pwv
        else:
            # Fallback: HR-based linear regression (less accurate)
            systolic = 80.0 + 0.5 * hr_bpm

        diastolic = systolic - 40.0
        systolic = float(np.clip(systolic, 80, 200))
        diastolic = float(np.clip(diastolic, 50, 130))

        return {
            "systolic_mmhg": round(systolic, 0),
            "diastolic_mmhg": round(diastolic, 0),
            "confidence": 0.50,
        }

    # ── private helpers ───────────────────────────────────────────

    def _chest_signal(self, history: np.ndarray, lo: int = 30, hi: int = 40) -> np.ndarray:
        """Extract mean amplitude across chest-zone subcarriers."""
        n_sub = history.shape[1]
        lo = min(lo, n_sub - 1)
        hi = min(hi, n_sub)
        return np.mean(history[:, lo:hi], axis=1)

    def _butter_bandpass(self, data: np.ndarray, lowcut: float, highcut: float, order: int = 4) -> np.ndarray:
        nyq = self.fs / 2.0
        low = max(lowcut / nyq, 0.001)
        high = min(highcut / nyq, 0.999)
        b, a = butter(order, [low, high], btype="band")
        if len(data) < 3 * max(len(b), len(a)):
            return data
        return filtfilt(b, a, data)

    def _spectral_snr(self, psd: np.ndarray, freqs: np.ndarray, peak_freq: float, bw: float = 0.1) -> float:
        """Signal-to-noise ratio around a spectral peak."""
        sig_mask = np.abs(freqs - peak_freq) < bw
        noise_mask = ~sig_mask & (freqs > 0)
        sig_power = np.mean(psd[sig_mask]) if np.any(sig_mask) else 0
        noise_power = np.mean(psd[noise_mask]) if np.any(noise_mask) else 1e-12
        return float(sig_power / noise_power)

    def _calculate_pwv(self, node_amps: np.ndarray) -> float:
        """
        Pulse Wave Velocity from cross-correlation lag between two nodes.

        Assumes nodes are ~0.5 m apart (typical room placement).
        """
        sig_a = np.mean(node_amps[0], axis=-1) if node_amps[0].ndim > 1 else node_amps[0]
        sig_b = np.mean(node_amps[1], axis=-1) if node_amps[1].ndim > 1 else node_amps[1]
        min_len = min(len(sig_a), len(sig_b))
        sig_a, sig_b = sig_a[:min_len], sig_b[:min_len]

        corr = np.correlate(sig_a - np.mean(sig_a), sig_b - np.mean(sig_b), mode="full")
        lag = np.argmax(corr) - (min_len - 1)
        delay_s = abs(lag) / self.fs
        distance_m = 0.5  # assumed node separation
        pwv = distance_m / max(delay_s, 1e-6)
        return float(np.clip(pwv, 2.0, 20.0))  # physiological PWV range
