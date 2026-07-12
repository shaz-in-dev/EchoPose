"""
inference/kinect/sync.py — Motion-spike cross-correlation for CSI/pose alignment.

The Kinect and the ESP32 aggregator run on independent clocks. This module
recovers the time offset between them by cross-correlating a strong motion event
(a clap, jump, or sharp arm swing) visible in both streams.

Algorithm:
  1. Compute a "motion energy" signal from the Kinect skeleton velocity.
  2. Compute a "Doppler energy" signal from the CSI amplitude variance.
  3. Cross-correlate the two signals to find the lag that maximises similarity.
  4. That lag is the offset: t_kinect = t_csi + offset.
  5. Resample one stream onto the other's grid for aligned windows.

Usage::
    correlator = SyncCorrelator()
    correlator.push_kinect(ts, velocity)   # call each Kinect frame
    correlator.push_csi(ts, doppler_energy) # call each CSI window
    offset, confidence = correlator.estimate_offset()
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


class SyncCorrelator:
    """Cross-correlation-based time-offset estimator for Kinect + CSI streams.

    Parameters
    ----------
    max_lag_s    : Maximum plausible offset to search, seconds.
    min_samples  : Minimum number of samples required in each buffer.
    """

    def __init__(self, max_lag_s: float = 5.0, min_samples: int = 30):
        self._max_lag = max_lag_s
        self._min_samples = min_samples
        self._kinect_ts:  List[float] = []
        self._kinect_val: List[float] = []
        self._csi_ts:     List[float] = []
        self._csi_val:    List[float] = []

    # ── Data ingestion ────────────────────────────────────────────────────────

    def push_kinect(self, timestamp_s: float, velocity: float) -> None:
        """Add a Kinect skeleton velocity sample (metres/second)."""
        self._kinect_ts.append(timestamp_s)
        self._kinect_val.append(float(velocity))

    def push_csi(self, timestamp_s: float, doppler_energy: float) -> None:
        """Add a CSI Doppler-energy sample (normalised amplitude variance)."""
        self._csi_ts.append(timestamp_s)
        self._csi_val.append(float(doppler_energy))

    def clear(self) -> None:
        self._kinect_ts.clear(); self._kinect_val.clear()
        self._csi_ts.clear();    self._csi_val.clear()

    # ── Offset estimation ─────────────────────────────────────────────────────

    def estimate_offset(self) -> Tuple[Optional[float], float]:
        """Estimate the time offset (seconds) between the two streams.

        Returns
        -------
        offset_s   : t_kinect ≈ t_csi + offset_s  (None if insufficient data)
        confidence : 0-1 float, pearson |r| of best-lag correlation
        """
        if (len(self._kinect_ts) < self._min_samples or
                len(self._csi_ts) < self._min_samples):
            return None, 0.0

        # Resample both onto a common 50 ms grid
        dt = 0.05
        t_start = max(self._kinect_ts[0], self._csi_ts[0])
        t_end   = min(self._kinect_ts[-1], self._csi_ts[-1])
        if t_end - t_start < 1.0:
            return None, 0.0

        grid = np.arange(t_start, t_end, dt)
        kin_rs = np.interp(grid, self._kinect_ts, self._kinect_val)
        csi_rs = np.interp(grid, self._csi_ts,    self._csi_val)

        # Normalise
        kin_rs = _normalise(kin_rs)
        csi_rs = _normalise(csi_rs)

        # Cross-correlation
        max_lag_samples = int(self._max_lag / dt)
        xcorr = np.correlate(kin_rs, csi_rs, mode="full")
        lags  = np.arange(-(len(csi_rs) - 1), len(kin_rs)) * dt

        # Restrict to ±max_lag
        mid = len(xcorr) // 2
        lo  = max(0, mid - max_lag_samples)
        hi  = min(len(xcorr), mid + max_lag_samples + 1)
        window_corr = xcorr[lo:hi]
        window_lags = lags[lo:hi]

        best_idx   = int(np.argmax(window_corr))
        best_lag   = float(window_lags[best_idx])
        confidence = float(window_corr[best_idx] / (len(grid) + 1e-9))
        confidence = min(abs(confidence), 1.0)

        return best_lag, round(confidence, 3)

    # ── Resampling ────────────────────────────────────────────────────────────

    @staticmethod
    def resample_onto_grid(
        src_ts:  np.ndarray,
        src_val: np.ndarray,
        grid_ts: np.ndarray,
        fill_value: float = 0.0,
    ) -> np.ndarray:
        """Interpolate src_val (sampled at src_ts) onto grid_ts.

        Points outside the source range are filled with fill_value.
        """
        out = np.full(len(grid_ts), fill_value, dtype=np.float32)
        mask = (grid_ts >= src_ts[0]) & (grid_ts <= src_ts[-1])
        out[mask] = np.interp(grid_ts[mask], src_ts, src_val).astype(np.float32)
        return out

    @staticmethod
    def apply_offset(timestamps: np.ndarray, offset_s: float) -> np.ndarray:
        """Shift timestamps by offset_s: t_aligned = t_original - offset_s."""
        return timestamps - offset_s


def _normalise(arr: np.ndarray) -> np.ndarray:
    mn, std = arr.mean(), arr.std()
    if std < 1e-9:
        return arr - mn
    return (arr - mn) / std
