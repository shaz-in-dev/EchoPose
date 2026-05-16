"""echopose_sdk.csi — WiFi CSI signal utilities.

Helpers for normalising, sanitising, and extracting features from raw
Channel State Information (CSI) amplitude arrays produced by EchoPose
ESP32-S3 nodes or compatible hardware.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np


# ── normalisation ──────────────────────────────────────────────────────────────

def normalize_subcarriers(
    amplitudes: np.ndarray,
    method: Literal["zscore", "minmax", "l2"] = "zscore",
    axis: int = -1,
    eps: float = 1e-8,
) -> np.ndarray:
    """Normalise CSI amplitude values across subcarriers.

    Parameters
    ----------
    amplitudes:
        Array of shape ``(…, n_subcarriers)`` or any shape where ``axis``
        contains the subcarrier dimension.
    method:
        ``"zscore"`` — zero-mean, unit variance per sample (default).
        ``"minmax"`` — scale to ``[0, 1]`` per sample.
        ``"l2"``     — L2-norm unit vector per sample.
    axis:
        Subcarrier axis.  Defaults to the last dimension.
    eps:
        Stability epsilon added to denominators.

    Returns
    -------
    np.ndarray
        Normalised amplitudes, same shape as input.
    """
    arr = np.asarray(amplitudes, dtype=np.float32)
    if method == "zscore":
        mean = arr.mean(axis=axis, keepdims=True)
        std = arr.std(axis=axis, keepdims=True) + eps
        return (arr - mean) / std
    if method == "minmax":
        mn = arr.min(axis=axis, keepdims=True)
        mx = arr.max(axis=axis, keepdims=True)
        return (arr - mn) / (mx - mn + eps)
    if method == "l2":
        norm = np.linalg.norm(arr, axis=axis, keepdims=True) + eps
        return arr / norm
    raise ValueError(f"Unknown normalisation method: {method!r}")


# ── sanitisation ───────────────────────────────────────────────────────────────

def sanitize_csi_bundle(
    bundle: Dict,
    expected_nodes: int = 3,
    expected_subcarriers: int = 64,
    fill_value: float = 0.0,
) -> Dict:
    """Fill missing / NaN amplitude values and clip outliers in a CSI bundle.

    Parameters
    ----------
    bundle:
        Dict with a ``"frames"`` list, each containing ``"amplitudes"`` and
        ``"node_id"``.
    expected_nodes:
        Number of nodes expected in the deployment.
    expected_subcarriers:
        Expected amplitude vector length (default 64).
    fill_value:
        Value used to pad missing subcarrier entries.

    Returns
    -------
    Dict
        A new bundle dict with sanitised amplitude arrays.  The original is
        not mutated.
    """
    import copy

    sanitized = copy.deepcopy(bundle)
    for frame in sanitized.get("frames", []):
        amps = frame.get("amplitudes", [])
        # Pad to expected length
        if len(amps) < expected_subcarriers:
            amps = amps + [fill_value] * (expected_subcarriers - len(amps))
        amps = amps[:expected_subcarriers]
        arr = np.array(amps, dtype=np.float32)
        # Replace NaN / Inf
        arr = np.where(np.isfinite(arr), arr, fill_value)
        # Clip 5-sigma outliers
        std = arr.std()
        mean = arr.mean()
        if std > 0:
            arr = np.clip(arr, mean - 5 * std, mean + 5 * std)
        frame["amplitudes"] = arr.tolist()
    return sanitized


# ── feature extraction ────────────────────────────────────────────────────────

def subcarrier_variance(amplitudes_sequence: np.ndarray) -> np.ndarray:
    """Compute per-subcarrier variance over a time window.

    Parameters
    ----------
    amplitudes_sequence:
        Shape ``(T, n_subcarriers)`` — a time-windowed sequence.

    Returns
    -------
    np.ndarray
        Shape ``(n_subcarriers,)`` — variance per subcarrier.
    """
    arr = np.asarray(amplitudes_sequence, dtype=np.float32)
    return arr.var(axis=0)


def extract_doppler_features(
    csi_window: np.ndarray,
    n_fft_bins: int = 16,
) -> np.ndarray:
    """Extract a compact Doppler feature vector from a CSI window.

    Combines time-domain statistics with low-frequency FFT energy to
    capture human micro-Doppler signatures.

    Parameters
    ----------
    csi_window:
        Shape ``(T, n_subcarriers)`` — windowed CSI amplitude sequence.
    n_fft_bins:
        Number of low-frequency FFT magnitude bins to include.

    Returns
    -------
    np.ndarray
        1-D feature vector of length ``3 * n_subcarriers + n_fft_bins``.
    """
    arr = np.asarray(csi_window, dtype=np.float64)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-8
    diff_energy = np.mean(np.diff(arr, axis=0) ** 2, axis=0)

    # FFT across time on mean-subtracted, averaged across subcarriers
    ts = arr.mean(axis=1) - arr.mean()
    fft_mag = np.abs(np.fft.rfft(ts))[:n_fft_bins]

    return np.concatenate([mean, std, diff_energy, fft_mag]).astype(np.float32)


def csi_correlation_matrix(
    amplitudes_sequence: np.ndarray,
) -> np.ndarray:
    """Compute the inter-subcarrier Pearson correlation matrix.

    Useful for detecting human presence (high correlation indicates
    reflections from a moving body).

    Parameters
    ----------
    amplitudes_sequence:
        Shape ``(T, n_subcarriers)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_subcarriers, n_subcarriers)`` — correlation matrix.
    """
    arr = np.asarray(amplitudes_sequence, dtype=np.float64)
    return np.corrcoef(arr.T)


def estimate_human_presence(
    amplitudes_sequence: np.ndarray,
    variance_threshold: float = 0.01,
) -> Dict[str, float]:
    """Heuristic presence score from CSI variance pattern.

    Parameters
    ----------
    amplitudes_sequence:
        Shape ``(T, n_subcarriers)`` — time-windowed CSI.
    variance_threshold:
        Subcarrier variance level considered "active" (human nearby).

    Returns
    -------
    dict with keys ``presence_score`` (0–1), ``active_subcarriers`` (int),
    ``mean_variance``.
    """
    arr = np.asarray(amplitudes_sequence, dtype=np.float32)
    var_per_sub = arr.var(axis=0)
    active = int(np.sum(var_per_sub > variance_threshold))
    n_sub = arr.shape[1]
    score = float(np.clip(active / n_sub, 0.0, 1.0))
    return {
        "presence_score": score,
        "active_subcarriers": active,
        "mean_variance": float(var_per_sub.mean()),
    }


# ── pilot-subcarrier interpolation ────────────────────────────────────────────

_ESP32_PILOT_INDICES: Tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 11, 25, 32, 39, 53, 59, 60, 61, 62, 63,
)
"""Pilot / guard / DC subcarrier indices for ESP32-S3 in 802.11n HT40 mode."""


def interpolate_pilot_subcarriers(
    amplitudes: np.ndarray,
    pilot_indices: Tuple[int, ...] = _ESP32_PILOT_INDICES,
) -> np.ndarray:
    """Replace pilot subcarriers with cubic-spline interpolated values.

    Pilot / guard subcarriers carry no channel information and should be
    interpolated from their neighbours before signal processing.

    Parameters
    ----------
    amplitudes:
        Shape ``(n_subcarriers,)`` — a single CSI frame.
    pilot_indices:
        Indices of pilot subcarriers to fill.  Defaults to ESP32-S3 HT40.

    Returns
    -------
    np.ndarray
        Shape ``(n_subcarriers,)`` with pilots replaced.
    """
    from scipy.interpolate import CubicSpline  # type: ignore

    arr = amplitudes.copy().astype(np.float64)
    n = len(arr)
    all_idx = np.arange(n)
    data_idx = np.array([i for i in all_idx if i not in set(pilot_indices)])
    cs = CubicSpline(data_idx, arr[data_idx])
    arr[list(pilot_indices)] = cs(list(pilot_indices))
    return arr.astype(np.float32)
