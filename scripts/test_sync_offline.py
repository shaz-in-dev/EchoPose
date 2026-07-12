"""
scripts/test_sync_offline.py — Offline proof-of-correctness for SyncCorrelator.

Generates synthetic Kinect-velocity and CSI-Doppler signals with a known
injected time offset, runs the cross-correlation, and asserts recovery within
tolerance. No hardware required.

Usage:
  python scripts/test_sync_offline.py
  python scripts/test_sync_offline.py --offset 1.5 --noise 0.2
"""

from __future__ import annotations

import argparse
import sys
import math
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))
from kinect.sync import SyncCorrelator


def make_spike_signal(
    n_samples: int,
    fps: float,
    spike_at_s: float,
    spike_width_s: float = 0.15,
    noise_std: float = 0.1,
    rng=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian spike + noise signal. Returns (timestamps, values)."""
    if rng is None:
        rng = np.random.default_rng(0)
    ts  = np.arange(n_samples) / fps
    val = np.exp(-0.5 * ((ts - spike_at_s) / spike_width_s) ** 2)
    val += rng.normal(0, noise_std, n_samples)
    return ts, val.astype(np.float64)


def run_test(
    true_offset_s: float = 0.75,
    noise_std:     float = 0.1,
    tolerance_s:   float = 0.1,
    fps_kinect:    float = 30.0,
    fps_csi:       float = 20.0,
    duration_s:    float = 10.0,
    spike_at_s:    float = 4.0,
) -> bool:
    rng = np.random.default_rng(42)

    n_kinect = int(duration_s * fps_kinect)
    n_csi    = int(duration_s * fps_csi)

    # Kinect signal: spike at spike_at_s, timestamps start at 0
    kin_ts, kin_val = make_spike_signal(n_kinect, fps_kinect, spike_at_s, noise_std=noise_std, rng=rng)

    # CSI signal: same spike but shifted by true_offset_s
    # t_kinect = t_csi + offset  ⟹  t_csi = t_kinect - offset
    csi_spike = spike_at_s - true_offset_s
    csi_ts, csi_val = make_spike_signal(n_csi, fps_csi, csi_spike, noise_std=noise_std, rng=rng)
    # CSI clock starts at a different wall-time (simulate independent epoch)
    csi_ts = csi_ts  # relative timestamps; keep both starting at 0

    correlator = SyncCorrelator(max_lag_s=3.0, min_samples=20)
    for t, v in zip(kin_ts, kin_val):
        correlator.push_kinect(float(t), float(v))
    for t, v in zip(csi_ts, csi_val):
        correlator.push_csi(float(t), float(v))

    offset, confidence = correlator.estimate_offset()

    if offset is None:
        print(f"  FAIL — correlator returned None (insufficient data)")
        return False

    error = abs(offset - true_offset_s)
    ok    = error <= tolerance_s

    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}]  true={true_offset_s:.3f}s  recovered={offset:.3f}s  "
          f"error={error:.3f}s  conf={confidence:.3f}  "
          f"(tolerance={tolerance_s:.3f}s)")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--offset",    type=float, default=0.75, help="Injected offset (s)")
    p.add_argument("--noise",     type=float, default=0.1,  help="Signal noise std")
    p.add_argument("--tolerance", type=float, default=0.10, help="Recovery tolerance (s)")
    args = p.parse_args()

    print("EchoPose — Offline sync cross-correlation test")
    print("=" * 55)

    all_pass = True
    test_cases = [
        (0.25, 0.05),
        (0.75, 0.10),
        (1.50, 0.15),
        (2.00, 0.20),
        (args.offset, args.noise),
    ]
    for offset, noise in test_cases:
        ok = run_test(true_offset_s=offset, noise_std=noise, tolerance_s=args.tolerance)
        all_pass = all_pass and ok

    print("=" * 55)
    if all_pass:
        print("All tests passed.")
        sys.exit(0)
    else:
        print("Some tests FAILED — check SyncCorrelator implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
