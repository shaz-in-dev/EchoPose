"""
pipeline/occupancy.py — Presence detection & occupancy analytics (Feature 11)

Combines skeleton availability, CSI energy, and vital-frequency signatures
to determine binary room occupancy and person count.
"""

import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger("rf_inference.occupancy")


class OccupancyAnalyzer:
    """Detects whether the room is occupied and estimates person count."""

    def __init__(self, fs: float = 20.0, energy_threshold: float = 0.05, vital_threshold: float = 0.01):
        self.fs = fs
        self.energy_threshold = energy_threshold
        self.vital_threshold = vital_threshold
        self._baseline_energy: Optional[float] = None

    def calibrate_empty_room(self, csi_amplitudes: np.ndarray) -> None:
        """Record baseline CSI energy when the room is known to be empty."""
        self._baseline_energy = float(np.mean(csi_amplitudes ** 2))
        logger.info(f"Occupancy baseline calibrated: energy={self._baseline_energy:.6f}")

    def detect_presence(
        self,
        skeletons: list,
        csi_amplitudes: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Multi-method presence detection.

        1. Skeleton presence (highest confidence)
        2. CSI energy above calibrated baseline
        3. Vital-frequency content (breathing + heartbeat in 0.2–3 Hz)
        """
        # Method 1: skeleton-based
        if skeletons and len(skeletons) > 0:
            has_valid = any(
                any(kp.get("confidence", 0) > 0.3 for kp in skel)
                for skel in skeletons
                if isinstance(skel, list)
            )
            if has_valid:
                num_people = len(skeletons)
                return {
                    "occupied": True,
                    "num_people": num_people,
                    "method": "skeleton",
                    "confidence": 0.95,
                }

        # M7: guard against _baseline_energy being None before any CSI computation
        if self._baseline_energy is None and csi_amplitudes is None:
            return {"occupied": False, "num_people": 0, "method": "csi_unavailable", "confidence": 0.0}

        presence_csi = False
        presence_vitals = False

        if csi_amplitudes is not None and csi_amplitudes.size > 0:
            # Method 2: CSI energy
            energy = float(np.mean(csi_amplitudes ** 2))
            baseline = self._baseline_energy if self._baseline_energy is not None else self.energy_threshold
            presence_csi = energy > baseline * 1.3

            # Method 3: vital-frequency content
            if csi_amplitudes.ndim >= 2:
                mean_amp = np.mean(csi_amplitudes, axis=0)
            else:
                mean_amp = csi_amplitudes
            if len(mean_amp) >= 10:
                fft_vals = np.abs(np.fft.rfft(mean_amp))
                freqs = np.fft.rfftfreq(len(mean_amp), d=1.0 / self.fs)
                vital_mask = (freqs >= 0.2) & (freqs <= 3.0)
                if np.any(vital_mask):
                    presence_vitals = float(np.max(fft_vals[vital_mask])) > self.vital_threshold

        occupied = presence_csi or presence_vitals
        return {
            "occupied": occupied,
            "num_people": 1 if occupied else 0,
            "method": "csi",
            "confidence": 0.75 if occupied else 0.80,
        }
