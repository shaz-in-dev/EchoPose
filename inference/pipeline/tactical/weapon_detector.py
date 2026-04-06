"""
pipeline/tactical/weapon_detector.py — Weapon detection (Feature 6)

Detects encumbered / armed targets by analysing gait load asymmetry,
torso CSI reflections (body armour / metal), and skeletal arm-swing
patterns.  Classifies: unarmed, handgun, rifle, heavy-load.
"""

import numpy as np
from scipy.signal import welch
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("rf_inference.tactical.weapon_detector")

SAMPLE_RATE = 20.0

_L_SHOULDER, _R_SHOULDER = 5, 6
_L_ELBOW, _R_ELBOW = 7, 8
_L_WRIST, _R_WRIST = 9, 10
_L_HIP, _R_HIP = 11, 12

_ASYM_RIFLE = 0.40
_ASYM_HANDGUN = 0.20
_ARMOR_REFLECTION_SHIFT = 0.15


class WeaponDetectionSystem:
    """Detect armed status from skeleton gait + CSI torso reflections."""

    def __init__(self, fs: float = SAMPLE_RATE):
        self.fs = fs
        self._skel_buf: list[np.ndarray] = []
        self._csi_buf: list[np.ndarray] = []
        self._max = int(fs * 5)

    def push(self, skeleton: List[Dict],
             csi_amplitudes: Optional[np.ndarray] = None) -> None:
        arr = np.array([[kp.get("x", 0), kp.get("y", 0), kp.get("z", 0)]
                        for kp in skeleton], dtype=np.float64)
        self._skel_buf.append(arr)
        if csi_amplitudes is not None:
            self._csi_buf.append(np.asarray(csi_amplitudes, dtype=np.float64))
        for b in (self._skel_buf, self._csi_buf):
            if len(b) > self._max:
                del b[: len(b) - self._max]

    def detect(self) -> Dict:
        """Analyse latest buffer for armed-status indicators."""
        if len(self._skel_buf) < int(self.fs * 2):
            return {"armed": False, "status": "buffering"}

        poses = np.array(self._skel_buf)
        gait_result = self._gait_asymmetry_analysis(poses)
        torso_result = self._torso_reflection(poses)
        arm_result = self._arm_swing_analysis(poses)

        weapon, conf = self._fuse(gait_result, torso_result, arm_result)

        return {
            "armed": weapon != "UNARMED",
            "weapon_type": weapon,
            "confidence": round(conf, 2),
            "gait_asymmetry": round(gait_result["asymmetry"], 3),
            "arm_swing_ratio": round(arm_result["swing_ratio"], 3),
            "body_armor_likelihood": round(torso_result["armor_prob"], 2),
            "threat_level": "HIGH" if weapon in ("RIFLE", "HEAVY_LOAD") else
                            "MEDIUM" if weapon == "HANDGUN" else "LOW",
        }

    # ── analysis modules ──────────────────────────────────────────

    def _gait_asymmetry_analysis(self, poses: np.ndarray) -> Dict:
        """Left-right hip vertical variance ratio → load indicator."""
        lh = poses[:, _L_HIP, 1]
        rh = poses[:, _R_HIP, 1]
        lv = float(np.var(lh))
        rv = float(np.var(rh))
        total = lv + rv + 1e-12
        asym = abs(lv - rv) / total
        return {"asymmetry": asym, "loaded_side": "left" if lv > rv else "right"}

    def _torso_reflection(self, poses: np.ndarray) -> Dict:
        """CSI torso-zone reflection analysis for armour / metal."""
        if not self._csi_buf or len(self._csi_buf) < 20:
            return {"armor_prob": 0.0}

        csi = np.array(self._csi_buf[-60:])
        n_sub = csi.shape[1]
        torso_zone = csi[:, n_sub // 3: 2 * n_sub // 3]

        mean_refl = float(np.mean(torso_zone))
        var_refl = float(np.var(torso_zone))

        # Metallic objects create high-variance, high-amplitude reflection
        armor_prob = float(np.clip(
            (var_refl - 0.01) / 0.05 * 0.5 + (mean_refl - 0.3) / 0.5 * 0.5,
            0.0, 1.0,
        ))
        return {"armor_prob": armor_prob, "mean_reflection": mean_refl}

    def _arm_swing_analysis(self, poses: np.ndarray) -> Dict:
        """Arm swing suppression indicates holding an object."""
        lw_y = poses[:, _L_WRIST, 1]
        rw_y = poses[:, _R_WRIST, 1]
        l_swing = float(np.std(lw_y))
        r_swing = float(np.std(rw_y))
        total = l_swing + r_swing + 1e-12
        ratio = min(l_swing, r_swing) / (max(l_swing, r_swing) + 1e-12)

        suppressed_side = "left" if l_swing < r_swing else "right"
        return {"swing_ratio": ratio, "suppressed_side": suppressed_side,
                "l_swing": l_swing, "r_swing": r_swing}

    def _fuse(self, gait: Dict, torso: Dict, arm: Dict) -> tuple:
        """Fuse sub-detectors into final weapon classification."""
        asym = gait["asymmetry"]
        ratio = arm["swing_ratio"]
        armor = torso["armor_prob"]

        if asym > _ASYM_RIFLE and ratio < 0.3:
            return ("RIFLE", min(0.6 + asym, 0.92))
        if asym > _ASYM_HANDGUN and ratio < 0.5:
            return ("HANDGUN", min(0.5 + asym, 0.85))
        if armor > 0.6:
            return ("HEAVY_LOAD", min(0.5 + armor * 0.3, 0.80))
        return ("UNARMED", max(0.85 - asym, 0.60))
