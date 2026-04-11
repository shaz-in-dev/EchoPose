from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import numpy as np


TARGET_SUBCARRIERS = 64


@dataclass(frozen=True)
class CanonicalCSIFrame:
    source: str
    timestamp_us: int
    node_id: int
    amplitudes: List[float]
    phases: List[float]



def _resample(values: Iterable[float], target_len: int = TARGET_SUBCARRIERS) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    if arr.size == 0:
        return np.zeros((target_len,), dtype=np.float32)
    if arr.size == target_len:
        return arr

    src_x = np.linspace(0.0, 1.0, num=arr.size, dtype=np.float32)
    dst_x = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(dst_x, src_x, arr).astype(np.float32)



def normalize_esp32_frame(frame: Dict[str, Any]) -> CanonicalCSIFrame:
    amps = _resample(frame.get("amplitudes", []), TARGET_SUBCARRIERS)
    phases = _resample(frame.get("phases", [0.0] * len(amps)), TARGET_SUBCARRIERS)
    return CanonicalCSIFrame(
        source="esp32",
        timestamp_us=int(frame.get("timestamp_us", 0)),
        node_id=int(frame.get("node_id", 0)),
        amplitudes=[float(x) for x in amps],
        phases=[float(x) for x in phases],
    )



def normalize_intel5300_frame(frame: Dict[str, Any]) -> CanonicalCSIFrame:
    """Normalize Intel 5300 style frame into canonical 64-subcarrier format.

    Supported Intel 5300 payload styles:
    - amplitudes/phases arrays.
    - i/q arrays where amplitudes/phases are derived from complex values.
    """

    if "amplitudes" in frame:
        amps = np.asarray(frame.get("amplitudes", []), dtype=np.float32)
        phases = np.asarray(frame.get("phases", np.zeros_like(amps)), dtype=np.float32)
    else:
        i = np.asarray(frame.get("i", []), dtype=np.float32)
        q = np.asarray(frame.get("q", []), dtype=np.float32)
        if i.size != q.size:
            raise ValueError("intel5300 frame i/q length mismatch")
        c = i + 1j * q
        amps = np.abs(c).astype(np.float32)
        phases = np.angle(c).astype(np.float32)

    amps = _resample(amps, TARGET_SUBCARRIERS)
    phases = _resample(phases, TARGET_SUBCARRIERS)

    return CanonicalCSIFrame(
        source="intel5300",
        timestamp_us=int(frame.get("timestamp_us", 0)),
        node_id=int(frame.get("node_id", 0)),
        amplitudes=[float(x) for x in amps],
        phases=[float(x) for x in phases],
    )



def canonicalize_frame(frame: Dict[str, Any], hardware: str) -> CanonicalCSIFrame:
    hw = hardware.lower()
    if hw == "esp32":
        return normalize_esp32_frame(frame)
    if hw in {"intel5300", "intel_5300", "intel"}:
        return normalize_intel5300_frame(frame)
    raise ValueError(f"unsupported hardware: {hardware}")
