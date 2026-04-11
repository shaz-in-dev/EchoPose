from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def summarize_confidence(confidences: Iterable[float]) -> Dict[str, float]:
    arr = np.asarray(list(confidences), dtype=np.float32)
    if arr.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }
