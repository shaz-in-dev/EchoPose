"""echopose_sdk.skeleton — COCO-format skeleton utilities.

Helpers for working with 17-keypoint COCO skeletons produced by the
EchoPose inference pipeline: bone-length computation, confidence
filtering, temporal smoothing, and body-height normalisation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── COCO-17 skeleton definition ───────────────────────────────────────────────

JOINT_NAMES: Tuple[str, ...] = (
    "nose",       # 0
    "left_eye",   # 1
    "right_eye",  # 2
    "left_ear",   # 3
    "right_ear",  # 4
    "left_shoulder",   # 5
    "right_shoulder",  # 6
    "left_elbow",      # 7
    "right_elbow",     # 8
    "left_wrist",      # 9
    "right_wrist",     # 10
    "left_hip",        # 11
    "right_hip",       # 12
    "left_knee",       # 13
    "right_knee",      # 14
    "left_ankle",      # 15
    "right_ankle",     # 16
)

COCO_BONES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (0, 2),                   # nose → eyes
    (1, 3), (2, 4),                   # eyes → ears
    (5, 6),                           # shoulder span
    (5, 7), (7, 9),                   # left arm
    (6, 8), (8, 10),                  # right arm
    (5, 11), (6, 12),                 # torso sides
    (11, 12),                         # hip span
    (11, 13), (13, 15),               # left leg
    (12, 14), (14, 16),               # right leg
)
"""17 COCO bone pairs (parent, child) as joint indices."""


# ── conversion helpers ────────────────────────────────────────────────────────

def keypoints_to_array(
    keypoints: Sequence[Dict],
    coords: Tuple[str, ...] = ("x", "y", "z"),
) -> np.ndarray:
    """Convert a list of keypoint dicts to a NumPy array.

    Parameters
    ----------
    keypoints:
        List of 17 dicts with at least the keys in ``coords`` and optionally
        ``"confidence"``.
    coords:
        Keys to extract, in order.  Default: ``("x", "y", "z")``.

    Returns
    -------
    np.ndarray
        Shape ``(17, len(coords))``.
    """
    return np.array(
        [[kp.get(c, 0.0) for c in coords] for kp in keypoints],
        dtype=np.float32,
    )


def array_to_keypoints(
    arr: np.ndarray,
    coords: Tuple[str, ...] = ("x", "y", "z"),
    confidences: Optional[Sequence[float]] = None,
) -> List[Dict]:
    """Convert a ``(17, D)`` NumPy array back to a list of keypoint dicts."""
    result = []
    for i, row in enumerate(arr):
        kp: Dict = {c: float(row[j]) for j, c in enumerate(coords)}
        if confidences is not None:
            kp["confidence"] = float(confidences[i])
        result.append(kp)
    return result


# ── measurements ─────────────────────────────────────────────────────────────

def bone_lengths(keypoints: np.ndarray) -> Dict[str, float]:
    """Compute Euclidean length of each COCO bone.

    Parameters
    ----------
    keypoints:
        Shape ``(17, 3)`` xyz array.

    Returns
    -------
    dict mapping ``"joint_a-joint_b"`` to length in the same units as the input.
    """
    lengths = {}
    for a, b in COCO_BONES:
        d = float(np.linalg.norm(keypoints[a] - keypoints[b]))
        lengths[f"{JOINT_NAMES[a]}-{JOINT_NAMES[b]}"] = d
    return lengths


def body_height(keypoints: np.ndarray) -> float:
    """Estimate body height as the distance from ankle mid-point to nose.

    Parameters
    ----------
    keypoints:
        Shape ``(17, 3)`` xyz array.

    Returns
    -------
    float — estimated body height in the same units as the input.
    """
    ankle_mid = (keypoints[15] + keypoints[16]) / 2.0
    head = keypoints[0]
    return float(np.linalg.norm(head - ankle_mid)) + 1e-8


def torso_height(keypoints: np.ndarray) -> float:
    """Spine length: mean-shoulder to mean-hip distance."""
    sh = (keypoints[5] + keypoints[6]) / 2.0
    hip = (keypoints[11] + keypoints[12]) / 2.0
    return float(np.linalg.norm(sh - hip)) + 1e-8


# ── filtering ─────────────────────────────────────────────────────────────────

def filter_by_confidence(
    keypoints: Sequence[Dict],
    threshold: float = 0.3,
    fill_with_mean: bool = True,
) -> List[Dict]:
    """Zero-out or fill low-confidence keypoints.

    Parameters
    ----------
    keypoints:
        List of 17 dicts with ``"confidence"`` key.
    threshold:
        Minimum confidence required to keep a keypoint.
    fill_with_mean:
        If ``True``, replace low-confidence joints with the mean position of
        high-confidence joints instead of zeroing them.

    Returns
    -------
    List[Dict] — filtered keypoints (new list, originals not mutated).
    """
    import copy

    filtered = copy.deepcopy(list(keypoints))
    high_conf = [
        kp for kp in filtered if kp.get("confidence", 1.0) >= threshold
    ]
    if fill_with_mean and high_conf:
        mx = np.mean([kp.get("x", 0) for kp in high_conf])
        my = np.mean([kp.get("y", 0) for kp in high_conf])
        mz = np.mean([kp.get("z", 0) for kp in high_conf])
    else:
        mx = my = mz = 0.0

    for kp in filtered:
        if kp.get("confidence", 1.0) < threshold:
            if fill_with_mean:
                kp["x"] = mx
                kp["y"] = my
                kp["z"] = mz
            else:
                kp["x"] = kp["y"] = kp["z"] = 0.0
            kp["confidence"] = 0.0
    return filtered


# ── normalisation ─────────────────────────────────────────────────────────────

def normalize_to_body_height(
    keypoints: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Centre and scale a skeleton so body height = 1.

    Centres at the mid-hip point and divides all coordinates by the estimated
    body height.  This is the standard normalisation used in pose metrics
    (e.g. body-normalised PCK).

    Parameters
    ----------
    keypoints:
        Shape ``(17, 3)`` xyz array.

    Returns
    -------
    (normalised_keypoints, scale_factor)
    """
    root = (keypoints[11] + keypoints[12]) / 2.0
    centred = keypoints - root
    scale = body_height(centred)
    return (centred / scale).astype(np.float32), scale


def align_root(keypoints: np.ndarray) -> np.ndarray:
    """Translate a skeleton so the mid-hip is at the origin."""
    root = (keypoints[11] + keypoints[12]) / 2.0
    return (keypoints - root).astype(np.float32)


# ── temporal smoothing ────────────────────────────────────────────────────────

def smooth_skeleton_sequence(
    sequence: np.ndarray,
    method: str = "ema",
    alpha: float = 0.3,
    window: int = 5,
) -> np.ndarray:
    """Smooth a sequence of skeletons over time.

    Parameters
    ----------
    sequence:
        Shape ``(T, 17, 3)`` — sequence of skeletons.
    method:
        ``"ema"``    — exponential moving average (low latency, default).
        ``"uniform"``— uniform sliding-window average.
        ``"gaussian"``— Gaussian-weighted sliding window.
    alpha:
        EMA decay factor (used only when ``method="ema"``).
    window:
        Window size for sliding-window methods.

    Returns
    -------
    np.ndarray
        Shape ``(T, 17, 3)`` — smoothed skeletons.
    """
    arr = np.asarray(sequence, dtype=np.float32)
    T = arr.shape[0]
    out = np.empty_like(arr)

    if method == "ema":
        out[0] = arr[0]
        for t in range(1, T):
            out[t] = alpha * arr[t] + (1 - alpha) * out[t - 1]
        return out

    if method in ("uniform", "gaussian"):
        half = window // 2
        if method == "gaussian":
            x = np.arange(window) - half
            weights = np.exp(-0.5 * (x / (window / 6)) ** 2)
            weights /= weights.sum()
        else:
            weights = np.ones(window) / window

        for t in range(T):
            lo = max(0, t - half)
            hi = min(T, t - half + window)
            w = weights[lo - (t - half): lo - (t - half) + (hi - lo)]
            w = w / w.sum()
            out[t] = (arr[lo:hi] * w[:, None, None]).sum(axis=0)
        return out

    raise ValueError(f"Unknown smoothing method: {method!r}")
