"""echopose_sdk.metrics — Pose estimation evaluation metrics.

Standard and body-normalised evaluation metrics used in WiFi CSI pose
estimation research.  All functions operate on NumPy arrays so they
compose naturally with the rest of the SDK.

Metric glossary
---------------
MPJPE   Mean Per-Joint Position Error (lower is better).
PCK     Percentage of Correct Keypoints at threshold *t*.
PA-MPJPE  Procrustes-aligned MPJPE (removes global rotation/scale).
MPJVE   Mean Per-Joint Velocity Error (temporal smoothness proxy).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ── core metrics ──────────────────────────────────────────────────────────────

def mpjpe(
    pred: np.ndarray,
    gt: np.ndarray,
    per_joint: bool = False,
) -> float | np.ndarray:
    """Mean Per-Joint Position Error (MPJPE).

    Parameters
    ----------
    pred, gt:
        Arrays of shape ``(…, J, 3)`` — predicted and ground-truth joint
        positions in any consistent unit.
    per_joint:
        If ``True``, return an array of shape ``(J,)`` instead of a scalar.

    Returns
    -------
    float (or ndarray if ``per_joint=True``) — mean Euclidean joint error.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    errors = np.linalg.norm(pred - gt, axis=-1)
    if per_joint:
        return errors.mean(axis=tuple(range(errors.ndim - 1)))
    return float(errors.mean())


def pck(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.1,
    per_joint: bool = False,
) -> float | np.ndarray:
    """Percentage of Correct Keypoints at absolute threshold *t*.

    Parameters
    ----------
    pred, gt:
        Arrays of shape ``(…, J, 3)``.
    threshold:
        Absolute distance threshold in the same units as ``pred``/``gt``.
    per_joint:
        If ``True``, return per-joint accuracy array of shape ``(J,)``.

    Returns
    -------
    float (or ndarray) in range ``[0, 1]``.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    correct = np.linalg.norm(pred - gt, axis=-1) < threshold
    if per_joint:
        return correct.mean(axis=tuple(range(correct.ndim - 1)))
    return float(correct.mean())


def body_normalized_pck(
    pred: np.ndarray,
    gt: np.ndarray,
    threshold: float = 0.1,
    root_pairs: Tuple[int, int] = (11, 12),
    head_idx: int = 0,
) -> float:
    """PCK normalised by per-sample body height.

    Each sample's threshold is ``threshold × body_height`` where body height
    is the distance from the ankle midpoint to the nose in the ground-truth
    skeleton.  This is the standard PCK used in academic pose benchmarks.

    Parameters
    ----------
    pred, gt:
        Arrays of shape ``(N, …, J, 3)``.  The joint dimension must be last-1.
    threshold:
        Fraction of body height (default 0.1 = 10%).
    root_pairs:
        Indices of the two hip joints used to define the pelvis root.
    head_idx:
        Index of the head/nose joint.

    Returns
    -------
    float in range ``[0, 1]``.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    # Flatten extra person dimension if present: (N,P,J,3) → (N*P, J, 3)
    if pred.ndim == 4:
        N, P, J, _ = pred.shape
        pred = pred.reshape(N * P, J, 3)
        gt = gt.reshape(N * P, J, 3)

    errors = np.linalg.norm(pred - gt, axis=-1)  # (N, J)

    # Estimate body height from GT: ankle midpoint → head
    ankle_mid = (gt[:, 15] + gt[:, 16]) / 2.0    # (N, 3) — works for COCO
    head = gt[:, head_idx]                         # (N, 3)
    heights = np.linalg.norm(head - ankle_mid, axis=-1) + 1e-8  # (N,)

    thresholds = threshold * heights[:, None]      # (N, J)
    correct = errors < thresholds
    return float(correct.mean())


def pa_mpjpe(
    pred: np.ndarray,
    gt: np.ndarray,
) -> float:
    """Procrustes-Aligned MPJPE (PA-MPJPE).

    Aligns predicted skeleton to ground truth via Procrustes analysis
    (optimal rotation + scale + translation) before computing MPJPE.
    Measures structural pose accuracy independent of global position error.

    Parameters
    ----------
    pred, gt:
        Arrays of shape ``(N, J, 3)`` or broadcastable.

    Returns
    -------
    float — PA-MPJPE.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    if pred.ndim == 4:
        N, P, J, _ = pred.shape
        pred = pred.reshape(N * P, J, 3)
        gt = gt.reshape(N * P, J, 3)

    aligned = _batch_procrustes(pred, gt)
    return float(np.mean(np.linalg.norm(aligned - gt, axis=-1)))


def mpjve(
    pred_seq: np.ndarray,
    gt_seq: np.ndarray,
) -> float:
    """Mean Per-Joint Velocity Error.

    Measures temporal smoothness of predicted joint trajectories.

    Parameters
    ----------
    pred_seq, gt_seq:
        Arrays of shape ``(T, J, 3)`` — time-ordered sequences.

    Returns
    -------
    float — mean velocity error across time steps and joints.
    """
    pred_v = np.diff(np.asarray(pred_seq, dtype=np.float64), axis=0)
    gt_v = np.diff(np.asarray(gt_seq, dtype=np.float64), axis=0)
    return float(np.mean(np.linalg.norm(pred_v - gt_v, axis=-1)))


def confidence_mae(
    pred_conf: Sequence[float],
    gt_conf: Sequence[float],
) -> float:
    """Mean Absolute Error between predicted and ground-truth confidence scores.

    Parameters
    ----------
    pred_conf, gt_conf:
        Sequences of per-joint confidence values.

    Returns
    -------
    float — MAE.
    """
    p = np.asarray(pred_conf, dtype=np.float64).ravel()
    g = np.asarray(gt_conf, dtype=np.float64).ravel()
    return float(np.mean(np.abs(p - g)))


# ── per-joint reporting ───────────────────────────────────────────────────────

def per_joint_error_table(
    pred: np.ndarray,
    gt: np.ndarray,
    joint_names: Optional[Sequence[str]] = None,
) -> List[Dict[str, float | str]]:
    """Return a per-joint error breakdown as a list of dicts.

    Suitable for printing or serialising as JSON.

    Parameters
    ----------
    pred, gt:
        Arrays of shape ``(N, J, 3)`` or ``(N, P, J, 3)``.
    joint_names:
        Optional joint name list of length J.  Defaults to COCO-17 names.

    Returns
    -------
    List of dicts with keys ``"joint"``, ``"mpjpe"``, ``"pck_01"``.
    """
    from .skeleton import JOINT_NAMES

    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    if pred.ndim == 4:
        N, P, J, _ = pred.shape
        pred = pred.reshape(N * P, J, 3)
        gt = gt.reshape(N * P, J, 3)

    J = pred.shape[-2]
    names = list(joint_names) if joint_names else list(JOINT_NAMES[:J])
    errors = np.linalg.norm(pred - gt, axis=-1)  # (N, J)
    correct = errors < 0.1

    rows = []
    for j in range(J):
        rows.append({
            "joint": names[j] if j < len(names) else f"joint_{j}",
            "mpjpe": round(float(errors[:, j].mean()), 4),
            "pck_01": round(float(correct[:, j].mean()), 4),
        })
    return rows


def summary_report(
    pred: np.ndarray,
    gt: np.ndarray,
    body_pck_threshold: float = 0.1,
) -> Dict[str, float]:
    """Compute a full summary metric dict for a single experiment.

    Parameters
    ----------
    pred, gt:
        Arrays of shape ``(N, J, 3)`` or ``(N, P, J, 3)``.
    body_pck_threshold:
        PCK threshold as a fraction of body height for body-normalised PCK.

    Returns
    -------
    Dict with keys: ``mpjpe``, ``pck_01_abs``, ``pck_05_abs``,
    ``body_pck_01``, ``body_pck_05``, ``pa_mpjpe``.
    """
    return {
        "mpjpe": mpjpe(pred, gt),
        "pck_01_abs": pck(pred, gt, threshold=0.1),
        "pck_05_abs": pck(pred, gt, threshold=0.5),
        "body_pck_01": body_normalized_pck(pred, gt, threshold=0.1),
        "body_pck_05": body_normalized_pck(pred, gt, threshold=0.5),
        "pa_mpjpe": pa_mpjpe(pred, gt),
    }


# ── internal helpers ──────────────────────────────────────────────────────────

def _batch_procrustes(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Batch Procrustes alignment: returns ``pred`` aligned to ``gt``."""
    # Centre
    pred_c = pred - pred.mean(axis=1, keepdims=True)
    gt_c = gt - gt.mean(axis=1, keepdims=True)

    # Scale
    pred_n = np.sqrt(np.sum(pred_c ** 2, axis=(1, 2), keepdims=True)) + 1e-8
    gt_n = np.sqrt(np.sum(gt_c ** 2, axis=(1, 2), keepdims=True)) + 1e-8
    pred_c = pred_c / pred_n
    gt_c = gt_c / gt_n

    # Optimal rotation via SVD
    M = np.einsum("nji,njk->nik", gt_c, pred_c)  # (N, 3, 3)
    U, _, Vt = np.linalg.svd(M)
    # Ensure proper rotation (det = +1)
    d = np.linalg.det(U @ Vt)
    S = np.eye(3)[None].repeat(M.shape[0], axis=0)
    S[:, 2, 2] = d
    R = np.einsum("nij,njk,nlk->nil", U, S, Vt)

    aligned = np.einsum("nji,nkj->nki", R, pred_c)
    # Re-scale to GT scale and re-centre
    aligned = aligned * gt_n + gt.mean(axis=1, keepdims=True)
    return aligned
