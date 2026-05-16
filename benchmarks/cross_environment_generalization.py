"""Cross-environment benchmark harness for EchoPose.

Expected manifest JSON format:
{
  "train": [{"features": "path.npy", "poses": "path.npy"}, ...],
  "test": {
    "room_b": [{"features": "...", "poses": "..."}],
    "room_c": [{"features": "...", "poses": "..."}]
  }
}
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _collect_unique_data_files(manifest: Dict[str, Any]) -> List[Path]:
    files: List[Path] = []
    for item in manifest["train"]:
        files.append(Path(item["features"]))
        files.append(Path(item["poses"]))
    for items in manifest["test"].values():
        for item in items:
            files.append(Path(item["features"]))
            files.append(Path(item["poses"]))
    return sorted(set(files), key=lambda p: str(p))


def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    # pred/gt: [N, P, J, 3]
    return float(np.mean(np.linalg.norm(pred - gt, axis=-1)))


def pck(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.1) -> float:
    d = np.linalg.norm(pred - gt, axis=-1)
    return float(np.mean(d < threshold))


def body_normalized_pck(
    pred: np.ndarray, gt: np.ndarray, threshold: float = 0.1
) -> float:
    """PCK where the threshold is scaled per sample by body height.

    Body height = distance from ankle midpoint (joints 15 & 16) to nose
    (joint 0) in ground-truth coordinates.  A prediction is "correct" for
    joint j in sample n when its error < threshold * body_height[n].

    Parameters
    ----------
    pred, gt : np.ndarray
        Shape ``(N, P, J, 3)`` — people P, joints J, xyz.
    threshold : float
        Fraction of body height used as the PCK threshold (default 0.1).
    """
    # Average across people to get (N, J, 3)
    pred_avg = pred.mean(axis=1)
    gt_avg = gt.mean(axis=1)
    errors = np.linalg.norm(pred_avg - gt_avg, axis=-1)   # (N, J)

    ankle_mid = (gt_avg[:, 15] + gt_avg[:, 16]) / 2.0    # (N, 3)
    head = gt_avg[:, 0]                                    # (N, 3)
    heights = np.linalg.norm(head - ankle_mid, axis=-1) + 1e-8  # (N,)

    thresholds = threshold * heights[:, None]              # (N, J)
    correct = errors < thresholds
    return float(correct.mean())


def load_pairs(items: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for item in items:
        xs.append(np.load(item["features"]))
        ys.append(np.load(item["poses"]))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _feature_depth(features: np.ndarray) -> np.ndarray:
    return np.mean(features, axis=(1, 2, 3), keepdims=False).astype(np.float64)


def _target_depth(poses_xyz: np.ndarray) -> np.ndarray:
    # Collapse [N, P, J, 3] -> [N] using mean z across people and joints.
    return np.mean(poses_xyz[..., 2], axis=(1, 2)).astype(np.float64)


def _fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    design = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _predict_depth(mode: str, f: np.ndarray, a: float = 0.0, b: float = 0.0) -> np.ndarray:
    if mode == "zero_z":
        return np.zeros_like(f)
    if mode == "mean_feature_z":
        return f
    if mode == "affine_feature_z":
        return a * f + b
    raise ValueError(f"Unknown baseline mode: {mode}")


def _cv_select_mode(train_features: np.ndarray, train_poses_xyz: np.ndarray) -> Tuple[str, float, float]:
    """Pick a simple depth baseline using train-only cross-validation."""
    f = _feature_depth(train_features)
    z = _target_depth(train_poses_xyz)
    n = f.shape[0]
    folds = min(5, max(2, n))

    candidates = ("zero_z", "mean_feature_z", "affine_feature_z")
    scores = {name: [] for name in candidates}

    for fold in range(folds):
        val_mask = (np.arange(n) % folds) == fold
        tr_mask = ~val_mask
        f_tr, z_tr = f[tr_mask], z[tr_mask]
        f_val, z_val = f[val_mask], z[val_mask]

        if f_tr.size == 0 or f_val.size == 0:
            continue

        a, b = _fit_affine(f_tr, z_tr)
        for name in candidates:
            z_hat = _predict_depth(name, f_val, a=a, b=b)
            rmse = float(np.sqrt(np.mean((z_hat - z_val) ** 2)))
            scores[name].append(rmse)

    mean_scores = {
        name: (float(np.mean(vals)) if vals else float("inf"))
        for name, vals in scores.items()
    }
    best_mode = min(mean_scores, key=mean_scores.get)
    a, b = _fit_affine(f, z)
    return best_mode, a, b


def baseline_predict(
    features: np.ndarray,
    mode: str,
    affine_a: float,
    affine_b: float,
    max_people: int = 3,
    num_joints: int = 17,
) -> np.ndarray:
    """Deterministic baseline predictor to validate benchmark plumbing."""
    n = features.shape[0]
    pred = np.zeros((n, max_people, num_joints, 3), dtype=np.float32)
    z_pred = _predict_depth(mode, _feature_depth(features), a=affine_a, b=affine_b)
    pred[..., 2] = z_pred[:, None, None]
    return pred


def run(manifest_path: Path, out_path: Path) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    x_train, y_train = load_pairs(manifest["train"])
    y_train_xyz = y_train[..., :3]
    baseline_mode, affine_a, affine_b = _cv_select_mode(x_train, y_train_xyz)

    room_metrics: Dict[str, Dict[str, float]] = {}
    for room, items in manifest["test"].items():
        x, y = load_pairs(items)
        pred = baseline_predict(
            x,
            mode=baseline_mode,
            affine_a=affine_a,
            affine_b=affine_b,
        )
        y3 = y[..., :3]
        room_metrics[room] = {
            "mpjpe": mpjpe(pred, y3),
            "pck@0.1": pck(pred, y3, 0.1),
            "body_pck@0.1": body_normalized_pck(pred, y3, 0.1),
            "samples": int(x.shape[0]),
        }

    unique_files = _collect_unique_data_files(manifest)
    evidence = {
        "algorithm": "sha256",
        "manifest_sha256": _sha256_file(manifest_path),
        "data_files_sha256": {str(p): _sha256_file(p) for p in unique_files},
        "baseline_id": f"{baseline_mode}-v2",
    }

    report: Dict[str, Any] = {
        "benchmark": "cross_environment_generalization",
        "protocol_version": "1.0",
        "rooms": room_metrics,
        "_evidence": evidence,
    }
    report["_report_sha256"] = _sha256_bytes(_canonical_json_bytes(report))

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-environment benchmark runner")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/baselines/cross_env/latest.json"))
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = run(args.manifest, args.out)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
