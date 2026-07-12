"""
scripts/train_with_splits.py — Training harness with honest evaluation splits.

Supports:
  - Leave-one-room-out (LORO): train on all rooms except one, test on held-out room
  - Leave-one-subject-out (LOSO): train on all subjects except one, test on held-out
  - Random split (baseline, intentionally NOT used for reported metrics)

The golden rule: test sessions must never appear in the training set.
Random splits violate this guarantee when sessions share rooms/subjects.
Always use LORO or LOSO for published numbers.

Usage:
  # Build a LORO split manifest from a directory of .npz session files:
  python scripts/train_with_splits.py --data data/sessions --split loro

  # Run LORO evaluation only (no training):
  python scripts/train_with_splits.py --data data/sessions --split loro --eval-only
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_session(path: Path) -> Dict[str, Any]:
    """Load a single session .npz file. Returns features, poses, metadata."""
    raw = np.load(path, allow_pickle=True)
    meta_str = str(raw.get("metadata", np.array("{}")))
    try:
        meta = json.loads(meta_str)
    except Exception:
        meta = {}
    # Derive room/subject from filename if metadata is absent
    stem = path.stem  # e.g. living_room__alice__20260101_120000
    parts = stem.split("__")
    meta.setdefault("room",    parts[0] if len(parts) > 0 else "unknown")
    meta.setdefault("subject", parts[1] if len(parts) > 1 else "unknown")
    return {
        "path":     path,
        "features": raw["features"],  # (N, 3, 64, 16)
        "poses":    raw["poses"],      # (N, 17, 4)
        "room":     meta["room"],
        "subject":  meta["subject"],
        "meta":     meta,
    }


def load_sessions(data_dir: Path) -> List[Dict]:
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        print(f"No .npz files found in {data_dir}")
        return []
    sessions = []
    for f in files:
        try:
            sessions.append(load_session(f))
        except Exception as exc:
            print(f"  Skipping {f.name}: {exc}")
    return sessions


# ── Split builders ────────────────────────────────────────────────────────────

def leave_one_room_out(sessions: List[Dict]) -> List[Tuple[str, List[Dict], List[Dict]]]:
    """Return list of (held_out_room, train_sessions, test_sessions)."""
    rooms = sorted({s["room"] for s in sessions})
    folds = []
    for room in rooms:
        train = [s for s in sessions if s["room"] != room]
        test  = [s for s in sessions if s["room"] == room]
        if train and test:
            folds.append((room, train, test))
    return folds


def leave_one_subject_out(sessions: List[Dict]) -> List[Tuple[str, List[Dict], List[Dict]]]:
    """Return list of (held_out_subject, train_sessions, test_sessions)."""
    subjects = sorted({s["subject"] for s in sessions})
    folds = []
    for subj in subjects:
        train = [s for s in sessions if s["subject"] != subj]
        test  = [s for s in sessions if s["subject"] == subj]
        if train and test:
            folds.append((subj, train, test))
    return folds


def concat_sessions(sessions: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate features and poses from multiple sessions."""
    features = np.concatenate([s["features"] for s in sessions], axis=0)
    poses    = np.concatenate([s["poses"]    for s in sessions], axis=0)
    return features, poses


# ── Metrics ───────────────────────────────────────────────────────────────────

def mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-joint position error (MPJPE) in same units as poses (metres).

    pred, gt: shape (N, 17, 3) — x,y,z only.
    """
    return float(np.mean(np.linalg.norm(pred[..., :3] - gt[..., :3], axis=-1)))


def pck_at_threshold(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.1) -> float:
    """Percentage of correct keypoints within threshold metres."""
    d = np.linalg.norm(pred[..., :3] - gt[..., :3], axis=-1)
    return float(np.mean(d < threshold)) * 100.0


def per_joint_mpjpe(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """Per-joint MPJPE dict keyed by COCO joint name."""
    COCO_JOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    errors = np.linalg.norm(pred[..., :3] - gt[..., :3], axis=-1)  # (N, 17)
    mean_per_joint = errors.mean(axis=0)  # (17,)
    return {name: round(float(err), 4) for name, err in zip(COCO_JOINTS, mean_per_joint)}


def validate_metrics_on_known_input():
    """Self-test: validate metric code returns expected values on synthetic data."""
    print("Validating metric correctness on synthetic data...")

    N, J = 100, 17
    rng  = np.random.default_rng(0)

    # Perfect prediction → MPJPE = 0
    gt   = rng.normal(0, 1, (N, J, 3)).astype(np.float32)
    pred = gt.copy()
    assert mpjpe(pred, gt) < 1e-5, "MPJPE(perfect) must be 0"

    # Constant offset of 0.1 m on every joint → MPJPE = 0.1 (exactly, along one axis)
    pred2 = gt + np.array([0.1, 0.0, 0.0])
    err   = mpjpe(pred2, gt)
    assert abs(err - 0.1) < 1e-4, f"MPJPE constant-offset must be 0.1, got {err}"

    # PCK@0.1 with offset 0.05 → all within threshold → 100%
    pred3 = gt + np.array([0.05, 0.0, 0.0])
    p     = pck_at_threshold(pred3, gt, 0.1)
    assert abs(p - 100.0) < 1e-4, f"PCK@0.1 must be 100%, got {p}"

    # PCK@0.1 with offset 0.15 → none within threshold → 0%
    pred4 = gt + np.array([0.15, 0.0, 0.0])
    p2    = pck_at_threshold(pred4, gt, 0.1)
    assert abs(p2) < 1e-4, f"PCK@0.1 must be 0%, got {p2}"

    print("  All metric validations passed.")


# ── Real-model training/prediction ─────────────────────────────────────────────
# Lazily imports torch so the zero-baseline path (and any environment without
# torch installed) keeps working without it.

def train_fold_model(x_train: np.ndarray, y_train: np.ndarray,
                      epochs: int, lr: float, batch_size: int, device: str):
    """Train a fresh PoseNetV2 on this fold's training sessions only.

    Trained per-fold (never on data that includes the held-out room/subject) to
    keep the LORO/LOSO guarantee honest — this is not a single globally-reused
    checkpoint.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from pipeline.pose_net_v2 import PoseNetV2, MAX_PEOPLE

    dev = torch.device(device)
    model = PoseNetV2().to(dev)

    # Ground-truth sessions store a single tracked person (N, 17, 4); pad to the
    # model's MAX_PEOPLE output slots so shapes line up, then train only the
    # first person's slot the same way it is read back out at inference time.
    x_t = torch.from_numpy(x_train).float()
    y_t = torch.from_numpy(y_train).float()
    y_padded = torch.zeros(len(y_t), MAX_PEOPLE, *y_t.shape[1:])
    y_padded[:, 0] = y_t

    loader = DataLoader(TensorDataset(x_t, y_padded), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(dev), yb.to(dev)
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def predict_with_model(model, x_test: np.ndarray, device: str) -> np.ndarray:
    """Run inference and return the first person's (N, 17, 3) xyz predictions."""
    import torch
    dev = torch.device(device)
    with torch.no_grad():
        x_t = torch.from_numpy(x_test).float().to(dev)
        out = model(x_t)  # [N, MAX_PEOPLE, 17, 4]
    return out[:, 0, :, :3].cpu().numpy()


# ── Evaluation loop ───────────────────────────────────────────────────────────

def evaluate_fold(
    train_sessions: List[Dict],
    test_sessions:  List[Dict],
    predictor=None,
    train_real_model: bool = False,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Evaluate one fold.

    predictor=None and train_real_model=False -> zero baseline (predicts the
    mean training pose; the honest "did we beat guessing the average?" floor).
    train_real_model=True -> trains a fresh PoseNetV2 on this fold's train
    sessions only, and evaluates it on the held-out test sessions.
    predictor=<callable> -> use a caller-supplied predictor instead.
    """
    x_train, y_train = concat_sessions(train_sessions)
    x_test,  y_test  = concat_sessions(test_sessions)

    if train_real_model:
        model = train_fold_model(x_train, y_train, epochs, lr, batch_size, device)
        pred = predict_with_model(model, x_test, device)
        predictor_label = "posenet_v2_per_fold"
    elif predictor is not None:
        pred = predictor(x_test)
        predictor_label = "custom"
    else:
        # Zero baseline: predict the mean y-position from training set
        mean_pose = y_train.mean(axis=0)[:, :3]
        pred = np.broadcast_to(mean_pose, (len(x_test), 17, 3)).copy()
        predictor_label = "zero_mean_baseline"

    return {
        "n_train":         len(x_train),
        "n_test":          len(x_test),
        "mpjpe_m":         round(mpjpe(pred, y_test), 4),
        "mpjpe_cm":        round(mpjpe(pred, y_test) * 100, 2),
        "pck_at_10cm_%":   round(pck_at_threshold(pred, y_test, 0.10), 2),
        "pck_at_15cm_%":   round(pck_at_threshold(pred, y_test, 0.15), 2),
        "per_joint_mpjpe": per_joint_mpjpe(pred, y_test),
        "predictor":       predictor_label,
    }


def run_evaluation(data_dir: Path, split: str, out_dir: Path,
                    train_real_model: bool = True,
                    epochs: int = 30, lr: float = 1e-3,
                    batch_size: int = 16, device: str = "cpu") -> None:
    sessions = load_sessions(data_dir)
    if not sessions:
        print("No sessions loaded. Collect data first with session_runner.py")
        return

    print(f"\nLoaded {len(sessions)} sessions from {data_dir}")
    rooms    = sorted({s["room"] for s in sessions})
    subjects = sorted({s["subject"] for s in sessions})
    print(f"  Rooms:    {rooms}")
    print(f"  Subjects: {subjects}")

    if split == "loro":
        folds = leave_one_room_out(sessions)
        split_label = "leave-one-room-out"
    elif split == "loso":
        folds = leave_one_subject_out(sessions)
        split_label = "leave-one-subject-out"
    else:
        raise ValueError(f"Unknown split: {split}")

    if not folds:
        print(f"Cannot build {split} folds — need at least 2 rooms or 2 subjects.")
        return

    mode = "training a fresh PoseNetV2 per fold" if train_real_model else "zero baseline"
    print(f"\nRunning {split_label} ({len(folds)} folds) — {mode}...")
    results = {}
    for held_out, train_s, test_s in folds:
        print(f"  Fold: held-out={held_out}  train={len(train_s)}  test={len(test_s)} sessions")
        results[held_out] = evaluate_fold(
            train_s, test_s,
            train_real_model=train_real_model,
            epochs=epochs, lr=lr, batch_size=batch_size, device=device,
        )

    # Summary
    all_mpjpe = [v["mpjpe_cm"] for v in results.values()]
    summary = {
        "split":           split_label,
        "n_folds":         len(folds),
        "mean_mpjpe_cm":   round(float(np.mean(all_mpjpe)), 2),
        "std_mpjpe_cm":    round(float(np.std(all_mpjpe)), 2),
        "min_mpjpe_cm":    round(float(np.min(all_mpjpe)), 2),
        "max_mpjpe_cm":    round(float(np.max(all_mpjpe)), 2),
        "per_fold":        results,
        "status":          ("posenet_v2_per_fold — trained fresh per fold, no leakage"
                             if train_real_model else
                             "zero_baseline — pass no --eval-only to train a real model"),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"results_{split}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults written to {out_path}")
    print(f"Mean MPJPE ({split_label}): {summary['mean_mpjpe_cm']:.2f} cm "
          f"± {summary['std_mpjpe_cm']:.2f} cm")


def main():
    p = argparse.ArgumentParser(description="EchoPose honest training-split evaluator")
    p.add_argument("--data",       type=Path, default=Path("data/sessions"),
                   help="Directory of .npz session files")
    p.add_argument("--split",      choices=["loro", "loso"], default="loro",
                   help="Split strategy: loro=leave-one-room-out, loso=leave-one-subject-out")
    p.add_argument("--out",        type=Path, default=Path("data/results"),
                   help="Output directory for results JSON")
    p.add_argument("--eval-only",  action="store_true",
                   help="Skip training, evaluate the zero-mean baseline only")
    p.add_argument("--epochs",     type=int, default=30, help="Epochs per fold (ignored with --eval-only)")
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device",     type=str, default="cpu")
    p.add_argument("--validate-metrics", action="store_true",
                   help="Run metric correctness self-test and exit")
    args = p.parse_args()

    if args.validate_metrics:
        validate_metrics_on_known_input()
        return

    run_evaluation(
        args.data, args.split, args.out,
        train_real_model=not args.eval_only,
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, device=args.device,
    )


if __name__ == "__main__":
    main()
