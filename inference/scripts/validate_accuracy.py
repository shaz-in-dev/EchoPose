"""
scripts/validate_accuracy.py — Accuracy validation for PoseNetV2

Computes standard pose estimation metrics:
  - MPJPE (Mean Per-Joint Position Error): mm-level 3D joint error
  - PCK   (Percentage of Correct Keypoints): fraction within threshold
  - Per-joint breakdown: error by COCO-17 keypoint

Usage:
    python -m scripts.validate_accuracy
    python -m scripts.validate_accuracy --data path/to/test_set.npz
    python -m scripts.validate_accuracy --threshold 0.1
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.pose_net_v2 import PoseNetV2, FEATURE_SHAPE, NUM_KEYPOINTS, MAX_PEOPLE

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def load_model(ckpt_path: Path, device: torch.device) -> PoseNetV2:
    model = PoseNetV2().to(device)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()
    return model


def compute_mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Mean Per-Joint Position Error (Euclidean distance in 3D).

    Args:
        pred: [N, MAX_PEOPLE, 17, 4] predicted keypoints (x, y, z, conf)
        gt:   [N, MAX_PEOPLE, 17, 4] ground-truth keypoints

    Returns:
        per_joint_error: [17] mean error per keypoint
    """
    # Use only xyz, ignore confidence channel
    pred_xyz = pred[:, :, :, :3]
    gt_xyz = gt[:, :, :, :3]
    # Confidence-weighted: only count joints where GT confidence > 0.5
    gt_conf = gt[:, :, :, 3]
    valid = gt_conf > 0.5

    dist = np.sqrt(np.sum((pred_xyz - gt_xyz) ** 2, axis=-1))  # [N, MAX_PEOPLE, 17]
    per_joint = []
    for j in range(NUM_KEYPOINTS):
        mask = valid[:, :, j]  # [N, MAX_PEOPLE]
        if mask.sum() > 0:
            per_joint.append(dist[:, :, j][mask].mean())
        else:
            per_joint.append(0.0)
    return np.array(per_joint)


def compute_pck(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.1) -> tuple:
    """Percentage of Correct Keypoints.

    A keypoint is correct if its Euclidean distance from GT is below `threshold`.

    Returns:
        overall_pck: float (0-1)
        per_joint_pck: [17] per joint accuracy
    """
    pred_xyz = pred[:, :, :, :3]
    gt_xyz = gt[:, :, :, :3]
    gt_conf = gt[:, :, :, 3]
    valid = gt_conf > 0.5

    dist = np.sqrt(np.sum((pred_xyz - gt_xyz) ** 2, axis=-1))
    correct = dist < threshold

    per_joint_pck = []
    total_correct = 0
    total_valid = 0
    for j in range(NUM_KEYPOINTS):
        mask = valid[:, :, j]
        n_valid = mask.sum()
        if n_valid > 0:
            n_correct = correct[:, :, j][mask].sum()
            per_joint_pck.append(n_correct / n_valid)
            total_correct += n_correct
            total_valid += n_valid
        else:
            per_joint_pck.append(0.0)

    overall = total_correct / total_valid if total_valid > 0 else 0.0
    return float(overall), np.array(per_joint_pck)


@torch.no_grad()
def evaluate(model: PoseNetV2, features: torch.Tensor, targets: torch.Tensor,
             device: torch.device, threshold: float) -> dict:
    """Run full evaluation and return metrics dict."""
    batch_size = 32
    all_preds = []
    n = features.shape[0]

    for i in range(0, n, batch_size):
        batch = features[i:i + batch_size].to(device)
        preds = model(batch)  # [B, MAX_PEOPLE, 17, 4]
        all_preds.append(preds.cpu().numpy())

    pred_np = np.concatenate(all_preds, axis=0)
    gt_np = targets.numpy()

    per_joint_mpjpe = compute_mpjpe(pred_np, gt_np)
    overall_pck, per_joint_pck = compute_pck(pred_np, gt_np, threshold)

    # Confidence accuracy: how well does predicted confidence match GT
    pred_conf = pred_np[:, :, :, 3]
    gt_conf = gt_np[:, :, :, 3]
    conf_mae = np.mean(np.abs(pred_conf - gt_conf))

    results = {
        "overall": {
            "mpjpe_mean": float(per_joint_mpjpe.mean()),
            "mpjpe_std": float(per_joint_mpjpe.std()),
            "pck": float(overall_pck),
            "pck_threshold": threshold,
            "confidence_mae": float(conf_mae),
            "num_samples": int(n),
        },
        "per_joint": {},
    }
    for j, name in enumerate(COCO_KEYPOINTS):
        results["per_joint"][name] = {
            "mpjpe": float(per_joint_mpjpe[j]),
            "pck": float(per_joint_pck[j]),
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Validate PoseNetV2 accuracy")
    parser.add_argument("--data", type=str, default=None, help="Path to test .npz dataset")
    parser.add_argument("--threshold", type=float, default=0.1, help="PCK threshold (default: 0.1)")
    parser.add_argument("--synth-size", type=int, default=256, help="Synthetic test set size")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(__file__).parent.parent / "models" / "pose_net.pt"
    model = load_model(ckpt_path, device)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Device: {device}")

    if args.data and Path(args.data).exists():
        data = np.load(args.data)
        features = torch.from_numpy(data["features"]).float()
        targets = torch.from_numpy(data["poses"]).float()
        print(f"Loaded test set: {args.data} ({len(features)} samples)")
    else:
        if args.data:
            print(f"Warning: {args.data} not found, using synthetic test data")
        features = torch.randn(args.synth_size, *FEATURE_SHAPE)
        targets = torch.rand(args.synth_size, MAX_PEOPLE, NUM_KEYPOINTS, 4)
        print(f"Using synthetic test data ({args.synth_size} samples)")

    results = evaluate(model, features, targets, device, args.threshold)

    print(f"\n{'='*50}")
    print(f"  PoseNetV2 Validation Results")
    print(f"{'='*50}")
    print(f"  Samples:        {results['overall']['num_samples']}")
    print(f"  MPJPE (mean):   {results['overall']['mpjpe_mean']:.4f}")
    print(f"  MPJPE (std):    {results['overall']['mpjpe_std']:.4f}")
    print(f"  PCK@{args.threshold}:       {results['overall']['pck']:.4f} ({results['overall']['pck']*100:.1f}%)")
    print(f"  Conf MAE:       {results['overall']['confidence_mae']:.4f}")
    print(f"{'='*50}")
    print(f"\n  Per-Joint Breakdown:")
    print(f"  {'Joint':<18s} {'MPJPE':>8s}  {'PCK':>8s}")
    print(f"  {'-'*36}")
    for name in COCO_KEYPOINTS:
        j = results["per_joint"][name]
        print(f"  {name:<18s} {j['mpjpe']:8.4f}  {j['pck']:8.4f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
