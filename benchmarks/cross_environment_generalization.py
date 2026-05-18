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


def load_pairs(items: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for item in items:
        xs.append(np.load(item["features"]))
        ys.append(np.load(item["poses"]))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def baseline_predict(features: np.ndarray, max_people: int = 3, num_joints: int = 17) -> np.ndarray:
    """Deterministic baseline predictor to validate benchmark plumbing."""
    n = features.shape[0]
    pred = np.zeros((n, max_people, num_joints, 3), dtype=np.float32)
    pred[..., 2] = np.mean(features, axis=(1, 2, 3), keepdims=False)[:, None, None]
    return pred


def run(manifest_path: Path, out_path: Path) -> Dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _, _ = load_pairs(manifest["train"])  # reserved for future model fit

    room_metrics: Dict[str, Dict[str, float]] = {}
    for room, items in manifest["test"].items():
        x, y = load_pairs(items)
        pred = baseline_predict(x)
        y3 = y[..., :3]
        room_metrics[room] = {
            "mpjpe": mpjpe(pred, y3),
            "pck@0.1": pck(pred, y3, 0.1),
            "samples": int(x.shape[0]),
        }

    unique_files = _collect_unique_data_files(manifest)
    evidence = {
        "algorithm": "sha256",
        "manifest_sha256": _sha256_file(manifest_path),
        "data_files_sha256": {str(p): _sha256_file(p) for p in unique_files},
        "baseline_id": "mean-z-v1",
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
