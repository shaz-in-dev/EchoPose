from __future__ import annotations

import hashlib
import json
import sys
import numpy as np
from pathlib import Path

import pytest


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_workspace_root()))

from benchmarks.cross_environment_generalization import run


def _canonical_json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), arr)


@pytest.fixture()
def sample_manifest(tmp_path: Path) -> Path:
    """Create minimal synthetic manifest + npy data files in tmp_path."""
    rng = np.random.default_rng(42)

    def _make_pair(name: str, n: int = 8):
        feat = rng.standard_normal((n, 3, 64, 16)).astype(np.float32)
        pose = rng.standard_normal((n, 3, 17, 4)).astype(np.float32)
        fp = tmp_path / f"{name}_feat.npy"
        pp = tmp_path / f"{name}_pose.npy"
        _write_npy(fp, feat)
        _write_npy(pp, pose)
        return {"features": str(fp), "poses": str(pp)}

    manifest = {
        "train": [_make_pair("train_a"), _make_pair("train_b")],
        "test": {
            "room_b": [_make_pair("room_b_0"), _make_pair("room_b_1")],
            "room_c": [_make_pair("room_c_0")],
        },
    }
    mpath = tmp_path / "manifest.json"
    mpath.write_text(json.dumps(manifest), encoding="utf-8")
    return mpath


def test_benchmark_report_has_evidence_and_valid_signature(sample_manifest: Path, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    report = run(sample_manifest, out)

    assert "_evidence" in report
    assert "rooms" in report
    assert "_report_sha256" in report

    expected_manifest_sha = hashlib.sha256(sample_manifest.read_bytes()).hexdigest()
    assert report["_evidence"]["manifest_sha256"] == expected_manifest_sha

    without_sig = dict(report)
    got_sig = without_sig.pop("_report_sha256")
    expected_sig = _sha256_bytes(_canonical_json_bytes(without_sig))
    assert got_sig == expected_sig

    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["_report_sha256"] == got_sig


def test_benchmark_rooms_contain_metrics(sample_manifest: Path, tmp_path: Path) -> None:
    report = run(sample_manifest, tmp_path / "r2.json")
    for room, metrics in report["rooms"].items():
        assert "mpjpe" in metrics
        assert "pck@0.1" in metrics
        assert metrics["samples"] > 0
        assert metrics["mpjpe"] >= 0.0
        assert 0.0 <= metrics["pck@0.1"] <= 1.0


def test_benchmark_evidence_contains_data_file_hashes(sample_manifest: Path, tmp_path: Path) -> None:
    report = run(sample_manifest, tmp_path / "r3.json")
    evidence = report["_evidence"]
    assert evidence["algorithm"] == "sha256"
    assert len(evidence["data_files_sha256"]) >= 2
    for path_str, sha in evidence["data_files_sha256"].items():
        assert len(sha) == 64  # sha256 hex digest
