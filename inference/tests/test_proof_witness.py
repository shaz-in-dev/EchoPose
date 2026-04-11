from __future__ import annotations

import sys
from pathlib import Path


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


sys.path.insert(0, str(_workspace_root()))

from v1.proof_system import (  # noqa: E402
    create_bundle_witness,
    create_manifest_witness,
    verify_bundle_witness,
    verify_manifest_witness,
)


def test_bundle_witness_roundtrip() -> None:
    bundle = {
        "frames": [
            {"node_id": 1, "amplitudes": [0.1] * 64},
            {"node_id": 2, "amplitudes": [0.2] * 64},
        ]
    }
    witness = create_bundle_witness(bundle)
    result = verify_bundle_witness(bundle, witness)
    assert result.passed


def test_bundle_witness_detects_tamper() -> None:
    bundle = {"frames": [{"node_id": 1, "amplitudes": [0.1] * 64}]}
    witness = create_bundle_witness(bundle)
    tampered = {"frames": [{"node_id": 1, "amplitudes": [0.2] * 64}]}
    result = verify_bundle_witness(tampered, witness)
    assert not result.passed


def test_manifest_witness_roundtrip() -> None:
    manifest_path = _workspace_root() / "data" / "baselines" / "cross_env" / "sample_manifest.json"
    witness = create_manifest_witness(manifest_path)
    result = verify_manifest_witness(manifest_path, witness)
    assert result.passed
