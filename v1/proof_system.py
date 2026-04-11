"""Minimal proof-oriented checks for EchoPose data integrity.

These checks are intentionally simple and deterministic so they can be run in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass(frozen=True)
class ProofResult:
    passed: bool
    reason: str


def _canonical_json_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def create_bundle_witness(bundle: Dict[str, Any]) -> Dict[str, str]:
    """Create a deterministic SHA-256 witness for a bundle payload."""
    return {
        "algorithm": "sha256",
        "bundle_sha256": _sha256_bytes(_canonical_json_bytes(bundle)),
    }


def verify_bundle_witness(bundle: Dict[str, Any], witness: Dict[str, Any]) -> ProofResult:
    """Verify that a witness digest matches the canonicalized bundle payload."""
    if witness.get("algorithm") != "sha256":
        return ProofResult(False, "unsupported witness algorithm")

    expected = witness.get("bundle_sha256")
    if not isinstance(expected, str) or not expected:
        return ProofResult(False, "witness missing bundle_sha256")

    got = _sha256_bytes(_canonical_json_bytes(bundle))
    if got != expected:
        return ProofResult(False, "bundle digest mismatch")

    return ProofResult(True, "bundle witness verified")


def create_manifest_witness(manifest_path: Path) -> Dict[str, str]:
    """Create a SHA-256 witness for a benchmark manifest file."""
    manifest_bytes = manifest_path.read_bytes()
    return {
        "algorithm": "sha256",
        "manifest_sha256": _sha256_bytes(manifest_bytes),
    }


def verify_manifest_witness(manifest_path: Path, witness: Dict[str, Any]) -> ProofResult:
    """Verify a witness against raw manifest file bytes."""
    if witness.get("algorithm") != "sha256":
        return ProofResult(False, "unsupported witness algorithm")

    expected = witness.get("manifest_sha256")
    if not isinstance(expected, str) or not expected:
        return ProofResult(False, "witness missing manifest_sha256")

    got = _sha256_bytes(manifest_path.read_bytes())
    if got != expected:
        return ProofResult(False, "manifest digest mismatch")

    return ProofResult(True, "manifest witness verified")


def verify_bundle_shape(bundle: Dict[str, Any], expected_nodes: int = 3, expected_subcarriers: int = 64) -> ProofResult:
    frames: List[Dict[str, Any]] = bundle.get("frames", [])
    if len(frames) < 1:
        return ProofResult(False, "bundle has no frames")

    seen_nodes = set()
    for frame in frames:
        node_id = frame.get("node_id")
        amps = frame.get("amplitudes")
        if node_id is None or not isinstance(amps, list):
            return ProofResult(False, "frame missing required fields")
        if len(amps) != expected_subcarriers:
            return ProofResult(False, f"unexpected subcarrier length: {len(amps)}")
        seen_nodes.add(node_id)

    if len(seen_nodes) > expected_nodes:
        return ProofResult(False, "node count exceeds expected topology")

    return ProofResult(True, "bundle shape verified")


def verify_pose_output(payload: Dict[str, Any]) -> ProofResult:
    skeletons = payload.get("skeletons")
    if not isinstance(skeletons, list):
        return ProofResult(False, "skeletons field missing")
    return ProofResult(True, "pose payload structure verified")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EchoPose deterministic proof witnesses")
    sub = parser.add_subparsers(dest="command", required=True)

    cbw = sub.add_parser("create-bundle-witness", help="Create witness from bundle JSON")
    cbw.add_argument("--bundle", type=Path, required=True)
    cbw.add_argument("--out", type=Path, required=True)

    vbw = sub.add_parser("verify-bundle-witness", help="Verify bundle JSON against witness")
    vbw.add_argument("--bundle", type=Path, required=True)
    vbw.add_argument("--witness", type=Path, required=True)

    cmw = sub.add_parser("create-manifest-witness", help="Create witness from manifest file")
    cmw.add_argument("--manifest", type=Path, required=True)
    cmw.add_argument("--out", type=Path, required=True)

    vmw = sub.add_parser("verify-manifest-witness", help="Verify manifest file against witness")
    vmw.add_argument("--manifest", type=Path, required=True)
    vmw.add_argument("--witness", type=Path, required=True)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "create-bundle-witness":
        bundle = _read_json(args.bundle)
        witness = create_bundle_witness(bundle)
        _write_json(args.out, witness)
        print(json.dumps(witness, indent=2))
        return 0

    if args.command == "verify-bundle-witness":
        bundle = _read_json(args.bundle)
        witness = _read_json(args.witness)
        result = verify_bundle_witness(bundle, witness)
        print(json.dumps({"passed": result.passed, "reason": result.reason}))
        return 0 if result.passed else 1

    if args.command == "create-manifest-witness":
        witness = create_manifest_witness(args.manifest)
        _write_json(args.out, witness)
        print(json.dumps(witness, indent=2))
        return 0

    if args.command == "verify-manifest-witness":
        witness = _read_json(args.witness)
        result = verify_manifest_witness(args.manifest, witness)
        print(json.dumps({"passed": result.passed, "reason": result.reason}))
        return 0 if result.passed else 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
