from __future__ import annotations

import hashlib
import json
from pathlib import Path

from benchmarks.cross_environment_generalization import run


def _canonical_json_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_benchmark_report_has_evidence_and_valid_signature(tmp_path: Path) -> None:
    manifest = Path("data/baselines/cross_env/sample_manifest.json")
    out = tmp_path / "cross_env_report.json"

    report = run(manifest, out)

    assert "_evidence" in report
    assert "rooms" in report
    assert "_report_sha256" in report

    expected_manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert report["_evidence"]["manifest_sha256"] == expected_manifest_sha

    without_sig = dict(report)
    got_sig = without_sig.pop("_report_sha256")
    expected_sig = _sha256_bytes(_canonical_json_bytes(without_sig))
    assert got_sig == expected_sig

    on_disk = json.loads(out.read_text(encoding="utf-8"))
    assert on_disk["_report_sha256"] == got_sig
