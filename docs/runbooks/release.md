# Release Runbook

## Preconditions

- CI is green.
- Benchmark baselines are updated when model or DSP logic changed.
- Release notes include limitations and known issues.
- Witness verification passes for benchmark manifest/report artifacts.

## Release Steps

1. Update version numbers for changed artifacts.
2. Tag release in Git (`vX.Y.Z`).
3. Trigger GitHub release.
4. Validate outputs:
   - Python package publish workflow
   - Docker image publish workflow
   - Rust crate publish workflow (if applicable)
   - Pages docs deploy workflow

## Benchmark Evidence And Witness Steps

1. Generate benchmark report with evidence block:

```bash
python benchmarks/cross_environment_generalization.py \
   --manifest data/baselines/cross_env/sample_manifest.json \
   --out data/baselines/cross_env/latest.json
```

2. Create manifest witness:

```bash
python -m v1.proof_system create-manifest-witness \
   --manifest data/baselines/cross_env/sample_manifest.json \
   --out data/baselines/cross_env/sample_manifest.witness.json
```

3. Verify manifest witness:

```bash
python -m v1.proof_system verify-manifest-witness \
   --manifest data/baselines/cross_env/sample_manifest.json \
   --witness data/baselines/cross_env/sample_manifest.witness.json
```

4. Verify report signature and evidence hashes before tagging release.

## Post-Release Validation

- Confirm container pull succeeds.
- Confirm package install succeeds.
- Confirm `/health` and `/analytics` endpoints return valid payloads.
- Confirm `data/baselines/cross_env/latest.json` has valid `_report_sha256` and matching `_evidence` hashes.
