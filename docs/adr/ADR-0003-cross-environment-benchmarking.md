# ADR-0003: Cross-Environment Benchmark Protocol

- Status: Accepted
- Date: 2026-04-11

## Context

In-domain performance alone is not enough for real deployments. Cross-room transfer must be measured and tracked.

## Decision

Introduce a manifest-driven benchmark harness at `benchmarks/cross_environment_generalization.py` and store outputs in `data/baselines/cross_env/`.

## Consequences

Positive:
- Reproducible transfer metrics.
- Comparable baseline tracking between releases.

Negative:
- Requires careful manifest and dataset lifecycle management.
