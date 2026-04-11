# ADR-0010: Comprehensive Test Strategy

- **Status:** Accepted
- **Date:** 2026-03-16
- **Decision Makers:** Core team

## Context

As EchoPose grew to 49+ Python modules across 9 directories, test coverage
became inconsistent.  Some modules had thorough tests while 65% had none.
Lack of tests slowed refactoring and allowed regressions.

## Decision

We adopt a **layered testing strategy**:

| Layer                 | Scope                               | Count Target |
|-----------------------|-------------------------------------|-------------|
| Unit tests            | Individual functions and classes     | 200+        |
| Integration tests     | Multi-module pipelines              | 20+         |
| Benchmark regression  | Performance/accuracy bounds          | 5+          |
| Proof-witness checks  | SHA-256 integrity in CI             | Per-release |

Test organisation:
- All tests live in `inference/tests/test_*.py`.
- pytest with `pythonpath = inference` (configured in `pytest.ini`).
- CI runs `pytest` on every push; benchmark evidence is signed and verified.

Module coverage prioritisation:
1. **Core pipeline** (pose, filter, emotion, occupancy) — highest impact.
2. **Tactical modules** (anti-jamming, fusion, tracking) — safety-critical.
3. **Research modules** (domain adaptation, frequency transfer) — prevent regressions.
4. **Infrastructure** (logger, metrics, security) — operational reliability.

## Consequences

- **Positive:** 255+ tests passing; every public module has coverage.
- **Positive:** CI catches regressions before merge.
- **Negative:** Test maintenance cost scales with module count.
- **Negative:** Some modules (GPU inference, OTA) need hardware-in-the-loop testing
  that can't run in CI.

## Alternatives Considered

| Alternative                | Why rejected                               |
|----------------------------|--------------------------------------------|
| Manual testing only        | Doesn't scale; regressions slip through    |
| Test per-PR only           | Misses cross-module interactions           |
| Property-based testing     | Planned for v2; hypothesis integration     |
