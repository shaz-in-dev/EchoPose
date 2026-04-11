# EchoPose Differentiation Strategy (No-Copy Path)

This document defines how EchoPose can be novel without copying other projects.

## 1. Positioning: Compete by Identity, Not by Star Count

Do not try to win by matching any other repository's raw metrics (stars, forks, crate count).
Win by solving deployment and reliability problems better.

EchoPose identity:
- Python-inference-first experimentation velocity.
- Operational deployment paths in one place (Compose + k8s + systemd).
- Tactical and situational analytics layer beyond skeleton-only output.
- Research modules that can graduate into production features.

## 2. What Not To Do

- Do not replicate repository structure just to look bigger.
- Do not copy ADRs, wording, UI style, or module naming.
- Do not claim medical-grade outputs without formal validation.
- Do not optimize for vanity metrics before reliability metrics.

## 3. Novelty Pillars for EchoPose

### EchoPose Naming Contract

All production-track module names should be EchoPose-native and capability-descriptive.

Examples:
- `signed_model_bundle` instead of competitor-style container names.
- `continual_personalization` instead of borrowed adaptation acronyms.
- `hardware_normalization` with explicit sensor profiles (ESP32, Intel 5300).

This keeps parity at the capability level while preserving a distinct architecture identity.

## Pillar A: Uncertainty-Aware RF Pose

Ship confidence and uncertainty as first-class outputs.

Deliverables:
- Per-joint uncertainty in `/ws/pose` payload.
- Domain-shift score endpoint (`/quality`).
- Auto-recalibration recommendation when drift is sustained.

Why this is novel:
- Most CSI demos output point estimates only.
- Confidence-aware rendering and alerting creates practical trust.

## Pillar B: Cross-Room Robustness as a Product Feature

Treat room transfer as a measurable product capability.

Deliverables:
- Benchmark suite: train Room A -> test Room B/C.
- Calibration profile versioning with hardware metadata.
- Drift-aware fallback mode when confidence drops.

Why this is novel:
- Real deployments fail on domain shift more than raw model architecture.

## Pillar C: Tactical RF Analytics Layer

Keep and deepen tactical modules as your differentiation axis.

Deliverables:
- Unified tactical event schema (anomaly, intent, anti-jam, crowd).
- Time-windowed incident timeline for operators.
- Replay with synchronized pose + tactical overlays.

Why this is novel:
- This is uncommon in CSI pose repos that focus only on keypoints.

## Pillar D: Operations-Grade Developer Experience

Make deployment and debugging dramatically easier than peers.

Deliverables:
- One-command local stack with health checks and seeded replay data.
- OpenTelemetry traces across aggregator -> inference -> UI.
- Golden-path runbooks for edge, on-prem, and cloud.

Why this is novel:
- Operational maturity is a moat when demos become deployments.

## 4. 30/60/90 Plan

## First 30 Days (Credibility)

- Remove or downgrade high-risk claims to "experimental" where needed.
- Add baseline CI for Python + Rust tests and container build.
- Publish a reproducible benchmark script and one public baseline report.

Exit criteria:
- Fresh clone can run smoke tests and start full stack.
- Benchmark report can be regenerated exactly from script.

## Day 31-60 (Differentiation)

- Productionize domain-shift monitor from research folder.
- Add uncertainty to stream payload and UI rendering.
- Add tactical event timeline and replay alignment.

Exit criteria:
- Drift detected in perturbed environment within defined threshold.
- UI visibly degrades confidence instead of hallucinating stable output.

## Day 61-90 (Defensible Novelty)

- Cross-room benchmark published with confidence metrics.
- SLO dashboard with p95/p99 latency and frame-drop counters.
- Release note with measured improvements and known limitations.

Exit criteria:
- Demonstrable improvement in transfer robustness.
- Operational KPIs visible and stable under load test.

## 5. Scoreboard That Matters

Track weekly:
- Reliability: p95/p99 latency, queue saturation, frame drop rate.
- Robustness: cross-room MPJPE delta, drift precision/recall.
- Trust: confidence-error calibration (ECE-like proxy).
- Ops: deployment success rate, mean time to recovery.

Avoid using stars/forks as primary success metrics during build phase.

## 6. Messaging Template

Use this narrative publicly:
- "EchoPose is an ops-first Wi-Fi sensing platform focused on robust deployment and uncertainty-aware tactical analytics."
- "We prioritize cross-environment reliability and confidence reporting over leaderboard-style demos."
- "Our roadmap is benchmark-driven: each claim is tied to a reproducible script and published metric."

## 7. Immediate Next 5 Tasks

1. Add `/quality` endpoint and uncertainty payload fields.
2. Add benchmark harness for cross-room transfer tests.
3. Add CI workflow and container build checks.
4. Add tactical event schema + replay index format.
5. Add one-page architecture and limitations doc for honest positioning.
