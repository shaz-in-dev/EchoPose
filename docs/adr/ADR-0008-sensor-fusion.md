# ADR-0008: Tactical Multi-Domain Sensor Fusion

- **Status:** Accepted
- **Date:** 2026-03-16
- **Decision Makers:** Core team

## Context

WiFi CSI alone has limitations: it struggles in metallic environments, is
range-limited (~15 m), and cannot classify targets beyond "person present."
Military and first-responder deployments require fusing WiFi with radar,
thermal, acoustic, and other feeds to produce a reliable Common Operating
Picture (COP).

## Decision

We implement `MultiDomainFusion` with:

1. **Modality-weighted ingestion** — each sensor type has a pre-tuned weight
   reflecting its typical accuracy (wifi_csi=0.35, radar=0.25, thermal=0.20,
   acoustic=0.10, visual/seismic=0.05 each).
2. **Nearest-neighbour track association** — new detections within 2 m of an
   existing track are fused; otherwise a new track is created.
3. **Confidence accumulation** — multi-source tracks accumulate confidence,
   capped at 0.99.
4. **Stale-track pruning** — tracks not updated within 10 s are dropped.

## Consequences

- **Positive:** Single API for all sensor types; easy to add new modalities.
- **Positive:** Cross-validated tracks are far more reliable than single-source.
- **Negative:** Simple nearest-neighbour association can fail in dense crowds.
- **Negative:** Modality weights are hand-tuned (future: learn from data).

## Alternatives Considered

| Alternative               | Why rejected                                 |
|---------------------------|----------------------------------------------|
| Kalman filter per track   | Good but adds complexity; planned for v2     |
| JPDA (joint probabilistic)| Overkill for current track densities         |
| No fusion                 | Single-modality is unreliable in the field   |
