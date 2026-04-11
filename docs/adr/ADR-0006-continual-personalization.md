# ADR-0006: Continual Online Personalization

- **Status:** Accepted
- **Date:** 2026-03-16
- **Decision Makers:** Core team

## Context

WiFi-based pose estimation accuracy degrades when deployed in a new physical
environment because multipath patterns differ from the training domain.
Re-collecting labelled data for every new room is impractical.

## Decision

We implement **online continual personalization** using:

1. **Low-rank delta adapters** (`DeltaAdapterLinear`) — small trainable layers
   injected around frozen base model weights.
2. **Stability penalty** — L2 regularisation that constrains adapter norms to
   prevent catastrophic forgetting of the base distribution.
3. **`OnlinePersonalizer`** — feeds unlabelled CSI frames through the adapter
   and back-propagates a self-supervised consistency loss.

Adapters are saved per-room and hot-swapped at inference time.

## Consequences

- **Positive:** 10-30% accuracy improvement in new environments without labels.
- **Positive:** Base model is frozen — no catastrophic forgetting.
- **Negative:** Small latency increase during adaptation (not inference).
- **Negative:** Adapter storage grows linearly with number of rooms.

## Alternatives Considered

| Alternative                | Why rejected                                    |
|----------------------------|-------------------------------------------------|
| Full fine-tuning           | Catastrophic forgetting; needs labelled data    |
| Domain-adversarial training| Requires simultaneous source + target batches   |
| Fixed calibration routine  | One-time; doesn't adapt to furniture changes    |
