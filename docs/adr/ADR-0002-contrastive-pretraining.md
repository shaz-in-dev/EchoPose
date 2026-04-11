# ADR-0002: Contrastive Self-Supervised Pretraining

- Status: Accepted
- Date: 2026-04-11

## Context

Labeled CSI-pose datasets are costly, and supervised-only training underperforms in new environments.

## Decision

Adopt contrastive pretraining over CSI windows using augmentation pairs and NT-Xent loss.
Initial implementation is in `inference/research/contrastive_pretrain.py`.

## Consequences

Positive:
- Better representation quality with limited labels.
- Reusable encoder initialization for downstream tasks.

Negative:
- Additional pretraining compute and hyperparameter tuning.
