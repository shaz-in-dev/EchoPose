"""Contrastive self-supervised pretraining for CSI tensors.

This module provides a production-usable baseline for representation learning:
- CSI augmentations tailored to amplitude tensors.
- Encoder + projection head.
- NT-Xent loss for positive/negative pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ContrastiveConfig:
    temperature: float = 0.2
    feature_dim: int = 128
    projection_dim: int = 64


class CSIEncoder(nn.Module):
    """Compact encoder for [B, N, S, D] CSI windows."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input expected as [B, 3, 64, 16]
        h = self.backbone(x).flatten(1)
        return self.fc(h)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CSIContrastiveModel(nn.Module):
    def __init__(self, cfg: ContrastiveConfig | None = None):
        super().__init__()
        self.cfg = cfg or ContrastiveConfig()
        self.encoder = CSIEncoder(out_dim=self.cfg.feature_dim)
        self.projector = ProjectionHead(self.cfg.feature_dim, self.cfg.projection_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.projector(self.encoder(x))
        return F.normalize(z, dim=1)


def csi_augment(x: torch.Tensor) -> torch.Tensor:
    """Apply CSI-specific augmentations for contrastive views."""
    y = x.clone()

    # Additive Gaussian noise
    y = y + 0.01 * torch.randn_like(y)

    # Random subcarrier masking (simulate packet loss)
    if torch.rand(1).item() < 0.5:
        mask_idx = torch.randint(0, y.shape[2], (4,))
        y[:, :, mask_idx, :] = 0.0

    # Mild gain jitter
    gain = 0.9 + 0.2 * torch.rand((y.shape[0], 1, 1, 1), device=y.device)
    y = y * gain

    return y


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Normalized temperature-scaled cross entropy loss (SimCLR-style)."""
    bsz = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature

    # Mask self-similarity
    mask = torch.eye(2 * bsz, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -9e15)

    # Positive indices: i <-> i+bsz
    targets = torch.arange(2 * bsz, device=z.device)
    targets = (targets + bsz) % (2 * bsz)

    return F.cross_entropy(sim, targets)


def contrastive_step(model: CSIContrastiveModel, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Single train step output tensors for training loop integration."""
    v1 = csi_augment(batch)
    v2 = csi_augment(batch)

    z1 = model(v1)
    z2 = model(v2)

    loss = nt_xent_loss(z1, z2, temperature=model.cfg.temperature)
    return loss, z1, z2
