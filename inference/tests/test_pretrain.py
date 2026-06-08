"""Tests for self-supervised CSI contrastive pretraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.contrastive_pretrain import (
    CSIContrastiveModel,
    CSIEncoder,
    ContrastiveConfig,
    ProjectionHead,
    csi_augment,
    contrastive_step,
    nt_xent_loss,
)


# ── CSIEncoder ────────────────────────────────────────────────────────────────

def test_encoder_output_shape():
    enc = CSIEncoder(out_dim=128)
    x = torch.randn(4, 3, 64, 16)
    out = enc(x)
    assert out.shape == (4, 128)


def test_encoder_custom_dim():
    enc = CSIEncoder(out_dim=64)
    x = torch.randn(2, 3, 64, 16)
    assert enc(x).shape == (2, 64)


# ── ProjectionHead ─────────────────────────────────────────────────────────────

def test_projection_head_shape():
    head = ProjectionHead(in_dim=128, out_dim=64)
    x = torch.randn(8, 128)
    assert head(x).shape == (8, 64)


# ── CSIContrastiveModel ────────────────────────────────────────────────────────

def test_model_forward_normalized():
    model = CSIContrastiveModel()
    x = torch.randn(6, 3, 64, 16)
    z = model(x)
    # Output should be L2-normalised
    norms = torch.norm(z, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_model_encode_shape():
    model = CSIContrastiveModel()
    x = torch.randn(4, 3, 64, 16)
    h = model.encode(x)
    assert h.shape == (4, 128)  # feature_dim default


def test_model_custom_config():
    cfg = ContrastiveConfig(temperature=0.1, feature_dim=64, projection_dim=32)
    model = CSIContrastiveModel(cfg)
    x = torch.randn(4, 3, 64, 16)
    z = model(x)
    assert z.shape == (4, 32)


# ── csi_augment() ─────────────────────────────────────────────────────────────

def test_augment_preserves_shape():
    x = torch.randn(8, 3, 64, 16)
    y = csi_augment(x)
    assert y.shape == x.shape


def test_augment_produces_different_views():
    torch.manual_seed(0)
    x = torch.randn(8, 3, 64, 16)
    v1 = csi_augment(x)
    v2 = csi_augment(x)
    assert not torch.allclose(v1, v2), "Two augmented views must differ"


def test_augment_preserves_dtype():
    x = torch.randn(4, 3, 64, 16).float()
    assert csi_augment(x).dtype == torch.float32


# ── nt_xent_loss() ────────────────────────────────────────────────────────────

def test_nt_xent_loss_is_scalar():
    z1 = F.normalize(torch.randn(8, 64), dim=1)
    z2 = F.normalize(torch.randn(8, 64), dim=1)
    loss = nt_xent_loss(z1, z2)
    assert loss.ndim == 0
    assert loss.item() > 0


def test_nt_xent_loss_perfect_case():
    z = F.normalize(torch.eye(8), dim=1)
    loss_same = nt_xent_loss(z, z, temperature=0.2)
    loss_diff = nt_xent_loss(z, F.normalize(torch.randn(8, 8), dim=1), temperature=0.2)
    assert loss_same < loss_diff, "Identical pairs should have lower loss than random"


def test_nt_xent_loss_temperature_sensitivity():
    z1 = F.normalize(torch.randn(8, 64), dim=1)
    z2 = F.normalize(torch.randn(8, 64), dim=1)
    loss_low  = nt_xent_loss(z1, z2, temperature=0.05)
    loss_high = nt_xent_loss(z1, z2, temperature=1.0)
    # Lower temperature → sharper distribution → higher loss on random pairs
    assert loss_low != loss_high


# ── contrastive_step() ────────────────────────────────────────────────────────

def test_contrastive_step_returns_loss_and_embeddings():
    model = CSIContrastiveModel()
    batch = torch.randn(8, 3, 64, 16)
    loss, z1, z2 = contrastive_step(model, batch)
    assert loss.item() > 0
    assert z1.shape == z2.shape
    assert z1.shape[0] == 8


def test_contrastive_step_gradients_flow():
    model = CSIContrastiveModel()
    batch = torch.randn(8, 3, 64, 16)
    loss, _, _ = contrastive_step(model, batch)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters())
    assert has_grad, "Gradients must flow through contrastive_step"


# ── pretrain script ────────────────────────────────────────────────────────────

def test_make_synthetic_shape():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from pretrain import make_synthetic
    data = make_synthetic(n=50)
    assert data.shape == (50, 3, 64, 16)
    assert data.dtype == torch.float32


def test_make_synthetic_not_uniform_noise():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from pretrain import make_synthetic
    data = make_synthetic(n=100)
    # Should have sine-wave structure, not pure noise
    # Variance across subcarriers (dim=2) should be significant
    var = data.var(dim=2).mean().item()
    assert var > 0.01


def test_pretrain_mock_full_run(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from pretrain import train

    args = argparse.Namespace(
        mock=True,
        mock_n=64,
        data_dir="data/csi",
        epochs=3,
        batch_size=16,
        lr=3e-4,
        temperature=0.2,
        feature_dim=128,
        projection_dim=64,
        output=str(tmp_path / "encoder.pt"),
        cpu=True,
    )
    train(args)
    assert (tmp_path / "encoder.pt").exists(), "Encoder checkpoint must be saved"


def test_pretrain_saved_encoder_loadable(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from pretrain import train

    out = str(tmp_path / "enc.pt")
    train(argparse.Namespace(
        mock=True, mock_n=32, data_dir="", epochs=1,
        batch_size=16, lr=1e-3, temperature=0.2,
        feature_dim=128, projection_dim=64, output=out, cpu=True,
    ))

    enc = CSIEncoder(out_dim=128)
    state = torch.load(out, map_location="cpu", weights_only=True)
    enc.load_state_dict(state)
    x = torch.randn(2, 3, 64, 16)
    out_tensor = enc(x)
    assert out_tensor.shape == (2, 128)
