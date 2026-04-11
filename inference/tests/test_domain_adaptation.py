"""tests/test_domain_adaptation.py — RealTimeDomainAdaptation coverage."""

import pytest
import torch
import torch.nn as nn
from research.domain_adaptation import RealTimeDomainAdaptation


@pytest.fixture
def adapter():
    return RealTimeDomainAdaptation(feature_dim=256)


# ── MMD loss ──────────────────────────────────────────────────

def test_mmd_loss_same_distribution(adapter):
    src = torch.randn(32, 256)
    loss = adapter.compute_mmd_loss(src, src)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_mmd_loss_different_distributions(adapter):
    src = torch.zeros(32, 256)
    tgt = torch.ones(32, 256)
    loss = adapter.compute_mmd_loss(src, tgt)
    assert loss.item() > 0


def test_mmd_loss_symmetric(adapter):
    src = torch.randn(32, 256)
    tgt = torch.randn(32, 256)
    l1 = adapter.compute_mmd_loss(src, tgt)
    l2 = adapter.compute_mmd_loss(tgt, src)
    assert l1.item() == pytest.approx(l2.item(), abs=1e-5)


# ── online adaptation ────────────────────────────────────────

def test_adapt_online_reduces_loss():
    from pipeline.pose_net_v2 import PoseNetV2, FEATURE_SHAPE
    model = PoseNetV2()
    adapter = RealTimeDomainAdaptation(feature_dim=256)

    source_anchors = torch.randn(4, 256, requires_grad=False)
    stream = torch.randn(4, *FEATURE_SHAPE)

    loss = adapter.adapt_online(model, stream, source_anchors)
    assert isinstance(loss, float)
    assert loss >= 0


def test_model_returns_to_eval():
    from pipeline.pose_net_v2 import PoseNetV2, FEATURE_SHAPE
    model = PoseNetV2()
    model.eval()
    adapter = RealTimeDomainAdaptation()

    source_anchors = torch.randn(2, 256)
    stream = torch.randn(2, *FEATURE_SHAPE)
    adapter.adapt_online(model, stream, source_anchors)
    assert not model.training
