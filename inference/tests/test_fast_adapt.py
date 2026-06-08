"""Tests for Micro-LoRA fast domain adaptation."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.fast_adapt import FastAdapter, LoRALinear, inject_lora
from pipeline.pose_net_v2 import PoseNetV2


# ── LoRALinear ─────────────────────────────────────────────────────────────────

def test_lora_linear_output_shape():
    base = nn.Linear(64, 128)
    lora = LoRALinear(base, rank=4, alpha=8.0)
    x = torch.randn(3, 64)
    out = lora(x)
    assert out.shape == (3, 128)


def test_lora_linear_base_is_frozen():
    base = nn.Linear(64, 128)
    lora = LoRALinear(base, rank=4)
    for p in lora.base.parameters():
        assert not p.requires_grad, "Base weight must be frozen"


def test_lora_linear_adapters_are_trainable():
    base = nn.Linear(64, 128)
    lora = LoRALinear(base, rank=4)
    assert lora.lora_a.requires_grad
    assert lora.lora_b.requires_grad


def test_lora_linear_delta_starts_near_zero():
    base = nn.Linear(64, 128)
    lora = LoRALinear(base, rank=4)
    x = torch.randn(8, 64)
    base_out = base(x)
    lora_out = lora(x)
    # lora_b is zero-initialized → delta should be ~0 at init
    assert torch.allclose(base_out, lora_out, atol=1e-4), "LoRA delta should be ~0 at init"


def test_lora_linear_scaling():
    base = nn.Linear(4, 4, bias=False)
    lora = LoRALinear(base, rank=2, alpha=4.0)
    assert lora.scaling == 4.0 / 2  # alpha / rank


# ── inject_lora() ──────────────────────────────────────────────────────────────

def test_inject_lora_wraps_linear_layers():
    model = PoseNetV2()
    inject_lora(model, rank=4)
    has_lora = any(isinstance(l, LoRALinear) for l in model.pose_head)
    assert has_lora, "inject_lora must replace at least one Linear with LoRALinear"


def test_inject_lora_freezes_non_adapter_non_bn_params():
    """All params except LoRA adapters and BN layers must be frozen."""
    from pipeline.fast_adapt import _BN_PREFIXES
    model = PoseNetV2()
    inject_lora(model, rank=4)
    for name, p in model.named_parameters():
        is_adapter = "lora_a" in name or "lora_b" in name
        is_bn      = ".".join(name.split(".")[:2]) in _BN_PREFIXES
        if not is_adapter and not is_bn:
            assert not p.requires_grad, f"{name} must be frozen after inject_lora"


def test_inject_lora_bn_layers_stay_trainable():
    """BatchNorm layers in ext_3/5/7 must remain trainable for encoder adaptation."""
    from pipeline.fast_adapt import _BN_PREFIXES
    model = PoseNetV2()
    inject_lora(model, rank=4)
    for name, p in model.named_parameters():
        if ".".join(name.split(".")[:2]) in _BN_PREFIXES:
            assert p.requires_grad, f"BN param {name} must be trainable"


def test_inject_lora_adapter_fraction_is_small():
    """Trainable params (adapters + BN) should be a tiny fraction of the model."""
    model = PoseNetV2()
    inject_lora(model, rank=8)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    frac = trainable / total
    assert frac < 0.02, f"Adapter+BN fraction {frac:.4f} should be <2% of total params"


def test_inject_lora_model_still_runs():
    model = PoseNetV2()
    inject_lora(model, rank=4)
    x = torch.randn(2, 3, 64, 16)
    out = model(x)
    assert out.shape == (2, 3, 17, 4)


def test_inject_lora_requires_pose_head():
    class NoHead(nn.Module):
        def forward(self, x): return x

    with pytest.raises(AttributeError):
        inject_lora(NoHead(), rank=4)


# ── FastAdapter ────────────────────────────────────────────────────────────────

def test_fast_adapter_initial_buffer_empty():
    adapter = FastAdapter(PoseNetV2(), rank=4)
    assert adapter.buffer_size == 0


def test_fast_adapter_push_frame():
    adapter = FastAdapter(PoseNetV2(), rank=4)
    adapter.push_frame(torch.randn(3, 64, 16))
    assert adapter.buffer_size == 1


def test_fast_adapter_push_multiple_frames():
    adapter = FastAdapter(PoseNetV2(), rank=4)
    for _ in range(10):
        adapter.push_frame(torch.randn(3, 64, 16))
    assert adapter.buffer_size == 10


def test_fast_adapter_buffer_capped_at_max():
    adapter = FastAdapter(PoseNetV2(), rank=4)
    for _ in range(FastAdapter.BUFFER_SIZE + 50):
        adapter.push_frame(torch.randn(3, 64, 16))
    assert adapter.buffer_size == FastAdapter.BUFFER_SIZE


def test_fast_adapt_returns_insufficient_when_few_frames():
    adapter = FastAdapter(PoseNetV2(), rank=4)
    for _ in range(10):
        adapter.push_frame(torch.randn(3, 64, 16))
    result = adapter.adapt(timeout_seconds=1.0)
    assert result["status"] == "insufficient_data"
    assert result["frames"] == 10


def test_fast_adapt_runs_with_enough_data():
    adapter = FastAdapter(PoseNetV2(), rank=4)
    for _ in range(20):
        adapter.push_frame(torch.randn(3, 64, 16))
    result = adapter.adapt(timeout_seconds=2.0)
    assert result["status"] == "complete"
    assert result["steps"] > 0
    assert isinstance(result["avg_loss"], float)
    assert result["frames_used"] == 20


def test_fast_adapt_model_returns_to_eval_after():
    adapter = FastAdapter(PoseNetV2(), rank=4)
    for _ in range(20):
        adapter.push_frame(torch.randn(3, 64, 16))
    adapter.adapt(timeout_seconds=1.0)
    assert not adapter.model.training, "model must be in eval mode after adapt()"


def test_fast_adapt_is_not_adapting_after():
    adapter = FastAdapter(PoseNetV2(), rank=4)
    for _ in range(20):
        adapter.push_frame(torch.randn(3, 64, 16))
    adapter.adapt(timeout_seconds=1.0)
    assert not adapter._is_adapting


# ── save/load adapters ─────────────────────────────────────────────────────────

def test_save_and_load_adapters(tmp_path):
    adapter = FastAdapter(PoseNetV2(), rank=4)
    for _ in range(20):
        adapter.push_frame(torch.randn(3, 64, 16))
    adapter.adapt(timeout_seconds=1.0)

    path = str(tmp_path / "lora.pt")
    adapter.save_adapters(path)
    assert Path(path).exists()

    # Mutate adapter weights, then reload and check they match saved values
    saved_a = {n: p.clone() for n, p in adapter.model.named_parameters() if "lora_a" in n}

    # Corrupt the weights
    for p in adapter.model.parameters():
        if p.requires_grad:
            p.data.fill_(999.0)

    adapter.load_adapters(path)
    for n, p in adapter.model.named_parameters():
        if "lora_a" in n and n in saved_a:
            assert torch.allclose(p, saved_a[n]), f"{n} not restored correctly"
