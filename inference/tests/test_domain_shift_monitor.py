"""tests/test_domain_shift_monitor.py — AutomaticDomainShiftDetection coverage."""

import pytest
import torch
import torch.nn as nn
from research.domain_shift_monitor import AutomaticDomainShiftDetection


class _DummyModel(nn.Module):
    """Tiny model for shift testing."""
    def __init__(self, output_val=0.5):
        super().__init__()
        self._val = output_val
        self.linear = nn.Linear(1, 1)  # need at least one param

    def forward(self, x):
        B = x.shape[0]
        return torch.full((B, 3, 17, 3), self._val)


@pytest.fixture
def monitor():
    return AutomaticDomainShiftDetection(variance_threshold=0.85)


# ── needs >= 2 models ─────────────────────────────────────────

def test_single_model_returns_false(monitor):
    m = _DummyModel()
    m.eval()
    x = torch.randn(1, 3, 64, 16)
    assert monitor.check_shift([m], x) is False


# ── all models agree → no shift ───────────────────────────────

def test_no_shift_when_agree(monitor):
    models = [_DummyModel(0.5), _DummyModel(0.5)]
    for m in models:
        m.eval()
    x = torch.randn(1, 3, 64, 16)
    assert monitor.check_shift(models, x) is False


# ── models disagree → shift detected ──────────────────────────

def test_shift_when_disagree(monitor):
    m1 = _DummyModel(0.0)
    m2 = _DummyModel(100.0)
    for m in [m1, m2]:
        m.eval()
    x = torch.randn(1, 3, 64, 16)
    # Push enough frames to fill rolling window
    for _ in range(40):
        result = monitor.check_shift([m1, m2], x)
    assert result is True


# ── rolling window ────────────────────────────────────────────

def test_rolling_window_capped(monitor):
    m1 = _DummyModel(0.5)
    m2 = _DummyModel(0.5)
    for m in [m1, m2]:
        m.eval()
    x = torch.randn(1, 3, 64, 16)
    for _ in range(50):
        monitor.check_shift([m1, m2], x)
    assert len(monitor.uncertainty_history) <= 30


# ── threshold configurability ──────────────────────────────────

def test_custom_threshold():
    mon = AutomaticDomainShiftDetection(variance_threshold=1e-6)
    m1 = _DummyModel(0.5)
    m2 = _DummyModel(0.51)
    for m in [m1, m2]:
        m.eval()
    x = torch.randn(1, 3, 64, 16)
    for _ in range(40):
        result = mon.check_shift([m1, m2], x)
    # With tiny threshold even small disagreement should trigger
    assert result is True
