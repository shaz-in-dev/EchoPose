from __future__ import annotations

import torch
import torch.nn as nn

from research.continual_personalization import DeltaAdapterLinear, OnlinePersonalizer


def test_delta_adapter_linear_forward_shape() -> None:
    m = DeltaAdapterLinear(8, 4, rank=2, adapter_gain=4.0)
    y = m(torch.randn(5, 8))
    assert y.shape == (5, 4)


def test_online_personalizer_update_runs() -> None:
    model = nn.Sequential(DeltaAdapterLinear(8, 4, rank=2, adapter_gain=4.0))
    personalizer = OnlinePersonalizer(model)
    loss_fn = nn.MSELoss()

    x = torch.randn(6, 8)
    y = torch.randn(6, 4)

    personalizer.calibrate_sensitivity(x, y, loss_fn)
    loss = personalizer.online_update(x, y, loss_fn)
    assert loss >= 0.0
