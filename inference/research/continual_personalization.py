from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn


@dataclass(frozen=True)
class PersonalizationConfig:
    rank: int = 4
    adapter_gain: float = 8.0
    stability_lambda: float = 0.05
    lr: float = 1e-3


class DeltaAdapterLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4, adapter_gain: float = 8.0):
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.rank = rank
        self.scaling = adapter_gain / max(1, rank)

        self.adapter_a = nn.Parameter(torch.zeros(in_features, rank))
        self.adapter_b = nn.Parameter(torch.zeros(rank, out_features))
        nn.init.normal_(self.adapter_a, std=0.01)
        nn.init.zeros_(self.adapter_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        delta = x @ self.adapter_a @ self.adapter_b
        return base_out + self.scaling * delta



def stability_penalty(
    named_params: Dict[str, torch.Tensor],
    sensitivity: Dict[str, torch.Tensor],
    reference: Dict[str, torch.Tensor],
    stability_lambda: float,
) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=next(iter(named_params.values())).device) if named_params else torch.tensor(0.0)
    for name, p in named_params.items():
        if name in sensitivity and name in reference:
            penalty = penalty + (sensitivity[name] * (p - reference[name]).pow(2)).sum()
    return stability_lambda * penalty


class OnlinePersonalizer:
    def __init__(self, model: nn.Module, cfg: PersonalizationConfig | None = None):
        self.model = model
        self.cfg = cfg or PersonalizationConfig()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        self.sensitivity: Dict[str, torch.Tensor] = {}
        self.reference: Dict[str, torch.Tensor] = {
            n: p.detach().clone() for n, p in self.model.named_parameters()
        }

    def calibrate_sensitivity(self, batch_x: torch.Tensor, batch_y: torch.Tensor, loss_fn: nn.Module) -> None:
        self.model.zero_grad(set_to_none=True)
        loss = loss_fn(self.model(batch_x), batch_y)
        loss.backward()

        self.sensitivity = {}
        for n, p in self.model.named_parameters():
            if p.grad is not None:
                self.sensitivity[n] = p.grad.detach().pow(2).clone()

    def online_update(self, batch_x: torch.Tensor, batch_y: torch.Tensor, loss_fn: nn.Module) -> float:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        pred = self.model(batch_x)
        task_loss = loss_fn(pred, batch_y)
        pen = stability_penalty(
            named_params={n: p for n, p in self.model.named_parameters()},
            sensitivity=self.sensitivity,
            reference=self.reference,
            stability_lambda=self.cfg.stability_lambda,
        )
        loss = task_loss + pen
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return float(loss.detach().cpu())
