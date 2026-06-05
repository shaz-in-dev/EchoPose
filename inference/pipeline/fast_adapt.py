"""Micro-LoRA fast domain adaptation for PoseNetV2.

Adapts EchoPose to a new room/environment in ~30 seconds using
self-supervised contrastive loss — no labeled data required.

Usage (from server.py):
    adapter = FastAdapter(model=posenet, rank=8, device=device)
    adapter.push_frame(csi_tensor)          # call each inference cycle
    result = adapter.adapt(timeout_seconds=30)   # call POST /adapt
    adapter.save_adapters("models/room_A.lora")
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("rf_inference.fast_adapt")


# ── LoRA layer ────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """nn.Linear wrapper with a trainable low-rank A×B delta.

    Output = W₀x  +  (alpha/rank) · x Aᵀ Bᵀ
    Only lora_a and lora_b are marked requires_grad=True.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0) -> None:
        super().__init__()
        self.base = base
        self.scaling = alpha / rank

        in_f, out_f = base.in_features, base.out_features
        self.lora_a = nn.Parameter(torch.randn(in_f, rank) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(rank, out_f))

        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * (x @ self.lora_a @ self.lora_b)


# ── Injection ─────────────────────────────────────────────────────────

# BatchNorm layers in the multi-scale feature extractors (ext_3.1, ext_5.1, ext_7.1).
# These are kept trainable so the contrastive loss through model.encoder() can
# backpropagate — BN stats shift with room multipath and adapt in seconds.
_BN_PREFIXES = frozenset({"ext_3.1", "ext_5.1", "ext_7.1"})


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0) -> nn.Module:
    """Replace every nn.Linear in model.pose_head with a LoRALinear.

    Frozen:   all weights except LoRA adapters and BatchNorm layers.
    Trainable: lora_a / lora_b matrices + BN weight/bias in ext_3/5/7.

    The BN layers must stay trainable so that contrastive loss through
    model.encoder() has a gradient path (BN stats shift with room multipath).

    Returns the same model (modified in-place).
    """
    if not hasattr(model, "pose_head"):
        raise AttributeError("inject_lora: model has no pose_head attribute")

    new_layers: list[nn.Module] = []
    for layer in model.pose_head:
        if isinstance(layer, nn.Linear):
            new_layers.append(LoRALinear(layer, rank=rank, alpha=alpha))
        else:
            new_layers.append(layer)
    model.pose_head = nn.Sequential(*new_layers)

    for name, p in model.named_parameters():
        is_adapter = "lora_a" in name or "lora_b" in name
        is_bn      = ".".join(name.split(".")[:2]) in _BN_PREFIXES
        if not is_adapter and not is_bn:
            p.requires_grad_(False)

    n_adapter = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total   = sum(p.numel() for p in model.parameters())
    logger.info(
        "LoRA injected: %d adapter+BN params / %d total (%.2f%%)",
        n_adapter, n_total, 100 * n_adapter / max(1, n_total),
    )
    return model


# ── Contrastive augmentation ──────────────────────────────────────────

def _augment_csi(x: torch.Tensor) -> torch.Tensor:
    """Two CSI-specific augmentations used as contrastive views."""
    y = x + 0.01 * torch.randn_like(x)
    if torch.rand(1).item() < 0.5:
        mask = torch.randint(0, y.shape[2], (4,))
        y[:, :, mask, :] = 0.0
    gain = (0.9 + 0.2 * torch.rand(1, device=x.device))
    return y * gain


# ── FastAdapter ───────────────────────────────────────────────────────

class FastAdapter:
    """Self-supervised LoRA adapter.

    Push CSI frames from the inference loop, then call adapt() to
    fine-tune only the adapter weights against the model's own encoder.
    No pose labels required.
    """

    BUFFER_SIZE = 500

    def __init__(
        self,
        model: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        lr: float = 5e-4,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.model = inject_lora(model, rank=rank, alpha=alpha).to(device)
        self.model.eval()
        adapter_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(adapter_params, lr=lr)
        self._buffer: deque[torch.Tensor] = deque(maxlen=self.BUFFER_SIZE)
        self._is_adapting = False

    # ── Public API ────────────────────────────────────────────────────

    def push_frame(self, csi: torch.Tensor) -> None:
        """Buffer a single CSI tensor [Nodes, Subs, Doppler]. Thread-safe (GIL-protected)."""
        self._buffer.append(csi.detach().cpu())

    def adapt(self, timeout_seconds: float = 30.0) -> dict:
        """Run LoRA fine-tuning synchronously for up to timeout_seconds.

        Returns stats dict: {status, steps, avg_loss, frames_used}.
        Safe to call from a FastAPI background task.
        """
        frames = list(self._buffer)
        if len(frames) < 16:
            return {"status": "insufficient_data", "frames": len(frames)}

        self._is_adapting = True
        self.model.train()
        deadline = time.monotonic() + timeout_seconds
        steps = 0
        total_loss = 0.0
        batch_size = min(32, len(frames))

        try:
            while time.monotonic() < deadline:
                idx = torch.randint(0, len(frames), (batch_size,))
                batch = torch.stack([frames[i] for i in idx.tolist()]).to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                loss = self._nt_xent(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0
                )
                self.optimizer.step()
                total_loss += loss.item()
                steps += 1
        finally:
            self.model.eval()
            self._is_adapting = False

        avg_loss = total_loss / max(1, steps)
        logger.info(
            "Adaptation complete: %d steps, avg_loss=%.4f, %d frames",
            steps, avg_loss, len(frames),
        )
        return {
            "status": "complete",
            "steps": steps,
            "avg_loss": round(avg_loss, 4),
            "frames_used": len(frames),
            "elapsed_s": round(timeout_seconds, 1),
        }

    def save_adapters(self, path: str) -> None:
        adapter_state = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        torch.save(adapter_state, path)
        logger.info("Saved LoRA adapters → %s", path)

    def load_adapters(self, path: str) -> None:
        state = torch.load(path, map_location=self.device, weights_only=True)
        current = self.model.state_dict()
        current.update(state)
        self.model.load_state_dict(current)
        logger.info("Loaded LoRA adapters from %s", path)

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    # ── Internal ──────────────────────────────────────────────────────

    def _nt_xent(self, x: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        """NT-Xent contrastive loss over encoder embeddings (no labels)."""
        if not hasattr(self.model, "encoder"):
            raise AttributeError("model must expose an encoder(x) method")

        z1 = F.normalize(self.model.encoder(_augment_csi(x)), dim=1)
        z2 = F.normalize(self.model.encoder(_augment_csi(x)), dim=1)

        bsz = z1.size(0)
        sim = torch.mm(z1, z2.t()) / temperature
        targets = torch.arange(bsz, device=self.device)
        return (F.cross_entropy(sim, targets) + F.cross_entropy(sim.t(), targets)) / 2
