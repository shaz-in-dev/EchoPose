"""Self-supervised CSI contrastive pretraining.

Trains a CSI encoder on unlabeled WiFi data using SimCLR-style NT-Xent loss.
No pose labels required — works with raw captured CSI or synthetic data.

Usage:
    # With real data (.npz files containing 'csi' key, shape [T, 3, 64, 16])
    python scripts/pretrain.py --data-dir data/csi/ --epochs 100

    # Without hardware (synthetic data)
    python scripts/pretrain.py --mock --epochs 50

    # Load the resulting encoder into downstream tasks:
    #   model.encoder.load_state_dict(torch.load("models/encoder.pt"))
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.contrastive_pretrain import (
    CSIContrastiveModel,
    ContrastiveConfig,
    contrastive_step,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("echopose.pretrain")


# ── Data loading ──────────────────────────────────────────────────────

def load_npz_dataset(data_dir: Path, max_frames: int = 100_000) -> torch.Tensor:
    frames: list[np.ndarray] = []
    for f in sorted(data_dir.glob("**/*.npz")):
        try:
            arr = np.load(f)["csi"].astype(np.float32)
        except (KeyError, Exception):
            logger.warning("Skipping %s (no 'csi' key or load error)", f)
            continue
        if arr.ndim == 4:
            frames.append(arr)
        else:
            logger.warning("Skipping %s: expected 4-D array, got shape %s", f, arr.shape)
    if not frames:
        raise FileNotFoundError(f"No usable .npz files found in {data_dir}")
    combined = np.concatenate(frames, axis=0)[:max_frames]
    logger.info("Loaded %d CSI frames from %s", len(combined), data_dir)
    return torch.from_numpy(combined)


def make_synthetic(n: int = 2000) -> torch.Tensor:
    """Synthetic [N, 3, 64, 16] CSI tensors for testing without hardware."""
    # Simulate multipath amplitude patterns (not uniform noise)
    t = torch.linspace(0, 4 * torch.pi, 64).unsqueeze(0).unsqueeze(0)
    base = torch.sin(t + torch.randn(n, 3, 1)).unsqueeze(-1).expand(n, 3, 64, 16)
    return base + 0.1 * torch.randn(n, 3, 64, 16)


# ── Training loop ─────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info("Device: %s", device)

    dataset = make_synthetic(args.mock_n) if args.mock else load_npz_dataset(Path(args.data_dir))
    dataset = dataset.to(device)
    n = len(dataset)
    logger.info("Dataset: %d frames, batch_size=%d, epochs=%d", n, args.batch_size, args.epochs)

    cfg = ContrastiveConfig(
        temperature=args.temperature,
        feature_dim=args.feature_dim,
        projection_dim=args.projection_dim,
    )
    model = CSIContrastiveModel(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    log_every = max(1, args.epochs // 20)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, steps = 0.0, 0
        perm = torch.randperm(n, device=device)

        for start in range(0, n - args.batch_size, args.batch_size):
            batch = dataset[perm[start : start + args.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = contrastive_step(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        scheduler.step()
        avg = total_loss / max(1, steps)

        if avg < best_loss:
            best_loss = avg
            torch.save(model.encoder.state_dict(), out_path)

        if epoch % log_every == 0 or epoch == args.epochs:
            logger.info(
                "Epoch %4d/%d  loss=%.4f  best=%.4f  lr=%.2e",
                epoch, args.epochs, avg, best_loss,
                scheduler.get_last_lr()[0],
            )

    logger.info("Pretraining complete. Best encoder saved → %s", out_path)
    logger.info(
        "Load into downstream model:\n"
        "    model.encoder.load_state_dict(torch.load('%s'))", out_path
    )


# ── CLI ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="EchoPose self-supervised CSI pretraining (no pose labels needed)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = p.add_mutually_exclusive_group()
    data.add_argument("--data-dir", default="data/csi",
                      help="Directory of .npz files with 'csi' key, shape [T, 3, 64, 16]")
    data.add_argument("--mock", action="store_true",
                      help="Use synthetic data — runs without hardware")
    p.add_argument("--mock-n", type=int, default=2000, metavar="N",
                   help="Number of synthetic frames (--mock only)")
    p.add_argument("--epochs",        type=int,   default=100)
    p.add_argument("--batch-size",    type=int,   default=64)
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--temperature",   type=float, default=0.2)
    p.add_argument("--feature-dim",   type=int,   default=128)
    p.add_argument("--projection-dim",type=int,   default=64)
    p.add_argument("--output",        default="models/encoder.pt",
                   help="Output path for best encoder checkpoint")
    p.add_argument("--cpu",           action="store_true", help="Force CPU even if GPU available")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
