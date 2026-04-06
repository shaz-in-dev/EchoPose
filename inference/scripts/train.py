"""
scripts/train.py — Training skeleton for PoseNetV2

This provides the training loop structure for when real labeled CSI-to-pose
datasets become available. Until then, it validates the model architecture
with synthetic data.

Usage:
    python -m scripts.train --epochs 10 --lr 1e-3
    python -m scripts.train --data path/to/dataset.npz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).parent.parent))
from pipeline.pose_net_v2 import PoseNetV2, FEATURE_SHAPE, NUM_KEYPOINTS, MAX_PEOPLE


class CSIPoseDataset(Dataset):
    """
    Loads CSI→pose pairs from an .npz file.
    Expected keys:
        features: float32 [N, nodes, subcarriers, doppler_bins]
        poses:    float32 [N, max_people, 17, 4]  (x, y, z, confidence)
    """

    def __init__(self, npz_path: Path | None = None, size: int = 256):
        if npz_path and npz_path.exists():
            data = np.load(npz_path)
            self.features = torch.from_numpy(data["features"]).float()
            self.poses = torch.from_numpy(data["poses"]).float()
        else:
            # Synthetic placeholder data for architecture validation
            self.features = torch.randn(size, *FEATURE_SHAPE)
            self.poses = torch.rand(size, MAX_PEOPLE, NUM_KEYPOINTS, 4)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.poses[idx]


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseNetV2().to(device)

    dataset = CSIPoseDataset(
        npz_path=Path(args.data) if args.data else None,
        size=args.synth_size,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    ckpt_dir = Path(__file__).parent.parent / "models"
    ckpt_dir.mkdir(exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)

            preds = model(features)  # [B, MAX_PEOPLE * 17 * 4]
            targets_flat = targets.view(targets.size(0), -1)

            loss = criterion(preds, targets_flat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.6f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_dir / "pose_net.pt")
            print(f"  -> Saved best checkpoint (loss={best_loss:.6f})")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PoseNetV2")
    parser.add_argument("--data", type=str, default=None, help="Path to .npz dataset")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--synth-size", type=int, default=256, help="Synthetic dataset size if no real data")
    args = parser.parse_args()

    train(args)
