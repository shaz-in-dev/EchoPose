"""
inference/research/domain_adaptation.py — Unsupervised Domain Adaptation (Feature 20)

Implements Maximum Mean Discrepancy (MMD) to continuously fine-tune the model
to new rooms (new multipath environments) without requiring labeled camera data.
"""

import torch
import torch.nn as nn

class RealTimeDomainAdaptation:
    """
    Self-supervising module that runs in parallel to inference.
    Maps features from a new environment back to the source distribution.
    """
    def __init__(self, feature_dim=256):
        self.feature_dim = feature_dim
        
    def compute_mmd_loss(self, source_features: torch.Tensor, target_features: torch.Tensor):
        """
        Maximum Mean Discrepancy (MMD) calculates the distance between two distributions.
        By minimizing this loss during online fine-tuning, the neural network learns to 
        extract "room-invariant" human signatures instead of overfitting to wall reflections.
        """
        # Linear MMD for high-speed online computation
        delta = source_features.mean(0) - target_features.mean(0)
        loss = delta.dot(delta)
        return loss
        
    def adapt_online(self, model: nn.Module, new_environment_stream: torch.Tensor, source_anchors: torch.Tensor):
        """
        Runs a quick backward pass to adjust Batch Norm and MLPs in a new room.
        WARNING: Highly experimental. This updates model weights in production!
        """
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        
        # Extract features (assume model has an encoder property as built in V2)
        target_features = model.encoder(new_environment_stream)
        
        # MMD Loss forces the new features to match the distribution of the original training lab
        loss = self.compute_mmd_loss(source_anchors, target_features)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        model.eval()
        return loss.item()
