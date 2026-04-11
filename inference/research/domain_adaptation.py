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
        self._optimizer = None
        self._optimizer_model_id = None
        
    def compute_mmd_loss(self, source_features: torch.Tensor, target_features: torch.Tensor):
        """
        Maximum Mean Discrepancy (MMD) calculates the distance between two distributions.
        """
        if source_features.shape != target_features.shape:
            raise ValueError(
                f"Feature shape mismatch: source {source_features.shape} vs target {target_features.shape}"
            )
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
        # Re-use optimizer across calls; recreate only if the model changes
        model_id = id(model)
        if self._optimizer is None or self._optimizer_model_id != model_id:
            self._optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
            self._optimizer_model_id = model_id
        
        # Extract features (assume model has an encoder property as built in V2)
        target_features = model.encoder(new_environment_stream)
        
        # MMD Loss forces the new features to match the distribution of the original training lab
        loss = self.compute_mmd_loss(source_anchors, target_features)
        
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        
        model.eval()
        return loss.item()
