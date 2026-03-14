"""
inference/research/domain_shift_monitor.py — Automatic Shift Detection (Feature 23)

Monitors the epistemic uncertainty of the model via deep ensembles 
to detect when the user moves the routers to a completely new room, 
triggering an automatic UI alert to recalibrate.
"""

import torch
import numpy as np
import logging

logger = logging.getLogger("rf_inference.domain_shift")

class AutomaticDomainShiftDetection:
    """
    Prevents silent failures. If the model is outputting garbage because the 
    walls moved, this detects it using ensemble variance rather than waiting for 
    the user to notice.
    """
    def __init__(self, variance_threshold=0.85):
        self.variance_threshold = variance_threshold
        # Keep a rolling window of uncertainty to avoid false alarms
        self.uncertainty_history = []
        
    def check_shift(self, ensemble_models: list, csi_input: torch.Tensor):
        """
        Queries an ensemble of N models (trained with different random seeds).
        In a familiar room, all N models agree on the pose (Variance ~ 0).
        In a new, unseen room, all N models guess wildly different poses (Variance > Threshold).
        """
        if len(ensemble_models) < 2:
            return False # Need at least 2 models for variance
            
        preds = []
        with torch.no_grad():
            for model in ensemble_models:
                # model must be in eval mode
                pose = model(csi_input) # [1, MAX_PEOPLE, 17, 3]
                preds.append(pose)
                
        # Stack predictions
        # Shape: [Num_Models, 1, MAX_PEOPLE, 17, 3]
        stacked = torch.stack(preds)
        
        # Calculate cross-model variance (Disagreement)
        # Average variance across joints/people
        ensemble_variance = torch.var(stacked, dim=0).mean().item()
        
        self.uncertainty_history.append(ensemble_variance)
        if len(self.uncertainty_history) > 30: # Rolling window of ~1.5 seconds at 20Hz
            self.uncertainty_history.pop(0)
            
        rolling_uncertainty = np.mean(self.uncertainty_history)
        
        if rolling_uncertainty > self.variance_threshold:
            logger.critical(f"DOMAIN SHIFT DETECTED! Ensemble Variance {rolling_uncertainty:.2f} > {self.variance_threshold}")
            logger.critical("Warning the UI to Trigger Recalibration Wizard...")
            return True
            
        return False
