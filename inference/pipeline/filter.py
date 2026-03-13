import numpy as np
from typing import List, Dict

class SkeletonFilter:
    """
    Applies Exponential Moving Average (EMA) smoothing to the 17 3D keypoints
    returned by the PoseNet model to prevent jitter/glitching.
    """
    def __init__(self, alpha: float = 0.4):
        """
        Args:
            alpha: Smoothing factor (0.0 to 1.0).
                   1.0 means no smoothing (instant updates).
                   Near 0.0 means heavy smoothing (sluggish).
        """
        self.alpha = alpha
        self.state = None  # Will hold the running [17, 3] array of (x,y,z)

    def filter(self, keypoints: List[Dict]) -> List[Dict]:
        """
        Ingests a list of 17 keypoint dicts, updates the EMA state,
        and returns the smoothed dicts.
        """
        if not keypoints or len(keypoints) != 17:
            return keypoints

        # Extract just x, y, z into a numpy array for vectorized math
        current_pts = np.zeros((17, 3), dtype=np.float32)
        for i, kp in enumerate(keypoints):
            current_pts[i, 0] = kp["x"]
            current_pts[i, 1] = kp["y"]
            current_pts[i, 2] = kp["z"]
        
        if self.state is None:
            self.state = current_pts.copy()
        else:
            # EMA formula: S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
            self.state = self.alpha * current_pts + (1.0 - self.alpha) * self.state

        # Reconstruct the list of dicts with the smoothed coordinates
        smoothed_kps = []
        for i in range(17):
            smoothed_kps.append({
                "x": float(self.state[i, 0]),
                "y": float(self.state[i, 1]),
                "z": float(self.state[i, 2]),
                "confidence": keypoints[i]["confidence"] # Keep original confidence
            })
            
        return smoothed_kps
