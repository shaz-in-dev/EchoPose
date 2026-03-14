import numpy as np
from typing import List, Dict

class SkeletonFilter:
    """
    Applies Exponential Moving Average (EMA) smoothing to the 17 3D keypoints
    returned by the PoseNet model to prevent jitter/glitching.
    """
    def __init__(self, alpha: float = 0.4, max_people: int = 3):
        self.alpha = alpha
        self.max_people = max_people
        self.states = [None] * max_people  # List of [17, 3] arrays

    def filter(self, all_keypoints: List[List[Dict]]) -> List[List[Dict]]:
        """
        Ingests a list of skeletons (each a list of 17 keypoint dicts),
        updates the EMA states, and returns all smoothed skeletons.
        """
        results = []
        for person_idx, keypoints in enumerate(all_keypoints):
            if person_idx >= self.max_people:
                break

            if not keypoints or len(keypoints) != 17:
                results.append(keypoints)
                continue

            # Extract just x, y, z into a numpy array for vectorized math
            current_pts = np.zeros((17, 3), dtype=np.float32)
            for i, kp in enumerate(keypoints):
                current_pts[i, 0] = kp["x"]
                current_pts[i, 1] = kp["y"]
                current_pts[i, 2] = kp["z"]
            
            if self.states[person_idx] is None:
                self.states[person_idx] = current_pts.copy()
            else:
                self.states[person_idx] = self.alpha * current_pts + (1.0 - self.alpha) * self.states[person_idx]

            # Reconstruct the list of dicts with the smoothed coordinates
            smoothed_kps = []
            for i in range(17):
                smoothed_kps.append({
                    "x": float(self.states[person_idx][i, 0]),
                    "y": float(self.states[person_idx][i, 1]),
                    "z": float(self.states[person_idx][i, 2]),
                    "confidence": keypoints[i]["confidence"]
                })
            results.append(smoothed_kps)
                
        return results
