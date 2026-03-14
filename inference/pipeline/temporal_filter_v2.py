"""
pipeline/temporal_filter_v2.py — Kalman-like Physics-Aware Smoothing (Feature 4)

Far superior to basic Exponential Moving Average (EMA).
Predicts where joints *should* be based on velocity, momentum, and physical limits,
blending predictions with actual measurements weighted by confidence scores.
"""

import numpy as np
from collections import deque
from math import exp

class TemporalPoseFilterV2:
    """ Physics-informed kalman filter for human biomechanics """
    def __init__(self, max_people=3):
        self.max_people = max_people
        # Keep tracking states for each person independently
        self.states = [None] * max_people
        self.velocities = [None] * max_people
        self.confidence_history = [deque(maxlen=10) for _ in range(max_people)]
        
    def filter(self, skeletons: list) -> list:
        """
        Applies physics-aware smoothing to an array of incoming skeletons.
        Each skeleton is a list of 17 keypoint dicts: {x, y, z, confidence}
        """
        smoothed_skeletons = []
        
        for p_idx in range(self.max_people):
            if p_idx >= len(skeletons):
                continue
                
            person = skeletons[p_idx]
            if not person:
                smoothed_skeletons.append([])
                continue
                
            # Convert incoming list of dicts to flat numpy arrays for fast math
            measured_coords = np.array([[kp["x"], kp["y"], kp["z"]] for kp in person], dtype=float)
            measured_confs  = np.array([kp["confidence"] for kp in person], dtype=float)
            
            # Initialize State First Frame
            if self.states[p_idx] is None:
                self.states[p_idx] = measured_coords
                self.velocities[p_idx] = np.zeros_like(measured_coords)
                self.confidence_history[p_idx].append(np.mean(measured_confs))
                
                smoothed_skeletons.append([{
                    "x": float(c[0]), "y": float(c[1]), "z": float(c[2]), "confidence": float(cf)
                } for c, cf in zip(measured_coords, measured_confs)])
                continue
                
            # --- The Kalman-like Math ---
            
            # 1. Physics: Predict where the skeleton SHOULD be based on previous momentum
            predicted = self.states[p_idx] + self.velocities[p_idx]
            
            # 2. Measurement Innovation (Difference between reality and physics prediction)
            innovation = measured_coords - predicted
            
            # 3. Dynamic Gain Control 
            # High confidence -> Trust measurement
            # Low confidence  -> Trust physics momentum
            # Using sigmoid activation smoothed by measurement confidence
            mean_conf = np.mean(measured_confs)
            gain = 1.0 / (1.0 + np.exp(-10.0 * (mean_conf - 0.5))) # Scales smoothly [0,1]
            gain_matrix = gain * measured_confs[:, np.newaxis] # Apply individual joint confidence weight
            
            # 4. Update the actual position state
            self.states[p_idx] = predicted + gain_matrix * innovation
            
            # 5. Update the momentum/velocity state (maintain 70% of old momentum, add 30% of new force)
            self.velocities[p_idx] = 0.7 * self.velocities[p_idx] + 0.3 * innovation
            
            # 6. Physical Anomaly Detection
            # If a joint "teleports" (moves more than 50% of the screen in 50ms), nuke it
            anomalies = np.linalg.norm(innovation, axis=1) > 0.5
            measured_confs[anomalies] = 0.0 # Force zero confidence rendering on glitches
            
            self.confidence_history[p_idx].append(np.mean(measured_confs))
            
            # 7. Format Output correctly for Three.js
            smoothed_skeletons.append([{
                "x": float(c[0]), "y": float(c[1]), "z": float(c[2]), "confidence": float(cf)
            } for c, cf in zip(self.states[p_idx], measured_confs)])
            
        return smoothed_skeletons
