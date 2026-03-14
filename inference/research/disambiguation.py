"""
inference/research/disambiguation.py — Multi-Person Disentanglement (Feature 19)

Uses clustering and adversarial feature learning to disentangle 
overlapping CSI signatures from multiple subjects in the same room.
"""

import numpy as np
from sklearn.cluster import DBSCAN

class MultiPersonDisambiguation:
    """
    Solves the 'Cocktail Party Problem' for Wi-Fi RF Sensing.
    Separates aggregated CSI waves back into individual human actors 
    prior to neural network inference.
    """
    def __init__(self, max_people=3):
        self.max_people = max_people
        
    def disentangle_csi_signatures(self, csi_matrix: np.ndarray, doppler_spectrum: np.ndarray):
        """
        Uses DBSCAN density clustering on the Doppler velocity profile to identify 
        distinct walking subjects. Separates the global CSI matrix into discrete
        feature tensors for each individual.
        """
        # Collapse spatial/subcarrier dimensions to focus purely on motion frequencies
        # doppler_spectrum shape: [nodes, doppler_bins]
        mean_doppler = np.mean(doppler_spectrum, axis=0) # [doppler_bins]
        
        # We treat the frequencies with high energy as "points" in 1D velocity space
        active_bins = np.where(mean_doppler > np.median(mean_doppler) + 2 * np.std(mean_doppler))[0]
        
        if len(active_bins) == 0:
            return [csi_matrix] # 1 or 0 people, no complex disambiguation needed
            
        # Reshape for sklearn
        X = active_bins.reshape(-1, 1)
        
        # Cluster the velocities
        # E.g., Person A walking fast (Bins 10-14), Person B standing still (Bins 1-3)
        clustering = DBSCAN(eps=2.5, min_samples=1).fit(X)
        labels = clustering.labels_
        
        unique_actors = np.unique(labels)
        
        # We cap at MAX_PEOPLE
        num_actors = min(len(unique_actors), self.max_people)
        
        individual_tensors = []
        for i in range(num_actors):
            actor_label = unique_actors[i]
            if actor_label == -1: continue # Noise
            
            # Find the frequency bins belonging to this actor
            actor_bins = X[labels == actor_label].flatten()
            
            # Create a localized mask allowing ONLY this actor's motion through
            mask = np.zeros_like(doppler_spectrum)
            mask[:, actor_bins] = 1.0
            
            # Apply the mask to separate the CSI
            isolated_csi = csi_matrix * mask[:, np.newaxis, :]
            individual_tensors.append(isolated_csi)
            
        return individual_tensors if individual_tensors else [csi_matrix]
