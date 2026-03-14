"""
inference/research/cross_polarization.py — Signal Fusing (Feature 17)

Combines Horizontal-Horizontal (HH), Horizontal-Vertical (HV), and 
Vertical-Vertical (VV) polarization states to drastically increase 
spatial resolution across intersecting targets.
"""

import numpy as np

class CrossPolarizationFusion:
    """
    NOVEL: First to use HH + HV + VV polarization CSI for multiperson pose.
    Usually requires hardware modding on Intel 5300 or custom SDRs.
    """
    
    def fuse_polarizations(self, hh: np.ndarray, hv: np.ndarray, vv: np.ndarray) -> np.ndarray:
        """
        Takes 3 complex CSI matrices and performs Principal Component Analysis (PCA)
        across the polarization dimension to extract the dominant human reflection
        vector independently of the antenna orientation.
        """
        if hh.shape != hv.shape or hv.shape != vv.shape:
            raise ValueError("All polarization matrices must share the same shape.")
            
        # Stack into [polarization, everything_else]
        stacked = np.stack([hh, hv, vv], axis=0) # [3, windows, subcarriers]
        
        # Calculate covariance across polarizations
        flat_stacked = stacked.reshape(3, -1)
        cov = np.cov(flat_stacked)
        
        # Eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # The principal eigenvector represents the dominant polarization alignment of the human body
        principal_vector = eigenvectors[:, 0]
        
        # Project the original signals onto the optimal polarization vector
        # This naturally suppresses clutter which usually has random polarization
        fused = (principal_vector[0] * hh) + (principal_vector[1] * hv) + (principal_vector[2] * vv)
        
        return np.abs(fused)
