"""
inference/research/cross_polarization.py — Polarization Fusion (Feature 17) [RESEARCH]

Combines Horizontal-Horizontal (HH), Horizontal-Vertical (HV), and 
Vertical-Vertical (VV) polarization states to increase spatial resolution.

HARDWARE NOTE: This requires multi-polarization antennas (e.g., Intel 5300 NIC
with hardware modifications or custom SDR). The standard ESP32-S3 single-chain
radio captures only a single polarization state and CANNOT produce the HH/HV/VV
inputs needed by this module. This is a research prototype for future hardware.
"""

import numpy as np

class CrossPolarizationFusion:
    """
    Research prototype: HH + HV + VV polarization CSI fusion for multiperson pose.
    Requires custom multi-polarization hardware not present in the standard ESP32-S3 BOM.
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
