"""
inference/research/frequency_transfer.py — Frequency Transfer (Feature 18)

Maps CSI into a normalized frequency-invariant space, allowing models trained 
on 5GHz to instantly generalize to 2.4GHz and 6GHz bands without retraining.
"""

import numpy as np

class FrequencyDomainTransferLearning:
    """
    NOVEL: Single Model Deployment across varying WiFi bands
    """
    
    def __init__(self):
        # The training domain center frequency (e.g., 5.2GHz WiFi Channel 36)
        self.source_frequency_hz = 5.2e9 
        
    def normalize_to_frequency_invariant_space(self, csi_amplitudes: np.ndarray, target_frequency_hz: float):
        """
        Transforms CSI to frequency-independent representation.
        Since Doppler shift (Hz) varies based on the carrier frequency for the SAME 
        human walking speed, we must normalize the frequency shifts into 
        carrier-agnostic Velocity (m/s) shifts.
        """
        c = 3e8 # Speed of light m/s
        
        # The lambda (wavelength) changes drastically between 2.4Ghz and 5Ghz
        source_lambda = c / self.source_frequency_hz
        target_lambda = c / target_frequency_hz
        
        # The scaling factor required to project the target physics back to the source physics
        scale_factor = target_lambda / source_lambda
        
        # To shift the FFT bins, we perform an interpolation. 
        # If scale_factor > 1 (e.g., 2.4GHz -> 5GHz), the Doppler shifts at 2.4GHz 
        # are "compressed" compared to 5GHz. We must stretch them out.
        
        target_bins = csi_amplitudes.shape[-1]
        original_x = np.arange(target_bins)
        
        # Scale the x-axis (frequency bins)
        mapped_x = original_x * scale_factor
        
        # Interpolate the amplitudes onto the new physical axis
        invariant_csi = np.zeros_like(csi_amplitudes)
        
        # In this implementation, assuming csi_amplitudes is 1D or the last axis is frequency bins
        if csi_amplitudes.ndim == 1:
            invariant_csi = np.interp(original_x, mapped_x, csi_amplitudes, left=0.0, right=0.0)
        else:
            # Apply along the last axis
            for idx in np.ndindex(csi_amplitudes.shape[:-1]):
                invariant_csi[idx] = np.interp(
                    original_x, mapped_x, csi_amplitudes[idx], left=0.0, right=0.0
                )
                
        return invariant_csi
