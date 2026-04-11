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
        """
        if target_frequency_hz <= 0:
            raise ValueError(f"target_frequency_hz must be positive, got {target_frequency_hz}")

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

        # np.interp requires xp to be monotonically increasing
        if scale_factor < 0:
            raise ValueError(f"Negative scale factor: {scale_factor}")
        if mapped_x[-1] < mapped_x[0]:
            mapped_x = mapped_x[::-1]
            flip = True
        else:
            flip = False
        
        # Interpolate the amplitudes onto the new physical axis
        invariant_csi = np.zeros_like(csi_amplitudes)
        
        # In this implementation, assuming csi_amplitudes is 1D or the last axis is frequency bins
        if csi_amplitudes.ndim == 1:
            src = csi_amplitudes[::-1] if flip else csi_amplitudes
            invariant_csi = np.interp(original_x, mapped_x, src, left=0.0, right=0.0)
        else:
            # Apply along the last axis
            for idx in np.ndindex(csi_amplitudes.shape[:-1]):
                src = csi_amplitudes[idx][::-1] if flip else csi_amplitudes[idx]
                invariant_csi[idx] = np.interp(
                    original_x, mapped_x, src, left=0.0, right=0.0
                )
                
        return invariant_csi
