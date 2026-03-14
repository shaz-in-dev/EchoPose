"""
pipeline/advanced_denoise.py — Multi-stage State-of-the-Art Denoising

Replaces the basic median background subtractor with:
1. Adaptive Wiener filtering
2. Daubechies Wavelet denoising
3. Spectral Subtraction (noise profile estimation)
4. STFT Time-Frequency decomposition
5. Outlier Rejection & Confidence Scoring
"""

import numpy as np
import scipy.signal as signal
import pywt
from collections import deque
from typing import List, Dict

class AdvancedDenoiser:
    """
    World-class CSI Signal Processing Pipeline.
    Outperforms standard filtering by fusing multi-domain mathematical transforms.
    """
    def __init__(self, num_nodes: int = 3, num_sub: int = 64, sample_hz: float = 20.0, stages: list = None):
        self.num_nodes = num_nodes
        self.num_sub = num_sub
        self.sample_hz = sample_hz
        self.window_size = 40
        self.fft_bins = 16
        
        self.stages = stages if stages else ['wiener', 'wavelet', 'spectral']
        
        # rolling buffers for each node's subcarriers
        self.buffers: Dict[int, List[deque]] = {
            nid: [deque(maxlen=self.window_size) for _ in range(num_sub)]
            for nid in range(num_nodes)
        }
        
    def push(self, node_id: int, amplitudes: List[float]) -> None:
        """Push a new frame into the window buffer."""
        buf = self.buffers.get(node_id)
        if buf is None: return
        for i, amp in enumerate(amplitudes[:self.num_sub]):
            buf[i].append(amp)

    def _apply_wiener(self, sig: np.ndarray) -> np.ndarray:
        """Adaptive Wiener filter to estimate the optimal linear filter for local noise."""
        # A 5-tap Wiener filter adapts to local variance.
        if len(sig) < 5: return sig
        return signal.wiener(sig, mysize=5)

    def _apply_wavelet(self, sig: np.ndarray) -> np.ndarray:
        """Daubechies Wavelet denoising using soft thresholding (Universal Threshold)."""
        if len(sig) < 8: return sig
        # Decompose signal using Daubechies 4 wavelet (db4)
        coeffs = pywt.wavedec(sig, 'db4', level=int(np.log2(len(sig)))-1)
        
        # Estimate noise variance from details (highest freq subband)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 
        uthresh = sigma * np.sqrt(2 * np.log(len(sig)))
        
        # Soft threshold the detail coefficients
        denoised_coeffs = [coeffs[0]] + [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs[1:]]
        
        # Reconstruct and strictly enforce length boundaries to prevent silent truncation data-loss
        reconstructed = pywt.waverec(denoised_coeffs, 'db4')
        if len(reconstructed) > len(sig):
            return reconstructed[:len(sig)]
        elif len(reconstructed) < len(sig):
            return np.pad(reconstructed, (0, len(sig) - len(reconstructed)), mode='edge')
        return reconstructed

    def _spectral_subtraction(self, sig: np.ndarray) -> np.ndarray:
        """
        Estimates the noise spectrum during periods of low variance and subtracts
        it from the phase-magnitude spectrum.
        """
        fs = self.sample_hz
        # Short-Time Fourier Transform (STFT)
        f, t, Zxx = signal.stft(sig, fs=fs, nperseg=min(16, len(sig)))
        
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # Estimate noise profile (lowest 15% power frames)
        frame_power = np.sum(mag**2, axis=0)
        noise_idx = np.argsort(frame_power)[:max(1, int(0.15 * len(frame_power)))]
        noise_profile = np.mean(mag[:, noise_idx], axis=1, keepdims=True)
        
        # Subtract noise profile with a flooring factor
        clean_mag = np.maximum(mag - 1.5 * noise_profile, 0.05 * noise_profile)
        
        # Reconstruct signal using Inverse STFT (ISTFT)
        _, clean_sig = signal.istft(clean_mag * np.exp(1j * phase), fs=fs)
        return clean_sig[:len(sig)]

    def compute_features(self) -> tuple[np.ndarray, dict]:
        """
        Processes multi-stage denoising across all stored buffers.
        Returns:
            features: [num_nodes, num_sub, FFT_BINS]
            confidence_scores: Dict mapping node -> average feature confidence
        """
        freqs    = np.fft.rfftfreq(self.window_size, d=1.0 / self.sample_hz)
        lo_idx   = int(np.searchsorted(freqs, 0.3))
        hi_idx   = int(np.searchsorted(freqs, 10.0))
        out_size = self.fft_bins

        features = np.zeros((self.num_nodes, self.num_sub, out_size), dtype=np.float32)
        node_confidence = {}

        for nid in range(self.num_nodes):
            buf = self.buffers[nid]
            confidences = []
            
            for sub in range(self.num_sub):
                data = list(buf[sub])
                if len(data) < 20: 
                    continue # Not enough data to confidently process
                    
                arr = np.array(data, dtype=np.float32)
                
                # Dynamic outlier rejection (Z-score > 3)
                mean, std = np.mean(arr), np.std(arr)
                if std > 0:
                    arr = np.where(np.abs((arr - mean) / std) > 3, mean, arr)

                # PIPELINE EXECUTION
                clean_sig = arr
                if 'wiener' in self.stages:
                    clean_sig = self._apply_wiener(clean_sig)
                if 'wavelet' in self.stages:
                    clean_sig = self._apply_wavelet(clean_sig)
                if 'spectral' in self.stages:
                    clean_sig = self._spectral_subtraction(clean_sig)
                
                # Final Feature Extraction (Doppler PSD via Hanning Window)
                window = np.hanning(len(clean_sig))
                spectrum = np.abs(np.fft.rfft(clean_sig * window, n=self.window_size)) ** 2
                band = spectrum[lo_idx:hi_idx]
                
                if len(band) > 0:
                    # Resize to dense feature vector space
                    resized = np.interp(
                        np.linspace(0, len(band) - 1, out_size),
                        np.arange(len(band)),
                        band
                    )
                    mx = resized.max()
                    if mx > 0: resized /= mx
                    features[nid, sub] = resized
                    
                # Compute subcarrier measurement confidence (Entropy of spectrum)
                if np.sum(spectrum) > 0:
                    prob = spectrum / np.sum(spectrum)
                    entropy = -np.sum(prob * np.log2(prob + 1e-12))
                    confidences.append(1.0 / (1.0 + entropy)) # Lower entropy = sharper Doppler = higher conf

            node_confidence[nid] = np.mean(confidences) if confidences else 0.0
            
        return features, node_confidence
