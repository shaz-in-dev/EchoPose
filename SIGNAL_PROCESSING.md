# Signal Processing Mathematics (EchoPose V2)

The EchoPose pipeline combines multi-domain mathematical transforms with deep learning for WiFi-based pose estimation.

## 1. Production Denoising Pipeline
The production signal processing chain is implemented in `inference/pipeline/advanced_denoise.py` and executes these stages in order (configurable via the `stages` parameter):

1. **Adaptive Wiener Filtering** — A 5-tap filter that adapts to local noise variance.
2. **Daubechies Wavelet Denoising (db4)** — Decomposes signals into frequency subbands and applies soft thresholding to remove noise while preserving micro-Doppler shifts from human limbs.
3. **Spectral Subtraction** — Estimates the noise spectrum from low-power frames and subtracts it from the signal via STFT/ISTFT.
4. **FFT Doppler Power Spectrum** — Extracts the final Doppler PSD features for neural network input.

## 2. Adversarial Robustness
When the direct line-of-sight (LOS) path is blocked, `RobustCSIProcessor` in `inference/pipeline/robust_processing.py` detects this via the Coefficient of Variation (CV) across subcarriers:

`CV = sqrt(Var(Amplitude)) / Mean(Amplitude)`

If `CV > 0.8`, the system automatically triggers **Multipath Exploitation** (`exploit_multipath()`), converting delayed signal bounces into primary feature vectors via an Inverse FFT Power Delay Profile (PDP).

## 3. Multi-Person Disambiguation
When multiple people are present, their Doppler signatures overlap. The `MultiPersonDisambiguation` class in `inference/research/disambiguation.py` applies **DBSCAN Density Clustering** on the Doppler velocity profile to separate individual motion signatures before inference.

This module is integrated into the production `FusionPipeline` (`inference/pipeline/fusion.py`).

## Research Modules (Experimental)
The following modules exist under `inference/research/` and are **not yet integrated** into the live inference pipeline. They are experimental implementations intended for future development:

| Module | Description | Status |
|--------|-------------|--------|
| `adversarial_cert.py` | Randomized smoothing robustness certification | Research only |
| `cross_polarization.py` | HH/HV/VV polarization fusion (requires custom hardware) | Research only |
| `frequency_transfer.py` | 2.4/5/6 GHz frequency-invariant transfer learning | Research only |
| `domain_adaptation.py` | MMD-based online domain adaptation | Research only |
| `domain_shift_monitor.py` | Ensemble variance domain-shift detection | Research only |
| `pinn.py` | Physics-Informed Neural Network training loss | Research only |
