"""
pipeline/denoise.py — FFT-based background subtraction & Doppler extraction

Converts a multi-node CSI bundle into a clean motion feature matrix.

Steps:
  1. Accumulate amplitude windows per subcarrier per node.
  2. Subtract the static background (median over last N frames = the "wall").
  3. Run a short-time FFT → Doppler spectrum (frequencies of motion).
  4. Return a feature tensor [nodes, subcarriers, doppler_bins].
"""

import numpy as np
from collections import deque
from typing import List, Dict

WINDOW_SIZE    = 40   # frames to keep in rolling window (~2 s at 20 Hz)
BACKGROUND_N   = 30   # frames used to estimate static background
FFT_BINS       = 16   # output Doppler frequency bins (0..10 Hz range)
MOTION_LO_HZ  = 0.3  # ignore DC / static reflections below this
MOTION_HI_HZ  = 10.0 # ignore high-freq noise above this


class CsiDenoiser:
    """
    Maintains a rolling amplitude buffer per node+subcarrier.
    Produces denoised Doppler feature vectors from synced bundles.
    """

    def __init__(self, num_nodes: int = 3, num_sub: int = 64, sample_hz: float = 20.0):
        self.num_nodes  = num_nodes
        self.num_sub    = num_sub
        self.sample_hz  = sample_hz
        # buffer[node_id][subcarrier] → deque of amplitude floats
        self.buffers: Dict[int, List[deque]] = {
            nid: [deque(maxlen=WINDOW_SIZE) for _ in range(num_sub)]
            for nid in range(num_nodes)
        }

    def push(self, node_id: int, amplitudes: List[float]) -> None:
        """Add one frame's amplitudes to the rolling buffer."""
        buf = self.buffers.get(node_id)
        if buf is None:
            return
        for i, amp in enumerate(amplitudes[:self.num_sub]):
            buf[i].append(amp)

    def compute_features(self) -> np.ndarray:
        """
        Returns ndarray [num_nodes, num_sub, FFT_BINS] of Doppler power.
        Values are zero if a node doesn't have enough data yet.
        """
        freqs    = np.fft.rfftfreq(WINDOW_SIZE, d=1.0 / self.sample_hz)
        lo_idx   = int(np.searchsorted(freqs, MOTION_LO_HZ))
        hi_idx   = int(np.searchsorted(freqs, MOTION_HI_HZ))
        out_size = FFT_BINS

        features = np.zeros((self.num_nodes, self.num_sub, out_size), dtype=np.float32)

        for nid in range(self.num_nodes):
            buf = self.buffers[nid]
            for sub in range(self.num_sub):
                data = list(buf[sub])
                if len(data) < BACKGROUND_N:
                    continue  # not enough data yet

                arr = np.array(data, dtype=np.float32)

                # 1. Background subtraction: remove median (static "wall")
                background  = np.median(arr[:BACKGROUND_N])
                motion_sig  = arr - background

                # 2. Hanning window to reduce spectral leakage
                window  = np.hanning(len(motion_sig))
                windowed = motion_sig * window

                # 3. FFT → power spectrum
                spectrum = np.abs(np.fft.rfft(windowed, n=WINDOW_SIZE)) ** 2

                # 4. Slice to motion band and resize to FFT_BINS
                band    = spectrum[lo_idx:hi_idx]
                if len(band) == 0:
                    continue
                resized = np.interp(
                    np.linspace(0, len(band) - 1, out_size),
                    np.arange(len(band)),
                    band,
                )

                # 5. Normalise per-subcarrier to [0,1]
                mx = resized.max()
                if mx > 0:
                    resized /= mx

                features[nid, sub] = resized

        return features  # shape: [nodes, subcarriers, doppler_bins]
