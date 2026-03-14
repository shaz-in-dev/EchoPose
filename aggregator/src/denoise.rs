use std::collections::{HashMap, VecDeque};
use crate::types::NUM_SUBCARRIERS;

pub const BACKGROUND_WINDOW: usize = 30;

/// Background subtractor that maintains a rolling median of CSI amplitudes.
/// This removes static reflections (walls/furniture) in real-time.
pub struct RollingDenoiser {
    // node_id -> [subcarrier_index] -> queue of amplitudes
    buffers: HashMap<u8, Vec<VecDeque<f32>>>,
}

impl RollingDenoiser {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    /// Subtracts the rolling background from the given amplitudes.
    /// Updates the rolling window with the new unfiltered values.
    pub fn denoise(&mut self, node_id: u8, amplitudes: &mut Vec<f32>) {
        let node_bufs = self.buffers.entry(node_id).or_insert_with(|| {
            (0..NUM_SUBCARRIERS)
                .map(|_| VecDeque::with_capacity(BACKGROUND_WINDOW))
                .collect()
        });

        for (i, amp) in amplitudes.iter_mut().enumerate() {
            if i >= NUM_SUBCARRIERS { break; }
            
            let buf = &mut node_bufs[i];
            
            // 1. Calculate background (median of the window)
            // If the buffer is empty or small, background is basically 0 or the current value
            let background = if buf.is_empty() {
                0.0
            } else {
                let mut sorted: Vec<f32> = buf.iter().copied().collect();
                sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                sorted[sorted.len() / 2]
            };

            // 2. Add current UNFILTERED value to the buffer for next time
            if buf.len() >= BACKGROUND_WINDOW {
                buf.pop_front();
            }
            buf.push_back(*amp);

            // 3. Subtract background (clamped to 0)
            *amp = (*amp - background).max(0.0);
        }
    }
}
