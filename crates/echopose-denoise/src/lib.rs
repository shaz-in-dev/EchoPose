use std::collections::{HashMap, VecDeque};
use echopose_types::NUM_SUBCARRIERS;

pub const BACKGROUND_WINDOW: usize = 30;

/// Background subtractor that maintains a rolling median of CSI amplitudes.
/// This removes static reflections (walls/furniture) in real-time.
pub struct RollingDenoiser {
    buffers: HashMap<u8, Vec<VecDeque<f32>>>,
}

impl RollingDenoiser {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    /// Subtracts the rolling background from the given amplitudes.
    pub fn denoise(&mut self, node_id: u8, amplitudes: &mut Vec<f32>) {
        let node_bufs = self.buffers.entry(node_id).or_insert_with(|| {
            (0..NUM_SUBCARRIERS)
                .map(|_| VecDeque::with_capacity(BACKGROUND_WINDOW))
                .collect()
        });

        for (i, amp) in amplitudes.iter_mut().enumerate() {
            if i >= NUM_SUBCARRIERS {
                break;
            }

            let buf = &mut node_bufs[i];

            let background = if buf.is_empty() {
                0.0
            } else {
                let mut sorted: Vec<f32> = buf.iter().copied().collect();
                sorted.sort_unstable_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                sorted[sorted.len() / 2]
            };

            if buf.len() >= BACKGROUND_WINDOW {
                buf.pop_front();
            }
            buf.push_back(*amp);

            *amp = (*amp - background).max(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn denoise_removes_static_component() {
        let mut d = RollingDenoiser::new();
        // Fill with constant background
        for _ in 0..BACKGROUND_WINDOW {
            let mut amps = vec![10.0; NUM_SUBCARRIERS];
            d.denoise(0, &mut amps);
        }
        // Now a frame with background + signal should subtract background
        let mut amps = vec![15.0; NUM_SUBCARRIERS];
        d.denoise(0, &mut amps);
        assert!((amps[0] - 5.0).abs() < 0.1);
    }

    #[test]
    fn denoise_clamps_to_zero() {
        let mut d = RollingDenoiser::new();
        for _ in 0..BACKGROUND_WINDOW {
            let mut amps = vec![10.0; NUM_SUBCARRIERS];
            d.denoise(0, &mut amps);
        }
        let mut amps = vec![5.0; NUM_SUBCARRIERS];
        d.denoise(0, &mut amps);
        assert!(amps[0] >= 0.0);
    }

    #[test]
    fn denoise_multi_node() {
        let mut d = RollingDenoiser::new();
        let mut a = vec![1.0; NUM_SUBCARRIERS];
        let mut b = vec![2.0; NUM_SUBCARRIERS];
        d.denoise(0, &mut a);
        d.denoise(1, &mut b);
        // Each node has independent buffers
        assert!(d.buffers.contains_key(&0));
        assert!(d.buffers.contains_key(&1));
    }
}
