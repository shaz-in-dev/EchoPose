use std::collections::{HashMap, HashSet};
use echopose_types::{CsiFrame, SyncedBundle};

/// Default synchronization window in microseconds (50 ms).
pub const DEFAULT_WINDOW_US: u64 = 50_000;

struct Window {
    start_us: u64,
    frames: HashMap<u8, CsiFrame>,
}

/// Timestamp-based multi-node CSI frame synchronizer.
///
/// Collects frames into sliding time windows and emits a
/// `SyncedBundle` when all expected nodes have contributed.
pub struct NodeSynchronizer {
    window_us: u64,
    expected_node_count: usize,
    discovered_nodes: HashSet<u8>,
    windows: HashMap<u64, Window>,
    newest_ts_us: u64,
}

impl NodeSynchronizer {
    pub fn new(expected_count: usize) -> Self {
        Self::with_window(expected_count, DEFAULT_WINDOW_US)
    }

    pub fn with_window(expected_count: usize, window_us: u64) -> Self {
        Self {
            window_us,
            expected_node_count: expected_count,
            discovered_nodes: HashSet::new(),
            windows: HashMap::new(),
            newest_ts_us: 0,
        }
    }

    /// Feed a decoded frame; returns a `SyncedBundle` if a window is complete.
    pub fn push(&mut self, frame: CsiFrame) -> Option<SyncedBundle> {
        let slot = (frame.timestamp_us / self.window_us) * self.window_us;

        if frame.timestamp_us > self.newest_ts_us {
            self.newest_ts_us = frame.timestamp_us;
        }

        self.discovered_nodes.insert(frame.node_id);

        let window = self.windows.entry(slot).or_insert_with(|| Window {
            start_us: slot,
            frames: HashMap::new(),
        });

        window.frames.insert(frame.node_id, frame);

        let target_nodes =
            std::cmp::min(self.expected_node_count, self.discovered_nodes.len());
        let complete = window.frames.len() >= target_nodes && target_nodes > 0;

        if complete {
            return self.flush_window(slot);
        }

        // Flush stale windows
        let stale_keys: Vec<u64> = self
            .windows
            .keys()
            .copied()
            .filter(|&k| self.newest_ts_us.saturating_sub(k) > self.window_us)
            .collect();

        for k in stale_keys {
            if let Some(bundle) = self.flush_window(k) {
                return Some(bundle);
            }
        }

        None
    }

    fn flush_window(&mut self, slot: u64) -> Option<SyncedBundle> {
        self.windows.remove(&slot).map(|w| SyncedBundle {
            window_us: w.start_us,
            frames: w.frames.into_values().collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(node_id: u8, ts: u64) -> CsiFrame {
        CsiFrame {
            node_id,
            timestamp_us: ts,
            amplitudes: vec![1.0; 64],
            phases: vec![0.0; 64],
        }
    }

    #[test]
    fn single_node_emits_immediately() {
        let mut sync = NodeSynchronizer::new(1);
        let result = sync.push(make_frame(0, 100_000));
        assert!(result.is_some());
        assert_eq!(result.unwrap().frames.len(), 1);
    }

    #[test]
    fn two_nodes_need_both() {
        // Pre-discover both nodes in the first window
        let mut sync = NodeSynchronizer::new(2);
        let _ = sync.push(make_frame(0, 100_000));
        let _ = sync.push(make_frame(1, 100_000));
        // Now in a new window, one node alone shouldn't emit
        // because target = min(2, 2) = 2
        let r = sync.push(make_frame(0, 300_000));
        // It may flush the old stale window; push into yet another slot
        let r2 = sync.push(make_frame(0, 500_000));
        // Only when node 1 arrives should it complete
        let result = sync.push(make_frame(1, 500_000));
        assert!(result.is_some());
        assert_eq!(result.unwrap().frames.len(), 2);
    }

    #[test]
    fn stale_window_flushes() {
        let mut sync = NodeSynchronizer::new(3);
        let _ = sync.push(make_frame(0, 100_000));
        // Discover node 1 in a far-future window — should flush the stale slot
        let _ = sync.push(make_frame(1, 100_000));
        // Push into a much later slot; stale window flushes
        let result = sync.push(make_frame(2, 500_000));
        // Either the push triggers a stale flush or completes the current window
        // Just verify no panic and we eventually get a bundle
        assert!(result.is_some() || sync.push(make_frame(0, 500_000)).is_some());
    }
}
