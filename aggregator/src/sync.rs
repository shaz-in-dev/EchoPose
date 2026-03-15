// ============================================================
// sync.rs — timestamp-based multi-node synchronisation
//
// Strategy: sliding 50ms time windows. Once all known nodes
// have contributed at least one frame to a window, emit a
// SyncedBundle to the output channel. Stale windows (>150ms
// old with incomplete data) are flushed with whatever arrived.
// ============================================================
use std::collections::{HashMap, HashSet};
use crate::types::{CsiFrame, SyncedBundle};

pub const WINDOW_US:      u64 = 50_000;   // 50 ms windows  → 20 Hz
pub const STALE_LIMIT_US: u64 = 50_000;   // flush after one window period — drives steady 20 Hz output

struct Window {
    start_us: u64,
    frames:   HashMap<u8, CsiFrame>, // node_id → latest frame in window
}

pub struct NodeSynchronizer {
    expected_node_count: usize,
    discovered_nodes:    HashSet<u8>,
    windows:             HashMap<u64, Window>, // slot_key → Window
    newest_ts_us:        u64,                  // highest timestamp seen = our clock
}

impl NodeSynchronizer {
    pub fn new(expected_count: usize) -> Self {
        Self {
            expected_node_count: expected_count,
            discovered_nodes:    HashSet::new(),
            windows:             HashMap::new(),
            newest_ts_us:        0,
        }
    }

    /// Feed a decoded frame; returns a SyncedBundle if the window is complete.
    pub fn push(&mut self, frame: CsiFrame) -> Option<SyncedBundle> {
        let slot = (frame.timestamp_us / WINDOW_US) * WINDOW_US;

        // Keep a running "clock" from the newest timestamp we've ever seen
        if frame.timestamp_us > self.newest_ts_us {
            self.newest_ts_us = frame.timestamp_us;
        }

        // Dynamically track new nodes
        self.discovered_nodes.insert(frame.node_id);

        let window = self.windows.entry(slot).or_insert_with(|| Window {
            start_us: slot,
            frames:   HashMap::new(),
        });

        // Keep newest frame per node per window
        window.frames.insert(frame.node_id, frame);

        // Complete if we hit the expected count, OR if we have frames from all nodes discovered so far
        let target_nodes = std::cmp::min(self.expected_node_count, self.discovered_nodes.len());
        let complete = window.frames.len() >= target_nodes && target_nodes > 0;

        if complete {
            return self.flush_window(slot);
        }

        // Flush stale windows using the newest-ever timestamp as the reference clock
        let stale_keys: Vec<u64> = self
            .windows
            .keys()
            .copied()
            .filter(|&k| self.newest_ts_us.saturating_sub(k) > STALE_LIMIT_US)
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
            frames:    w.frames.into_values().collect(),
        })
    }
}
