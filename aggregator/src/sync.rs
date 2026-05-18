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

fn read_window_us() -> u64 {
    std::env::var("AGGREGATOR_SYNC_WINDOW_MS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(50)
        * 1_000 // convert ms → µs
}

pub static WINDOW_US: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
pub static STALE_LIMIT_US: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

pub fn init_window_us() {
    let w = WINDOW_US.get_or_init(read_window_us);
    STALE_LIMIT_US.get_or_init(|| *w);
}

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
        let window_us = *WINDOW_US.get().unwrap_or(&50_000);
        let stale_limit_us = *STALE_LIMIT_US.get().unwrap_or(&50_000);

        // Cap how far ahead a new timestamp can push the reference clock (5 windows max)
        let max_forward_jump = 5 * window_us;
        let capped_ts = frame.timestamp_us.min(self.newest_ts_us.saturating_add(max_forward_jump));
        if capped_ts > self.newest_ts_us {
            self.newest_ts_us = capped_ts;
        }

        let slot = (capped_ts / window_us) * window_us;

        // Reject stale frames before inserting (prevents phantom detections from
        // nodes that drop out and rejoin with a backlog of buffered frames).
        let frame_age_us = self.newest_ts_us.saturating_sub(capped_ts);
        let max_age_us = 2 * window_us;
        if frame_age_us > max_age_us {
            return None;
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
            .filter(|&k| self.newest_ts_us.saturating_sub(k) > stale_limit_us)
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
