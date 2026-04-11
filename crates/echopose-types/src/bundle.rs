use serde::{Deserialize, Serialize};
use crate::frame::CsiFrame;

/// A synchronized bundle of frames from all nodes,
/// aligned to the same 50 ms time window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncedBundle {
    pub window_us: u64,
    pub frames: Vec<CsiFrame>,
}
