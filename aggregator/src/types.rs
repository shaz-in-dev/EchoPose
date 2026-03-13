// ============================================================
// types.rs — shared data structures
// ============================================================
use serde::{Deserialize, Serialize};

pub const CSI_MAGIC: u32     = 0x43534931; // "CSI1"
pub const NUM_SUBCARRIERS: usize = 64;

/// Binary-compatible with the ESP32 struct (little-endian, packed)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct RawCsiFrame {
    pub magic:           u32,
    pub node_id:         u16,
    pub timestamp_us:    u64,
    pub num_subcarriers: u16,
    pub iq_data:         [i16; NUM_SUBCARRIERS * 2], // interleaved I,Q
}

impl RawCsiFrame {
    pub const FRAME_SIZE: usize = std::mem::size_of::<RawCsiFrame>();

    /// Parse from raw UDP bytes. Returns None if magic or size invalid.
    pub fn from_bytes(buf: &[u8]) -> Option<Self> {
        if buf.len() < Self::FRAME_SIZE {
            return None;
        }
        // SAFETY: the struct is repr(C, packed) and we've verified size
        let frame = unsafe { std::ptr::read_unaligned(buf.as_ptr() as *const RawCsiFrame) };
        if u32::from_le(frame.magic) != CSI_MAGIC {
            return None;
        }
        Some(frame)
    }

    /// Complex amplitude for subcarrier i: sqrt(I² + Q²)
    pub fn amplitude(&self, i: usize) -> f32 {
        let re = self.iq_data[i * 2]     as f32;
        let im = self.iq_data[i * 2 + 1] as f32;
        (re * re + im * im).sqrt()
    }

    /// Phase angle for subcarrier i: atan2(Q, I)
    pub fn phase(&self, i: usize) -> f32 {
        let re = self.iq_data[i * 2]     as f32;
        let im = self.iq_data[i * 2 + 1] as f32;
        im.atan2(re)
    }
}

/// A decoded, heap-owned CSI frame with derived features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsiFrame {
    pub node_id:      u8,
    pub timestamp_us: u64,
    pub amplitudes:   Vec<f32>,   // length = NUM_SUBCARRIERS
    pub phases:       Vec<f32>,
}

impl From<&RawCsiFrame> for CsiFrame {
    fn from(raw: &RawCsiFrame) -> Self {
        let n = u16::from_le(raw.num_subcarriers) as usize;
        let amps  = (0..n).map(|i| raw.amplitude(i)).collect();
        let phases = (0..n).map(|i| raw.phase(i)).collect();
        CsiFrame {
            node_id:      u16::from_le(raw.node_id) as u8,
            timestamp_us: u64::from_le(raw.timestamp_us),
            amplitudes:   amps,
            phases,
        }
    }
}

/// A synchronized bundle of frames from all nodes,
/// aligned to the same 50 ms time window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncedBundle {
    pub window_us: u64,          // window start timestamp
    pub frames:    Vec<CsiFrame>, // one per node (may be <3 if a node dropped)
}
