# EchoPose Architecture

EchoPose V2 is a full-stack WiFi CSI pose estimation system, decoupled into a high-performance Rust aggregation layer and a Python/ONNX inference layer.

## The Tri-Node Geometry
The system requires exactly 3 ESP32-S3 devices running custom native C firmware. 
- **Node 0 (Tx/Rx):** Central Illuminator
- **Node 1 (Rx):** Left Receiver
- **Node 2 (Rx):** Right Receiver

These form a uniform bounding box around the target room.

## Data Flow (Latency: <40ms End-to-End)
1. **Physical Layer (ESP32-S3):** Captures 802.11n CSI matrices at 20Hz.
2. **Transport Layer (Network):** UDP broadcast strips standard TCP overhead, streaming directly to the aggregator.
3. **Aggregation Layer (Rust):** Synchronizes rogue UDP packets into atomic `[3, 64]` timestamped tensors based on `mac_id`. Out-of-order packets are dropped to ensure strict temporal linearity.
4. **Signal Processing Pipeline (Python):** Wavelet, Wiener, and Spectral Subtraction filters denoise the ambient room reflections.
5. **Neural Inference (ONNX):** Multi-scale CNNs combined with Temporal LSTMs extract 17 3D skeletal keypoints.
6. **Presentation Layer (JS/Three.js):** 60FPS fluid rendering mapped to browser WebSockets.

## Scaling Limits
- Maximum Concurrent Persons tracked: 3 (via DBSCAN disambiguation + multi-head regression)
- Subcarrier Depth: 64 per antenna (ESP32-S3 hardware limit)
- Refresh Rate: Max 20Hz 

## Research Modules
Several experimental modules exist under `inference/research/` for future development.
These are **not** integrated into the live pipeline — see `SIGNAL_PROCESSING.md` for details.

To dive deeper into the mathematics behind the pipeline, read `SIGNAL_PROCESSING.md`.
