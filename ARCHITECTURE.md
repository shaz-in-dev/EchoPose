# EchoPose Architecture

EchoPose V3 represents the absolute pinnacle of zero-vision human tracking. We have completely decoupled the stack into a high-performance Rust hardware layer and an asynchronously scaled Python/PyTorch inference layer.

## The Tri-Node Geometry
The system requires exactly 3 ESP32-S3 devices running custom native C firmware. 
- **Node 0 (Tx/Rx):** Central Illuminator
- **Node 1 (Rx):** Left Receiver
- **Node 2 (Rx):** Right Receiver

These form a uniform bounding box around the target room.

## Data Flow (Latency: <40ms End-to-End)
1. **Physical Layer (ESP32-S3):** Captures 802.11n CSI matrices at 100Hz.
2. **Transport Layer (Network):** UDP broadcast strips standard TCP overhead, streaming directly to the aggregator.
3. **Aggregation Layer (Rust):** Synchronizes rogue UDP packets into atomic `[3, 64]` timestamped tensors based on `mac_id`. Out-of-order packets are dropped to ensure strict temporal linearity.
4. **Signal Processing Pipeline (Python):** Wavelet, Wiener, and Spectral Subtraction filters denoise the ambient room reflections.
5. **Neural Inference (PyTorch/TensorRT):** Multi-scale CNNs combined with Temporal LSTMs extract 17 3D skeletal keypoints.
6. **Presentation Layer (React/Three.js):** 60FPS fluid rendering mapped to browser WebSockets.

## Scaling Limits
- Maximum Concurrent Persons tracked: 3 (Hardcoded limit for real-time physics constraints)
- Subcarrier Depth: Up to 1024 (Current ESP32 limit is 64 per antenna)
- Refresh Rate: Max 100Hz 

To dive deeper into the mathematics behind the pipeline, read `SIGNAL_PROCESSING.md`.
