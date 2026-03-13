# EchoPose (formerly RF-Inference Mesh)

An advanced, full-stack Wi-Fi sensing system that treats radio waves like invisible sonar. EchoPose detects, tracks, and renders human poses **through walls** in real-time using only ESP32-S3 nodes and your standard computer hardware.

## V2 Features

EchoPose V2 is a hardened, production-ready system featuring:
- **Lightning-Fast ONNX Inference:** Heavily optimized CPU/Edge execution using `onnxruntime`.
- **Buttery-Smooth Tracking:** Temporal Exponential Moving Average (EMA) filters eliminate signal jitter.
- **Dynamic Node Discovery:** The Rust aggregator automatically detects and registers ESP32 nodes as they power on.
- **Room Environment Calibration:** A built-in `/calibrate` engine learns the room's static noise floor (walls, furniture) and subtracts it from live traffic.
- **Over-The-Air (OTA) Updates:** Flash ESP32 firmware directly over the Wi-Fi mesh.
- **Session Recording:** Save tracking data to local JSON files and replay them in the 3D dashboard.

---

## Architecture Stack

| Layer | Technology | Role |
|-------|------------|------|
| **Firmware** | C / ESP-IDF | Runs on ESP32-S3s. Captures 64-subcarrier I/Q CSI at 20 Hz. Streams binary UDP. |
| **Aggregator** | Rust / Axum | Receives UDP, aligns frames into 50ms windows, calibrates background noise, broadcasts via WS. |
| **Inference** | Python / ONNX | FFT background subtraction → Doppler features → ONNX PoseNet → 17 COCO keypoints. |
| **UI** | JS / Three.js | Connects to inference WS, renders real-time 3D skeleton + CSI Heatmap + Records Sessions. |

---

## Quick Start (Simulation Mode)

Want to see it in action without ESP32 hardware? Use the included Python simulator.

### 1. Start the Aggregator (Rust)
```powershell
cd aggregator
cargo run --release
```

### 2. Start the Inference Engine (Python)
```bash
cd inference
pip install -r requirements.txt
python server.py
```

### 3. Start the Hardware Simulator
```bash
cd scripts
python mock_esp32_mesh.py
```

### 4. Launch the Dashboard
Open `ui/index.html` in your web browser, click **Connect**, and watch the 3D skeleton track the simulated Doppler pulses!

---

## Production Deployment

For real-world hardware deployment:

1. **Configure:** Edit the central `.env` file in the project root.
2. **Flash:** Build and flash the `firmware/` C project to your ESP32-S3 nodes. Set `CONFIG_HOST_IP` to your Aggregator's IP.
3. **Deploy Backend:** Run `docker-compose up -d --build` to launch the Rust and Python servers in production Gunicorn/Release containers.
4. **Calibrate:** Access the UI, clear the room, and hit the `/calibrate` endpoint to subtract static reflections.

---

## Hardware Requirements

| Part | Qty | Purpose |
|------|-----|---------|
| ESP32-S3 (U.FL) | 3–6 | CSI capture nodes |
| SMA antennas | 3–6 | Better directional gain |
| Dedicated 2.4 GHz router | 1 | Silent AP (no other traffic) |
| Host PC (GPU optional) | 1 | Runs aggregator + inference |

---

## CSI Frame Format (binary, little-endian)

```
Bytes  0–3    magic           uint32   0x43534931 ("CSI1")
Bytes  4–5    node_id         uint16
Bytes  6–13   timestamp_us    uint64   µs since ESP boot
Bytes 14–15   num_subcarriers uint16   (always 64)
Bytes 16–N    iq_data         int16[]  interleaved I, Q pairs
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
