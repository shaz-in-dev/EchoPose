# EchoPose

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.88+-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![C](https://img.shields.io/badge/C-ESP--IDF-A8B9CC?logo=c&logoColor=white)](https://www.espressif.com/en/products/sdks/esp-idf)
[![JavaScript](https://img.shields.io/badge/JavaScript-UI-F7DF1E?logo=javascript&logoColor=black)](https://developer.mozilla.org/docs/Web/JavaScript)
[![PyPI](https://img.shields.io/pypi/v/echopose-sdk?logo=pypi&logoColor=white)](https://pypi.org/project/echopose-sdk/)
[![crates.io](https://img.shields.io/crates/v/echopose-types?logo=rust&logoColor=white)](https://crates.io/crates/echopose-types)
[![GHCR](https://img.shields.io/badge/GHCR-live-181717?logo=github&logoColor=white)](https://github.com/shaz-in-dev/EchoPose/pkgs/container/echopose-aggregator)

An ops-focused, full-stack Wi-Fi sensing system that treats radio waves like invisible sonar. EchoPose detects, tracks, and renders human poses in real time using ESP32-S3 nodes and commodity compute.

## Why EchoPose Exists

EchoPose is built for operations-first Wi-Fi sensing with rapid iteration and measurable robustness:
- **Operational deployment first:** Docker Compose, Kubernetes, and systemd paths in one repo.
- **Analytics beyond pose:** activity, fall-risk, occupancy, and tactical situational layers.
- **Research-to-production bridge:** experimental modules live beside a running end-to-end stack.

## Current Status

- **Automated tests:** 256 collected, 256 passing in CI.
- **Architecture records:** 10 ADRs in [docs/adr/README.md](docs/adr/README.md).
- **Rust workspace:** 6 crates (shared types + denoise + sync + localize + aggregator + WASM).
- **WASM pipeline:** browser bridge implemented in [ui/wasm_bridge.js](ui/wasm_bridge.js); build scripts in [scripts/build_wasm.ps1](scripts/build_wasm.ps1) and [scripts/build_wasm.sh](scripts/build_wasm.sh).
- **Publishing:** release workflows for Python, Rust, and Docker are active and release-triggered.
- **Registry releases:** [`echopose-sdk` on PyPI](https://pypi.org/project/echopose-sdk/) is live, EchoPose crates are live on crates.io, and container images are published to GHCR.

## Maturity Tracks

- [Differentiation strategy](DIFFERENTIATION_STRATEGY.md)
- [Level-up master plan](LEVEL_UP_MASTER_PLAN.md)
- [Docs hub](docs/README.md)
- [ADR index](docs/adr/README.md)
- [Release runbook](docs/runbooks/release.md)
- [Cross-environment benchmark harness](benchmarks/cross_environment_generalization.py)
- [Signed model bundle module](inference/research/signed_model_bundle.py)
- [Continual personalization module](inference/research/continual_personalization.py)
- [Hardware normalization module](inference/pipeline/hardware_normalization.py)
- [Browser WASM runtime](ui/wasm/README.md)
- [Legacy v1 and proof system](v1/README.md)
- [Python package scaffold (`echopose-sdk`)](echopose_sdk/README.md)

## V2 Features

EchoPose V2 is a production-track Wi-Fi CSI pose estimation system featuring:
- **Trained PoseNetV2 Model:** Multi-scale CNN + LSTM + Attention architecture with a trained PyTorch checkpoint (`models/pose_net.pt`).
- **ONNX Inference Path:** CPU/Edge execution via `onnxruntime` with automatic fallback to PyTorch if ONNX is unavailable.
- **Buttery-Smooth Tracking:** Temporal Exponential Moving Average (EMA) filters eliminate signal jitter.
- **Dynamic Node Discovery:** The Rust aggregator automatically detects and registers ESP32 nodes as they power on.
- **Room Environment Calibration:** A built-in `/calibrate` engine learns the room's static noise floor (walls, furniture) and subtracts it from live traffic.
- **Over-The-Air (OTA) Updates:** Flash ESP32 firmware directly over the Wi-Fi mesh.
- **Session Recording:** Save tracking data to local JSON files and replay them in the 3D dashboard.
- **Rate Limiting & API Key Auth:** All REST endpoints are protected by per-IP rate limiting (60 req/s) and API key verification.
- **Accuracy Validation:** Built-in `scripts/validate_accuracy.py` computes MPJPE, PCK, and per-joint metrics.

### Health Metrics & Vitals (NEW)
- **Heart Rate Detection:** Chest micro-Doppler FFT extracts 40–180 bpm HR from CSI subcarriers 30–40.
- **Respiratory Rate:** Thorax motion analysis detects 6–60 breaths/min via bandpass + Welch PSD.
- **SpO2 / Temperature / Blood Pressure Proxies:** Experimental inference-only research signals.

> **Medical Safety Notice:** EchoPose is not a medical device and must not be used for diagnosis, treatment, or emergency clinical decisions.

### Activity & Analytics (NEW)
- **Gait Analysis:** Walking speed, stride length, cadence, and gait symmetry from hip keypoint trajectories.
- **Activity Classification:** Standing / Walking / Running / Sitting / Lying from skeleton geometry + Doppler.
- **Fall Detection:** Centre-of-mass velocity monitoring with balance risk scoring and CRITICAL alerts.
- **Exercise Counting:** Rep detection for push-ups, squats, jumping jacks, sit-ups via joint angle cycles.
- **Gesture Recognition:** Wave, point, raise, swipe detection from wrist trajectory analysis.
- **Occupancy Analytics:** Multi-method presence detection (skeleton + CSI energy + vital frequencies).
- **Sleep Stage Classification:** Awake / N1 / N2 / N3 / REM from immobility + HRV + breathing regularity.
- **Emotion & Stress Estimation:** 0–100 stress score from HR elevation, breathing rate, and postural cues.
- **Health Anomaly Alerts:** Context-aware vital sign monitoring with NORMAL / WARNING / CRITICAL levels.

### Advanced Security & Situational Awareness (NEW)
- **Through-Wall Target Tracking:** Multi-target detection and tracking through solid structures using CSI Doppler signatures.
- **Crowd Analytics:** Real-time crowd density estimation, flow direction, and spectral occupancy analysis.
- **Posture Classification:** Detect standing, crouching, prone, crawling, and surrender postures from skeleton geometry.
- **Acoustic Event Detection:** Impulse and anomaly detection from CSI amplitude transients (gunshot-like, explosion-like, glass-break).
- **Behavioural Anomaly Scanning:** Calibrated baseline comparison to flag unusual movement or environmental changes.
- **Intent Prediction:** Movement trajectory and posture-based intent forecasting (approaching, retreating, loitering, evading).
- **Anti-Jamming & RF Integrity:** Real-time detection of signal jamming, spoofing, and interference with alert logging.
- **Coverage Planning:** Sensor placement optimization with wall-aware signal propagation modeling.
- **Multi-Domain Sensor Fusion:** Combines WiFi CSI, acoustic, and RF modalities into unified situational tracks.
- **Gait Biometrics:** Walking pattern fingerprinting for person re-identification across sessions.
- **Indoor Mapping:** Progressive environment reconstruction from CSI reflection patterns.
- **Stealth & Low-Observable Mode:** Reduced emission profiles and encrypted data channels for sensitive deployments.
- **Perimeter Intrusion Detection:** Zone-based alerting for unauthorized entry via weapon/threat-class signal signatures.

---

## Model Architecture & Accuracy

### PoseNetV2 Architecture

```
Input: [B, 3, 64, 16]  (nodes × subcarriers × doppler_bins)
  ├─ Multi-Scale 1D CNN (kernel 3/5/7) → 192 channels
  ├─ LSTM (2-layer, 256-hidden) → temporal modeling
  ├─ Multi-Head Attention (8 heads) → spatial disentangling
  └─ Pose Regression Head → [B, 3, 17, 4]
Output: 3 people × 17 COCO keypoints × {x, y, z, confidence}
```

### Metrics (Synthetic Validation, 256 samples)

| Metric | Value |
|--------|-------|
| MPJPE (mean) | 0.4937 |
| MPJPE (std) | 0.0054 |
| PCK@0.1 | 0.4% |
| Confidence MAE | 0.2534 |

> **Note:** These metrics are from synthetic (random) test data — they validate that the model architecture and training pipeline function correctly end-to-end. Real-world accuracy depends on a labeled CSI→pose dataset collected with your specific hardware setup. See [Training with Real Data](#training-with-real-data) below.

## Maturity Snapshot

- The stack is functional end-to-end (firmware -> Rust aggregation -> Python inference -> Web UI).
- Some modules are research-grade and require further validation for production claims.
- Novelty focus for upcoming releases: domain shift robustness, uncertainty-aware outputs, and reproducible benchmarking.

### Robustness & Edge Cases

EchoPose handles the following challenging scenarios:

| Scenario | Mitigation |
|----------|------------|
| **Node dropout** | `FusionPipeline.robustness` tracks per-node health; graceful degradation to 2 or 1 node |
| **NaN / Inf in CSI** | `denoise.rs` uses NaN-safe sorting; Python pipeline clips and replaces invalid values |
| **Rate limiting / DoS** | Per-IP token-bucket limiter at 60 req/s on all REST endpoints; 429 response on excess |
| **Multipath interference** | Multi-scale CNN extracts features at 3 resolutions; attention layer disentangles overlapping reflections |
| **Person occlusion** | Multi-person orthogonal pose heads; disambiguation module resolves crossing paths |
| **Signal jitter** | TemporalPoseFilterV2 applies EMA smoothing with configurable alpha |
| **Missing subcarriers** | Rolling median denoiser in Rust aggregator fills gaps before inference |
| **Stale connections** | WebSocket connection manager prunes dead clients automatically |

---

## Architecture Stack

| Layer | Technology | Role |
|-------|------------|------|
| **Firmware** | C / ESP-IDF | Runs on ESP32-S3s. Captures 64-subcarrier I/Q CSI at 20 Hz. Streams binary UDP. |
| **Aggregator** | Rust / Axum | Receives UDP, aligns frames into 50ms windows, calibrates background noise, broadcasts via WS. |
| **Inference** | Python / PyTorch | FFT background subtraction → Doppler features → PoseNetV2 → 17 COCO keypoints. |
| **UI** | JS / Three.js | Connects to inference WS, renders real-time 3D skeleton + CSI Heatmap + Records Sessions. |

---

## Quick Start

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
Open `ui/index.html` in your web browser, click **Connect**, and watch the 3D skeleton.

---

## Training with Real Data

### Step 1: Collect a Dataset

Capture CSI frames while simultaneously recording ground-truth poses (e.g., from a camera-based system like OpenPose or MediaPipe). Save as `.npz`:

```python
np.savez("dataset.npz",
    features=csi_array,   # float32 [N, 3, 64, 16]
    poses=pose_array       # float32 [N, 3, 17, 4]
)
```

### Step 2: Train

```bash
cd inference
python -m scripts.train --data path/to/dataset.npz --epochs 50 --batch-size 16 --lr 1e-3
```

The best checkpoint is saved to `models/pose_net.pt` automatically.

### Step 3: Validate

```bash
python -m scripts.validate_accuracy --data path/to/test_set.npz --threshold 0.1
```

### Step 4: Export to ONNX (Optional)

```bash
python -m scripts.export_onnx
```

---

## Production Deployment

For real-world hardware deployment:

1. **Configure:** Set environment variables (see `.env.example`):
   - `ECHOPOSE_API_TOKEN` — API key for authenticated endpoints
   - `AGGREGATOR_WS_URI` — WebSocket URI for the Rust aggregator
   - `INFERENCE_DEVICE` — `cpu`, `cuda`, or `auto`
2. **Flash:** Build and flash the `firmware/` C project to your ESP32-S3 nodes. Set `CONFIG_HOST_IP` to your Aggregator's IP.
3. **Deploy Backend:** Run `docker-compose up -d --build` to launch the Rust and Python servers.
4. **Calibrate:** Access the UI, clear the room, and hit the `/calibrate` endpoint to subtract static reflections.

---

## Hardware Requirements

| Part | Qty | Est. Cost | Purpose |
|------|-----|-----------|---------|
| ESP32-S3 (U.FL) | 3 | ~$10 ea | CSI capture nodes |
| SMA antennas | 3 | ~$5 ea | Directional gain |
| Dedicated 2.4 GHz router | 1 | ~$30 | Silent AP (no other traffic) |
| Host PC (GPU optional) | 1 | existing | Runs aggregator + inference |
| **Total** | | **~$75–100** | |

### Performance

| Metric | Value |
|--------|-------|
| End-to-end latency | < 40 ms (CPU), < 15 ms (GPU) |
| CSI capture rate | 20 Hz per node |
| Max tracked people | 3 simultaneous |
| Subcarrier resolution | 64 per frame |
| WebSocket throughput | 1000+ concurrent UI clients (server_v2) |

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

## REST API

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/health` | GET | — | Server health & connected client count |
| `/analytics` | GET | — | Latest health metrics, activity, vitals, alerts snapshot |
| `/tactical` | GET | — | Latest security & situational awareness snapshot |
| `/ws/pose` | WS | — | Real-time skeleton + analytics + tactical stream |
| `/ingest` | POST | API key | Submit CSI bundle for inference (server_v2) |

### `/analytics` Response Shape

```json
{
  "vitals": { "heart_rate": {...}, "respiratory_rate": {...}, "spo2": {...}, "temperature": {...}, "blood_pressure": {...} },
  "activity": { "activity": "walking", "confidence": 0.78 },
  "gait": { "walking_speed_ms": 1.2, "cadence_steps_min": 110, ... },
  "fall": { "fall_detected": false, "fall_risk": "LOW" },
  "gestures": { "left_hand": "idle", "right_hand": "wave" },
  "sleep": { "sleep_stage": "AWAKE", "confidence": 0.88 },
  "occupancy": { "occupied": true, "num_people": 2 },
  "emotion": { "stress_level": "CALM", "stress_score": 12.5 },
  "health_alerts": { "alert_level": "NORMAL", "anomalies": [] }
}
```

### `/tactical` Response Shape

```json
{
  "targets": [{"id": 0, "x": 1.2, "y": -0.3, "z": 0.8, "doppler": 0.15, "confidence": 0.72}],
  "crowd": {"density": 3.5, "flow_direction_deg": 45.0, "spectral_occupancy": 0.28},
  "anomalies": {"is_anomaly": false, "deviation": 0.12},
  "acoustic": {"events": []},
  "intent": {"label": "APPROACHING", "confidence": 0.65},
  "anti_jamming": {"jamming_detected": false, "snr": 18.5},
  "coverage": {"sensor_count": 3, "covered_pct": 0.82},
  "tactical_activity": {"activity": "STANDING", "confidence": 0.91}
}
```

---

- **Rate Limiting:** Token-bucket per-IP limiter (60 req/s) on all HTTP endpoints
- **API Key Auth:** `X-EchoPose-Token` header required on data-ingestion endpoints
- **Payload Validation:** Pydantic models enforce CSI bundle structure and size limits
- **Encryption at Rest:** AES-256 Fernet encryption for session data storage
- **CORS:** Configurable allowed origins via `ALLOWED_ORIGINS` env var

---

## Licence

Copyright (c) 2026 Muhammed Shazin Sadhik Kunhi Parambath. All rights reserved.

EchoPose is **source-available but not free for commercial use**.

- **Personal / academic / non-commercial use** — Free under the [Source-Available Licence](LICENSE)
- **Commercial / enterprise / defence / healthcare use** — [Commercial Licence Required](COMMERCIAL_LICENSE.md)

You may **not** sell, resell, sublicense, or monetise this software without written permission.

Contact **shazin2889@gmail.com** for commercial licensing.
