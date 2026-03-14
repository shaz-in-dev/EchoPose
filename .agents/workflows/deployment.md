---
description: How to Deploy the EchoPose Enterprise V3 Stack
---

# EchoPose Enterprise V3 Deployment Guide

This workflow provisions the high-throughput Rust Aggregator, the PyTorch/TensorRT inference engine, and the React UI dashboard.

## Prerequisites
- A Linux or Windows Host with Docker & `docker-compose` installed.
- (Optional but Recommended) NVIDIA Toolkit for GPU acceleration.
- ESP32-S3 boards flashed with the `firmware/main/main.c` build.

## Step 1: Initialize Zero-Trust Secrets
Before booting the servers, you must generate encryption keys and API tokens to satisfy the `inference/security.py` strict enforcement rules.
Create a `.env` file in the root of the project:

```bash
# .env
EXPECTED_NODES=3

# Aggregator Configuration
AGGREGATOR_UDP_PORT=5005
AGGREGATOR_HTTP_PORT=3000

# Inference Server Configuration
INFERENCE_WS_PORT=8765
INFERENCE_DEVICE=auto

# Enterprise Security Keys
ECHOPOSE_API_TOKEN=your_secure_random_token_here
ECHOPOSE_SESSION_KEY=your_fernet_aes_key_here

# Inference CORS Restrictions
ALLOWED_ORIGINS=http://localhost:8000,http://localhost:3000,https://your-production-domain.com
```

> [!TIP]
> You can generate a valid Fernet AES-256 key via Python:
> `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode('utf-8'))"`

## Step 2: Boot the Aggregator Engine
The Rust aggregator must start first so it can bind the UDP socket (`0.0.0.0:5005`) to listen to the ESP32 routers.

// turbo
```bash
cd aggregator
cargo build --release
./target/release/aggregator
```

> [!NOTE]
> Ensure port 5005/UDP is completely unblocked in your firewall.

## Step 3: Launch High-Throughput Inference
With the aggregator fanning out WebSocket bundles on `:3000`, boot the `server_v2.py` pipeline.
This handles the Multi-Person Disambiguation, Advanced Wavelet Denoising, and PoseNet ML processing.

// turbo
```bash
cd inference
pip install -r requirements.txt
python server_v2.py
```

> [!IMPORTANT]
> If you are using Docker instead of bare metal, you can skip steps 2 and 3 and simply run:
> `bash deployment/deploy.sh`

## Step 4: Access the UI Dashboard
Start the React + Three-Fiber frontend visualizer.
```bash
cd ui
npm install
npm run start
```
Navigate your browser to `http://localhost:8000` (or whatever origin you compiled the UI to).

## Step 5: Perform Static Room Calibration
1. Ensure the room has absolutely **zero moving humans**.
2. Click the `/calibrate` button in the UI or hit the REST endpoint:
   `curl -X POST http://localhost:3000/calibrate`
3. Wait exactly 5.0 seconds. The Rust engine will formulate the static multipath noise floor to erase internal wall reflections.
4. Walk into the room. The system is active.
