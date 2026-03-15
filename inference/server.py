"""
inference/server.py — FastAPI + WebSocket inference server

                         WS:8080 (from aggregator)
                              │
                         ┌────────────┐
                         │  Fusion    │  pull CSI bundles from aggregator
                         │  Denoiser  │  FFT background subtraction
                         │  PoseNet   │  → 17 keypoints
                         └────┬───────┘
                              │ WS:8765
                         ┌────▼───────┐
                         │  UI /      │  Three.js dashboard
                         │  clients   │
                         └────────────┘

Environment variables:
  AGGREGATOR_WS   ws://localhost:3000/ws
  INFERENCE_PORT  8765
"""

import asyncio
import json
import os
import time
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

import numpy as np
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from pipeline.fusion import FusionPipeline
from pipeline.pose   import PoseEstimator
from pipeline.temporal_filter_v2 import TemporalPoseFilterV2

from monitoring.metrics import SystemMetrics
from custom_logger import StructuredLogger


# Load central config
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

AGGREGATOR_WS  = os.getenv("AGGREGATOR_WS_URI", "ws://localhost:3000/ws")
INFERENCE_PORT = int(os.getenv("INFERENCE_WS_PORT", "8765"))
DEVICE         = os.getenv("INFERENCE_DEVICE", "auto")

app = FastAPI(title="RF-Mesh Inference", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:8080,http://127.0.0.1:8000,http://127.0.0.1:8080").split(","),
    allow_methods=["GET", "POST", "OPTIONS"], 
    allow_headers=["*"]
)

estimator = PoseEstimator()
skeleton_filter = TemporalPoseFilterV2(max_people=3)
class ConnectionManager:
    """Thread-safe WebSocket connection manager."""
    def __init__(self):
        self.active_connections: set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active_connections.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active_connections.discard(ws)

    async def broadcast(self, message: str):
        dead_connections = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                dead_connections.add(connection)
        for dead in dead_connections:
            self.disconnect(dead)

    @property
    def count(self):
        return len(self.active_connections)

manager = ConnectionManager()


import logging
from logging.handlers import RotatingFileHandler

# ── Logging Setup ──────────────────────────────────────────────────
log_dir = Path(__file__).resolve().parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

logger = logging.getLogger("rf_inference")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Rotating file handler (10 MB per file, max 5 files)
file_handler = RotatingFileHandler(log_dir / "inference.log", maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ── Feature 6 & 7: Enterprise Observability ────────────────────────
sys_metrics = SystemMetrics(port=9090)
struct_logger = StructuredLogger(log_dir="logs")

# ── Background task: pull from aggregator, infer, broadcast ──────

@retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(5)
)
async def connect_and_process(fusion_pipeline):
    async with websockets.connect(AGGREGATOR_WS, ping_interval=20, ping_timeout=20, open_timeout=10) as ws:
        logger.info(f"Connected to aggregator @ {AGGREGATOR_WS}")
        async for raw in ws:
            try:
                start_time = time.time()
                bundle = json.loads(raw)
                features  = fusion_pipeline.process_bundle(bundle)
                skeletons = estimator.predict(features)
                smoothed_skeletons = skeleton_filter.filter(skeletons)
                
                # Extract pipeline metrics for logging
                mean_conf = np.mean([kp["confidence"] for s in smoothed_skeletons for kp in s]) if smoothed_skeletons and smoothed_skeletons[0] else 0.0
                node_health = fusion_pipeline.robustness.node_health
                
                # Extract amplitudes for the UI Heatmap
                amps_dict = {}
                for f in bundle.get("frames", []):
                    amps_dict[f["node_id"]] = {"amplitudes": f.get("amplitudes", [])}
                
                payload = json.dumps({
                    "window_us": bundle.get("window_us"),
                    "skeletons": smoothed_skeletons,
                    "amplitudes": amps_dict,
                    "num_frames": len(bundle.get("frames", [])),
                })

                await manager.broadcast(payload)
                
                # Observability updates
                latency_ms = (time.time() - start_time) * 1000
                sys_metrics.record_inference(latency_ms, mean_conf)
                sys_metrics.record_node_health(node_health)
                struct_logger.log_inference(latency_ms, mean_conf, [], node_health)

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from aggregator: {e}")
                sys_metrics.record_drop()
                continue
            except Exception as e:
                logger.error(f"Inference error during processing: {e}")
                struct_logger.log_error("Inference Error", str(e))

async def aggregator_loop():
    """Wrapper to maintain infinite resilience beyond individual retries."""
    fusion_pipeline = FusionPipeline()
    while True:
        try:
            await connect_and_process(fusion_pipeline)
        except Exception as e:
            logger.error(f"Aggregator WS exhausted all retries! Connection failed: {e}. Rebounding in 10s...")
            await asyncio.sleep(10)


@app.on_event("startup")
async def startup():
    asyncio.create_task(aggregator_loop())


# ── UI WebSocket endpoint ─────────────────────────────────────────
@app.websocket("/ws/pose")
async def ws_pose(ws: WebSocket):
    await manager.connect(ws)
    logger.info(f"UI client connected. Total: {manager.count}")
    try:
        while True:
            await ws.receive_text()   # keep-alive
    except WebSocketDisconnect:
        manager.disconnect(ws)
        logger.info(f"UI client disconnected. Total: {manager.count}")


# ── REST ──────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "ui_clients": manager.count}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=INFERENCE_PORT, reload=False, log_config=None)
