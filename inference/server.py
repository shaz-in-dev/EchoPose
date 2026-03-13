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
from pathlib import Path

import numpy as np
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from pipeline.fusion import FusionPipeline
from pipeline.pose   import PoseEstimator
from pipeline.filter import SkeletonFilter


# Load central config
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

AGGREGATOR_WS  = os.getenv("AGGREGATOR_WS_URI", "ws://localhost:3000/ws")
INFERENCE_PORT = int(os.getenv("INFERENCE_WS_PORT", "8765"))
DEVICE         = os.getenv("INFERENCE_DEVICE", "auto")

app = FastAPI(title="RF-Mesh Inference", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://localhost:8080", "http://127.0.0.1:8000", "http://127.0.0.1:8080"], # Restricted for production security
    allow_methods=["GET", "POST", "OPTIONS"], 
    allow_headers=["*"]
)

estimator = PoseEstimator()
skeleton_filter = SkeletonFilter(alpha=0.4)
# All connected UI clients
_ui_clients: list[WebSocket] = []


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

# ── Background task: pull from aggregator, infer, broadcast ──────
async def aggregator_loop():
    """Connects to the Rust aggregator WS and runs inference on each bundle."""
    fusion_pipeline = FusionPipeline()  # Thread-safe pipeline per worker
    backoff = 1.0
    while True:
        try:
            async with websockets.connect(AGGREGATOR_WS, ping_interval=20, ping_timeout=20, open_timeout=10) as ws:
                logger.info(f"Connected to aggregator @ {AGGREGATOR_WS}")
                backoff = 1.0
                async for raw in ws:
                    try:
                        bundle = json.loads(raw)
                        features  = fusion_pipeline.process_bundle(bundle)
                        keypoints = estimator.predict(features)
                        smoothed_keypoints = skeleton_filter.filter(keypoints)

                        payload = json.dumps({
                            "window_us":  bundle.get("window_us"),
                            "keypoints":  smoothed_keypoints,
                            "num_frames": len(bundle.get("frames", [])),
                        })

                        # Broadcast to all connected UI clients
                        dead = []
                        for client in _ui_clients:
                            try:
                                await client.send_text(payload)
                            except Exception:
                                dead.append(client)
                        for d in dead:
                            _ui_clients.remove(d)

                    except json.JSONDecodeError:
                        logger.warning("Malformed JSON received from aggregator. Skipping frame.")
                    except Exception as e:
                        logger.error(f"Inference error during processing: {e}")

        except Exception as e:
            logger.error(f"Aggregator connection error: {e}. Retrying in {backoff:.0f}s…")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)


@app.on_event("startup")
async def startup():
    asyncio.create_task(aggregator_loop())


# ── UI WebSocket endpoint ─────────────────────────────────────────
@app.websocket("/ws/pose")
async def ws_pose(ws: WebSocket):
    await ws.accept()
    _ui_clients.append(ws)
    logger.info(f"UI client connected. Total: {len(_ui_clients)}")
    try:
        while True:
            await ws.receive_text()   # keep-alive
    except WebSocketDisconnect:
        _ui_clients.remove(ws)
        logger.info(f"UI client disconnected. Total: {len(_ui_clients)}")


# ── REST ──────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "ui_clients": len(_ui_clients)}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=INFERENCE_PORT, reload=False, log_config=None)
