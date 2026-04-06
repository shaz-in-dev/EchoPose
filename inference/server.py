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
from contextlib import asynccontextmanager
from pathlib import Path
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

import numpy as np
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from pipeline.fusion import FusionPipeline
from pipeline.pose   import PoseEstimator
from pipeline.temporal_filter_v2 import TemporalPoseFilterV2
from pipeline.vitals import VitalsExtractor
from pipeline.activity import ActivityClassifier
from pipeline.fall_detector import FallDetector
from pipeline.sleep_analyzer import SleepAnalyzer
from pipeline.gesture import GestureRecognizer
from pipeline.occupancy import OccupancyAnalyzer
from pipeline.emotion import EmotionDetector
from pipeline.health_alerts import HealthAnomalyDetector

from monitoring.metrics import SystemMetrics
from custom_logger import StructuredLogger
from security import limiter, verify_api_key, IncomingCSIBundle


# Load central config
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

AGGREGATOR_WS  = os.getenv("AGGREGATOR_WS_URI", "ws://localhost:3000/ws")
INFERENCE_PORT = int(os.getenv("INFERENCE_WS_PORT", "8765"))
DEVICE         = os.getenv("INFERENCE_DEVICE", "auto")

@asynccontextmanager
async def lifespan(app):
    task = asyncio.create_task(aggregator_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="RF-Mesh Inference", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:8080,http://127.0.0.1:8000,http://127.0.0.1:8080").split(","),
    allow_methods=["GET", "POST", "OPTIONS"], 
    allow_headers=["*"]
)

estimator = PoseEstimator()
skeleton_filter = TemporalPoseFilterV2(max_people=3)
vitals_extractor = VitalsExtractor()
activity_classifier = ActivityClassifier()
fall_detector = FallDetector()
sleep_analyzer = SleepAnalyzer()
gesture_recognizer = GestureRecognizer()
occupancy_analyzer = OccupancyAnalyzer()
emotion_detector = EmotionDetector()
health_alerter = HealthAnomalyDetector()

# Latest analytics snapshot (updated each inference cycle)
_latest_analytics: dict = {}
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
                features, per_person = fusion_pipeline.process_bundle(bundle)
                skeletons = estimator.predict(features, per_person_features=per_person)
                smoothed_skeletons = skeleton_filter.filter(skeletons)
                
                # ── Health & Activity Analytics ───────────────────
                # Feed raw CSI amplitudes to vitals extractor
                for f in bundle.get("frames", []):
                    amps = np.array(f.get("amplitudes", []), dtype=np.float64)
                    if amps.size:
                        vitals_extractor.push(amps)

                vitals = vitals_extractor.extract_all(features)

                # Feed first skeleton to activity / fall / gesture / sleep modules
                if smoothed_skeletons and len(smoothed_skeletons) > 0:
                    first_skel = smoothed_skeletons[0]
                    activity_classifier.push_skeleton(first_skel)
                    fall_detector.push_skeleton(first_skel)
                    gesture_recognizer.push_skeleton(first_skel)
                    sleep_analyzer.push_motion(first_skel)

                    hr_val = vitals.get("heart_rate", {}).get("heart_rate")
                    rr_val = vitals.get("respiratory_rate", {}).get("respiratory_rate")
                    sleep_analyzer.push_vitals(hr_val, rr_val)
                    if hr_val and rr_val:
                        emotion_detector.update_baselines(hr_val, rr_val)

                activity = activity_classifier.classify_activity()
                gait = activity_classifier.analyze_gait()
                fall = fall_detector.detect()
                gestures = gesture_recognizer.recognize()
                sleep = sleep_analyzer.classify()
                occupancy = occupancy_analyzer.detect_presence(smoothed_skeletons, features)
                emotion = emotion_detector.estimate_stress(
                    vitals.get("heart_rate", {}).get("heart_rate"),
                    vitals.get("respiratory_rate", {}).get("respiratory_rate"),
                    smoothed_skeletons[0] if smoothed_skeletons else None,
                )
                alerts = health_alerter.check(
                    hr=vitals.get("heart_rate", {}).get("heart_rate"),
                    rr=vitals.get("respiratory_rate", {}).get("respiratory_rate"),
                    spo2=vitals.get("spo2", {}).get("spo2"),
                    activity=activity.get("activity", "sitting"),
                )

                analytics = {
                    "vitals": vitals,
                    "activity": activity,
                    "gait": gait,
                    "fall": fall,
                    "gestures": gestures,
                    "sleep": sleep,
                    "occupancy": occupancy,
                    "emotion": emotion,
                    "health_alerts": alerts,
                }
                global _latest_analytics
                _latest_analytics = analytics

                # Extract pipeline metrics for logging
                all_kp_confs = [kp["confidence"] for s in smoothed_skeletons for kp in s]
                mean_conf = np.mean(all_kp_confs) if all_kp_confs else 0.0
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
                    "simulation": estimator.is_simulation,
                    "analytics": analytics,
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
            logger.warning(f"Aggregator WS exhausted all retries! Connection failed: {e}. Rebounding in 10s...")
            await asyncio.sleep(10)



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
async def health(request: Request):
    limiter.check_rate_limit(request.client.host)
    return {"status": "ok", "ui_clients": manager.count}


@app.get("/analytics")
async def analytics(request: Request):
    """Return the latest health metrics, activity, and alert snapshot."""
    limiter.check_rate_limit(request.client.host)
    return _latest_analytics or {"status": "no_data"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=INFERENCE_PORT, reload=False, log_config=None)
