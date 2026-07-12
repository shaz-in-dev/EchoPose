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
import torch
import uvicorn
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pydantic import ValidationError

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
from pipeline.disaster_response import DisasterResponseEngine
from pipeline.disaster_bridge import attach_disaster_context

from pipeline.tactical import (
    TacticalTargetTracker,
    ConcealmentDetector,
    WeaponDetectionSystem,
    CrowdDensityAnalyzer,
    TacticalActivityClassifier,
    AnomalyScanner,
    BehavioralIntentPredictor,
    AntiJammingDefense,
    MultiDomainFusion,
)

from monitoring.metrics import SystemMetrics
from custom_logger import StructuredLogger
from security import limiter, verify_api_key, IncomingCSIBundle, LicenseTier, require_tier
from integrations.homeassistant import HAConfig, HomeAssistantBridge
from integrations.matter import MatterBridge
from integrations.webhooks import WebhookManager, EventType
from integrations.session_store import SessionStore
from pipeline.fast_adapt import FastAdapter
from pipeline.pose_net_v2 import PoseNetV2


# Load central config
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

AGGREGATOR_WS  = os.getenv("AGGREGATOR_WS_URI", "ws://localhost:3000/ws")
_raw_port = int(os.getenv("INFERENCE_WS_PORT", "8765"))
if not 1 <= _raw_port <= 65535:
    raise ValueError(f"INFERENCE_WS_PORT={_raw_port} is out of the valid range 1-65535")
INFERENCE_PORT = _raw_port
DEVICE         = os.getenv("INFERENCE_DEVICE", "auto")
_EXPECTED_NODES    = int(os.getenv("EXPECTED_NODES", "3"))
_EXPECTED_AMP_SIZE = _EXPECTED_NODES * 64 * 16   # nodes × subcarriers × doppler_bins

@asynccontextmanager
async def lifespan(app):
    global _ha_bridge, _matter_bridge

    # Caregiver alert system (always starts — reads channels from .env)
    await _webhook_manager.start()

    # Restore last_activity_ts from DB so inactivity timer survives restarts
    saved_activity = _session_store.get_state("last_activity_ts")
    if saved_activity:
        _webhook_manager._last_activity_ts = float(saved_activity)
        logger.info("Restored last_activity_ts from session DB (%.0f s ago)",
                    time.time() - _webhook_manager._last_activity_ts)

    # Home Assistant MQTT bridge (opt-in via HA_MQTT_BROKER env var)
    ha_cfg = HAConfig.from_env()
    if ha_cfg:
        _ha_bridge = HomeAssistantBridge(ha_cfg)
        await _ha_bridge.start()

    # Matter protocol bridge (opt-in via MATTER_ENABLED=true)
    _matter_bridge = MatterBridge.from_env()
    if _matter_bridge:
        await _matter_bridge.start()

    # LoRA fast-adapter (opt-in via LORA_ADAPT=true)
    if os.getenv("LORA_ADAPT", "").lower() == "true":
        _init_fast_adapter()

    task = asyncio.create_task(aggregator_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await _webhook_manager.stop()
    if _ha_bridge:
        await _ha_bridge.stop()
    if _matter_bridge:
        await _matter_bridge.stop()


app = FastAPI(title="RF-Mesh Inference", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:8080,http://127.0.0.1:8000,http://127.0.0.1:8080").split(","),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    allow_credentials=False,  # keep False; set True only with explicit origins, never with wildcard
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
disaster_engine = DisasterResponseEngine()

# Tactical analytics modules
tactical_tracker = TacticalTargetTracker()
concealment_detector = ConcealmentDetector()
weapon_detector = WeaponDetectionSystem()
crowd_analyzer = CrowdDensityAnalyzer()
tactical_activity = TacticalActivityClassifier()
anomaly_scanner = AnomalyScanner()
intent_predictor = BehavioralIntentPredictor()
anti_jam = AntiJammingDefense()
sensor_fusion = MultiDomainFusion()

# Latest tactical snapshot
_latest_tactical: dict = {}

# Latest analytics snapshot (updated each inference cycle)
_latest_analytics: dict = {}

# Latest disaster snapshot (updated each inference cycle)
_latest_disaster: dict = {}

# ── Home Assistant MQTT bridge ────────────────────────────────────────
_ha_bridge: HomeAssistantBridge | None = None

# ── Matter protocol bridge ────────────────────────────────────────────
_matter_bridge: MatterBridge | None = None

# ── Caregiver alert / webhook manager ────────────────────────────────
_webhook_manager = WebhookManager()

# ── Session history store (SQLite — survives server restarts) ─────────
_session_store = SessionStore()

# ── LoRA fast-adapt ───────────────────────────────────────────────────
_fast_adapter: FastAdapter | None = None

def _init_fast_adapter() -> None:
    global _fast_adapter
    _posenet = PoseNetV2()
    ckpt = Path("models/pose_net.pt")
    if ckpt.exists():
        _posenet.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    _fast_adapter = FastAdapter(
        model=_posenet,
        rank=int(os.getenv("LORA_RANK", "8")),
        device="cpu",
    )

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
            except Exception as exc:
                logger.debug("Dropping dead WebSocket connection: %s", exc)
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
sys_metrics = SystemMetrics(port=int(os.getenv("METRICS_PORT", "9090")))
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
                IncomingCSIBundle(**bundle)  # raises ValidationError on malformed/oversized bundle
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

                # ── Tactical Analytics ────────────────────────────
                for f in bundle.get("frames", []):
                    amps = np.array(f.get("amplitudes", []), dtype=np.float64)
                    if amps.size:
                        tactical_tracker.push(amps)
                        concealment_detector.push(amps)
                        crowd_analyzer.push(amps)
                        anomaly_scanner.push(amps)
                        anti_jam.push(amps)

                if smoothed_skeletons and len(smoothed_skeletons) > 0:
                    first_skel = smoothed_skeletons[0]
                    tactical_activity.push_skeleton(first_skel)
                    weapon_detector.push(first_skel)
                    stress_score = (emotion.get("stress_score") or 0) / 100.0
                    tac_act = tactical_activity.classify().get("activity", "STANDING")
                    intent_predictor.push(first_skel, stress_score, tac_act)

                    # Feed WiFi detections to sensor fusion
                    wifi_dets = []
                    tgt = tactical_tracker.detect()
                    for t in tgt.get("targets", []):
                        wifi_dets.append({
                            "x": t.get("range_m", 5), "y": 0, "z": 0,
                            "confidence": t.get("confidence", 0.5),
                            "classification": t.get("classification", "UNKNOWN"),
                        })
                    if wifi_dets:
                        sensor_fusion.ingest("wifi_csi", wifi_dets)

                tactical_data = {
                    "targets": tactical_tracker.detect(),
                    "concealment": concealment_detector.scan(),
                    "weapon": weapon_detector.detect(),
                    "crowd": crowd_analyzer.estimate(
                        room_area_m2=float(os.getenv("ROOM_AREA_M2", "50.0")),
                        skeleton_count=len(smoothed_skeletons),
                    ),
                    "tactical_activity": tactical_activity.classify(),
                    "anomalies": anomaly_scanner.scan(),
                    "intent": intent_predictor.predict(),
                    "anti_jam": anti_jam.check(),
                    "fusion": sensor_fusion.get_cop(),
                }
                global _latest_tactical
                _latest_tactical = tactical_data

                analytics, disaster_data = attach_disaster_context(
                    analytics=analytics,
                    tactical=tactical_data,
                    engine=disaster_engine,
                )
                global _latest_analytics
                _latest_analytics = analytics
                global _latest_disaster
                _latest_disaster = disaster_data

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
                    "tactical": tactical_data,
                    "disaster": disaster_data,
                })

                await manager.broadcast(payload)

                # Home Assistant MQTT publish
                if _ha_bridge:
                    await _ha_bridge.publish(analytics, len(smoothed_skeletons))

                # Matter protocol publish (Apple Home / Google Home / Alexa)
                if _matter_bridge:
                    await _matter_bridge.publish(analytics, len(smoothed_skeletons))

                # Caregiver alerts (fall, inactivity, vitals, presence)
                await _webhook_manager.process_frame(analytics, len(smoothed_skeletons))

                # Persist analytics to SQLite session store
                _session_store.maybe_log_vitals(analytics, len(smoothed_skeletons))
                _session_store.set_state("last_activity_ts", _webhook_manager._last_activity_ts)

                # Push raw CSI into LoRA adapter buffer for future /adapt calls
                if _fast_adapter is not None:
                    for f in bundle.get("frames", []):
                        amps = np.array(f.get("amplitudes", []), dtype=np.float32)
                        if amps.size == _EXPECTED_AMP_SIZE:
                            _fast_adapter.push_frame(
                                torch.from_numpy(amps.reshape(_EXPECTED_NODES, 64, 16))
                            )

                # Observability updates
                latency_ms = (time.time() - start_time) * 1000
                sys_metrics.record_inference(latency_ms, mean_conf)
                sys_metrics.record_node_health(node_health)
                struct_logger.log_inference(latency_ms, mean_conf, [], node_health)

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from aggregator: {e}")
                sys_metrics.record_drop()
                continue
            except ValidationError as e:
                logger.error(f"Malformed CSI bundle from aggregator: {e}")
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
_TRUSTED_PROXIES: frozenset[str] = frozenset(
    x.strip() for x in os.getenv("TRUSTED_PROXIES", "127.0.0.1,::1").split(",") if x.strip()
)

def _client_ip(request: Request) -> str:
    """Extract real client IP, only trusting X-Forwarded-For from known proxies.

    Without this guard, any client can spoof X-Forwarded-For to bypass per-IP
    rate limiting. Set TRUSTED_PROXIES to your nginx/k8s ingress IP(s).

    When request.client is None (behind a Unix-socket proxy or certain load
    balancers), X-Forwarded-For is the only source available and is used as-is.
    """
    peer = request.client.host if request.client else None
    # Trust X-Forwarded-For when: (a) peer is from a known proxy, or
    # (b) peer is None (we have no TCP-level IP to validate against)
    if peer is None or peer in _TRUSTED_PROXIES:
        forwarded = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return peer or "unknown"


@app.get("/health")
async def health():
    # No rate limiting — K8s liveness/readiness probes hit this endpoint
    # from the same IP every few seconds and would trip the limiter.
    return {"status": "ok", "ui_clients": manager.count}


@app.get("/analytics")
async def analytics(request: Request, _: None = Depends(verify_api_key)):
    """Return the latest health metrics, activity, and alert snapshot."""
    await limiter.check_rate_limit(_client_ip(request))
    return _latest_analytics or {"status": "no_data"}


@app.get("/tactical")
async def tactical(
    request: Request,
    _key:   None = Depends(verify_api_key),
    _tier:  None = Depends(require_tier(LicenseTier.ENTERPRISE)),
):
    """Return the latest tactical analytics snapshot. Requires Enterprise license."""
    await limiter.check_rate_limit(_client_ip(request))
    return _latest_tactical or {"status": "no_data"}


@app.get("/disaster")
async def disaster(request: Request, _: None = Depends(verify_api_key)):
    """Return the latest disaster-response analytics snapshot."""
    await limiter.check_rate_limit(_client_ip(request))
    return _latest_disaster or {"status": "no_data"}


@app.post("/webhooks/register")
async def webhook_register(request: Request, _: None = Depends(verify_api_key)):
    """Register a webhook URL to receive caregiver alerts.

    Body: { "url": "https://...", "name": "My Phone", "events": ["fall_detected"], "secret": "opt" }
    Events: fall_detected | person_entered | person_left | inactivity_alert |
            health_alert | vitals_critical | anomaly
    """
    await limiter.check_rate_limit(_client_ip(request))
    body = await request.json()
    if "url" not in body:
        from fastapi import HTTPException
        raise HTTPException(status_code=422, detail="'url' is required")
    wid = _webhook_manager.register(
        url=body["url"],
        name=body.get("name", ""),
        events=set(body.get("events", [])),
        secret=body.get("secret"),
    )
    return {"id": wid, "status": "registered"}


@app.get("/webhooks")
async def webhook_list(request: Request, _: None = Depends(verify_api_key)):
    """List all registered webhook endpoints and active alert channels."""
    await limiter.check_rate_limit(_client_ip(request))
    return {
        "webhooks": _webhook_manager.list_webhooks(),
        "channels": _webhook_manager.channel_summary(),
    }


@app.delete("/webhooks/{webhook_id}")
async def webhook_delete(webhook_id: str, request: Request, _: None = Depends(verify_api_key)):
    """Remove a registered webhook by ID."""
    await limiter.check_rate_limit(_client_ip(request))
    removed = _webhook_manager.unregister(webhook_id)
    return {"status": "removed" if removed else "not_found"}


@app.get("/status/caregiver")
async def caregiver_status(request: Request):
    """Quick caregiver dashboard — no auth required, safe to expose.

    Returns last activity time, person count, vitals, and alert channel status.
    """
    await limiter.check_rate_limit(_client_ip(request))
    analytics = _latest_analytics or {}
    vitals = (analytics.get("vitals") or {})
    hr = (vitals.get("heart_rate") or {}).get("heart_rate")
    rr = (vitals.get("respiratory_rate") or {}).get("respiratory_rate")
    channels = _webhook_manager.channel_summary()
    return {
        "person_detected":      channels["last_activity_ago_s"] < 30,
        "last_activity_ago_s":  channels["last_activity_ago_s"],
        "heart_rate":           round(hr, 1) if hr else None,
        "respiratory_rate":     round(rr, 1) if rr else None,
        "activity":             (analytics.get("activity") or {}).get("activity"),
        "fall_detected":        (analytics.get("fall") or {}).get("fall_detected", False),
        "alert_channels":       {k: v for k, v in channels.items() if k != "last_activity_ago_s"},
    }


@app.get("/history/events")
async def history_events(request: Request, hours: float = 24.0, limit: int = 200):
    """Return recent caregiver alert events from the SQLite session store.

    ?hours=24  — look-back window (default 24 h)
    ?limit=200 — max rows returned
    """
    await limiter.check_rate_limit(_client_ip(request))
    return {"events": _session_store.get_events(hours=hours, limit=limit)}


@app.get("/history/vitals")
async def history_vitals(request: Request, hours: float = 24.0, limit: int = 500):
    """Return vitals timeline from the SQLite session store.

    Snapshots are taken every SESSION_VITALS_INTERVAL_S seconds (default 5 min).
    """
    await limiter.check_rate_limit(_client_ip(request))
    return {"vitals": _session_store.get_vitals(hours=hours, limit=limit)}


@app.get("/history/summary")
async def history_summary(request: Request):
    """Return session store summary — total events, last fall, DB path."""
    await limiter.check_rate_limit(_client_ip(request))
    return _session_store.summary()


@app.get("/license")
async def license_info(request: Request):
    """Return the active license tier. No auth required — safe to expose publicly."""
    await limiter.check_rate_limit(_client_ip(request))
    from security import get_license_tier, _LICENSE_MODE
    tier = get_license_tier()
    return {
        "tier":  tier.name,
        "label": tier.label(),
        "mode":  _LICENSE_MODE,
        "features": {
            "basic_analytics": True,
            "ha_mqtt":         True,
            "matter":          True,
            "fast_adapt":      tier >= LicenseTier.PROFESSIONAL,
            "tactical":        tier >= LicenseTier.ENTERPRISE,
            "defense_modules": tier >= LicenseTier.DEFENSE,
        },
    }


@app.get("/matter/pairing")
async def matter_pairing(request: Request):
    """Return the Matter commissioning QR code and manual pairing code.

    Scan the QR code in Apple Home, Google Home, or the Alexa app to add
    EchoPose as a native Matter sensor hub. Only needs to be done once.
    """
    await limiter.check_rate_limit(_client_ip(request))
    if _matter_bridge is None:
        return {"status": "disabled", "detail": "Start server with MATTER_ENABLED=true"}
    return await _matter_bridge.get_pairing()


@app.get("/matter/status")
async def matter_status(request: Request):
    """Return Matter bridge status and commissioning state."""
    await limiter.check_rate_limit(_client_ip(request))
    if _matter_bridge is None:
        return {"status": "disabled"}
    return await _matter_bridge.get_status()


@app.post("/adapt")
async def adapt(
    request: Request,
    _key:   None = Depends(verify_api_key),
    _tier:  None = Depends(require_tier(LicenseTier.PROFESSIONAL)),
):
    """Trigger a 30-second self-supervised LoRA adaptation to the current room.

    Requires LORA_ADAPT=true at startup. Uses the last ~500 buffered CSI frames.
    """
    await limiter.check_rate_limit(_client_ip(request))
    if _fast_adapter is None:
        return {"status": "disabled", "detail": "Start server with LORA_ADAPT=true to enable"}
    if _fast_adapter.buffer_size < 16:
        return {"status": "insufficient_data", "frames": _fast_adapter.buffer_size}

    timeout = float(request.query_params.get("timeout", "30"))
    timeout = max(5.0, min(120.0, timeout))

    result = await asyncio.get_event_loop().run_in_executor(
        None, lambda: _fast_adapter.adapt(timeout_seconds=timeout)
    )
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=INFERENCE_PORT, reload=False, log_config=None)
