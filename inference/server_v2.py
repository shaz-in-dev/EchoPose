"""
inference/server_v2.py — High-Throughput Async Pipeline (Feature 11)

An entirely non-blocking async implementation.
Moves CPU/GPU bound ML inference into thread pools so the asyncio event loop 
can handle thousands of concurrent WebSocket connections and UI commands without dropping frames.
"""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
import os
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import websockets
from websockets.exceptions import WebSocketException
from pipeline.fusion import FusionPipeline
from pipeline.pose import PoseEstimator as DistributedInference
from pipeline.fall_detector import FallDetector as _FallDetector
from security import limiter, verify_api_key
from integrations.homeassistant import HAConfig, HomeAssistantBridge
from integrations.matter import MatterBridge
from integrations.webhooks import WebhookManager

_ha_bridge_v2:      HomeAssistantBridge | None = None
_matter_bridge_v2:  MatterBridge        | None = None
_webhook_manager_v2 = WebhookManager()

logger = logging.getLogger("rf_inference.async_server")

AGGREGATOR_WS = os.getenv("AGGREGATOR_WS_URI", "ws://localhost:3000/ws")

class HighThroughputServer:
    """Manages concurrent UI clients and non-blocking inference decoupling.

    NOTE: This server prioritises throughput over analytics richness.
    Vitals extraction, activity classification, emotion, and tactical modules
    are NOT run here — use server.py for those features.
    Fall detection IS included because it drives critical caregiver alerts.
    """
    def __init__(self):
        self.clients = set()
        self.fusion  = FusionPipeline()
        self.model   = DistributedInference()
        self.fall_detector = _FallDetector()
        self.bundle_queue  = asyncio.Queue(maxsize=100)
        
    async def handle_client(self, ws: WebSocket):
        await ws.accept()
        self.clients.add(ws)
        
        try:
            # Create per-client concurrent tasks
            tasks = [
                self._receive_ui_commands(ws),
                self._health_ping(ws)
            ]
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.info(f"Client disconnected: {e}")
        finally:
            self.clients.discard(ws)

    async def _infer_continuously(self):
        """
        Background infinite loop doing the heavy lifting asynchronously.
        Pulls from the aggregator queue, runs NN inference in a separate Thread,
        and broadcasts back to the async loop.
        """
        while True:
            # Await next available bundle without blocking I/O
            bundle = await self.bundle_queue.get()
            
            # 1. Feature fusion (Fast, runs in-loop)
            features, per_person = self.fusion.process_bundle(bundle)
            
            # 2. ML Inference (Slow, dispatch to ThreadPoolExecutor so UI doesn't freeze)
            # Pass all per-person feature tensors for multi-person inference
            inference_input = per_person if per_person else [features]
            skeletons = await self.model.batch_inference(inference_input)
            
            # 3. Broadcast to all active clients concurrently
            payload = json.dumps({"skeletons": skeletons[0] if skeletons else []})
            tasks = [client.send_text(payload) for client in self.clients]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            # 4. Minimal analytics — fall detection only (enough for caregiver alerts)
            person_count = len(skeletons[0]) if skeletons and skeletons[0] else 0
            if skeletons and skeletons[0]:
                self.fall_detector.push_skeleton(skeletons[0][0])
            fall_result = self.fall_detector.detect()
            minimal_analytics = {"fall": fall_result}

            if _ha_bridge_v2:
                await _ha_bridge_v2.publish(minimal_analytics, person_count)
            if _matter_bridge_v2:
                await _matter_bridge_v2.publish(minimal_analytics, person_count)
            await _webhook_manager_v2.process_frame(minimal_analytics, person_count)

            self.bundle_queue.task_done()

    async def _receive_ui_commands(self, ws: WebSocket):
        import httpx
        while True:
            cmd = await ws.receive_text()
            if cmd == "calibrate":
                logger.info("Triggering System Recalibration")
                try:
                    async with httpx.AsyncClient() as client:
                        await client.post("http://localhost:3000/calibrate")
                except Exception as e:
                    logger.error(f"Failed to relay calibrate command: {e}")

    async def _health_ping(self, ws: WebSocket):
        while True:
            await asyncio.sleep(10)
            await ws.send_json({"ping": "health_check"})

    @retry(
        wait=wait_exponential(multiplier=1, max=30),
        stop=stop_after_attempt(10),
        retry=retry_if_exception_type((ConnectionRefusedError, OSError, WebSocketException)),
    )
    async def _aggregator_ws_feed(self):
        """Connect to the Rust aggregator WebSocket and feed bundles into the queue."""
        async with websockets.connect(AGGREGATOR_WS, ping_interval=20, ping_timeout=10) as conn:
            logger.info(f"Connected to aggregator at {AGGREGATOR_WS}")
            async for raw in conn:
                bundle = json.loads(raw)
                if not self.bundle_queue.full():
                    await self.bundle_queue.put(bundle)

    async def aggregator_loop(self):
        """Maintain infinite resilience beyond individual retries."""
        while True:
            try:
                await self._aggregator_ws_feed()
            except Exception as e:
                logger.warning(f"Aggregator WS exhausted all retries: {e}. Rebounding in 10s...")
                await asyncio.sleep(10)

server = HighThroughputServer()

import ipaddress as _ipaddress

_TRUSTED_PROXIES_V2: frozenset = frozenset(
    x.strip() for x in os.getenv("TRUSTED_PROXIES", "127.0.0.1,::1").split(",") if x.strip()
)

def _client_ip(request: Request) -> str:
    peer = request.client.host if request.client else None
    if peer is None or peer in _TRUSTED_PROXIES_V2:
        forwarded = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        if forwarded:
            return forwarded
    return peer or "unknown"

@asynccontextmanager
async def lifespan(_: FastAPI):
    global _ha_bridge_v2, _matter_bridge_v2

    await _webhook_manager_v2.start()

    ha_cfg = HAConfig.from_env()
    if ha_cfg:
        _ha_bridge_v2 = HomeAssistantBridge(ha_cfg)
        await _ha_bridge_v2.start()

    _matter_bridge_v2 = MatterBridge.from_env()
    if _matter_bridge_v2:
        await _matter_bridge_v2.start()

    worker   = asyncio.create_task(server._infer_continuously())
    agg_task = asyncio.create_task(server.aggregator_loop())
    try:
        yield
    finally:
        worker.cancel()
        agg_task.cancel()
        await asyncio.gather(worker, agg_task, return_exceptions=True)
        await _webhook_manager_v2.stop()
        if _ha_bridge_v2:
            await _ha_bridge_v2.stop()
        if _matter_bridge_v2:
            await _matter_bridge_v2.stop()


app = FastAPI(title="EchoPose V2 High-Throughput Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:8080").split(","),
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.websocket("/ws/pose")
async def pose_stream(ws: WebSocket):
    await server.handle_client(ws)
    
@app.post("/ingest")
async def ingest_bundle(bundle: dict, request: Request, _key: str = Depends(verify_api_key)):
    """Aggregator sends bundles here via HTTP/WS. Queue it for non-blocking processing."""
    await limiter.check_rate_limit(_client_ip(request))
    if not server.bundle_queue.full():
        await server.bundle_queue.put(bundle)
    return {"status": "queued"}

@app.get("/health")
async def health():
    # No rate limiting — K8s liveness probes hit this from same IP continuously.
    return {
        "status": "ok",
        "ui_clients": len(server.clients),
        "queue_depth": server.bundle_queue.qsize(),
    }

@app.get("/matter/pairing")
async def matter_pairing_v2(request: Request):
    await limiter.check_rate_limit(_client_ip(request))
    if _matter_bridge_v2 is None:
        return {"status": "disabled", "detail": "Start with MATTER_ENABLED=true"}
    return await _matter_bridge_v2.get_pairing()


@app.get("/matter/status")
async def matter_status_v2(request: Request):
    await limiter.check_rate_limit(_client_ip(request))
    if _matter_bridge_v2 is None:
        return {"status": "disabled"}
    return await _matter_bridge_v2.get_status()


@app.get("/license")
async def license_info_v2(request: Request):
    """Active license tier and feature availability."""
    await limiter.check_rate_limit(_client_ip(request))
    from security import get_license_tier, LicenseTier, _LICENSE_MODE
    tier = get_license_tier()
    return {
        "tier":  tier.name,
        "label": tier.label(),
        "mode":  _LICENSE_MODE,
        "features": {
            "basic_analytics": True,
            "ha_mqtt":         True,
            "matter":          True,
            "alerts":          True,
            "fast_adapt":      tier >= LicenseTier.PROFESSIONAL,
            "tactical":        tier >= LicenseTier.ENTERPRISE,
            "defense_modules": tier >= LicenseTier.DEFENSE,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
