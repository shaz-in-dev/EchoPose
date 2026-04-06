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
from gpu_server import DistributedInference
from security import limiter, verify_api_key

logger = logging.getLogger("rf_inference.async_server")

AGGREGATOR_WS = os.getenv("AGGREGATOR_WS_URI", "ws://localhost:3000/ws")

class HighThroughputServer:
    """Manages concurrent UI clients and non-blocking inference decoupling"""
    def __init__(self):
        self.clients = set()
        self.fusion = FusionPipeline()
        self.model = DistributedInference()
        self.bundle_queue = asyncio.Queue(maxsize=100)
        
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

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Spin up background workers and stop them cleanly during shutdown.
    worker = asyncio.create_task(server._infer_continuously())
    agg_task = asyncio.create_task(server.aggregator_loop())
    try:
        yield
    finally:
        worker.cancel()
        agg_task.cancel()
        await asyncio.gather(worker, agg_task, return_exceptions=True)


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
    limiter.check_rate_limit(request.client.host)
    if not server.bundle_queue.full():
        await server.bundle_queue.put(bundle)
    return {"status": "queued"}

@app.get("/health")
async def health(request: Request):
    limiter.check_rate_limit(request.client.host)
    return {
        "status": "ok",
        "ui_clients": len(server.clients),
        "queue_depth": server.bundle_queue.qsize(),
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
