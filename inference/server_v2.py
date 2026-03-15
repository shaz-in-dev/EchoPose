"""
inference/server_v2.py — High-Throughput Async Pipeline (Feature 11)

An entirely non-blocking async implementation.
Moves CPU/GPU bound ML inference into thread pools so the asyncio event loop 
can handle thousands of concurrent WebSocket connections and UI commands without dropping frames.
"""

import asyncio
import json
import logging
from faststream import FastStream
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import os
from pipeline.fusion import FusionPipeline
from gpu_server import DistributedInference

logger = logging.getLogger("rf_inference.async_server")
app = FastAPI(title="EchoPose V2 High-Throughput Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "http://localhost:8000,http://localhost:8080").split(","),
    allow_methods=["GET", "POST", "OPTIONS"], 
    allow_headers=["*"]
)

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
            features = self.fusion.process_bundle(bundle)
            
            # 2. ML Inference (Slow, dispatch to ThreadPoolExecutor so UI doesn't freeze)
            skeletons = await asyncio.to_thread(self.model.batch_inference, [features])
            
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

server = HighThroughputServer()

@app.on_event("startup")
async def startup():
    # Spin up the background worker threads immediately
    asyncio.create_task(server._infer_continuously())

@app.websocket("/ws/pose")
async def pose_stream(ws: WebSocket):
    await server.handle_client(ws)
    
@app.post("/ingest")
async def ingest_bundle(bundle: dict):
    """Aggregator sends bundles here via HTTP/WS. Queue it for non-blocking processing."""
    if not server.bundle_queue.full():
        await server.bundle_queue.put(bundle)
    return {"status": "queued"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
