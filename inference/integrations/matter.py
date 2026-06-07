"""EchoPose Matter protocol integration.

Manages the Node.js matter-bridge subprocess and pushes EchoPose analytics
to it via localhost HTTP, which the bridge translates into live Matter
attribute updates readable by Apple Home, Google Home, and Amazon Alexa.

Enable:
    MATTER_ENABLED=true         # required
    MATTER_HTTP_PORT=7788       # optional, must match bridge
    MATTER_PORT=5540            # UDP port for Matter protocol
    MATTER_PASSCODE=20202021    # pairing passcode
    MATTER_DISCRIMINATOR=3840   # 12-bit value 0-4095

First run:
    1. Start EchoPose with MATTER_ENABLED=true
    2. Call GET /matter/pairing to get the QR code
    3. Scan the QR code in Apple Home / Google Home / Alexa app
    4. Done — sensors appear as native devices

Pairing is permanent (stored in matter-bridge/matter-storage/).
Subsequent restarts reconnect automatically.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger("rf_inference.matter")

_BRIDGE_DIR = Path(__file__).resolve().parent.parent.parent / "matter-bridge"


class MatterBridge:
    """Async client for the EchoPose Matter bridge subprocess.

    Lifecycle:
        await bridge.start()           # install deps if needed, launch Node.js bridge
        await bridge.publish(...)      # called every inference cycle
        await bridge.get_pairing()     # → {qrPairingCode, manualPairingCode, qrCodeDataUrl}
        await bridge.stop()            # graceful shutdown
    """

    def __init__(
        self,
        bridge_dir: Path = _BRIDGE_DIR,
        http_port: int = 7788,
        host: str = "",
    ) -> None:
        self.bridge_dir = bridge_dir
        self.http_port = http_port
        _host = host or os.getenv("MATTER_HOST", "127.0.0.1")
        self._base_url = f"http://{_host}:{http_port}"
        self._process: Optional[subprocess.Popen] = None
        self._client:  Optional[httpx.AsyncClient] = None
        self._ready = False

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> Optional["MatterBridge"]:
        if os.getenv("MATTER_ENABLED", "").lower() != "true":
            return None
        return cls(
            http_port=int(os.getenv("MATTER_HTTP_PORT", "7788")),
            host=os.getenv("MATTER_HOST", "127.0.0.1"),
        )

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        if not self.bridge_dir.exists():
            logger.error(
                "Matter bridge not found at %s — "
                "make sure matter-bridge/ exists in the project root",
                self.bridge_dir,
            )
            return

        self._client = httpx.AsyncClient(base_url=self._base_url, timeout=5.0)

        # If bridge is already running (e.g. started externally), just attach
        if await self._ping():
            logger.info("Matter bridge already running on port %d", self.http_port)
            self._ready = True
            return

        await self._install_deps()
        await self._spawn()

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None
            self._ready = False
            logger.info("Matter bridge stopped")

    # ── Publishing ─────────────────────────────────────────────────────────────

    async def publish(self, analytics: dict, person_count: int) -> None:
        """Push the latest inference snapshot to the Matter bridge."""
        if not self._ready or self._client is None:
            return

        vitals = analytics.get("vitals") or {}
        hr   = (vitals.get("heart_rate")       or {}).get("heart_rate")
        rr   = (vitals.get("respiratory_rate") or {}).get("respiratory_rate")
        fall = bool((analytics.get("fall") or {}).get("fall_detected", False))
        act  = str((analytics.get("activity") or {}).get("activity", "unknown"))
        stress = (analytics.get("emotion") or {}).get("stress_score")

        payload = {
            "presence":      person_count > 0,
            "person_count":  person_count,
            "heart_rate":    round(hr, 1)     if hr     is not None else None,
            "rr":            round(rr, 1)     if rr     is not None else None,
            "fall_detected": fall,
            "activity":      act,
            "stress_score":  round(stress, 1) if stress is not None else None,
        }

        try:
            await self._client.post("/state", json=payload)
        except Exception as exc:
            logger.debug("Matter publish error: %s", exc)
            self._ready = False   # will re-check on next cycle

    # ── Info endpoints ─────────────────────────────────────────────────────────

    async def get_pairing(self) -> dict:
        """Return QR code and manual pairing code for commissioning."""
        if not self._client:
            return {"error": "bridge not started"}
        try:
            r = await self._client.get("/pairing")
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            return {"error": str(exc)}

    async def get_status(self) -> dict:
        if not self._client:
            return {"running": False, "commissioned": False}
        try:
            r = await self._client.get("/status")
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            return {"running": False, "error": str(exc)}

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _install_deps(self) -> None:
        node_modules = self.bridge_dir / "node_modules"
        if node_modules.exists():
            return
        logger.info("Installing matter-bridge npm dependencies (first run, ~30 s) …")
        try:
            proc = await asyncio.create_subprocess_exec(
                "npm", "install",
                cwd=str(self.bridge_dir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.error("npm install failed:\n%s", stderr.decode())
            else:
                logger.info("matter-bridge dependencies installed")
        except FileNotFoundError:
            logger.error("Node.js not found — install Node.js ≥18 to enable Matter support")

    async def _spawn(self) -> None:
        log_path = self.bridge_dir / "bridge.log"
        log_fh = open(log_path, "a")

        env = {
            **os.environ,
            "MATTER_HTTP_PORT":     str(self.http_port),
            "MATTER_PORT":          os.getenv("MATTER_PORT", "5540"),
            "MATTER_PASSCODE":      os.getenv("MATTER_PASSCODE", "20202021"),
            "MATTER_DISCRIMINATOR": os.getenv("MATTER_DISCRIMINATOR", "3840"),
            "MATTER_STORAGE_DIR":   str(self.bridge_dir / "matter-storage"),
        }

        self._process = subprocess.Popen(
            ["node", "src/bridge.js"],
            cwd=str(self.bridge_dir),
            stdout=log_fh,
            stderr=log_fh,
            env=env,
        )
        logger.info("Matter bridge launched (pid=%d) — log: %s", self._process.pid, log_path)

        # Poll until HTTP API responds (up to 15 s — matter.js startup is slow)
        for attempt in range(30):
            await asyncio.sleep(0.5)
            if self._process.poll() is not None:
                logger.error("Matter bridge exited early (code=%d) — check %s", self._process.returncode, log_path)
                return
            if await self._ping():
                self._ready = True
                status = await self.get_status()
                logger.info(
                    "Matter bridge ready — commissioned=%s, devices=%s",
                    status.get("commissioned"),
                    status.get("devices"),
                )
                return

        logger.error("Matter bridge did not respond within 15 s — check %s", log_path)

    async def _ping(self) -> bool:
        if self._client is None:
            return False
        try:
            r = await self._client.get("/health", timeout=1.0)
            return r.status_code == 200
        except Exception:
            return False
