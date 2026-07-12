"""
scripts/session_runner.py — Guided data-collection session runner.

Walks the subject through a scripted capture protocol, logging a timestamp
for each phase. Connects to both the aggregator (CSI) and an optional
PoseSource (Kinect or mock) and records aligned .npz datasets.

Capture protocol (one session ≈ 15 minutes):
  1. Sync clap  — sharp motion to establish time alignment
  2. Stand still — 30 s baseline
  3. Walk        — 60 s walking back and forth
  4. Sit         — 60 s seated
  5. Stand up / sit down cycles — 30 s
  6. Lie down    — 30 s
  7. Fall (controlled, optional) — 5 s

Usage:
  python scripts/session_runner.py --subject alice --room living_room
  python scripts/session_runner.py --subject bob   --room bedroom --mock
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))

from kinect.pose_source import BodyFrame, PoseSource
from kinect.mock_kinect  import MockKinectSource
from kinect.recorder     import AlignedRecorder
from kinect.sync         import SyncCorrelator
from kinect.transform    import CoordTransform

try:
    import websockets
    _HAS_WS = True
except ImportError:
    _HAS_WS = False

AGGREGATOR_URL = os.getenv("AGGREGATOR_WS_URI", "ws://localhost:3000/ws")
DATA_DIR       = Path(os.getenv("SESSION_DATA_DIR", "data/sessions"))

# ── Protocol definition ────────────────────────────────────────────────────────

PROTOCOL: List[Dict] = [
    {"name": "sync_clap",        "label": "Do a SHARP CLAP now and hold still",   "duration_s": 5},
    {"name": "stand_baseline",   "label": "Stand still, arms at sides",            "duration_s": 30},
    {"name": "walk",             "label": "Walk normally back and forth",           "duration_s": 60},
    {"name": "sit",              "label": "Sit down and stay seated",               "duration_s": 60},
    {"name": "stand_sit_cycles", "label": "Stand up and sit down 3 times",         "duration_s": 30},
    {"name": "lie_down",         "label": "Lie down on the floor / bed",           "duration_s": 30},
    {"name": "recovery_stand",   "label": "Stand up slowly",                       "duration_s": 15},
]


# ── Session runner ────────────────────────────────────────────────────────────

class SessionRunner:
    def __init__(
        self,
        subject:  str,
        room:     str,
        pose_source: PoseSource,
        mock_csi: bool = False,
    ):
        self.subject    = subject
        self.room       = room
        self._source    = pose_source
        self._mock_csi  = mock_csi

        ts = time.strftime("%Y%m%d_%H%M%S")
        session_id   = f"{room}__{subject}__{ts}"
        self._outpath = DATA_DIR / f"{session_id}.npz"

        self._metadata: Dict = {
            "subject":    subject,
            "room":       room,
            "session_id": session_id,
            "protocol":   [p["name"] for p in PROTOCOL],
            "timestamps": {},
        }

        self._correlator = SyncCorrelator()
        self._recorder   = AlignedRecorder(
            output_path=self._outpath,
            transform=CoordTransform.identity(),
            metadata=self._metadata,
        )

        # CSI state
        self._latest_csi:    Optional[np.ndarray] = None
        self._latest_csi_ts: float = 0.0
        self._phase: str = "idle"

    # ── Entry point ───────────────────────────────────────────────────────────

    async def run(self) -> None:
        print(f"\n{'='*60}")
        print(f"  EchoPose Session Runner")
        print(f"  Subject: {self.subject}   Room: {self.room}")
        print(f"  Output:  {self._outpath}")
        print(f"{'='*60}\n")

        self._recorder.open()
        self._source.open()

        # Start CSI listener in background
        if _HAS_WS and not self._mock_csi:
            asyncio.create_task(self._csi_listener())
        else:
            asyncio.create_task(self._mock_csi_generator())

        # Start skeleton recorder in background
        asyncio.create_task(self._skeleton_recorder())

        # Run the guided protocol
        for step in PROTOCOL:
            await self._run_step(step)

        # Wrap up
        self._source.close()
        self._recorder.close()

        n = self._recorder.window_count
        print(f"\n{'='*60}")
        print(f"  Session complete — {n} aligned windows saved")
        print(f"  File: {self._outpath}")
        print(f"{'='*60}\n")

    async def _run_step(self, step: Dict) -> None:
        name      = step["name"]
        label     = step["label"]
        duration  = step["duration_s"]

        print(f"\n>>> PHASE: {label.upper()}")
        print(f"    Duration: {duration} s")
        input("    Press ENTER when ready...")

        t_start = time.time()
        self._phase = name
        self._metadata["timestamps"][name] = {"start": t_start}

        for remaining in range(duration, 0, -1):
            print(f"\r    {remaining:3d}s remaining — phase: {name}    ", end="", flush=True)
            await asyncio.sleep(1)

        self._metadata["timestamps"][name]["end"] = time.time()
        print(f"\r    Phase '{name}' complete.                              ")

    # ── Background workers ────────────────────────────────────────────────────

    async def _csi_listener(self) -> None:
        """Pull CSI bundles from the aggregator and update _latest_csi."""
        while True:
            try:
                async with websockets.connect(AGGREGATOR_URL) as ws:
                    async for raw in ws:
                        bundle = json.loads(raw)
                        amps = self._extract_csi(bundle)
                        if amps is not None:
                            self._latest_csi    = amps
                            self._latest_csi_ts = time.time()
            except Exception:
                await asyncio.sleep(1)

    async def _mock_csi_generator(self) -> None:
        """Generate synthetic CSI for offline testing."""
        rng = np.random.default_rng(0)
        while True:
            amps = rng.normal(1.0, 0.1, (3, 64, 16)).astype(np.float32)
            # Inject a spike at t≈2 s
            age = time.time() - (self._recorder._open_ts or time.time())
            if 1.9 < age < 2.1:
                amps += rng.normal(0, 2.0, amps.shape)
            self._latest_csi    = amps
            self._latest_csi_ts = time.time()
            await asyncio.sleep(0.05)  # 20 Hz

    async def _skeleton_recorder(self) -> None:
        """Read frames from the PoseSource and pair them with the latest CSI."""
        loop = asyncio.get_event_loop()
        while True:
            frame = await loop.run_in_executor(None, self._source.read_one)
            if frame is None:
                await asyncio.sleep(0.01)
                continue

            # Push to sync correlator
            if frame.first_body():
                vel = _body_velocity(frame)
                self._correlator.push_kinect(frame.timestamp_s, vel)

            if self._latest_csi is not None:
                csi_age = time.time() - self._latest_csi_ts
                if csi_age < 0.5:  # only pair if CSI is fresh
                    self._recorder.add_window(
                        csi_window=self._latest_csi,
                        body_frame=frame,
                        timestamp_s=frame.timestamp_s,
                    )

            await asyncio.sleep(1 / 35)  # ~30 fps poll

    @staticmethod
    def _extract_csi(bundle: Dict) -> Optional[np.ndarray]:
        """Convert aggregator bundle to (3, 64, 16) CSI array."""
        frames = bundle.get("frames", [])
        if len(frames) < 3:
            return None
        nodes = sorted(frames, key=lambda f: f.get("node_id", ""))[:3]
        arr = np.zeros((3, 64, 16), dtype=np.float32)
        for i, f in enumerate(nodes):
            amps = np.array(f.get("amplitudes", []), dtype=np.float32)
            if amps.size >= 64 * 16:
                arr[i] = amps[:64 * 16].reshape(64, 16)
        return arr


def _body_velocity(frame: BodyFrame) -> float:
    """Rough kinetic energy proxy from spine joint movement."""
    if not frame.first_body():
        return 0.0
    joints = frame.first_body()
    if len(joints) < 4:
        return 0.0
    # Use Head joint (3)
    j = joints[3]
    return abs(j.x) + abs(j.y) + abs(j.z)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="EchoPose guided data-collection session runner")
    p.add_argument("--subject", required=True, help="Subject identifier (e.g. alice)")
    p.add_argument("--room",    required=True, help="Room identifier (e.g. living_room)")
    p.add_argument("--mock",    action="store_true",
                   help="Use mock Kinect + mock CSI (no hardware needed)")
    p.add_argument("--fps",     type=int, default=30, help="Kinect target FPS")
    args = p.parse_args()

    if args.mock:
        source = MockKinectSource(fps=args.fps, emit_spike=True)
    else:
        try:
            from pykinect2 import PyKinectV2, PyKinectRuntime
            # PyKinect2Source is defined in kinect/pykinect_source.py once adapter arrives
            from kinect.pykinect_source import PyKinect2Source
            source = PyKinect2Source()
        except ImportError:
            print("PyKinect2 not found — falling back to mock source.")
            print("Install the Kinect for Windows SDK 2.0 and pip install pykinect2.")
            source = MockKinectSource(fps=args.fps, emit_spike=True)

    runner = SessionRunner(
        subject=args.subject,
        room=args.room,
        pose_source=source,
        mock_csi=args.mock,
    )
    asyncio.run(runner.run())


if __name__ == "__main__":
    main()
