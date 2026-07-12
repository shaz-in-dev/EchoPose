"""
scripts/csi_inspector.py — Live CSI signal inspector.

Connects to the aggregator WebSocket and renders real-time diagnostics:
  - Amplitude heatmap per node × subcarrier
  - Per-node frame rate and dropout rate
  - Doppler energy signal (aggregate motion proxy)
  - Background noise floor vs current amplitude

Requires: pip install websockets numpy rich
Optional: pip install matplotlib  (for graphical heatmap mode)

Usage:
  # Text mode (rich terminal):
  python scripts/csi_inspector.py

  # Matplotlib mode:
  python scripts/csi_inspector.py --mode plot

  # Custom aggregator URL:
  python scripts/csi_inspector.py --url ws://192.168.1.10:3000/ws
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from collections import defaultdict, deque
from typing import Dict, List, Optional

import numpy as np

SUBCARRIERS = 64


# ── Stats tracker ─────────────────────────────────────────────────────────────

class NodeStats:
    def __init__(self, node_id: str, window: int = 100):
        self.node_id    = node_id
        self.frame_times: deque = deque(maxlen=window)
        self.drops       = 0
        self.last_amps:  Optional[np.ndarray] = None
        self.baseline:   Optional[np.ndarray] = None
        self._baseline_samples: List[np.ndarray] = []
        self._calibrated = False
        self._doppler:   deque = deque(maxlen=200)

    def push(self, amplitudes: List[float]) -> None:
        now = time.monotonic()
        self.frame_times.append(now)
        arr = np.array(amplitudes[:SUBCARRIERS], dtype=np.float32)
        if arr.size < SUBCARRIERS:
            self.drops += 1
            return
        self.last_amps = arr
        if not self._calibrated:
            self._baseline_samples.append(arr)
            if len(self._baseline_samples) >= 30:
                self.baseline = np.mean(self._baseline_samples, axis=0)
                self._calibrated = True
        if self.baseline is not None:
            self._doppler.append(float(np.var(arr - self.baseline)))

    @property
    def fps(self) -> float:
        ts = list(self.frame_times)
        if len(ts) < 2:
            return 0.0
        elapsed = ts[-1] - ts[0]
        return (len(ts) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def doppler_energy(self) -> float:
        return float(np.mean(list(self._doppler))) if self._doppler else 0.0

    @property
    def snr_db(self) -> float:
        if self.last_amps is None or self.baseline is None:
            return 0.0
        signal = float(np.mean(np.abs(self.last_amps - self.baseline)))
        noise  = float(np.std(self.baseline)) + 1e-9
        return 20 * np.log10(signal / noise + 1e-9)


# ── Text inspector (rich) ─────────────────────────────────────────────────────

async def run_text_inspector(url: str) -> None:
    try:
        from rich.live import Live
        from rich.table import Table
        from rich.text import Text
        from rich.console import Console
        from rich import box
    except ImportError:
        print("Install 'rich' for the text UI:  pip install rich")
        return

    import websockets

    console = Console()
    stats: Dict[str, NodeStats] = {}
    total_bundles = 0

    def make_table() -> Table:
        t = Table(title="EchoPose — CSI Inspector", box=box.ROUNDED, expand=True)
        t.add_column("Node",    style="cyan",  width=10)
        t.add_column("FPS",     style="green", width=8)
        t.add_column("Drops",   style="red",   width=8)
        t.add_column("Doppler", style="yellow",width=10)
        t.add_column("SNR dB",  style="magenta",width=10)
        t.add_column("Calibrated", width=12)
        t.add_column("Subcarrier Amplitude (mini-bar)", min_width=40)
        for nid, ns in stats.items():
            amp_bar = _mini_bar(ns.last_amps) if ns.last_amps is not None else "—"
            t.add_row(
                nid,
                f"{ns.fps:.1f}",
                str(ns.drops),
                f"{ns.doppler_energy:.4f}",
                f"{ns.snr_db:.1f}",
                "✔" if ns._calibrated else "…",
                amp_bar,
            )
        t.caption = f"Bundles received: {total_bundles}  |  Nodes: {len(stats)}"
        return t

    with Live(make_table(), refresh_per_second=5, console=console) as live:
        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    console.log(f"[green]Connected to {url}[/green]")
                    async for raw in ws:
                        bundle = json.loads(raw)
                        for frame in bundle.get("frames", []):
                            nid  = str(frame.get("node_id", "?"))
                            amps = frame.get("amplitudes", [])
                            if nid not in stats:
                                stats[nid] = NodeStats(nid)
                            stats[nid].push(amps)
                        total_bundles += 1
                        live.update(make_table())
            except Exception as exc:
                console.log(f"[red]Disconnected: {exc}. Reconnecting in 3 s...[/red]")
                await asyncio.sleep(3)


def _mini_bar(amps: np.ndarray, width: int = 48) -> str:
    """Render amplitude array as a compact unicode bar graph."""
    blocks = " ▁▂▃▄▅▆▇█"
    n = len(amps)
    bucket = max(1, n // width)
    result = []
    mn, mx = amps.min(), amps.max()
    rng = mx - mn or 1.0
    for i in range(0, n, bucket):
        val = float(amps[i:i+bucket].mean())
        idx = int((val - mn) / rng * (len(blocks) - 1))
        result.append(blocks[idx])
    return "".join(result)


# ── Matplotlib inspector ──────────────────────────────────────────────────────

async def run_plot_inspector(url: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
    except ImportError:
        print("Install 'matplotlib':  pip install matplotlib")
        return

    import websockets

    stats: Dict[str, NodeStats] = {}
    lock = asyncio.Lock()

    fig, axes = plt.subplots(2, 3, figsize=(14, 6))
    fig.suptitle("EchoPose — CSI Inspector")
    fig.tight_layout(pad=2.0)

    async def fetch():
        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    async for raw in ws:
                        bundle = json.loads(raw)
                        async with lock:
                            for frame in bundle.get("frames", []):
                                nid  = str(frame.get("node_id", "?"))
                                amps = frame.get("amplitudes", [])
                                if nid not in stats:
                                    stats[nid] = NodeStats(nid)
                                stats[nid].push(amps)
            except Exception:
                await asyncio.sleep(3)

    def update(_):
        for ax in axes.flat:
            ax.clear()
        node_list = sorted(stats.keys())[:6]
        for i, nid in enumerate(node_list):
            ax = axes.flat[i]
            ns = stats[nid]
            if ns.last_amps is not None:
                ax.bar(range(len(ns.last_amps)), ns.last_amps, color="steelblue", width=1)
                if ns.baseline is not None:
                    ax.plot(range(len(ns.baseline)), ns.baseline, "r--", linewidth=1)
            ax.set_title(f"Node {nid} | {ns.fps:.1f} Hz | Doppler {ns.doppler_energy:.4f}")
            ax.set_xlabel("Subcarrier")
            ax.set_ylabel("Amplitude")
        return axes.flat

    loop = asyncio.get_event_loop()
    loop.create_task(fetch())
    ani = animation.FuncAnimation(fig, update, interval=200)
    plt.show()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="EchoPose CSI Inspector")
    p.add_argument("--url",  default="ws://localhost:3000/ws",
                   help="Aggregator WebSocket URL")
    p.add_argument("--mode", choices=["text", "plot"], default="text",
                   help="Display mode: text (rich) or plot (matplotlib)")
    args = p.parse_args()

    if args.mode == "plot":
        asyncio.run(run_plot_inspector(args.url))
    else:
        asyncio.run(run_text_inspector(args.url))


if __name__ == "__main__":
    main()
