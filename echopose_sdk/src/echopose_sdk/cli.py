"""echopose-sdk — command-line interface.

Subcommands
-----------
validate    Validate a CSI bundle JSON file.
inspect     Print a summary of a bundle (per-node stats).
metrics     Compute pose metrics between a prediction and ground-truth file.
stream      Stream and replay frames from a JSONL session file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# ── validate ──────────────────────────────────────────────────────────────────

def _cmd_validate(args: argparse.Namespace) -> None:
    from .validation import validate_bundle

    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        print(json.dumps({"error": f"File not found: {bundle_path}"}))
        sys.exit(1)

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    ok, msg = validate_bundle(
        payload,
        expected_nodes=args.expected_nodes,
        expected_subcarriers=args.expected_subcarriers,
    )
    print(json.dumps({"valid": ok, "reason": msg, "file": str(bundle_path)}))
    sys.exit(0 if ok else 1)


# ── inspect ───────────────────────────────────────────────────────────────────

def _cmd_inspect(args: argparse.Namespace) -> None:
    import numpy as np
    from .csi import subcarrier_variance, estimate_human_presence

    bundle_path = Path(args.bundle)
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))

    frames = payload.get("frames", [])
    if not frames:
        print(json.dumps({"error": "No frames in bundle"}))
        sys.exit(1)

    report = {
        "file": str(bundle_path),
        "timestamp_ms": payload.get("timestamp_ms"),
        "n_frames": len(frames),
        "nodes": [],
    }

    node_ids = sorted({f.get("node_id") for f in frames})
    if args.node is not None:
        node_ids = [n for n in node_ids if str(n) == str(args.node)]

    for nid in node_ids:
        node_frames = [f for f in frames if f.get("node_id") == nid]
        amps = [f.get("amplitudes", []) for f in node_frames if "amplitudes" in f]
        node_info: dict = {"node_id": nid, "n_frames": len(node_frames)}
        if amps:
            arr = np.array(amps, dtype=np.float32)
            node_info["subcarriers"] = arr.shape[-1] if arr.ndim >= 2 else "?"
            node_info["amplitude_mean"] = round(float(np.mean(arr)), 4)
            node_info["amplitude_std"] = round(float(np.std(arr)), 4)
            var = subcarrier_variance(arr)
            node_info["subcarrier_variance_mean"] = round(float(np.mean(var)), 4)
            node_info["human_present"] = bool(estimate_human_presence(arr))
        report["nodes"].append(node_info)

    indent = 2 if args.pretty else None
    print(json.dumps(report, indent=indent))


# ── metrics ───────────────────────────────────────────────────────────────────

def _cmd_metrics(args: argparse.Namespace) -> None:
    import numpy as np
    from .metrics import summary_report, per_joint_error_table

    pred = np.load(args.pred)
    gt = np.load(args.gt)

    report = summary_report(pred, gt)
    if args.per_joint:
        report["per_joint"] = per_joint_error_table(pred, gt)

    indent = 2 if args.pretty else None
    print(json.dumps(report, indent=indent))


# ── stream ────────────────────────────────────────────────────────────────────

def _cmd_stream(args: argparse.Namespace) -> None:
    import time
    from .streaming import BundleReader

    source = Path(args.source)
    fps = args.fps
    interval = 1.0 / fps if fps > 0 else 0

    with BundleReader(source, skip_invalid=True) as reader:
        for i, bundle in enumerate(reader):
            if args.limit and i >= args.limit:
                break
            ts = bundle.get("timestamp_ms", bundle.get("ts", "?"))
            n_frames = len(bundle.get("frames", []))
            print(f"[{ts}] {n_frames} frame(s)")
            if interval:
                time.sleep(interval)


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="echopose-sdk",
        description="EchoPose SDK command-line tools",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # validate
    p_val = sub.add_parser("validate", help="Validate a CSI bundle JSON file")
    p_val.add_argument("bundle", help="Path to bundle JSON")
    p_val.add_argument("--expected-nodes", type=int, default=3)
    p_val.add_argument("--expected-subcarriers", type=int, default=64)
    p_val.set_defaults(func=_cmd_validate)

    # inspect
    p_ins = sub.add_parser("inspect", help="Inspect and summarise a bundle file")
    p_ins.add_argument("bundle", help="Path to bundle JSON")
    p_ins.add_argument("--node", default=None, help="Filter to a specific node_id")
    p_ins.add_argument("--pretty", action="store_true")
    p_ins.set_defaults(func=_cmd_inspect)

    # metrics
    p_met = sub.add_parser("metrics", help="Compute pose metrics (.npy files)")
    p_met.add_argument("--pred", required=True, help="Predicted poses (.npy, shape N,J,3)")
    p_met.add_argument("--gt", required=True, help="Ground-truth poses (.npy, shape N,J,3)")
    p_met.add_argument("--per-joint", action="store_true", help="Include per-joint breakdown")
    p_met.add_argument("--pretty", action="store_true")
    p_met.set_defaults(func=_cmd_metrics)

    # stream
    p_str = sub.add_parser("stream", help="Stream bundles from a JSONL session file")
    p_str.add_argument("source", help="Path to .jsonl or .jsonl.gz file")
    p_str.add_argument("--fps", type=float, default=0, help="Replay frame rate (0 = as fast as possible)")
    p_str.add_argument("--limit", type=int, default=0, help="Max bundles to stream (0 = all)")
    p_str.set_defaults(func=_cmd_stream)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
