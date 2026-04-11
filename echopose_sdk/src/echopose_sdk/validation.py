from __future__ import annotations

from typing import Any, Dict, List, Tuple


def validate_bundle(bundle: Dict[str, Any], expected_nodes: int = 3, expected_subcarriers: int = 64) -> Tuple[bool, str]:
    frames: List[Dict[str, Any]] = bundle.get("frames", [])
    if not frames:
        return False, "bundle has no frames"

    nodes = set()
    for frame in frames:
        if "node_id" not in frame or "amplitudes" not in frame:
            return False, "frame missing node_id or amplitudes"
        amps = frame["amplitudes"]
        if not isinstance(amps, list) or len(amps) != expected_subcarriers:
            return False, "invalid amplitudes length"
        nodes.add(frame["node_id"])

    if len(nodes) > expected_nodes:
        return False, "too many nodes for configured topology"
    return True, "ok"
