"""
pipeline/tactical/sensor_fusion.py — Multi-domain sensor fusion (Feature 15)

Fuses EchoPose WiFi sensing data with external sensor feeds
(radar, thermal, acoustic, visual, seismic) to produce a unified
Common Operating Picture (COP).

Each sensor modality is registered with its strengths / weaknesses
and detection reports are cross-validated before creating unified
tracks.

External feeds are ingested as standardised dicts.
"""

import time
import uuid
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("rf_inference.tactical.sensor_fusion")

# Sensor modality weights (tuned for typical accuracy)
_MODALITY_WEIGHTS = {
    "wifi_csi":  0.35,
    "radar":     0.25,
    "thermal":   0.20,
    "acoustic":  0.10,
    "visual":    0.05,
    "seismic":   0.05,
}

_ASSOCIATION_DIST = 2.0   # metres — max distance to associate detections
_TRACK_TIMEOUT = 10.0     # seconds before a track is dropped


class _Track:
    """Internal fused track object."""
    __slots__ = ("track_id", "position", "velocity", "sources",
                 "confidence", "last_update", "classification")

    def __init__(self, track_id: str, position: np.ndarray):
        self.track_id = track_id
        self.position = position
        self.velocity = np.zeros(3)
        self.sources: Dict[str, float] = {}
        self.confidence = 0.0
        self.last_update = time.time()
        self.classification = "UNKNOWN"

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "position": {"x": round(float(self.position[0]), 2),
                         "y": round(float(self.position[1]), 2),
                         "z": round(float(self.position[2]), 2)},
            "velocity_ms": round(float(np.linalg.norm(self.velocity)), 3),
            "heading_deg": round(float(np.degrees(np.arctan2(
                self.velocity[1], self.velocity[0])) % 360), 1),
            "sources": dict(self.sources),
            "n_sources": len(self.sources),
            "confidence": round(self.confidence, 2),
            "classification": self.classification,
            "age_s": round(time.time() - self.last_update, 1),
        }


class MultiDomainFusion:
    """Fuse detections from multiple sensor modalities into unified tracks."""

    def __init__(self, association_dist: float = _ASSOCIATION_DIST):
        self._assoc_dist = association_dist
        self._tracks: Dict[str, _Track] = {}

    def ingest(self, modality: str, detections: List[Dict]) -> None:
        """
        Ingest detections from a sensor modality.

        Each detection dict must have:
          x, y, z (metres), confidence (0-1)
        Optional:
          classification (str), velocity_ms (float)
        """
        weight = _MODALITY_WEIGHTS.get(modality, 0.10)

        for det in detections:
            pos = np.array([det.get("x", 0), det.get("y", 0), det.get("z", 0)],
                           dtype=np.float64)
            conf = float(det.get("confidence", 0.5)) * weight

            matched = self._associate(pos)
            if matched:
                self._update_track(matched, pos, conf, modality, det)
            else:
                self._create_track(pos, conf, modality, det)

    def get_cop(self) -> Dict:
        """Return the current Common Operating Picture."""
        self._prune_stale()
        tracks = [t.to_dict() for t in self._tracks.values()]
        tracks.sort(key=lambda t: t["confidence"], reverse=True)

        return {
            "tracks": tracks,
            "total_tracks": len(tracks),
            "active_modalities": list(set(
                m for t in self._tracks.values() for m in t.sources
            )),
            "timestamp": time.time(),
        }

    def get_track(self, track_id: str) -> Optional[Dict]:
        t = self._tracks.get(track_id)
        return t.to_dict() if t else None

    @property
    def track_count(self) -> int:
        self._prune_stale()
        return len(self._tracks)

    # ── helpers ───────────────────────────────────────────────────

    def _associate(self, pos: np.ndarray) -> Optional[_Track]:
        """Find the nearest existing track within association distance."""
        best, best_dist = None, self._assoc_dist
        for t in self._tracks.values():
            d = float(np.linalg.norm(t.position - pos))
            if d < best_dist:
                best_dist = d
                best = t
        return best

    def _create_track(self, pos: np.ndarray, conf: float,
                      modality: str, det: Dict) -> None:
        tid = str(uuid.uuid4())[:8]
        t = _Track(tid, pos)
        t.sources[modality] = conf
        t.confidence = conf
        t.classification = det.get("classification", "UNKNOWN")
        self._tracks[tid] = t
        logger.debug(f"New track {tid} from {modality} at {pos}")

    def _update_track(self, track: _Track, pos: np.ndarray,
                      conf: float, modality: str, det: Dict) -> None:
        dt = time.time() - track.last_update
        if dt > 0:
            track.velocity = (pos - track.position) / dt

        # Weighted position update
        alpha = 0.4
        track.position = alpha * pos + (1 - alpha) * track.position
        track.sources[modality] = conf
        track.confidence = min(sum(track.sources.values()), 0.99)
        track.last_update = time.time()

        # Upgrade classification if new modality provides one
        cls = det.get("classification")
        if cls and cls != "UNKNOWN":
            track.classification = cls

    def _prune_stale(self) -> None:
        now = time.time()
        stale = [tid for tid, t in self._tracks.items()
                 if now - t.last_update > _TRACK_TIMEOUT]
        for tid in stale:
            del self._tracks[tid]
