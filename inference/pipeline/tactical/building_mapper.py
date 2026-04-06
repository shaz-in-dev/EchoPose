"""
pipeline/tactical/building_mapper.py — Building layout reconstruction (Feature 2)

Reconstructs a coarse 3-D floor-plan from multi-node WiFi CSI
reflection patterns.  Identifies wall positions, material composition,
room boundaries, and large obstacles.

Method:
  1. Collect CSI from ≥3 nodes at known positions.
  2. Build a 2-D attenuation grid via tomographic inversion.
  3. Threshold the grid to locate walls / obstacles.
  4. Segment rooms via connected-component labelling.
"""

import numpy as np
from scipy.ndimage import label as cc_label, uniform_filter
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("rf_inference.tactical.building_mapper")

# Empirical attenuation coefficients (dB/m) at 2.4 GHz
_ATT_AIR = 0.0
_ATT_DRYWALL = 3.0
_ATT_CONCRETE = 12.0
_ATT_METAL = 30.0
_ATT_WOOD = 5.0

_GRID_RES = 0.25        # metres per cell
_WALL_THRESHOLD = 6.0   # dB — above this we infer a wall


class BuildingMapper:
    """Reconstruct building layout from multi-node CSI."""

    def __init__(self, grid_size: Tuple[float, float] = (20.0, 20.0),
                 resolution: float = _GRID_RES):
        self.gx = int(grid_size[0] / resolution)
        self.gy = int(grid_size[1] / resolution)
        self.res = resolution
        self._node_positions: List[np.ndarray] = []
        self._attenuation_grid = np.zeros((self.gx, self.gy), dtype=np.float64)
        self._samples = 0

    def set_node_positions(self, positions: List[Tuple[float, float]]) -> None:
        """Register known transmitter/receiver node positions (metres)."""
        self._node_positions = [np.array(p) for p in positions]
        logger.info(f"Registered {len(positions)} node positions.")

    def accumulate(self, node_csi: Dict[int, np.ndarray]) -> None:
        """
        Feed one frame of per-node CSI.  node_csi maps node_index →
        amplitude array.  Requires ≥2 nodes per call.
        """
        ids = sorted(node_csi.keys())
        if len(ids) < 2 or len(self._node_positions) < 2:
            return

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if ids[i] >= len(self._node_positions) or ids[j] >= len(self._node_positions):
                    continue
                a_amp = node_csi[ids[i]]
                b_amp = node_csi[ids[j]]
                a_pow = float(np.mean(a_amp ** 2)) + 1e-12
                b_pow = float(np.mean(b_amp ** 2)) + 1e-12
                loss_db = 10.0 * np.log10(a_pow / b_pow)
                self._project_ray(
                    self._node_positions[ids[i]],
                    self._node_positions[ids[j]],
                    abs(loss_db),
                )
        self._samples += 1

    def reconstruct(self) -> Dict:
        """Return the current floor-plan estimate."""
        if self._samples < 5:
            return {"status": "accumulating", "samples": self._samples}

        smoothed = uniform_filter(self._attenuation_grid, size=3)
        wall_mask = smoothed > _WALL_THRESHOLD

        # Connected-component room segmentation
        room_mask = ~wall_mask
        labelled, n_rooms = cc_label(room_mask)

        rooms = []
        for rid in range(1, n_rooms + 1):
            ys, xs = np.where(labelled == rid)
            rooms.append({
                "room_id": rid,
                "area_m2": round(float(len(xs)) * self.res ** 2, 1),
                "centre": [round(float(np.mean(xs)) * self.res, 1),
                           round(float(np.mean(ys)) * self.res, 1)],
            })

        walls = self._extract_walls(wall_mask)
        materials = self._classify_materials(smoothed, wall_mask)

        return {
            "rooms": rooms,
            "walls": walls,
            "materials": materials,
            "grid_resolution_m": self.res,
            "grid_shape": [self.gx, self.gy],
            "confidence": round(min(self._samples / 50, 0.90), 2),
        }

    # ── helpers ───────────────────────────────────────────────────

    def _project_ray(self, p1: np.ndarray, p2: np.ndarray, loss_db: float) -> None:
        """Bresenham-style ray projection onto the attenuation grid."""
        c1 = (p1 / self.res).astype(int)
        c2 = (p2 / self.res).astype(int)
        pts = self._bresenham(c1[0], c1[1], c2[0], c2[1])
        if not pts:
            return
        per_cell = loss_db / len(pts)
        for x, y in pts:
            if 0 <= x < self.gx and 0 <= y < self.gy:
                self._attenuation_grid[x, y] += per_cell

    @staticmethod
    def _bresenham(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
        pts = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            pts.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return pts

    def _extract_walls(self, wall_mask: np.ndarray) -> List[Dict]:
        """Return list of wall segments from the binary mask."""
        labelled, n = cc_label(wall_mask)
        walls = []
        for wid in range(1, n + 1):
            ys, xs = np.where(labelled == wid)
            walls.append({
                "wall_id": wid,
                "length_m": round(float(len(xs)) * self.res, 1),
                "start": [int(np.min(xs)), int(np.min(ys))],
                "end": [int(np.max(xs)), int(np.max(ys))],
            })
        return walls

    def _classify_materials(self, grid: np.ndarray,
                            wall_mask: np.ndarray) -> List[Dict]:
        """Classify wall material from attenuation magnitude."""
        labelled, n = cc_label(wall_mask)
        mats = []
        for wid in range(1, n + 1):
            cells = grid[labelled == wid]
            mean_att = float(np.mean(cells))
            if mean_att > 25:
                mat = "metal"
            elif mean_att > 10:
                mat = "concrete"
            elif mean_att > 4:
                mat = "wood"
            else:
                mat = "drywall"
            mats.append({"wall_id": wid, "material": mat,
                         "attenuation_db_m": round(mean_att, 1)})
        return mats
