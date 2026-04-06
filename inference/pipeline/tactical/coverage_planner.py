"""
pipeline/tactical/coverage_planner.py — Stealth infiltration & coverage mapping (Feature 14)

Models enemy WiFi sensor coverage and identifies blind spots,
shadow zones, and optimal movement paths.  Useful for planning
covert movement through monitored areas.

Input:  known/estimated sensor positions + building geometry.
Output: coverage heat-map, blind-spot polygons, recommended path.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger("rf_inference.tactical.coverage_planner")

_GRID_RES = 0.5       # metres per cell
_MAX_RANGE = 15.0      # metres — effective detection range per node
_WALL_ATTENUATION = 0.6  # multiplier per wall crossing


class CoveragePlanner:
    """Model sensor coverage and find blind spots for movement planning."""

    def __init__(self, area_size: Tuple[float, float] = (30.0, 30.0),
                 resolution: float = _GRID_RES):
        self.res = resolution
        self.nx = int(area_size[0] / resolution)
        self.ny = int(area_size[1] / resolution)
        self._sensor_positions: List[np.ndarray] = []
        self._wall_segments: List[Tuple[np.ndarray, np.ndarray]] = []

    def set_sensors(self, positions: List[Tuple[float, float]]) -> None:
        """Register estimated enemy sensor positions (metres)."""
        self._sensor_positions = [np.array(p) for p in positions]
        logger.info(f"Coverage planner: {len(positions)} sensors registered.")

    def add_wall(self, start: Tuple[float, float], end: Tuple[float, float]) -> None:
        """Add a wall segment (blocks/attenuates sensor coverage)."""
        self._wall_segments.append((np.array(start), np.array(end)))

    def compute_coverage(self) -> Dict:
        """Build coverage heat-map and identify blind spots."""
        if not self._sensor_positions:
            return {"status": "no_sensors"}

        grid = np.zeros((self.nx, self.ny), dtype=np.float64)

        for sensor in self._sensor_positions:
            self._paint_sensor(grid, sensor)

        # Normalise to 0–1 (0 = no coverage, 1 = full coverage)
        if np.max(grid) > 0:
            grid /= np.max(grid)

        blind = grid < 0.1
        partial = (grid >= 0.1) & (grid < 0.5)

        blind_zones = self._extract_zones(blind)
        path = self._plan_path(grid)

        return {
            "coverage_pct": round(float(np.mean(grid > 0.1)) * 100, 1),
            "blind_spot_pct": round(float(np.mean(blind)) * 100, 1),
            "blind_zones": blind_zones,
            "recommended_path": path,
            "grid_shape": [self.nx, self.ny],
            "resolution_m": self.res,
        }

    def query_point(self, x: float, y: float) -> Dict:
        """Check coverage level at a specific coordinate."""
        ci = int(x / self.res)
        cj = int(y / self.res)
        if 0 <= ci < self.nx and 0 <= cj < self.ny:
            grid = np.zeros((self.nx, self.ny))
            for s in self._sensor_positions:
                self._paint_sensor(grid, s)
            if np.max(grid) > 0:
                grid /= np.max(grid)
            level = float(grid[ci, cj])
            return {"x": x, "y": y, "coverage": round(level, 3),
                    "status": "BLIND" if level < 0.1 else
                              "PARTIAL" if level < 0.5 else "COVERED"}
        return {"x": x, "y": y, "coverage": 0.0, "status": "OUT_OF_BOUNDS"}

    # ── helpers ───────────────────────────────────────────────────

    def _paint_sensor(self, grid: np.ndarray, sensor: np.ndarray) -> None:
        """Add one sensor's coverage to the grid (inverse-square, wall-attenuated)."""
        sx = int(sensor[0] / self.res)
        sy = int(sensor[1] / self.res)
        r_cells = int(_MAX_RANGE / self.res)

        for dx in range(-r_cells, r_cells + 1):
            for dy in range(-r_cells, r_cells + 1):
                cx, cy = sx + dx, sy + dy
                if not (0 <= cx < self.nx and 0 <= cy < self.ny):
                    continue
                dist = np.sqrt(dx ** 2 + dy ** 2) * self.res
                if dist > _MAX_RANGE or dist < 0.1:
                    continue
                strength = 1.0 / (dist ** 2 + 1.0)

                # Attenuate for each wall crossing
                cell_pos = np.array([cx * self.res, cy * self.res])
                wall_crossings = self._count_wall_crossings(sensor, cell_pos)
                strength *= _WALL_ATTENUATION ** wall_crossings

                grid[cx, cy] += strength

    def _count_wall_crossings(self, p1: np.ndarray, p2: np.ndarray) -> int:
        """Count how many wall segments the ray from p1→p2 crosses."""
        count = 0
        for ws, we in self._wall_segments:
            if self._segments_intersect(p1, p2, ws, we):
                count += 1
        return count

    @staticmethod
    def _segments_intersect(a1: np.ndarray, a2: np.ndarray,
                            b1: np.ndarray, b2: np.ndarray) -> bool:
        """Test if line segment a1→a2 intersects segment b1→b2."""
        d1 = a2 - a1
        d2 = b2 - b1
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cross) < 1e-12:
            return False
        t = ((b1[0] - a1[0]) * d2[1] - (b1[1] - a1[1]) * d2[0]) / cross
        u = ((b1[0] - a1[0]) * d1[1] - (b1[1] - a1[1]) * d1[0]) / cross
        return 0 <= t <= 1 and 0 <= u <= 1

    def _extract_zones(self, mask: np.ndarray) -> List[Dict]:
        """Convert blind-spot mask to zone list."""
        from scipy.ndimage import label as cc_label
        labelled, n = cc_label(mask)
        zones = []
        for zid in range(1, min(n + 1, 50)):
            ys, xs = np.where(labelled == zid)
            zones.append({
                "zone_id": zid,
                "area_m2": round(float(len(xs)) * self.res ** 2, 1),
                "centre": [round(float(np.mean(xs)) * self.res, 1),
                           round(float(np.mean(ys)) * self.res, 1)],
            })
        return zones

    def _plan_path(self, grid: np.ndarray) -> List[Dict]:
        """
        Greedy minimum-exposure path from bottom-left to top-right.

        Uses a simple gradient-descent on the coverage grid.
        For production use, replace with A* or Dijkstra.
        """
        start = (0, 0)
        goal = (self.nx - 1, self.ny - 1)
        path = [start]
        current = list(start)

        visited = set()
        visited.add(tuple(current))

        for _ in range(self.nx * self.ny):
            if tuple(current) == goal:
                break
            best_next = None
            best_cost = float("inf")
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (1, 1), (-1, 1), (1, -1)]:
                nx_, ny_ = current[0] + dx, current[1] + dy
                if not (0 <= nx_ < self.nx and 0 <= ny_ < self.ny):
                    continue
                if (nx_, ny_) in visited:
                    continue
                # Cost = coverage level + distance-to-goal penalty
                cost = grid[nx_, ny_] + 0.01 * np.sqrt(
                    (nx_ - goal[0]) ** 2 + (ny_ - goal[1]) ** 2)
                if cost < best_cost:
                    best_cost = cost
                    best_next = (nx_, ny_)
            if best_next is None:
                break
            current = list(best_next)
            visited.add(tuple(current))
            path.append(tuple(current))

        return [{"x": round(p[0] * self.res, 1),
                 "y": round(p[1] * self.res, 1)} for p in path[::max(1, len(path) // 20)]]
