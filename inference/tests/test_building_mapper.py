"""tests/test_building_mapper.py — BuildingMapper coverage."""

import pytest
import numpy as np
from pipeline.tactical.building_mapper import BuildingMapper


@pytest.fixture
def mapper():
    m = BuildingMapper(grid_size=(10.0, 10.0), resolution=0.5)
    m.set_node_positions([(1, 1), (9, 1), (5, 9)])
    return m


# ── accumulation ──────────────────────────────────────────────

def test_accumulate_needs_2_nodes(mapper):
    mapper.accumulate({0: np.ones(64)})
    assert mapper._samples == 0


def test_accumulate_increments(mapper):
    mapper.accumulate({0: np.ones(64) * 0.5, 1: np.ones(64) * 0.3})
    assert mapper._samples == 1


def test_reconstruct_needs_samples(mapper):
    result = mapper.reconstruct()
    assert result["status"] == "accumulating"


def test_reconstruct_after_enough_samples(mapper):
    for _ in range(10):
        csi = {
            0: np.random.rand(64) * 0.5 + 0.1,
            1: np.random.rand(64) * 0.3 + 0.1,
            2: np.random.rand(64) * 0.4 + 0.1,
        }
        mapper.accumulate(csi)
    result = mapper.reconstruct()
    assert "rooms" in result
    assert "walls" in result
    assert "materials" in result
    assert "confidence" in result


# ── grid geometry ──────────────────────────────────────────────

def test_grid_shape():
    m = BuildingMapper(grid_size=(20.0, 20.0), resolution=0.25)
    assert m.gx == 80
    assert m.gy == 80


# ── bresenham ──────────────────────────────────────────────────

def test_bresenham_horizontal():
    pts = BuildingMapper._bresenham(0, 0, 5, 0)
    assert len(pts) == 6
    assert pts[0] == (0, 0)
    assert pts[-1] == (5, 0)


def test_bresenham_vertical():
    pts = BuildingMapper._bresenham(0, 0, 0, 5)
    assert len(pts) == 6


def test_bresenham_diagonal():
    pts = BuildingMapper._bresenham(0, 0, 3, 3)
    assert pts[0] == (0, 0)
    assert pts[-1] == (3, 3)


def test_bresenham_single_point():
    pts = BuildingMapper._bresenham(2, 2, 2, 2)
    assert pts == [(2, 2)]


# ── material classification ───────────────────────────────────

def test_classify_materials(mapper):
    # Build a small grid with known values
    grid = np.zeros((mapper.gx, mapper.gy))
    wall_mask = np.zeros((mapper.gx, mapper.gy), dtype=bool)
    wall_mask[5, 5] = True
    grid[5, 5] = 30.0  # metal
    mats = mapper._classify_materials(grid, wall_mask)
    assert len(mats) >= 1
    assert mats[0]["material"] == "metal"


# ── node positions ────────────────────────────────────────────

def test_set_node_positions(mapper):
    assert len(mapper._node_positions) == 3
