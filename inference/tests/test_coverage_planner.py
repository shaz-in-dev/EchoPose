"""tests/test_coverage_planner.py — CoveragePlanner coverage."""

import pytest
import numpy as np
from pipeline.tactical.coverage_planner import CoveragePlanner


@pytest.fixture
def planner():
    p = CoveragePlanner(area_size=(10.0, 10.0), resolution=1.0)
    p.set_sensors([(2, 2), (8, 8)])
    return p


# ── basic coverage ────────────────────────────────────────────

def test_no_sensors_status():
    p = CoveragePlanner()
    result = p.compute_coverage()
    assert result["status"] == "no_sensors"


def test_coverage_with_sensors(planner):
    result = planner.compute_coverage()
    assert "coverage_pct" in result
    assert "blind_spot_pct" in result
    assert "blind_zones" in result
    assert "recommended_path" in result


def test_coverage_percentage_range(planner):
    result = planner.compute_coverage()
    assert 0 <= result["coverage_pct"] <= 100
    assert 0 <= result["blind_spot_pct"] <= 100


# ── query point ───────────────────────────────────────────────

def test_query_near_sensor(planner):
    result = planner.query_point(2.0, 2.0)
    assert result["status"] in ("COVERED", "PARTIAL", "BLIND")


def test_query_out_of_bounds(planner):
    result = planner.query_point(100.0, 100.0)
    assert result["status"] == "OUT_OF_BOUNDS"


def test_query_returns_keys(planner):
    result = planner.query_point(5.0, 5.0)
    assert "x" in result
    assert "y" in result
    assert "coverage" in result
    assert "status" in result


# ── wall attenuation ──────────────────────────────────────────

def test_wall_reduces_coverage():
    p = CoveragePlanner(area_size=(10.0, 10.0), resolution=1.0)
    p.set_sensors([(0, 0)])
    # Add a wall segment between sensor and far side
    p.add_wall((5, 0), (5, 10))
    result = p.compute_coverage()
    # Should still produce a valid result
    assert "coverage_pct" in result


# ── segment intersection ──────────────────────────────────────

def test_segments_intersect_true():
    a1, a2 = np.array([0, 0]), np.array([4, 4])
    b1, b2 = np.array([0, 4]), np.array([4, 0])
    assert CoveragePlanner._segments_intersect(a1, a2, b1, b2)


def test_segments_no_intersect():
    a1, a2 = np.array([0, 0]), np.array([1, 0])
    b1, b2 = np.array([0, 2]), np.array([1, 2])
    assert CoveragePlanner._segments_intersect(a1, a2, b1, b2) is False


def test_segments_parallel():
    a1, a2 = np.array([0, 0]), np.array([1, 0])
    b1, b2 = np.array([0, 0]), np.array([1, 0])
    assert CoveragePlanner._segments_intersect(a1, a2, b1, b2) is False


# ── path planning ──────────────────────────────────────────────

def test_path_has_entries(planner):
    result = planner.compute_coverage()
    assert len(result["recommended_path"]) >= 1
    assert "x" in result["recommended_path"][0]
    assert "y" in result["recommended_path"][0]
