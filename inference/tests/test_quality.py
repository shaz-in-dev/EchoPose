"""tests/test_quality.py — echopose_sdk quality module coverage."""

import pytest
import numpy as np

import sys, os
# echopose_sdk lives at workspace root, not under inference/
_workspace = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_workspace, "echopose_sdk", "src"))

from echopose_sdk.quality import summarize_confidence


def test_normal_confidences():
    result = summarize_confidence([0.8, 0.9, 0.7])
    assert result["mean"] == pytest.approx(0.8, abs=0.01)
    assert result["min"] == pytest.approx(0.7, abs=0.01)
    assert result["max"] == pytest.approx(0.9, abs=0.01)


def test_empty_list():
    result = summarize_confidence([])
    assert result["mean"] == 0.0
    assert result["min"] == 0.0
    assert result["max"] == 0.0


def test_single_value():
    result = summarize_confidence([0.5])
    assert result["mean"] == pytest.approx(0.5, abs=0.01)
    assert result["min"] == pytest.approx(0.5, abs=0.01)
    assert result["max"] == pytest.approx(0.5, abs=0.01)


def test_generator_input():
    result = summarize_confidence(x / 10 for x in range(1, 6))
    assert result["min"] == pytest.approx(0.1, abs=0.01)
    assert result["max"] == pytest.approx(0.5, abs=0.01)


def test_all_zeros():
    result = summarize_confidence([0.0, 0.0, 0.0])
    assert result["mean"] == 0.0


def test_all_ones():
    result = summarize_confidence([1.0, 1.0])
    assert result["mean"] == pytest.approx(1.0)
    assert result["min"] == pytest.approx(1.0)
