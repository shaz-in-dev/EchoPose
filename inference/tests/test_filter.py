"""tests/test_filter.py — SkeletonFilter EMA smoothing coverage."""

import pytest
import numpy as np
from pipeline.filter import SkeletonFilter


def _make_keypoints(offset=0.0):
    return [
        {"x": 0.5 + offset, "y": 0.1 * i, "z": 0.5, "confidence": 0.9}
        for i in range(17)
    ]


@pytest.fixture
def filt():
    return SkeletonFilter(alpha=0.4, max_people=3)


def test_single_person_passthrough(filt):
    kps = [_make_keypoints()]
    result = filt.filter(kps)
    assert len(result) == 1
    assert len(result[0]) == 17


def test_first_frame_copies_state(filt):
    kps = [_make_keypoints()]
    result = filt.filter(kps)
    for i in range(17):
        assert abs(result[0][i]["x"] - kps[0][i]["x"]) < 1e-5


def test_ema_smoothing(filt):
    # Feed same frame twice, then a shifted frame — result should lag
    kps_normal = [_make_keypoints(0.0)]
    filt.filter(kps_normal)
    filt.filter(kps_normal)

    kps_shifted = [_make_keypoints(0.5)]
    result = filt.filter(kps_shifted)
    # x should be between 0.5 and 1.0 (0.4 * 1.0 + 0.6 * 0.5 = 0.7)
    assert 0.5 < result[0][0]["x"] < 1.0


def test_multi_person(filt):
    kps = [_make_keypoints(0.0), _make_keypoints(0.2)]
    result = filt.filter(kps)
    assert len(result) == 2
    assert len(result[1]) == 17


def test_max_people_limit():
    filt = SkeletonFilter(alpha=0.4, max_people=1)
    kps = [_make_keypoints(0.0), _make_keypoints(0.5)]
    result = filt.filter(kps)
    assert len(result) == 1


def test_empty_keypoints(filt):
    result = filt.filter([])
    assert result == []


def test_wrong_keypoint_count(filt):
    bad = [{"x": 0, "y": 0, "z": 0, "confidence": 0.5}] * 10
    result = filt.filter([bad])
    assert result == [bad]


def test_confidence_preserved(filt):
    kps = [_make_keypoints()]
    result = filt.filter(kps)
    assert result[0][0]["confidence"] == 0.9


def test_alpha_zero_no_update():
    filt = SkeletonFilter(alpha=0.0, max_people=1)
    kps_a = [_make_keypoints(0.0)]
    filt.filter(kps_a)
    kps_b = [_make_keypoints(1.0)]
    result = filt.filter(kps_b)
    # alpha=0 means state never updates from initial
    assert abs(result[0][0]["x"] - 0.5) < 1e-5


def test_alpha_one_immediate():
    filt = SkeletonFilter(alpha=1.0, max_people=1)
    kps_a = [_make_keypoints(0.0)]
    filt.filter(kps_a)
    kps_b = [_make_keypoints(1.0)]
    result = filt.filter(kps_b)
    assert abs(result[0][0]["x"] - 1.5) < 1e-5
