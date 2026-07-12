"""
inference/kinect/joint_mapping.py — Kinect-v2 (25 joints) → COCO-17 mapping.

Kinect joint enum index → COCO keypoint index mapping table, plus helpers
to convert a 25-joint Kinect body into the 17-joint COCO skeleton EchoPose
expects in training data.

COCO-17 order (0-indexed):
  0  nose           9  left_wrist
  1  left_eye      10  right_wrist
  2  right_eye     11  left_hip
  3  left_ear      12  right_hip
  4  right_ear     13  left_knee
  5  left_shoulder 14  right_knee
  6  right_shoulder 15 left_ankle
  7  left_elbow    16  right_ankle
  8  right_elbow

Kinect-v2 joint enum (0-indexed):
  0  SpineBase      13  KneeLeft
  1  SpineMid       14  AnkleLeft
  2  Neck           15  FootLeft
  3  Head           16  HipRight
  4  ShoulderLeft   17  KneeRight
  5  ElbowLeft      18  AnkleRight
  6  WristLeft      19  FootRight
  7  HandLeft       20  SpineShoulder
  8  ShoulderRight  21  HandTipLeft
  9  ElbowRight     22  ThumbLeft
  10 WristRight     23  HandTipRight
  11 HandRight      24  ThumbRight
  12 HipLeft
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from .pose_source import Joint

# COCO index → (primary Kinect index, fallback Kinect index or None)
# Primary is used when confidence >= 0.5; fallback when primary is inferred/not-tracked.
# COCO joints that have no natural Kinect equivalent (eyes, ears) are derived
# from Head (index 3) with a fixed offset — marked with a negative offset flag.
KINECT_TO_COCO: List[Tuple[int, Optional[int]]] = [
    # (primary_kinect_idx, fallback_kinect_idx)
    (3,  None),  # 0  nose       ← Head (approximated)
    (3,  None),  # 1  left_eye   ← Head (no eye joints in Kinect-v2)
    (3,  None),  # 2  right_eye  ← Head
    (3,  None),  # 3  left_ear   ← Head
    (3,  None),  # 4  right_ear  ← Head
    (4,  20),    # 5  left_shoulder  ← ShoulderLeft  / SpineShoulder fallback
    (8,  20),    # 6  right_shoulder ← ShoulderRight / SpineShoulder fallback
    (5,  4),     # 7  left_elbow  ← ElbowLeft  / ShoulderLeft fallback
    (9,  8),     # 8  right_elbow ← ElbowRight / ShoulderRight fallback
    (6,  5),     # 9  left_wrist  ← WristLeft  / ElbowLeft fallback
    (10, 9),     # 10 right_wrist ← WristRight / ElbowRight fallback
    (12, 0),     # 11 left_hip    ← HipLeft    / SpineBase fallback
    (16, 0),     # 12 right_hip   ← HipRight   / SpineBase fallback
    (13, 12),    # 13 left_knee   ← KneeLeft   / HipLeft fallback
    (17, 16),    # 14 right_knee  ← KneeRight  / HipRight fallback
    (14, 13),    # 15 left_ankle  ← AnkleLeft  / KneeLeft fallback
    (18, 17),    # 16 right_ankle ← AnkleRight / KneeRight fallback
]

# Confidence threshold: below this we use the fallback joint (if available)
_FALLBACK_THRESHOLD = 0.5

# Derived-from-Head joints (COCO indices 1-4: eyes and ears) get low confidence
_HEAD_DERIVED_CONFIDENCE = 0.2


class JointMapper:
    """Convert a 25-joint Kinect body into a 17-joint COCO body.

    Usage::
        mapper = JointMapper()
        coco_body = mapper.map(kinect_25_joints)
    """

    def __init__(self, fallback_threshold: float = _FALLBACK_THRESHOLD):
        self._thresh = fallback_threshold

    def map(self, kinect_joints: List[Joint]) -> List[Joint]:
        """Map 25 Kinect joints → 17 COCO joints.

        Joints with no valid mapping are returned with confidence=0.0 and
        coordinates equal to the fallback (or zeroed if no fallback).
        """
        if len(kinect_joints) < 25:
            raise ValueError(
                f"Expected 25 Kinect joints, got {len(kinect_joints)}. "
                "Pass the full 25-joint Kinect body."
            )

        coco: List[Joint] = []
        for coco_idx, (primary, fallback) in enumerate(KINECT_TO_COCO):
            primary_j = kinect_joints[primary]
            # For head-derived joints, reduce confidence
            if coco_idx in (1, 2, 3, 4):
                coco.append(Joint(
                    x=primary_j.x,
                    y=primary_j.y,
                    z=primary_j.z,
                    confidence=min(primary_j.confidence, _HEAD_DERIVED_CONFIDENCE),
                ))
                continue

            if primary_j.confidence >= self._thresh or fallback is None:
                coco.append(Joint(
                    x=primary_j.x,
                    y=primary_j.y,
                    z=primary_j.z,
                    confidence=primary_j.confidence,
                ))
            else:
                fb_j = kinect_joints[fallback]
                coco.append(Joint(
                    x=fb_j.x,
                    y=fb_j.y,
                    z=fb_j.z,
                    confidence=fb_j.confidence * 0.7,  # downgrade: derived, not measured
                ))

        assert len(coco) == 17, "Mapping bug: expected 17 COCO joints"
        return coco

    def map_to_array(self, kinect_joints: List[Joint]):
        """Return shape (17, 4) numpy array: [x, y, z, confidence]."""
        import numpy as np
        coco = self.map(kinect_joints)
        return np.array([[j.x, j.y, j.z, j.confidence] for j in coco], dtype=np.float32)
