"""EchoPose Kinect data-collection pipeline.

Provides:
  PoseSource          — abstract interface all sources implement
  BodyFrame           — data shape emitted by every PoseSource
  MockKinectSource    — synthetic source for offline development
  JointMapper         — Kinect-25 → COCO-17 joint mapping
  CoordTransform      — Kinect camera-space → world-space transform
  SyncCorrelator      — motion-spike cross-correlation for CSI/pose alignment
  AlignedRecorder     — writes paired (CSI, pose) .npz datasets
"""

from .pose_source import PoseSource, BodyFrame
from .mock_kinect import MockKinectSource
from .joint_mapping import JointMapper, KINECT_TO_COCO
from .transform import CoordTransform
from .sync import SyncCorrelator
from .recorder import AlignedRecorder

__all__ = [
    "PoseSource", "BodyFrame",
    "MockKinectSource",
    "JointMapper", "KINECT_TO_COCO",
    "CoordTransform",
    "SyncCorrelator",
    "AlignedRecorder",
]
