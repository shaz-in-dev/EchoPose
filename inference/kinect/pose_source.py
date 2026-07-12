"""
inference/kinect/pose_source.py — Abstract PoseSource interface.

Every skeleton source (Kinect, mock, future RGB-D cameras) implements this.
The rest of the pipeline only ever sees PoseSource + BodyFrame — swapping the
physical source requires zero changes downstream.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Iterator, List, Optional


@dataclass
class Joint:
    """One body joint in Kinect camera space (metres) or COCO image space (normalised)."""
    x: float
    y: float
    z: float
    confidence: float  # 0.0 = not tracked, 0.5 = inferred, 1.0 = tracked

    def as_tuple(self):
        return (self.x, self.y, self.z, self.confidence)


@dataclass
class BodyFrame:
    """One frame of skeleton data as emitted by a PoseSource.

    Attributes
    ----------
    timestamp_s : float
        Wall-clock time (seconds) when the frame was captured.
    bodies : list[list[Joint]]
        Up to 6 bodies; each body is a list of 25 Kinect joints
        (before joint mapping) or 17 COCO joints (after mapping).
    frame_index : int
        Monotonically increasing counter from the source.
    source_id : str
        Human-readable source tag, e.g. "kinect_v2" or "mock".
    """
    timestamp_s: float
    bodies: List[List[Joint]]
    frame_index: int = 0
    source_id: str = "unknown"

    @property
    def person_count(self) -> int:
        return len(self.bodies)

    def first_body(self) -> Optional[List[Joint]]:
        return self.bodies[0] if self.bodies else None


class PoseSource(abc.ABC):
    """Abstract skeleton source.

    Usage pattern::

        source = MockKinectSource(fps=30)
        source.open()
        for frame in source.stream():
            process(frame)
        source.close()
    """

    @abc.abstractmethod
    def open(self) -> None:
        """Initialise hardware / open connection."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release hardware / close connection."""

    @abc.abstractmethod
    def stream(self) -> Iterator[BodyFrame]:
        """Yield BodyFrames continuously until close() is called."""

    @abc.abstractmethod
    def read_one(self) -> Optional[BodyFrame]:
        """Return the next BodyFrame, or None if no frame is ready."""

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
