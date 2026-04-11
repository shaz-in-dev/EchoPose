"""Disaster-response heuristics over CSI-derived activity/tactical streams.

This module generates structured alerts for emergency scenarios such as:
- possible trapped-person low-motion events,
- sudden crowd surges,
- potential collapse patterns from abrupt posture transitions.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List


@dataclass
class DisasterAlert:
    level: str
    code: str
    message: str


class DisasterResponseEngine:
    def __init__(self) -> None:
        self._recent_occupancy: List[int] = []

    def _push_occupancy(self, num_people: int) -> None:
        self._recent_occupancy.append(num_people)
        if len(self._recent_occupancy) > 20:
            self._recent_occupancy.pop(0)

    def evaluate(self, analytics: Dict, tactical: Dict) -> Dict:
        alerts: List[DisasterAlert] = []

        occupancy = analytics.get("occupancy", {})
        activity = analytics.get("activity", {})
        fall = analytics.get("fall", {})
        anomalies = tactical.get("anomalies", {})

        num_people = int(occupancy.get("num_people", 0))
        self._push_occupancy(num_people)

        # Crowd surge detection
        if len(self._recent_occupancy) >= 5:
            prev = self._recent_occupancy[-5]
            if num_people - prev >= 5:
                alerts.append(
                    DisasterAlert(
                        level="WARNING",
                        code="CROWD_SURGE",
                        message="Rapid occupancy increase detected; verify evacuation routes.",
                    )
                )

        # Potential collapse/fall cluster
        if fall.get("fall_detected"):
            alerts.append(
                DisasterAlert(
                    level="CRITICAL",
                    code="FALL_EVENT",
                    message="Possible casualty event detected from sudden posture change.",
                )
            )

        # Trapped-person low motion with anomaly deviation
        if activity.get("activity") in {"lying", "sitting"} and anomalies.get("is_anomaly"):
            alerts.append(
                DisasterAlert(
                    level="WARNING",
                    code="LOW_MOTION_ANOMALY",
                    message="Persistent low-motion anomaly; possible trapped-person scenario.",
                )
            )

        level = "NORMAL"
        if any(a.level == "CRITICAL" for a in alerts):
            level = "CRITICAL"
        elif alerts:
            level = "WARNING"

        return {
            "disaster_level": level,
            "alerts": [asdict(a) for a in alerts],
            "recent_occupancy": list(self._recent_occupancy),
        }
