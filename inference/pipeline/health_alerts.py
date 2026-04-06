"""
pipeline/health_alerts.py — Anomaly detection & health alerts (Feature 14)

Monitors vital signs in context of current activity and fires
NORMAL / WARNING / CRITICAL alerts when values leave safe ranges.
"""

import time
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("rf_inference.health_alerts")

# Normal ranges per activity context
_RANGES = {
    "sleeping": {"hr": (40, 80), "rr": (10, 20), "spo2": (95, 100)},
    "sitting":  {"hr": (50, 100), "rr": (12, 24), "spo2": (95, 100)},
    "standing": {"hr": (55, 110), "rr": (12, 24), "spo2": (94, 100)},
    "walking":  {"hr": (70, 140), "rr": (16, 30), "spo2": (93, 100)},
    "running":  {"hr": (100, 185), "rr": (20, 45), "spo2": (92, 100)},
    "lying":    {"hr": (40, 90), "rr": (10, 20), "spo2": (95, 100)},
}

# Danger thresholds independent of activity
_CRITICAL_HR_LOW, _CRITICAL_HR_HIGH = 30, 200
_CRITICAL_RR_LOW, _CRITICAL_RR_HIGH = 6, 40
_CRITICAL_SPO2 = 88


class HealthAnomalyDetector:
    """Context-aware health anomaly detection and alerting."""

    def __init__(self):
        self._last_alert_ts: float = 0.0
        self._cooldown_s: float = 10.0

    def check(
        self,
        hr: Optional[float] = None,
        rr: Optional[float] = None,
        spo2: Optional[float] = None,
        activity: str = "sitting",
    ) -> Dict:
        """
        Evaluate vitals against context-appropriate ranges.

        Returns anomalies list and an alert_level: NORMAL / WARNING / CRITICAL.
        """
        anomalies: List[str] = []
        alert_level = "NORMAL"

        ranges = _RANGES.get(activity, _RANGES["sitting"])

        if hr is not None:
            if hr < _CRITICAL_HR_LOW or hr > _CRITICAL_HR_HIGH:
                anomalies.append(f"HR critically out of range: {hr:.0f} bpm")
                alert_level = "CRITICAL"
            elif not ranges["hr"][0] <= hr <= ranges["hr"][1]:
                anomalies.append(f"HR out of expected range for {activity}: {hr:.0f} bpm")
                alert_level = max(alert_level, "WARNING", key=_severity)

        if rr is not None:
            if rr < _CRITICAL_RR_LOW or rr > _CRITICAL_RR_HIGH:
                anomalies.append(f"RR critically out of range: {rr:.0f} bpm")
                alert_level = "CRITICAL"
            elif not ranges["rr"][0] <= rr <= ranges["rr"][1]:
                anomalies.append(f"RR abnormal for {activity}: {rr:.0f} bpm")
                alert_level = max(alert_level, "WARNING", key=_severity)

        if spo2 is not None:
            if spo2 < _CRITICAL_SPO2:
                anomalies.append(f"SpO2 critically low: {spo2:.1f}%")
                alert_level = "CRITICAL"
            elif not ranges["spo2"][0] <= spo2 <= ranges["spo2"][1]:
                anomalies.append(f"SpO2 below expected: {spo2:.1f}%")
                alert_level = max(alert_level, "WARNING", key=_severity)

        # Dangerous combinations
        if hr is not None and spo2 is not None and hr > 120 and spo2 < 92:
            anomalies.append("Tachycardia + Hypoxia — seek medical attention")
            alert_level = "CRITICAL"

        if alert_level != "NORMAL":
            now = time.time()
            if now - self._last_alert_ts > self._cooldown_s:
                self._last_alert_ts = now
                logger.warning(f"Health alert [{alert_level}]: {anomalies}")

        return {
            "anomalies_detected": len(anomalies) > 0,
            "anomalies": anomalies,
            "alert_level": alert_level,
        }


def _severity(level: str) -> int:
    return {"NORMAL": 0, "WARNING": 1, "CRITICAL": 2}.get(level, 0)
