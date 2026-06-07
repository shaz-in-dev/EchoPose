"""Home Assistant MQTT integration with auto-discovery.

Full sensor coverage for all EchoPose analytics modules. Sensors appear
in HA automatically — no YAML config required.

Required:
  HA_MQTT_BROKER=192.168.1.10

Optional:
  HA_MQTT_PORT=1883
  HA_MQTT_USERNAME=
  HA_MQTT_PASSWORD=
  HA_DISCOVERY_PREFIX=homeassistant
  HA_STATE_PREFIX=echopose
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger("rf_inference.homeassistant")

_DEVICE = {
    "identifiers": ["echopose"],
    "name": "EchoPose",
    "model": "EchoPose WiFi CSI",
    "manufacturer": "EchoPose",
    "sw_version": "0.2.0",
    "configuration_url": "http://localhost:8765",
}

# ── Entity registry ────────────────────────────────────────────────────────────
# (unique_id, component, name, device_class, unit, icon, state_class, expire_after_s)
# expire_after: HA marks the sensor unavailable if no update arrives within this many seconds.
_ENTITIES: list[tuple] = [
    # ── Presence & occupancy ──────────────────────────────────────────
    ("echopose_presence",        "binary_sensor", "Presence",         "occupancy", None,      None,                 None,          None),
    ("echopose_person_count",    "sensor",        "Person Count",     None,        "persons", "mdi:account-group",  "measurement", None),

    # ── Vital signs ───────────────────────────────────────────────────
    ("echopose_heart_rate",      "sensor",        "Heart Rate",       None,        "bpm",     "mdi:heart-pulse",    "measurement", None),
    ("echopose_respiratory_rate","sensor",        "Respiratory Rate", None,        "brpm",    "mdi:lungs",          "measurement", None),
    ("echopose_spo2",            "sensor",        "SpO2",             None,        "%",       "mdi:water-percent",  "measurement", None),

    # ── Motion & activity ─────────────────────────────────────────────
    ("echopose_activity",        "sensor",        "Activity",         "enum",      None,      "mdi:run",            None,          None),
    ("echopose_gait_speed",      "sensor",        "Gait Speed",       None,        "m/s",     "mdi:walk",           "measurement", None),
    ("echopose_gesture",         "sensor",        "Last Gesture",     None,        None,      "mdi:hand-wave",      None,          None),
    ("echopose_sleep_stage",     "sensor",        "Sleep Stage",      "enum",      None,      "mdi:sleep",          None,          None),

    # ── Emotion & wellbeing ───────────────────────────────────────────
    ("echopose_stress_score",    "sensor",        "Stress Score",     None,        "%",       "mdi:brain",          "measurement", None),

    # ── Alerts ───────────────────────────────────────────────────────
    ("echopose_fall_detected",   "binary_sensor", "Fall Detected",    "problem",   None,      "mdi:alert",          None,          30),
    ("echopose_anomaly",         "binary_sensor", "Anomaly Detected", "motion",    None,      "mdi:eye-circle",     None,          60),
    ("echopose_health_alert",    "sensor",        "Health Alert",     None,        None,      "mdi:medical-bag",    None,          None),
]


@dataclass
class HAConfig:
    broker: str
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    discovery_prefix: str = "homeassistant"
    state_prefix: str = "echopose"

    @classmethod
    def from_env(cls) -> Optional["HAConfig"]:
        broker = os.getenv("HA_MQTT_BROKER", "").strip()
        if not broker:
            return None
        return cls(
            broker=broker,
            port=int(os.getenv("HA_MQTT_PORT", "1883")),
            username=os.getenv("HA_MQTT_USERNAME") or None,
            password=os.getenv("HA_MQTT_PASSWORD") or None,
            discovery_prefix=os.getenv("HA_DISCOVERY_PREFIX", "homeassistant"),
            state_prefix=os.getenv("HA_STATE_PREFIX", "echopose"),
        )


class HomeAssistantBridge:
    """Async MQTT bridge: publishes all EchoPose analytics as HA auto-discovered entities.

    Features:
    - 13 entities covering all analytics modules (vitals, activity, alerts, etc.)
    - LWT so HA marks entities unavailable if EchoPose goes offline
    - Auto-reconnect with exponential back-off; re-publishes discovery on reconnect
    - state_class=measurement for history graphs; expire_after for transient alerts
    """

    def __init__(self, cfg: HAConfig) -> None:
        self.cfg = cfg
        self._client = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connected = False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        import paho.mqtt.client as mqtt

        self._loop = asyncio.get_running_loop()
        avail_topic = f"{self.cfg.state_prefix}/availability"

        client = mqtt.Client(
            client_id="echopose-bridge",
            clean_session=True,
            protocol=mqtt.MQTTv311,
        )

        # Last Will Testament — HA marks all entities unavailable if we disconnect uncleanly
        client.will_set(avail_topic, payload="offline", qos=1, retain=True)

        if self.cfg.username:
            client.username_pw_set(self.cfg.username, self.cfg.password)

        # Auto-reconnect: 1 s initial delay, doubles up to 30 s
        client.reconnect_delay_set(min_delay=1, max_delay=30)

        client.on_connect    = self._on_connect
        client.on_disconnect = self._on_disconnect
        self._client = client

        try:
            await self._loop.run_in_executor(
                None,
                lambda: client.connect(self.cfg.broker, self.cfg.port, keepalive=60),
            )
            client.loop_start()
            for _ in range(20):
                if self._connected:
                    break
                await asyncio.sleep(0.1)
        except Exception as exc:
            logger.warning("HA MQTT: cannot reach %s:%d — %s", self.cfg.broker, self.cfg.port, exc)
            self._client = None
            return

        if not self._connected:
            logger.warning("HA MQTT: no CONNACK from broker within 2 s")

    async def stop(self) -> None:
        if self._client:
            try:
                await self._publish(f"{self.cfg.state_prefix}/availability", "offline", retain=True, qos=1)
            except Exception:
                pass
            self._client.loop_stop()
            self._client.disconnect()
            self._connected = False

    # ── paho callbacks ─────────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, rc) -> None:
        if rc == 0:
            self._connected = True
            logger.info("HA MQTT connected to %s:%d", self.cfg.broker, self.cfg.port)
            # Re-publish discovery on every connect (handles broker restarts)
            if self._loop:
                asyncio.run_coroutine_threadsafe(self._publish_discovery(), self._loop)
        else:
            _RC = {1: "wrong protocol", 2: "bad client id", 3: "server unavailable",
                   4: "bad credentials", 5: "not authorised"}
            logger.warning("HA MQTT broker refused connection: %s (rc=%d)", _RC.get(rc, "unknown"), rc)

    def _on_disconnect(self, client, userdata, rc) -> None:
        self._connected = False
        if rc != 0:
            logger.info("HA MQTT disconnected (rc=%d), paho will reconnect automatically", rc)

    # ── Discovery ──────────────────────────────────────────────────────────────

    async def _publish_discovery(self) -> None:
        avail_topic = f"{self.cfg.state_prefix}/availability"

        for row in _ENTITIES:
            uid, component, name, device_class, unit, icon, state_class, expire_after = row
            state_topic = f"{self.cfg.state_prefix}/{uid}/state"

            payload: dict[str, Any] = {
                "name": name,
                "unique_id": uid,
                "state_topic": state_topic,
                "device": _DEVICE,
                "availability": [{"topic": avail_topic, "payload_available": "online", "payload_not_available": "offline"}],
            }

            if device_class:
                payload["device_class"] = device_class
            if unit:
                payload["unit_of_measurement"] = unit
            if icon:
                payload["icon"] = icon
            if state_class:
                payload["state_class"] = state_class
            if expire_after:
                payload["expire_after"] = expire_after

            if component == "binary_sensor":
                payload["payload_on"] = "ON"
                payload["payload_off"] = "OFF"

            # Activity and sleep stage: send options list for HA "Select" rendering
            if uid == "echopose_activity":
                payload["options"] = ["walking", "running", "sitting", "standing", "lying", "falling", "unknown"]
            if uid == "echopose_sleep_stage":
                payload["options"] = ["awake", "light", "deep", "rem", "unknown"]

            disc_topic = f"{self.cfg.discovery_prefix}/{component}/{uid}/config"
            await self._publish(disc_topic, json.dumps(payload), retain=True, qos=1)

        await self._publish(avail_topic, "online", retain=True, qos=1)
        logger.debug("HA MQTT discovery published (%d entities)", len(_ENTITIES))

    # ── State publishing ───────────────────────────────────────────────────────

    async def publish(self, analytics: dict, person_count: int) -> None:
        """Push the latest analytics snapshot to all HA sensor topics."""
        if not self._connected or self._client is None:
            return

        vitals   = analytics.get("vitals")   or {}
        activity = analytics.get("activity") or {}
        gait     = analytics.get("gait")     or {}
        fall     = analytics.get("fall")     or {}
        gestures = analytics.get("gestures") or {}
        sleep    = analytics.get("sleep")    or {}
        emotion  = analytics.get("emotion")  or {}
        alerts   = analytics.get("health_alerts") or []

        hr    = (vitals.get("heart_rate")     or {}).get("heart_rate")
        rr    = (vitals.get("respiratory_rate") or {}).get("respiratory_rate")
        spo2  = (vitals.get("spo2")           or {}).get("spo2")
        stress = emotion.get("stress_score")
        gait_speed = gait.get("gait_speed") or gait.get("speed")
        gesture_name = gestures.get("gesture") or gestures.get("name") or "none"
        sleep_stage  = sleep.get("stage") or sleep.get("sleep_stage") or "unknown"
        act_label    = activity.get("activity") or "unknown"
        fall_flag    = bool(fall.get("fall_detected", False))
        anomaly_flag = bool(analytics.get("anomaly") or (alerts and any(a.get("severity") == "critical" for a in alerts)))

        # Most recent health alert label (empty string = no alert)
        if alerts:
            top_alert = alerts[0]
            alert_str = top_alert.get("type") or top_alert.get("message") or "alert"
        else:
            alert_str = "none"

        def _fmt(val: Any, decimals: int = 1) -> str:
            return str(round(val, decimals)) if val is not None else "unavailable"

        states: dict[str, str] = {
            "echopose_presence":         "ON" if person_count > 0 else "OFF",
            "echopose_person_count":     str(person_count),
            "echopose_heart_rate":       _fmt(hr),
            "echopose_respiratory_rate": _fmt(rr),
            "echopose_spo2":             _fmt(spo2),
            "echopose_activity":         act_label,
            "echopose_gait_speed":       _fmt(gait_speed, 2),
            "echopose_gesture":          gesture_name,
            "echopose_sleep_stage":      sleep_stage,
            "echopose_stress_score":     _fmt(stress),
            "echopose_fall_detected":    "ON" if fall_flag else "OFF",
            "echopose_anomaly":          "ON" if anomaly_flag else "OFF",
            "echopose_health_alert":     alert_str,
        }

        for uid, state in states.items():
            await self._publish(f"{self.cfg.state_prefix}/{uid}/state", state)

        # Publish JSON attributes for vitals (used by HA template sensors / dashboards)
        vitals_attrs = {
            "hr_confidence":   (vitals.get("heart_rate") or {}).get("confidence"),
            "rr_confidence":   (vitals.get("respiratory_rate") or {}).get("confidence"),
            "activity_confidence": activity.get("confidence"),
            "stress_arousal":  emotion.get("arousal"),
            "stress_valence":  emotion.get("valence"),
            "gait_cadence":    gait.get("cadence"),
            "gait_stride":     gait.get("stride_length"),
            "person_count":    person_count,
            "alerts":          [a.get("type") for a in alerts] if alerts else [],
        }
        await self._publish(
            f"{self.cfg.state_prefix}/attributes",
            json.dumps({k: v for k, v in vitals_attrs.items() if v is not None}),
        )

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _publish(self, topic: str, payload: str, retain: bool = False, qos: int = 0) -> None:
        if self._client is None or self._loop is None:
            return
        client = self._client
        await self._loop.run_in_executor(
            None,
            lambda: client.publish(topic, payload, qos=qos, retain=retain),
        )
