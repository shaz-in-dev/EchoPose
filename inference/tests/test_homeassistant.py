"""Tests for Home Assistant MQTT integration."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Config ─────────────────────────────────────────────────────────────────────

def test_config_returns_none_when_no_broker():
    from integrations.homeassistant import HAConfig
    env = {k: v for k, v in os.environ.items() if k != "HA_MQTT_BROKER"}
    with patch.dict(os.environ, env, clear=True):
        assert HAConfig.from_env() is None


def test_config_returns_config_when_broker_set():
    from integrations.homeassistant import HAConfig
    with patch.dict(os.environ, {
        "HA_MQTT_BROKER": "192.168.1.10",
        "HA_MQTT_PORT": "1884",
        "HA_MQTT_USERNAME": "user",
        "HA_MQTT_PASSWORD": "pass",
    }):
        cfg = HAConfig.from_env()
        assert cfg is not None
        assert cfg.broker == "192.168.1.10"
        assert cfg.port == 1884
        assert cfg.username == "user"
        assert cfg.password == "pass"


def test_config_defaults():
    from integrations.homeassistant import HAConfig
    with patch.dict(os.environ, {"HA_MQTT_BROKER": "localhost"}):
        cfg = HAConfig.from_env()
        assert cfg.port == 1883
        assert cfg.discovery_prefix == "homeassistant"
        assert cfg.state_prefix == "echopose"


# ── Entity registry ────────────────────────────────────────────────────────────

def test_all_entities_have_required_fields():
    from integrations.homeassistant import _ENTITIES
    for row in _ENTITIES:
        uid, component, name, *_ = row
        assert uid.startswith("echopose_"), f"{uid} must start with echopose_"
        assert component in ("sensor", "binary_sensor"), f"{uid}: bad component"
        assert name, f"{uid}: name is empty"
        assert len(row) == 8, f"{uid}: row must have 8 fields"


def test_entity_components_are_valid():
    from integrations.homeassistant import _ENTITIES
    components = {row[1] for row in _ENTITIES}
    assert components.issubset({"sensor", "binary_sensor"})


def test_binary_sensors_have_device_class():
    from integrations.homeassistant import _ENTITIES
    for uid, component, _, device_class, *_ in _ENTITIES:
        if component == "binary_sensor":
            assert device_class, f"{uid} binary_sensor must have a device_class"


def test_measurement_sensors_have_state_class():
    from integrations.homeassistant import _ENTITIES
    numeric_units = {"bpm", "brpm", "%", "m/s", "persons"}
    for uid, component, _, _, unit, _, state_class, _ in _ENTITIES:
        if unit in numeric_units:
            assert state_class == "measurement", f"{uid} with unit '{unit}' needs state_class=measurement"


# ── Discovery payload ──────────────────────────────────────────────────────────

def _make_bridge():
    from integrations.homeassistant import HAConfig, HomeAssistantBridge
    cfg = HAConfig(broker="localhost")
    bridge = HomeAssistantBridge(cfg)
    bridge._connected = True
    bridge._loop = asyncio.new_event_loop()
    bridge._client = object()  # publish() guards against None; set any truthy sentinel
    return bridge


def test_discovery_payload_required_keys():
    from integrations.homeassistant import _ENTITIES, _DEVICE, HAConfig, HomeAssistantBridge
    bridge = _make_bridge()
    published = {}

    async def fake_publish(topic, payload, retain=False, qos=0):
        if "/config" in topic:
            published[topic] = json.loads(payload)

    bridge._publish = fake_publish
    bridge._loop.run_until_complete(bridge._publish_discovery())
    bridge._loop.close()

    for topic, payload in published.items():
        assert "name" in payload
        assert "unique_id" in payload
        assert "state_topic" in payload
        assert "device" in payload
        assert "availability" in payload


def test_discovery_publishes_availability_online():
    from integrations.homeassistant import HAConfig, HomeAssistantBridge
    bridge = _make_bridge()
    avail_published = []

    async def fake_publish(topic, payload, retain=False, qos=0):
        if "availability" in topic and payload == "online":
            avail_published.append(topic)

    bridge._publish = fake_publish
    bridge._loop.run_until_complete(bridge._publish_discovery())
    bridge._loop.close()
    assert avail_published, "availability=online must be published"


# ── publish() ─────────────────────────────────────────────────────────────────

def _full_analytics():
    return {
        "vitals": {
            "heart_rate": {"heart_rate": 72.5, "confidence": 0.9},
            "respiratory_rate": {"respiratory_rate": 16.0},
            "spo2": {"spo2": 98.2},
        },
        "activity": {"activity": "walking", "confidence": 0.88},
        "gait": {"gait_speed": 1.2, "cadence": 105, "stride_length": 0.7},
        "fall": {"fall_detected": False},
        "gestures": {"gesture": "wave"},
        "sleep": {"stage": "awake"},
        "emotion": {"stress_score": 34.1, "arousal": 0.4, "valence": 0.6},
        "health_alerts": [],
    }


@pytest.mark.asyncio
async def test_publish_covers_all_entities():
    from integrations.homeassistant import _ENTITIES, HAConfig, HomeAssistantBridge
    bridge = _make_bridge()
    published_topics = set()

    async def fake_publish(topic, payload, retain=False, qos=0):
        published_topics.add(topic)

    bridge._publish = fake_publish
    await bridge.publish(_full_analytics(), person_count=2)

    for uid, *_ in _ENTITIES:
        expected = f"echopose/{uid}/state"
        assert expected in published_topics, f"Missing state publish for {uid}"


@pytest.mark.asyncio
async def test_publish_presence_on_when_people():
    bridge = _make_bridge()
    states = {}

    async def fake_publish(topic, payload, **kw):
        states[topic] = payload

    bridge._publish = fake_publish
    await bridge.publish(_full_analytics(), person_count=3)
    assert states.get("echopose/echopose_presence/state") == "ON"


@pytest.mark.asyncio
async def test_publish_presence_off_when_empty():
    bridge = _make_bridge()
    states = {}

    async def fake_publish(topic, payload, **kw):
        states[topic] = payload

    bridge._publish = fake_publish
    await bridge.publish(_full_analytics(), person_count=0)
    assert states.get("echopose/echopose_presence/state") == "OFF"


@pytest.mark.asyncio
async def test_publish_fall_on_when_detected():
    bridge = _make_bridge()
    states = {}

    async def fake_publish(topic, payload, **kw):
        states[topic] = payload

    bridge._publish = fake_publish
    a = _full_analytics()
    a["fall"]["fall_detected"] = True
    await bridge.publish(a, person_count=1)
    assert states.get("echopose/echopose_fall_detected/state") == "ON"


@pytest.mark.asyncio
async def test_publish_unavailable_when_no_vitals():
    bridge = _make_bridge()
    states = {}

    async def fake_publish(topic, payload, **kw):
        states[topic] = payload

    bridge._publish = fake_publish
    await bridge.publish({}, person_count=0)
    assert states.get("echopose/echopose_heart_rate/state") == "unavailable"
    assert states.get("echopose/echopose_respiratory_rate/state") == "unavailable"


@pytest.mark.asyncio
async def test_publish_noop_when_disconnected():
    bridge = _make_bridge()
    bridge._connected = False
    called = []

    async def fake_publish(*a, **kw):
        called.append(1)

    bridge._publish = fake_publish
    await bridge.publish(_full_analytics(), person_count=1)
    assert not called


@pytest.mark.asyncio
async def test_publish_noop_when_no_client():
    bridge = _make_bridge()
    bridge._client = None
    called = []

    async def fake_publish(*a, **kw):
        called.append(1)

    bridge._publish = fake_publish
    await bridge.publish(_full_analytics(), person_count=1)
    assert not called
