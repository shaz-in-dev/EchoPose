"""Tests for Matter protocol bridge integration."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Factory ────────────────────────────────────────────────────────────────────

def test_from_env_disabled_by_default():
    from integrations.matter import MatterBridge
    env = {k: v for k, v in os.environ.items() if k != "MATTER_ENABLED"}
    with patch.dict(os.environ, env, clear=True):
        assert MatterBridge.from_env() is None


def test_from_env_disabled_explicit_false():
    from integrations.matter import MatterBridge
    with patch.dict(os.environ, {"MATTER_ENABLED": "false"}):
        assert MatterBridge.from_env() is None


def test_from_env_enabled():
    from integrations.matter import MatterBridge
    with patch.dict(os.environ, {"MATTER_ENABLED": "true", "MATTER_HTTP_PORT": "7799"}):
        bridge = MatterBridge.from_env()
        assert bridge is not None
        assert bridge.http_port == 7799
        assert bridge._base_url == "http://127.0.0.1:7799"


# ── publish() ─────────────────────────────────────────────────────────────────

def _analytics():
    return {
        "vitals": {
            "heart_rate": {"heart_rate": 75.0},
            "respiratory_rate": {"respiratory_rate": 14.0},
        },
        "fall": {"fall_detected": False},
        "activity": {"activity": "sitting"},
        "emotion": {"stress_score": 22.0},
    }


@pytest.mark.asyncio
async def test_publish_noop_when_not_ready():
    from integrations.matter import MatterBridge
    bridge = MatterBridge()
    bridge._ready = False
    bridge._client = MagicMock()
    bridge._client.post = AsyncMock()
    await bridge.publish(_analytics(), person_count=0)
    bridge._client.post.assert_not_called()


@pytest.mark.asyncio
async def test_publish_sends_correct_payload():
    from integrations.matter import MatterBridge
    bridge = MatterBridge()
    bridge._ready = True

    sent = []

    async def fake_post(url, json=None):
        sent.append(json)
        r = MagicMock()
        r.raise_for_status = MagicMock()
        return r

    bridge._client = MagicMock()
    bridge._client.post = fake_post

    await bridge.publish(_analytics(), person_count=2)

    assert len(sent) == 1
    p = sent[0]
    assert p["presence"] is True
    assert p["person_count"] == 2
    assert p["heart_rate"] == 75.0
    assert p["rr"] == 14.0
    assert p["fall_detected"] is False
    assert p["activity"] == "sitting"
    assert p["stress_score"] == 22.0


@pytest.mark.asyncio
async def test_publish_presence_false_when_no_people():
    from integrations.matter import MatterBridge
    bridge = MatterBridge()
    bridge._ready = True
    sent = []

    async def fake_post(url, json=None):
        sent.append(json)
        return MagicMock(raise_for_status=MagicMock())

    bridge._client = MagicMock()
    bridge._client.post = fake_post

    await bridge.publish({}, person_count=0)
    assert sent[0]["presence"] is False


@pytest.mark.asyncio
async def test_publish_null_vitals_when_missing():
    from integrations.matter import MatterBridge
    bridge = MatterBridge()
    bridge._ready = True
    sent = []

    async def fake_post(url, json=None):
        sent.append(json)
        return MagicMock(raise_for_status=MagicMock())

    bridge._client = MagicMock()
    bridge._client.post = fake_post

    await bridge.publish({}, person_count=1)
    assert sent[0]["heart_rate"] is None
    assert sent[0]["rr"] is None


@pytest.mark.asyncio
async def test_publish_silently_handles_error():
    from integrations.matter import MatterBridge
    import httpx
    bridge = MatterBridge()
    bridge._ready = True

    async def fail_post(url, json=None):
        raise httpx.ConnectError("refused")

    bridge._client = MagicMock()
    bridge._client.post = fail_post

    # Must not raise — bridge failures are non-fatal
    await bridge.publish(_analytics(), person_count=1)
    assert bridge._ready is False  # marks self as not ready


# ── get_pairing / get_status ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_pairing_no_client():
    from integrations.matter import MatterBridge
    bridge = MatterBridge()
    bridge._client = None
    result = await bridge.get_pairing()
    assert "error" in result


@pytest.mark.asyncio
async def test_get_status_no_client():
    from integrations.matter import MatterBridge
    bridge = MatterBridge()
    bridge._client = None
    result = await bridge.get_status()
    assert result.get("running") is False


@pytest.mark.asyncio
async def test_ping_returns_false_when_no_client():
    from integrations.matter import MatterBridge
    bridge = MatterBridge()
    bridge._client = None
    assert await bridge._ping() is False


@pytest.mark.asyncio
async def test_ping_returns_false_on_connection_error():
    from integrations.matter import MatterBridge
    import httpx
    bridge = MatterBridge()

    async def fail_get(*args, **kwargs):
        raise httpx.ConnectError("refused")

    bridge._client = MagicMock()
    bridge._client.get = fail_get
    assert await bridge._ping() is False
