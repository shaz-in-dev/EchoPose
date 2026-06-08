"""Tests for the caregiver webhook / alert system."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from integrations.webhooks import AlertEvent, EventType, WebhookManager


# ── Helpers ───────────────────────────────────────────────────────────────────

def _manager() -> WebhookManager:
    m = WebhookManager()
    m._client = MagicMock()   # prevent real HTTP calls
    return m


def _analytics(
    fall=False, hr=None, rr=None, alerts=None, activity="sitting"
) -> dict:
    return {
        "fall":   {"fall_detected": fall, "confidence": 0.95},
        "vitals": {
            "heart_rate":       {"heart_rate": hr} if hr else {},
            "respiratory_rate": {"respiratory_rate": rr} if rr else {},
        },
        "activity":      {"activity": activity},
        "health_alerts": alerts or [],
    }


# ── Registry ──────────────────────────────────────────────────────────────────

def test_register_returns_id():
    m = _manager()
    wid = m.register(url="https://example.com/hook", name="test")
    assert isinstance(wid, str)
    assert len(wid) == 8


def test_register_multiple_unique_ids():
    m = _manager()
    ids = {m.register(url=f"https://example.com/{i}") for i in range(10)}
    assert len(ids) == 10


def test_list_webhooks_empty():
    m = _manager()
    assert m.list_webhooks() == []


def test_list_webhooks_after_register():
    m = _manager()
    m.register(url="https://a.com", name="A")
    m.register(url="https://b.com", name="B")
    result = m.list_webhooks()
    assert len(result) == 2
    urls = {w["url"] for w in result}
    assert urls == {"https://a.com", "https://b.com"}


def test_unregister_existing():
    m = _manager()
    wid = m.register(url="https://example.com")
    assert m.unregister(wid) is True
    assert m.list_webhooks() == []


def test_unregister_nonexistent_returns_false():
    m = _manager()
    assert m.unregister("deadbeef") is False


def test_channel_summary_no_channels():
    m = _manager()
    summary = m.channel_summary()
    assert summary["webhooks"] == 0
    assert summary["pushover"] is False
    assert summary["telegram"] is False
    assert "last_activity_ago_s" in summary


# ── env loading ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_load_env_registers_webhook_urls():
    m = WebhookManager()
    m._client = MagicMock()
    with patch.dict(os.environ, {"WEBHOOK_URLS": "https://a.com,https://b.com"}):
        m._load_env()
    assert len(m.list_webhooks()) == 2


@pytest.mark.asyncio
async def test_load_env_sets_pushover():
    m = WebhookManager()
    m._client = MagicMock()
    with patch.dict(os.environ, {
        "PUSHOVER_APP_TOKEN": "app123",
        "PUSHOVER_USER_KEY":  "user456",
    }):
        m._load_env()
    assert m._pushover == ("app123", "user456")


@pytest.mark.asyncio
async def test_load_env_sets_telegram():
    m = WebhookManager()
    m._client = MagicMock()
    with patch.dict(os.environ, {
        "TELEGRAM_BOT_TOKEN": "bot:TOKEN",
        "TELEGRAM_CHAT_ID":   "12345",
    }):
        m._load_env()
    assert m._telegram == ("bot:TOKEN", "12345")


@pytest.mark.asyncio
async def test_load_env_ignores_empty_strings():
    m = WebhookManager()
    m._client = MagicMock()
    with patch.dict(os.environ, {"WEBHOOK_URLS": ",,  ,"}):
        m._load_env()
    assert m.list_webhooks() == []


# ── Fall detection ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_fall_detected_fires_alert():
    m = _manager()
    fired = []

    async def fake_dispatch(event):
        fired.append(event.event_type)

    m._dispatch_all = fake_dispatch
    await m.process_frame(_analytics(fall=True), person_count=1)
    assert EventType.FALL_DETECTED in fired


@pytest.mark.asyncio
async def test_fall_not_fired_when_false():
    m = _manager()
    fired = []

    async def fake_dispatch(event):
        fired.append(event.event_type)

    m._dispatch_all = fake_dispatch
    await m.process_frame(_analytics(fall=False), person_count=1)
    assert EventType.FALL_DETECTED not in fired


@pytest.mark.asyncio
async def test_fall_alert_message_is_human_readable():
    m = _manager()
    events = []

    async def capture(event):
        events.append(event)

    m._dispatch_all = capture
    await m.process_frame(_analytics(fall=True), person_count=1)
    assert events
    assert "fall" in events[0].message.lower()
    assert events[0].severity == "critical"


# ── Presence change ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_person_entered_fires_on_0_to_1():
    m = _manager()
    m._last_person_count = 0
    fired = []

    async def fake_dispatch(event):
        fired.append(event.event_type)

    m._dispatch_all = fake_dispatch
    await m.process_frame(_analytics(), person_count=1)
    assert EventType.PERSON_ENTERED in fired


@pytest.mark.asyncio
async def test_person_left_fires_on_1_to_0():
    m = _manager()
    m._last_person_count = 1
    fired = []

    async def fake_dispatch(event):
        fired.append(event.event_type)

    m._dispatch_all = fake_dispatch
    await m.process_frame(_analytics(), person_count=0)
    assert EventType.PERSON_LEFT in fired


@pytest.mark.asyncio
async def test_no_presence_event_when_count_stable():
    m = _manager()
    m._last_person_count = 2
    fired = []

    async def fake_dispatch(event):
        fired.append(event.event_type)

    m._dispatch_all = fake_dispatch
    await m.process_frame(_analytics(), person_count=2)
    assert EventType.PERSON_ENTERED not in fired
    assert EventType.PERSON_LEFT    not in fired


# ── Vitals critical ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_vitals_critical_fires_on_high_hr():
    m = _manager()
    fired = []

    async def fake_dispatch(event):
        fired.append(event.event_type)

    m._dispatch_all = fake_dispatch
    await m.process_frame(_analytics(hr=145), person_count=1)
    assert EventType.VITALS_CRITICAL in fired


@pytest.mark.asyncio
async def test_vitals_critical_fires_on_low_hr():
    m = _manager()
    fired = []

    async def fake_dispatch(event):
        fired.append(event.event_type)

    m._dispatch_all = fake_dispatch
    await m.process_frame(_analytics(hr=40), person_count=1)
    assert EventType.VITALS_CRITICAL in fired


@pytest.mark.asyncio
async def test_vitals_normal_hr_does_not_fire():
    m = _manager()
    fired = []

    async def fake_dispatch(event):
        fired.append(event.event_type)

    m._dispatch_all = fake_dispatch
    await m.process_frame(_analytics(hr=72), person_count=1)
    assert EventType.VITALS_CRITICAL not in fired


@pytest.mark.asyncio
async def test_vitals_critical_message_includes_hr():
    m = _manager()
    events = []

    async def capture(event):
        events.append(event)

    m._dispatch_all = capture
    await m.process_frame(_analytics(hr=150), person_count=1)
    critical = [e for e in events if e.event_type == EventType.VITALS_CRITICAL]
    assert critical
    assert "150" in critical[0].message


# ── Inactivity ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_inactivity_fires_after_threshold():
    m = _manager()
    # Set last activity far in the past
    m._last_activity_ts = time.time() - (4 * 3600 + 60)  # 4h 1m ago
    m._last_fired = {}  # clear debounce
    fired = []

    async def fake_dispatch(event):
        fired.append(event.event_type)

    m._dispatch_all = fake_dispatch
    await m.process_frame(_analytics(), person_count=0)
    assert EventType.INACTIVITY in fired


@pytest.mark.asyncio
async def test_inactivity_not_fired_when_recent_activity():
    m = _manager()
    m._last_activity_ts = time.time() - 60  # 1 minute ago — well within threshold
    fired = []

    async def fake_dispatch(event):
        fired.append(event.event_type)

    m._dispatch_all = fake_dispatch
    await m.process_frame(_analytics(), person_count=0)
    assert EventType.INACTIVITY not in fired


@pytest.mark.asyncio
async def test_inactivity_resets_on_presence():
    m = _manager()
    old_ts = time.time() - 10000
    m._last_activity_ts = old_ts

    await m.process_frame(_analytics(), person_count=1)
    assert m._last_activity_ts > old_ts


# ── Debounce ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_debounce_prevents_rapid_repeat():
    m = _manager()
    fired = []

    async def fake_dispatch(event):
        fired.append(event.event_type)

    m._dispatch_all = fake_dispatch

    # First call — should fire
    await m.process_frame(_analytics(fall=True), person_count=1)
    count_after_first = fired.count(EventType.FALL_DETECTED)

    # Immediate second call — must be suppressed by debounce
    await m.process_frame(_analytics(fall=True), person_count=1)
    count_after_second = fired.count(EventType.FALL_DETECTED)

    assert count_after_first == 1
    assert count_after_second == 1  # no change


@pytest.mark.asyncio
async def test_debounce_expires_after_window():
    m = _manager()
    fired = []

    async def fake_dispatch(event):
        fired.append(event.event_type)

    m._dispatch_all = fake_dispatch

    # Fire once
    await m.process_frame(_analytics(fall=True), person_count=1)
    # Manually expire the debounce
    m._last_fired[EventType.FALL_DETECTED] = time.time() - 9999

    # Should fire again now
    await m.process_frame(_analytics(fall=True), person_count=1)
    assert fired.count(EventType.FALL_DETECTED) == 2


# ── HTTP dispatch ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_http_webhook_post_called():
    m = _manager()
    posted = []

    async def fake_post(url, json=None, headers=None):
        posted.append((url, json))
        r = MagicMock()
        r.raise_for_status = MagicMock()
        return r

    m._client.post = fake_post
    wid = m.register(url="https://example.com/hook")

    event = AlertEvent(
        event_type=EventType.FALL_DETECTED,
        message="Test fall",
        severity="critical",
        details={},
        person_count=1,
    )
    await m._dispatch_all(event)

    assert len(posted) == 1
    assert posted[0][0] == "https://example.com/hook"
    payload = posted[0][1]
    assert payload["event"] == EventType.FALL_DETECTED
    assert payload["severity"] == "critical"


@pytest.mark.asyncio
async def test_http_webhook_hmac_header_added_when_secret():
    m = _manager()
    headers_sent = {}

    async def fake_post(url, json=None, headers=None):
        headers_sent.update(headers or {})
        r = MagicMock()
        r.raise_for_status = MagicMock()
        return r

    m._client.post = fake_post
    m.register(url="https://example.com", secret="mysecret")

    event = AlertEvent(
        event_type=EventType.FALL_DETECTED,
        message="test",
        severity="critical",
        details={},
        person_count=1,
    )
    await m._dispatch_all(event)
    assert "X-EchoPose-Signature" in headers_sent
    assert headers_sent["X-EchoPose-Signature"].startswith("sha256=")


@pytest.mark.asyncio
async def test_http_webhook_skips_non_subscribed_event():
    m = _manager()
    posted = []

    async def fake_post(url, json=None, headers=None):
        posted.append(url)
        r = MagicMock()
        r.raise_for_status = MagicMock()
        return r

    m._client.post = fake_post
    # Subscribe only to person_entered, not fall
    m.register(url="https://example.com", events={EventType.PERSON_ENTERED})

    event = AlertEvent(
        event_type=EventType.FALL_DETECTED,
        message="test",
        severity="critical",
        details={},
        person_count=1,
    )
    await m._dispatch_all(event)
    assert not posted  # should not be called


@pytest.mark.asyncio
async def test_http_webhook_error_does_not_raise():
    """Failed webhooks must never crash the inference loop."""
    import httpx
    m = _manager()

    async def always_fail(*args, **kwargs):
        raise httpx.ConnectError("refused")

    m._client.post = always_fail
    m.register(url="https://example.com")

    event = AlertEvent(
        event_type=EventType.FALL_DETECTED,
        message="test",
        severity="critical",
        details={},
        person_count=1,
    )
    # Should not raise
    await m._dispatch_all(event)


# ── Pushover ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pushover_called_when_configured():
    m = _manager()
    m._pushover = ("app_tok", "user_key")
    sent = []

    async def fake_post(url, data=None, json=None, headers=None):
        sent.append(url)
        return MagicMock()

    m._client.post = fake_post

    event = AlertEvent(
        event_type=EventType.FALL_DETECTED,
        message="Fall!",
        severity="critical",
        details={},
        person_count=1,
    )
    await m._dispatch_all(event)
    assert any("pushover.net" in u for u in sent)


@pytest.mark.asyncio
async def test_pushover_not_called_when_not_configured():
    m = _manager()
    m._pushover = None
    sent = []

    async def fake_post(url, *args, **kwargs):
        sent.append(url)
        return MagicMock()

    m._client.post = fake_post

    event = AlertEvent(
        event_type=EventType.FALL_DETECTED,
        message="test",
        severity="critical",
        details={},
        person_count=1,
    )
    await m._dispatch_all(event)
    assert not any("pushover.net" in u for u in sent)


# ── Telegram ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_telegram_called_when_configured():
    m = _manager()
    m._telegram = ("bot:TOKEN", "12345")
    sent = []

    async def fake_post(url, json=None, data=None, headers=None):
        sent.append(url)
        return MagicMock()

    m._client.post = fake_post

    event = AlertEvent(
        event_type=EventType.FALL_DETECTED,
        message="Fall!",
        severity="critical",
        details={},
        person_count=1,
    )
    await m._dispatch_all(event)
    assert any("telegram.org" in u for u in sent)


# ── Health alert ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_alert_fires():
    m = _manager()
    fired = []

    async def fake_dispatch(event):
        fired.append(event.event_type)

    m._dispatch_all = fake_dispatch
    analytics = _analytics(alerts=[{"type": "tachycardia", "severity": "warning"}])
    await m.process_frame(analytics, person_count=1)
    assert EventType.HEALTH_ALERT in fired


# ── channel_summary ───────────────────────────────────────────────────────────

def test_channel_summary_reflects_config():
    m = _manager()
    m.register("https://example.com")
    m._pushover = ("tok", "usr")
    m._telegram = ("bot", "chat")
    summary = m.channel_summary()
    assert summary["webhooks"]  == 1
    assert summary["pushover"]  is True
    assert summary["telegram"]  is True
