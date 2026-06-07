"""EchoPose caregiver alert system.

Sends real-time push notifications when important events happen —
fall detected, no activity for hours, abnormal heart rate, etc.

Designed for the primary use case: a son or daughter monitoring
an elderly parent at home, no cameras required.

Supported channels:
  - Pushover    → instant phone alert (https://pushover.net, free)
  - Telegram    → free global messaging via bot
  - HTTP        → any webhook URL (IFTTT, Zapier, n8n, Home Assistant)

Configure in .env:
  PUSHOVER_APP_TOKEN=...
  PUSHOVER_USER_KEY=...
  TELEGRAM_BOT_TOKEN=...
  TELEGRAM_CHAT_ID=...
  WEBHOOK_URLS=https://hook.example.com/alert
  INACTIVITY_ALERT_MINUTES=240   (alert if no movement for 4 hours)
  WEBHOOK_DEBOUNCE_SECONDS=60    (min gap between repeated alerts)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

import httpx

logger = logging.getLogger("rf_inference.webhooks")


# ── Event types ───────────────────────────────────────────────────────────────

class EventType:
    FALL_DETECTED   = "fall_detected"
    PERSON_ENTERED  = "person_entered"
    PERSON_LEFT     = "person_left"
    INACTIVITY      = "inactivity_alert"
    HEALTH_ALERT    = "health_alert"
    VITALS_CRITICAL = "vitals_critical"
    ANOMALY         = "anomaly"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class WebhookConfig:
    id:     str
    url:    str
    name:   str
    events: set        # empty set = subscribe to all events
    secret: Optional[str] = None   # HMAC-SHA256 signing secret


@dataclass
class AlertEvent:
    event_type:   str
    message:      str
    severity:     str    # "info" | "warning" | "critical"
    details:      dict
    person_count: int
    timestamp:    float = field(default_factory=time.time)


# ── Per-event debounce windows ─────────────────────────────────────────────────

_DEBOUNCE: dict[str, float] = {
    EventType.FALL_DETECTED:   60,    # 1 min  — critical, but don't spam
    EventType.PERSON_ENTERED:  30,    # 30 s
    EventType.PERSON_LEFT:     30,    # 30 s
    EventType.INACTIVITY:    1800,    # 30 min — re-alert if still inactive
    EventType.HEALTH_ALERT:   300,    # 5 min
    EventType.VITALS_CRITICAL:300,    # 5 min
    EventType.ANOMALY:        120,    # 2 min
}

_INACTIVITY_THRESHOLD = float(os.getenv("INACTIVITY_ALERT_MINUTES", "240")) * 60  # seconds
_GLOBAL_DEBOUNCE      = float(os.getenv("WEBHOOK_DEBOUNCE_SECONDS", "60"))


# ── Manager ───────────────────────────────────────────────────────────────────

class WebhookManager:
    """Tracks sensor events and dispatches alert notifications.

    Call process_frame() every inference cycle.
    """

    def __init__(self) -> None:
        self._webhooks:   dict[str, WebhookConfig] = {}
        self._last_fired: dict[str, float]         = {}
        self._last_person_count = 0
        self._last_activity_ts  = time.time()
        self._client: Optional[httpx.AsyncClient]  = None

        # Optional direct channels
        self._pushover: Optional[tuple[str, str]] = None  # (app_token, user_key)
        self._telegram: Optional[tuple[str, str]] = None  # (bot_token, chat_id)

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, connect=5.0),
            headers={"User-Agent": "EchoPose/0.2.0"},
        )
        self._load_env()

    async def stop(self) -> None:
        if self._client:
            await self._client.aclose()

    def _load_env(self) -> None:
        for url in (u.strip() for u in os.getenv("WEBHOOK_URLS", "").split(",") if u.strip()):
            self.register(url=url, name="env-configured")

        tok = os.getenv("PUSHOVER_APP_TOKEN", "").strip()
        usr = os.getenv("PUSHOVER_USER_KEY", "").strip()
        if tok and usr:
            self._pushover = (tok, usr)
            logger.info("Pushover alerts configured")

        btok = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        cid  = os.getenv("TELEGRAM_CHAT_ID",   "").strip()
        if btok and cid:
            self._telegram = (btok, cid)
            logger.info("Telegram alerts configured")

    # ── Webhook registry ───────────────────────────────────────────────────────

    def register(
        self,
        url:    str,
        name:   str = "",
        events: set | None = None,
        secret: str | None = None,
    ) -> str:
        wid = uuid4().hex[:8]
        self._webhooks[wid] = WebhookConfig(
            id=wid, url=url,
            name=name or url,
            events=set(events) if events else set(),
            secret=secret,
        )
        logger.info("Webhook registered: %s → %s", wid, url)
        return wid

    def unregister(self, wid: str) -> bool:
        return self._webhooks.pop(wid, None) is not None

    def list_webhooks(self) -> list[dict]:
        return [
            {"id": w.id, "name": w.name, "url": w.url, "events": sorted(w.events)}
            for w in self._webhooks.values()
        ]

    def channel_summary(self) -> dict:
        return {
            "webhooks":  len(self._webhooks),
            "pushover":  self._pushover is not None,
            "telegram":  self._telegram is not None,
            "last_activity_ago_s": round(time.time() - self._last_activity_ts),
        }

    # ── Frame processor ────────────────────────────────────────────────────────

    async def process_frame(self, analytics: dict, person_count: int) -> None:
        """Evaluate every inference frame for alertable events."""
        now = time.time()

        # Track last known activity
        if person_count > 0:
            self._last_activity_ts = now

        fall   = analytics.get("fall")    or {}
        vitals = analytics.get("vitals")  or {}
        alerts = analytics.get("health_alerts") or []
        hr     = (vitals.get("heart_rate") or {}).get("heart_rate")

        # ── Fall detected ──────────────────────────────────────────
        if fall.get("fall_detected"):
            await self._maybe_fire(EventType.FALL_DETECTED, AlertEvent(
                event_type=EventType.FALL_DETECTED,
                message=(
                    "⚠️ Fall detected — please check on your loved one immediately. "
                    f"{person_count} person(s) detected in the room."
                ),
                severity="critical",
                details={
                    "confidence":  fall.get("confidence"),
                    "heart_rate":  hr,
                    "person_count": person_count,
                },
                person_count=person_count,
            ))

        # ── Presence change ────────────────────────────────────────
        if person_count > 0 and self._last_person_count == 0:
            await self._maybe_fire(EventType.PERSON_ENTERED, AlertEvent(
                event_type=EventType.PERSON_ENTERED,
                message="✅ Someone is in the monitored area",
                severity="info",
                details={"person_count": person_count},
                person_count=person_count,
            ))
        elif person_count == 0 and self._last_person_count > 0:
            await self._maybe_fire(EventType.PERSON_LEFT, AlertEvent(
                event_type=EventType.PERSON_LEFT,
                message="📍 The monitored area is now empty",
                severity="info",
                details={},
                person_count=0,
            ))
        self._last_person_count = person_count

        # ── Inactivity ─────────────────────────────────────────────
        idle = now - self._last_activity_ts
        if idle >= _INACTIVITY_THRESHOLD:
            hours = idle / 3600
            await self._maybe_fire(EventType.INACTIVITY, AlertEvent(
                event_type=EventType.INACTIVITY,
                message=(
                    f"⏰ No movement detected for {hours:.1f} hour(s). "
                    "A wellness check may be needed."
                ),
                severity="warning",
                details={"inactive_hours": round(hours, 2)},
                person_count=0,
            ))

        # ── Abnormal vitals ────────────────────────────────────────
        if hr is not None and (hr < 45 or hr > 130):
            await self._maybe_fire(EventType.VITALS_CRITICAL, AlertEvent(
                event_type=EventType.VITALS_CRITICAL,
                message=f"💔 Abnormal heart rate detected: {hr:.0f} bpm — please check in.",
                severity="critical",
                details={"heart_rate": hr},
                person_count=person_count,
            ))

        # ── Health alerts ──────────────────────────────────────────
        if alerts:
            top = alerts[0]
            await self._maybe_fire(EventType.HEALTH_ALERT, AlertEvent(
                event_type=EventType.HEALTH_ALERT,
                message=f"🏥 Health alert: {top.get('type', 'anomaly detected')}",
                severity=top.get("severity", "warning"),
                details={"alerts": [a.get("type") for a in alerts[:3]]},
                person_count=person_count,
            ))

    # ── Dispatch ───────────────────────────────────────────────────────────────

    async def _maybe_fire(self, event_type: str, event: AlertEvent) -> None:
        now     = time.time()
        debounce = max(_GLOBAL_DEBOUNCE, _DEBOUNCE.get(event_type, 60))
        if now - self._last_fired.get(event_type, 0) < debounce:
            return
        self._last_fired[event_type] = now
        logger.info("Alert fired: %s — %s", event_type, event.message)
        await self._dispatch_all(event)

    async def _dispatch_all(self, event: AlertEvent) -> None:
        payload = {
            "event":        event.event_type,
            "message":      event.message,
            "severity":     event.severity,
            "details":      event.details,
            "person_count": event.person_count,
            "timestamp":    event.timestamp,
            "device":       "EchoPose Home Hub",
        }

        tasks: list = []

        for wh in self._webhooks.values():
            if not wh.events or event.event_type in wh.events:
                tasks.append(self._post_http(wh, payload))

        if self._pushover:
            tasks.append(self._pushover_send(
                *self._pushover,
                title="EchoPose Alert",
                message=event.message,
                priority=1 if event.severity == "critical" else 0,
            ))

        if self._telegram:
            tasks.append(self._telegram_send(
                *self._telegram,
                text=f"*EchoPose Home Monitor*\n\n{event.message}",
            ))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.debug("Alert dispatch error: %s", r)

    async def _post_http(self, wh: WebhookConfig, payload: dict) -> None:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if wh.secret:
            body = json.dumps(payload, sort_keys=True).encode()
            sig  = hmac.new(wh.secret.encode(), body, hashlib.sha256).hexdigest()
            headers["X-EchoPose-Signature"] = f"sha256={sig}"

        for attempt in range(3):
            try:
                r = await self._client.post(wh.url, json=payload, headers=headers)
                r.raise_for_status()
                return
            except Exception as exc:
                if attempt == 2:
                    logger.warning("HTTP webhook %s failed: %s", wh.name, exc)
                else:
                    await asyncio.sleep(2 ** attempt)

    async def _pushover_send(
        self, app_token: str, user_key: str,
        title: str, message: str, priority: int = 0,
    ) -> None:
        try:
            await self._client.post(
                "https://api.pushover.net/1/messages.json",
                data={
                    "token":    app_token,
                    "user":     user_key,
                    "title":    title,
                    "message":  message,
                    "priority": priority,
                    "sound":    "siren" if priority >= 1 else "pushover",
                    # Emergency priority requires retry/expire
                    **({"retry": 60, "expire": 3600} if priority >= 2 else {}),
                },
            )
        except Exception as exc:
            logger.debug("Pushover error: %s", exc)

    async def _telegram_send(self, bot_token: str, chat_id: str, text: str) -> None:
        try:
            await self._client.post(
                f"https://api.telegram.org/bot{bot_token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"},
            )
        except Exception as exc:
            logger.debug("Telegram error: %s", exc)
