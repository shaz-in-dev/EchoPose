"""EchoPose caregiver alerts client.

Simple, high-level Python API for sons/daughters monitoring elderly parents.
No cameras. No subscriptions. Just WiFi signals and peace of mind.

Quick start:

    import asyncio
    from echopose_sdk import CaregiverAlerts

    async def main():
        monitor = CaregiverAlerts(
            server_url="http://raspberry-pi.local:8765",
            api_token="your-api-token",    # from ECHOPOSE_API_TOKEN in .env
        )

        # Phone alerts via Pushover (https://pushover.net — free app)
        await monitor.add_pushover(app_token="...", user_key="...")

        # Or Telegram
        await monitor.add_telegram(bot_token="...", chat_id="...")

        # Or any HTTP endpoint (IFTTT, Zapier, n8n, Home Assistant)
        await monitor.add_webhook("https://maker.ifttt.com/trigger/fall/json/with/key/...")

        # Check current status
        status = await monitor.get_status()
        print(f"Mum is: {status['activity']}")
        print(f"Heart rate: {status['heart_rate']} bpm")
        print(f"Last seen: {status['last_activity_ago_s']}s ago")

        # Clean up
        await monitor.close()

    asyncio.run(main())
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

logger = logging.getLogger("echopose_sdk.alerts")


class CaregiverAlerts:
    """Client for the EchoPose caregiver alert system.

    Wraps the /webhooks/* and /status/caregiver REST endpoints.

    All methods are async and require an active event loop.
    For synchronous use, wrap calls with asyncio.run().
    """

    def __init__(self, server_url: str, api_token: str = "") -> None:
        if not _HTTPX_AVAILABLE:
            raise ImportError("httpx is required: pip install httpx")
        self._base    = server_url.rstrip("/")
        self._headers = {"X-EchoPose-Token": api_token} if api_token else {}
        self._client: Optional[httpx.AsyncClient] = None
        self._webhook_ids: list[str] = []

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._base,
                headers=self._headers,
                timeout=10.0,
            )
        return self._client

    # ── Alert channel registration ─────────────────────────────────────────────

    async def add_webhook(
        self,
        url: str,
        name: str = "",
        events: list[str] | None = None,
        secret: str | None = None,
    ) -> str:
        """Register an HTTP webhook to receive alert POSTs.

        Returns the webhook ID (use with remove_webhook to unregister).

        events: list of event names to subscribe to, or None for all events.
            fall_detected | person_entered | person_left | inactivity_alert |
            health_alert  | vitals_critical | anomaly
        """
        client = await self._ensure_client()
        body: dict = {"url": url, "name": name}
        if events:
            body["events"] = events
        if secret:
            body["secret"] = secret
        r = await client.post("/webhooks/register", json=body)
        r.raise_for_status()
        wid = r.json()["id"]
        self._webhook_ids.append(wid)
        return wid

    async def add_pushover(self, app_token: str, user_key: str) -> None:
        """Configure Pushover phone alerts on the server.

        Get your tokens at https://pushover.net (free).
        """
        # Pushover is configured server-side via .env; this is a convenience
        # method that validates the tokens and logs the setup.
        client = await self._ensure_client()
        try:
            r = await httpx.AsyncClient(timeout=5.0).post(
                "https://api.pushover.net/1/users/validate.json",
                data={"token": app_token, "user": user_key},
            )
            data = r.json()
            if data.get("status") != 1:
                raise ValueError(f"Pushover validation failed: {data.get('errors', data)}")
            logger.info("Pushover tokens validated successfully")
        except Exception as exc:
            raise ValueError(f"Could not validate Pushover credentials: {exc}") from exc

        logger.info(
            "Set PUSHOVER_APP_TOKEN=%s and PUSHOVER_USER_KEY=%s in your .env "
            "then restart EchoPose to enable phone alerts.",
            app_token[:6] + "***",
            user_key[:6] + "***",
        )

    async def add_telegram(self, bot_token: str, chat_id: str) -> None:
        """Validate Telegram bot credentials.

        Get a bot token from @BotFather on Telegram.
        Then message your bot once and run:
            GET https://api.telegram.org/bot<TOKEN>/getUpdates
        to find your chat_id.
        """
        try:
            r = await httpx.AsyncClient(timeout=5.0).get(
                f"https://api.telegram.org/bot{bot_token}/getMe"
            )
            data = r.json()
            if not data.get("ok"):
                raise ValueError(f"Telegram error: {data}")
            bot_name = data["result"].get("username", "unknown")
            logger.info("Telegram bot validated: @%s", bot_name)
        except Exception as exc:
            raise ValueError(f"Could not validate Telegram bot: {exc}") from exc

        logger.info(
            "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID=%s in your .env "
            "then restart EchoPose.",
            chat_id,
        )

    async def remove_webhook(self, webhook_id: str) -> bool:
        """Unregister a webhook by ID."""
        client = await self._ensure_client()
        r = await client.delete(f"/webhooks/{webhook_id}")
        r.raise_for_status()
        self._webhook_ids = [w for w in self._webhook_ids if w != webhook_id]
        return r.json().get("status") == "removed"

    async def list_webhooks(self) -> list[dict]:
        """Return all registered webhooks and active alert channels."""
        client = await self._ensure_client()
        r = await client.get("/webhooks")
        r.raise_for_status()
        return r.json()

    # ── Status ─────────────────────────────────────────────────────────────────

    async def get_status(self) -> dict:
        """Return the current sensor status — safe to call from a dashboard or widget.

        Returns:
            person_detected      bool
            last_activity_ago_s  int    seconds since last movement
            heart_rate           float  bpm, or None
            respiratory_rate     float  breaths/min, or None
            activity             str    e.g. "sitting", "walking"
            fall_detected        bool
            alert_channels       dict   {"pushover": bool, "telegram": bool, ...}
        """
        client = await self._ensure_client()
        r = await client.get("/status/caregiver")
        r.raise_for_status()
        return r.json()

    async def get_analytics(self) -> dict:
        """Return the full analytics snapshot (requires API token)."""
        client = await self._ensure_client()
        r = await client.get("/analytics")
        r.raise_for_status()
        return r.json()

    async def get_license(self) -> dict:
        """Return active license tier and feature availability."""
        client = await self._ensure_client()
        r = await client.get("/license")
        r.raise_for_status()
        return r.json()

    async def health_check(self) -> bool:
        """Return True if the EchoPose server is reachable."""
        try:
            client = await self._ensure_client()
            r = await client.get("/health", timeout=3.0)
            return r.status_code == 200
        except Exception:
            return False

    # ── Context manager ────────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "CaregiverAlerts":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()
