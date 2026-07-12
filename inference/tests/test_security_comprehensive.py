"""
tests/test_security_comprehensive.py

Covers security.py gaps: Fernet roundtrip, verify_api_key, rate limiter
async behaviour + stale-client cleanup, dev-mode bypass, and the
_client_ip proxy helper in server_v2.
"""

from __future__ import annotations

import asyncio
import os
import time
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_request(host: str = "1.2.3.4", forwarded_for: str | None = None):
    req = MagicMock()
    req.client = MagicMock()
    req.client.host = host
    headers: dict[str, str] = {}
    if forwarded_for:
        headers["X-Forwarded-For"] = forwarded_for
    req.headers = headers
    return req


def _make_request_no_client(forwarded_for: str = "10.0.0.1"):
    req = MagicMock()
    req.client = None
    req.headers = {"X-Forwarded-For": forwarded_for}
    return req


# ── Fernet encryption roundtrip ───────────────────────────────────────────────

class TestFernetEncryption:
    def setup_method(self):
        # Import after env may have been patched in other tests — do it fresh
        import importlib
        import security
        importlib.reload(security)
        self.mod = security

    def test_encrypt_decrypt_roundtrip(self):
        payload = {"user_id": "abc123", "role": "admin", "score": 42}
        encrypted = self.mod.encrypt_session_data(payload)
        assert isinstance(encrypted, bytes)
        recovered = self.mod.decrypt_session_data(encrypted)
        assert recovered == payload

    def test_encrypt_produces_different_ciphertext_each_time(self):
        payload = {"x": 1}
        c1 = self.mod.encrypt_session_data(payload)
        c2 = self.mod.encrypt_session_data(payload)
        # Fernet uses random IV — ciphertexts should differ
        assert c1 != c2

    def test_decrypt_rejects_tampered_bytes(self):
        from cryptography.fernet import InvalidToken
        payload = {"secret": "value"}
        encrypted = self.mod.encrypt_session_data(payload)
        tampered = bytearray(encrypted)
        tampered[20] ^= 0xFF
        with pytest.raises((InvalidToken, Exception)):
            self.mod.decrypt_session_data(bytes(tampered))


# ── verify_api_key ─────────────────────────────────────────────────────────────

class TestVerifyApiKey:
    def test_valid_token_accepted(self, monkeypatch):
        import security
        monkeypatch.setattr(security, "VALID_TOKENS", {"test-secret-token"})
        monkeypatch.setattr(security, "_DEV_AUTH_DISABLED", False)
        result = security.verify_api_key("test-secret-token")
        assert result == "test-secret-token"

    def test_invalid_token_rejected(self, monkeypatch):
        import security
        monkeypatch.setattr(security, "VALID_TOKENS", {"real-token"})
        monkeypatch.setattr(security, "_DEV_AUTH_DISABLED", False)
        with pytest.raises(HTTPException) as exc:
            security.verify_api_key("wrong-token")
        assert exc.value.status_code == 403

    def test_missing_token_rejected(self, monkeypatch):
        import security
        monkeypatch.setattr(security, "VALID_TOKENS", {"real-token"})
        monkeypatch.setattr(security, "_DEV_AUTH_DISABLED", False)
        with pytest.raises(HTTPException) as exc:
            security.verify_api_key(None)
        assert exc.value.status_code == 403

    def test_dev_mode_bypasses_auth(self, monkeypatch):
        import security
        monkeypatch.setattr(security, "VALID_TOKENS", set())
        monkeypatch.setattr(security, "_DEV_AUTH_DISABLED", True)
        result = security.verify_api_key(None)
        assert result == "dev-no-auth"

    def test_dev_mode_bypass_accepts_any_key(self, monkeypatch):
        import security
        monkeypatch.setattr(security, "_DEV_AUTH_DISABLED", True)
        result = security.verify_api_key("totally-random")
        assert result == "dev-no-auth"


# ── RateLimiter async behaviour ───────────────────────────────────────────────

class TestRateLimiterAsync:
    def _run(self, coro):
        return asyncio.run(coro)

    def test_allows_up_to_limit(self):
        from security import RateLimiter
        rl = RateLimiter(requests_per_second=5)
        for _ in range(5):
            assert self._run(rl.check_rate_limit("1.2.3.4")) is True

    def test_blocks_over_limit(self):
        from security import RateLimiter
        rl = RateLimiter(requests_per_second=3)
        for _ in range(3):
            self._run(rl.check_rate_limit("5.5.5.5"))
        with pytest.raises(HTTPException) as exc:
            self._run(rl.check_rate_limit("5.5.5.5"))
        assert exc.value.status_code == 429

    def test_different_ips_are_independent(self):
        from security import RateLimiter
        rl = RateLimiter(requests_per_second=2)
        for _ in range(2):
            self._run(rl.check_rate_limit("192.168.1.1"))
        # Should not affect a different IP
        assert self._run(rl.check_rate_limit("192.168.1.2")) is True

    def test_stale_clients_are_pruned(self):
        from security import RateLimiter
        rl = RateLimiter(requests_per_second=100)
        # Simulate a client that hasn't been seen in >60 s
        rl.clients["old.client"] = [time.time() - 65]
        rl.last_cleanup = time.time() - 61  # trigger cleanup
        self._run(rl.check_rate_limit("new.client"))
        assert "old.client" not in rl.clients

    def test_concurrent_calls_dont_corrupt_state(self):
        from security import RateLimiter
        rl = RateLimiter(requests_per_second=100)

        async def _burst():
            tasks = [rl.check_rate_limit("concurrent") for _ in range(10)]
            return await asyncio.gather(*tasks, return_exceptions=True)

        results = asyncio.run(_burst())
        errors = [r for r in results if isinstance(r, Exception)]
        assert len(errors) == 0  # all 10 within limit of 100


# ── IncomingCSIBundle validation ──────────────────────────────────────────────

class TestIncomingCSIBundle:
    def test_valid_bundle_accepted(self):
        from security import IncomingCSIBundle
        b = IncomingCSIBundle(
            window_us=50000,
            frames=[{"node_id": 0, "amplitudes": [1.0] * 64}],
        )
        assert b.window_us == 50000

    def test_empty_frames_rejected(self):
        from security import IncomingCSIBundle
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            IncomingCSIBundle(window_us=50000, frames=[])

    def test_too_many_frames_rejected(self):
        from security import IncomingCSIBundle
        from pydantic import ValidationError
        frames = [{"node_id": 0, "amplitudes": [1.0] * 64}] * 201
        with pytest.raises(ValidationError):
            IncomingCSIBundle(window_us=1000, frames=frames)

    def test_malformed_frame_missing_node_id(self):
        from security import IncomingCSIBundle
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            IncomingCSIBundle(
                window_us=1000,
                frames=[{"amplitudes": [1.0] * 64}],
            )

    def test_too_many_subcarriers_rejected(self):
        from security import IncomingCSIBundle
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            IncomingCSIBundle(
                window_us=1000,
                frames=[{"node_id": 0, "amplitudes": [1.0] * 1025}],
            )


# ── _client_ip proxy helper (H8 regression) ──────────────────────────────────

class TestClientIpExtraction:
    @pytest.fixture(autouse=True)
    def _import(self):
        from server_v2 import _client_ip
        self.fn = _client_ip

    def test_direct_connection(self):
        req = _make_request(host="10.0.0.5")
        assert self.fn(req) == "10.0.0.5"

    def test_none_client_falls_back_to_forwarded_for(self):
        req = _make_request_no_client(forwarded_for="203.0.113.1")
        assert self.fn(req) == "203.0.113.1"

    def test_forwarded_for_with_proxy_chain(self):
        req = _make_request_no_client(forwarded_for="203.0.113.1, 10.1.1.1, 172.16.0.1")
        assert self.fn(req) == "203.0.113.1"

    def test_no_client_no_header_returns_unknown(self):
        req = MagicMock()
        req.client = None
        req.headers = {}
        assert self.fn(req) == "unknown"
