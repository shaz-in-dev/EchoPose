"""
inference/security.py — Enterprise Security Hardening

Covers:
  - Rate limiting (token bucket)
  - API key validation (X-EchoPose-Token)
  - License tier enforcement (X-EchoPose-License)
  - Payload validation (Pydantic)
  - Session encryption (Fernet / AES-256)

License tiers (set via ECHOPOSE_LICENSE_KEY env var):
  COMMUNITY   — no key required; basic pose + analytics
  PROFESSIONAL — key required; + fast adapt, gait biometrics
  ENTERPRISE  — key required; + full tactical suite
  DEFENSE     — key required; + weapon detection, classified modules
"""

from __future__ import annotations

from fastapi import Request, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from typing import Callable, Optional
import asyncio
import enum
import hashlib
import hmac
import os
import json
import re
import secrets
import time
from pydantic import BaseModel, model_validator
import logging
from cryptography.fernet import Fernet

logger = logging.getLogger("rf_inference.security")

API_KEY_HEADER = APIKeyHeader(name="X-EchoPose-Token", auto_error=False)

_raw_token = os.getenv("ECHOPOSE_API_TOKEN", "")
_is_production = os.getenv("ECHOPOSE_ENV", "development").lower() == "production"
if not _raw_token or _raw_token == "change_me_in_production":
    if _is_production:
        raise RuntimeError(
            "ECHOPOSE_API_TOKEN is not set or uses the insecure default. "
            "Set a strong token via the ECHOPOSE_API_TOKEN environment variable."
        )
    logger.warning(
        "ECHOPOSE_API_TOKEN is not set or uses the insecure default. "
        "Set a strong token via environment variable before production deployment."
    )
VALID_TOKENS = {_raw_token} if _raw_token else set()

# H3: In dev mode (no token set, not production), disable auth so endpoints aren't locked out
_DEV_AUTH_DISABLED = not _raw_token and not _is_production

# ── License Tier System ────────────────────────────────────────────────────────

class LicenseTier(enum.IntEnum):
    """Feature tiers in ascending order of capability."""
    COMMUNITY    = 0   # No key required — basic pose + analytics
    PROFESSIONAL = 1   # Fast adapt, gait biometrics, HA/Matter
    ENTERPRISE   = 2   # Full tactical suite, multi-site
    DEFENSE      = 3   # Weapon detection, classified modules

    @classmethod
    def from_short(cls, short: str) -> "LicenseTier":
        _MAP = {"CM": cls.COMMUNITY, "PR": cls.PROFESSIONAL, "EN": cls.ENTERPRISE, "DF": cls.DEFENSE}
        t = _MAP.get(short.upper())
        if t is None:
            raise ValueError(f"Unknown tier code: {short}")
        return t

    def label(self) -> str:
        return self.name.title()


# Key format:  EP-{TIER}-{8-char-id}-{8-char-hmac}
# Example:     EP-PR-A1B2C3D4-E5F6G7H8
_LICENSE_KEY_RE = re.compile(r"^EP-(CM|PR|EN|DF)-([A-Z0-9]{8})-([A-Z0-9]{8})$")

_LICENSE_SECRET = os.getenv("ECHOPOSE_LICENSE_SECRET", "dev-secret-change-me")
# "permissive" = log warnings but never block (good for dev)
# "enforced"   = reject requests that exceed the tier
# Default follows ECHOPOSE_ENV the same way the API token does above: production
# deployments enforce tiers unless the operator explicitly opts back into
# permissive mode. Without this, a forgotten env var would ship every paid
# feature unlocked to every deployment, license key or not.
_default_license_mode = "enforced" if _is_production else "permissive"
_LICENSE_MODE = os.getenv("ECHOPOSE_LICENSE_MODE", _default_license_mode).lower()


def _compute_hmac(tier_short: str, key_id: str) -> str:
    """Return first 8 uppercase hex chars of HMAC-SHA256(tier:id, secret)."""
    msg = f"{tier_short}:{key_id}".encode()
    sig = hmac.new(_LICENSE_SECRET.encode(), msg, hashlib.sha256).hexdigest()
    return sig[:8].upper()


def verify_license_key(key: str) -> LicenseTier:
    """Validate a license key string and return its tier.

    Raises ValueError with a user-readable message on failure.
    """
    m = _LICENSE_KEY_RE.match(key.strip().upper())
    if not m:
        raise ValueError(
            "Invalid license key format. Expected EP-{TIER}-{ID}-{HMAC}. "
            "Purchase a key at https://github.com/shaz-in-dev/EchoPose"
        )
    tier_short, key_id, provided_hmac = m.group(1), m.group(2), m.group(3)
    expected_hmac = _compute_hmac(tier_short, key_id)
    if not hmac.compare_digest(provided_hmac, expected_hmac):
        raise ValueError("License key signature is invalid. Contact support.")
    return LicenseTier.from_short(tier_short)


# Parse the license key at startup
_raw_license_key = os.getenv("ECHOPOSE_LICENSE_KEY", "").strip()
_active_tier: LicenseTier = LicenseTier.COMMUNITY

if _raw_license_key:
    try:
        _active_tier = verify_license_key(_raw_license_key)
        logger.info("License key accepted — tier: %s", _active_tier.label())
    except ValueError as _lic_err:
        if _LICENSE_MODE == "enforced":
            raise RuntimeError(f"License key rejected: {_lic_err}")
        logger.warning("License key invalid (%s) — running as Community tier", _lic_err)
else:
    if _is_production and _LICENSE_MODE == "enforced":
        logger.warning(
            "No ECHOPOSE_LICENSE_KEY set in enforced mode. "
            "Restricted features will return 402. Buy at https://github.com/shaz-in-dev/EchoPose"
        )


def get_license_tier() -> LicenseTier:
    """Return the active license tier for this deployment."""
    return _active_tier


def require_tier(minimum: LicenseTier):
    """FastAPI dependency factory — raises 402 if active tier is below minimum.

    Usage:
        @app.get("/adapt")
        async def adapt(_: None = Depends(require_tier(LicenseTier.PROFESSIONAL))):
            ...
    """
    def _check():
        if _LICENSE_MODE != "enforced":
            return  # permissive mode — never block
        if _active_tier < minimum:
            raise HTTPException(
                status_code=402,
                detail=(
                    f"This feature requires a {minimum.label()} license or higher. "
                    f"Your active tier: {_active_tier.label()}. "
                    "Purchase a license at https://github.com/shaz-in-dev/EchoPose"
                ),
            )
    return _check


class RateLimiter:
    """Token Bucket rate limiter to prevent DoS attacks on the inference pipeline"""
    def __init__(self, requests_per_second: int = 50):
        self.rps = requests_per_second
        self.clients = {}
        self.last_cleanup = time.time()
        self._lock = asyncio.Lock()  # H2: prevent race conditions on self.clients

    def _cleanup_stale_clients(self, now):
        """Prevents memory leak by pruning idle IPs every 60 seconds"""
        if now - self.last_cleanup > 60.0:
            stale = [ip for ip, timestamps in self.clients.items() if not timestamps or now - timestamps[-1] > 60.0]
            for ip in stale:
                del self.clients[ip]
            self.last_cleanup = now

    async def check_rate_limit(self, client_ip: str):  # H2: async with lock
        async with self._lock:
            now = time.time()
            self._cleanup_stale_clients(now)

            if client_ip not in self.clients:
                self.clients[client_ip] = [now]
                return True

            # Clean up old requests outside the 1-second window
            self.clients[client_ip] = [t for t in self.clients[client_ip] if now - t < 1.0]

            if len(self.clients[client_ip]) >= self.rps:
                logger.warning(f"Rate limit exceeded for IP: {client_ip}")
                raise HTTPException(status_code=429, detail="Too Many Requests")

            self.clients[client_ip].append(now)
            return True

limiter = RateLimiter(requests_per_second=60) # Allow 60Hz Max

def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Validates incoming REST connection tokens"""
    # H3: Skip auth in dev mode when no token is configured
    if _DEV_AUTH_DISABLED:
        return "dev-no-auth"
    if not api_key or api_key not in VALID_TOKENS:
        raise HTTPException(status_code=403, detail="Invalid or missing API token")
    return api_key

class IncomingCSIBundle(BaseModel):
    """Strict Input Validation using Pydantic"""
    window_us: int
    frames: list
    
    @model_validator(mode="before")
    @classmethod
    def validate_physics(cls, values):
        frames = values.get('frames')
        if not frames:
            raise ValueError("Empty CSI bundle payload")
        if len(frames) > 200:
            raise ValueError("Exceeded maximum frames per bundle (200) - Possible injection attack")
        # Validate that frames contain expected structures
        for frame in frames:
            if "node_id" not in frame or "amplitudes" not in frame:
                raise ValueError("Malformed CSI frame syntax")
            if len(frame["amplitudes"]) > 1024:
                 raise ValueError("Exceeded maximum subcarriers per node (1024)")
        return values
        
# ── Encryption at Rest ─────────────────────────────────────────────
_fernet_key_env = os.getenv("ECHOPOSE_SESSION_KEY")
if _fernet_key_env:
    # H1: Wrap Fernet key loading with a clear error message on bad key format
    try:
        cipher_suite = Fernet(_fernet_key_env.encode())
        FERNET_KEY = _fernet_key_env
    except Exception:
        raise RuntimeError(
            "ECHOPOSE_SESSION_KEY must be a valid Fernet key. "
            "Generate one with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
else:
    FERNET_KEY = Fernet.generate_key().decode('utf-8')
    cipher_suite = Fernet(FERNET_KEY.encode('utf-8'))

def encrypt_session_data(data: dict) -> bytes:
    """Encrypt JSON session data using AES-256 for at-rest storage"""
    json_bytes = json.dumps(data).encode('utf-8')
    return cipher_suite.encrypt(json_bytes)

def decrypt_session_data(encrypted_bytes: bytes) -> dict:
    """Decrypt stored session data back to dict"""
    json_bytes = cipher_suite.decrypt(encrypted_bytes)
    return json.loads(json_bytes.decode('utf-8'))
