"""Tests for EchoPose license key tier system."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── LicenseTier ────────────────────────────────────────────────────────────────

def test_tier_ordering():
    from security import LicenseTier
    assert LicenseTier.COMMUNITY    < LicenseTier.PROFESSIONAL
    assert LicenseTier.PROFESSIONAL < LicenseTier.ENTERPRISE
    assert LicenseTier.ENTERPRISE   < LicenseTier.DEFENSE


def test_tier_from_short_valid():
    from security import LicenseTier
    assert LicenseTier.from_short("CM") == LicenseTier.COMMUNITY
    assert LicenseTier.from_short("PR") == LicenseTier.PROFESSIONAL
    assert LicenseTier.from_short("EN") == LicenseTier.ENTERPRISE
    assert LicenseTier.from_short("DF") == LicenseTier.DEFENSE


def test_tier_from_short_case_insensitive():
    from security import LicenseTier
    assert LicenseTier.from_short("pr") == LicenseTier.PROFESSIONAL
    assert LicenseTier.from_short("Df") == LicenseTier.DEFENSE


def test_tier_from_short_invalid():
    from security import LicenseTier
    with pytest.raises(ValueError):
        LicenseTier.from_short("XX")


def test_tier_label():
    from security import LicenseTier
    assert LicenseTier.PROFESSIONAL.label() == "Professional"
    assert LicenseTier.DEFENSE.label()      == "Defense"


# ── verify_license_key() ───────────────────────────────────────────────────────

def _make_key(tier: str, secret: str = "test-secret") -> str:
    """Generate a real key using the same algorithm as scripts/generate_license.py."""
    import hashlib, hmac, secrets as sec
    key_id = sec.token_hex(4).upper()
    msg    = f"{tier}:{key_id}".encode()
    sig    = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()[:8].upper()
    return f"EP-{tier}-{key_id}-{sig}"


def test_verify_valid_professional_key():
    from security import verify_license_key, LicenseTier, _LICENSE_SECRET
    key  = _make_key("PR", _LICENSE_SECRET)
    tier = verify_license_key(key)
    assert tier == LicenseTier.PROFESSIONAL


def test_verify_valid_enterprise_key():
    from security import verify_license_key, LicenseTier, _LICENSE_SECRET
    key  = _make_key("EN", _LICENSE_SECRET)
    tier = verify_license_key(key)
    assert tier == LicenseTier.ENTERPRISE


def test_verify_valid_defense_key():
    from security import verify_license_key, LicenseTier, _LICENSE_SECRET
    key  = _make_key("DF", _LICENSE_SECRET)
    tier = verify_license_key(key)
    assert tier == LicenseTier.DEFENSE


def test_verify_bad_format_raises():
    from security import verify_license_key
    with pytest.raises(ValueError, match="format"):
        verify_license_key("NOT-A-KEY")


def test_verify_wrong_tier_code_raises():
    from security import verify_license_key
    with pytest.raises(ValueError):
        verify_license_key("EP-XX-A1B2C3D4-E5F6G7H8")


def test_verify_bad_hmac_raises():
    from security import verify_license_key, _LICENSE_SECRET
    good = _make_key("PR", _LICENSE_SECRET)
    # Corrupt the HMAC portion (last 8 chars)
    bad  = good[:-8] + "00000000"
    with pytest.raises(ValueError, match="signature"):
        verify_license_key(bad)


def test_verify_wrong_secret_raises():
    from security import verify_license_key
    key = _make_key("PR", "wrong-secret")
    with pytest.raises(ValueError):
        verify_license_key(key)


def test_verify_key_is_case_insensitive():
    from security import verify_license_key, LicenseTier, _LICENSE_SECRET
    key  = _make_key("PR", _LICENSE_SECRET).lower()
    tier = verify_license_key(key)
    assert tier == LicenseTier.PROFESSIONAL


# ── require_tier() ─────────────────────────────────────────────────────────────

def test_require_tier_permissive_never_raises():
    """In permissive mode, require_tier must not block anything."""
    from security import require_tier, LicenseTier
    with patch.dict(os.environ, {"ECHOPOSE_LICENSE_MODE": "permissive"}):
        checker = require_tier(LicenseTier.DEFENSE)
        checker()   # should not raise, even though active tier may be COMMUNITY


def test_require_tier_enforced_blocks_lower_tier():
    from security import require_tier, LicenseTier
    import importlib, security as sec_mod
    from fastapi import HTTPException

    original_mode  = sec_mod._LICENSE_MODE
    original_tier  = sec_mod._active_tier

    sec_mod._LICENSE_MODE  = "enforced"
    sec_mod._active_tier   = LicenseTier.COMMUNITY

    try:
        checker = require_tier(LicenseTier.PROFESSIONAL)
        with pytest.raises(HTTPException) as exc_info:
            checker()
        assert exc_info.value.status_code == 402
        assert "Professional" in exc_info.value.detail
    finally:
        sec_mod._LICENSE_MODE = original_mode
        sec_mod._active_tier  = original_tier


def test_require_tier_enforced_allows_equal_tier():
    from security import require_tier, LicenseTier
    import security as sec_mod

    original_mode = sec_mod._LICENSE_MODE
    original_tier = sec_mod._active_tier

    sec_mod._LICENSE_MODE = "enforced"
    sec_mod._active_tier  = LicenseTier.PROFESSIONAL

    try:
        checker = require_tier(LicenseTier.PROFESSIONAL)
        checker()   # should not raise
    finally:
        sec_mod._LICENSE_MODE = original_mode
        sec_mod._active_tier  = original_tier


def test_require_tier_enforced_allows_higher_tier():
    from security import require_tier, LicenseTier
    import security as sec_mod

    original_mode = sec_mod._LICENSE_MODE
    original_tier = sec_mod._active_tier

    sec_mod._LICENSE_MODE = "enforced"
    sec_mod._active_tier  = LicenseTier.DEFENSE   # highest tier

    try:
        checker = require_tier(LicenseTier.ENTERPRISE)
        checker()   # should not raise
    finally:
        sec_mod._LICENSE_MODE = original_mode
        sec_mod._active_tier  = original_tier


# ── get_license_tier() ────────────────────────────────────────────────────────

def test_get_license_tier_returns_tier():
    from security import get_license_tier, LicenseTier
    tier = get_license_tier()
    assert isinstance(tier, LicenseTier)


# ── generate_license.py script ────────────────────────────────────────────────

def test_generate_key_valid_format():
    import re
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
    from generate_license import generate_key
    key = generate_key("PR", "test-secret")
    assert re.match(r"^EP-PR-[A-Z0-9]{8}-[A-Z0-9]{8}$", key)


def test_generate_and_verify_roundtrip():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
    from generate_license import generate_key, verify_key
    secret = "my-test-secret-123"
    for tier in ["CM", "PR", "EN", "DF"]:
        key    = generate_key(tier, secret)
        result = verify_key(key, secret)
        assert result["valid"] is True
        assert result["tier"] == tier


def test_verify_key_wrong_secret_returns_invalid():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
    from generate_license import generate_key, verify_key
    key    = generate_key("PR", "real-secret")
    result = verify_key(key, "wrong-secret")
    assert result["valid"] is False


def test_generate_keys_are_unique():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
    from generate_license import generate_key
    keys = {generate_key("PR", "secret") for _ in range(20)}
    assert len(keys) == 20  # all unique
