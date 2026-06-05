#!/usr/bin/env python3
"""EchoPose license key generator.

Usage (run from project root):
    python scripts/generate_license.py --tier PR
    python scripts/generate_license.py --tier EN --count 5
    python scripts/generate_license.py --verify EP-PR-A1B2C3D4-E5F6G7H8

The ECHOPOSE_LICENSE_SECRET env var must be set to your private signing secret.
NEVER share this secret — anyone with it can forge keys.

Tier codes:
    CM  Community    (free, no key needed — this is for documentation only)
    PR  Professional
    EN  Enterprise
    DF  Defense
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import re
import secrets
import sys


_LICENSE_KEY_RE = re.compile(r"^EP-(CM|PR|EN|DF)-([A-Z0-9]{8})-([A-Z0-9]{8})$")

_TIER_LABELS = {
    "CM": "Community",
    "PR": "Professional",
    "EN": "Enterprise",
    "DF": "Defense",
}


def _get_secret() -> str:
    s = os.getenv("ECHOPOSE_LICENSE_SECRET", "").strip()
    if not s or s == "dev-secret-change-me":
        print(
            "ERROR: ECHOPOSE_LICENSE_SECRET is not set or uses the insecure default.\n"
            "Set a strong secret:  export ECHOPOSE_LICENSE_SECRET=$(openssl rand -hex 32)",
            file=sys.stderr,
        )
        sys.exit(1)
    return s


def _compute_hmac(tier_short: str, key_id: str, secret: str) -> str:
    msg = f"{tier_short}:{key_id}".encode()
    sig = hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()
    return sig[:8].upper()


def generate_key(tier_short: str, secret: str) -> str:
    key_id = secrets.token_hex(4).upper()
    sig    = _compute_hmac(tier_short, key_id, secret)
    return f"EP-{tier_short}-{key_id}-{sig}"


def verify_key(key: str, secret: str) -> dict:
    m = _LICENSE_KEY_RE.match(key.strip().upper())
    if not m:
        return {"valid": False, "reason": "Format mismatch — expected EP-{TIER}-{ID}-{HMAC}"}
    tier_short, key_id, provided_hmac = m.group(1), m.group(2), m.group(3)
    expected_hmac = _compute_hmac(tier_short, key_id, secret)
    if not hmac.compare_digest(provided_hmac, expected_hmac):
        return {"valid": False, "reason": "HMAC signature mismatch"}
    return {
        "valid": True,
        "tier":  tier_short,
        "label": _TIER_LABELS.get(tier_short, "Unknown"),
        "key_id": key_id,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="EchoPose license key generator")
    sub = p.add_subparsers(dest="cmd")

    gen = sub.add_parser("generate", help="Generate new license key(s)")
    gen.add_argument("--tier",  required=True, choices=["CM", "PR", "EN", "DF"],
                     help="License tier")
    gen.add_argument("--count", type=int, default=1, help="Number of keys to generate")

    ver = sub.add_parser("verify", help="Verify an existing license key")
    ver.add_argument("key", help="Key to verify")

    # Convenience: allow `python generate_license.py --tier PR` without subcommand
    p.add_argument("--tier",   choices=["CM", "PR", "EN", "DF"], help="Tier (shorthand for generate)")
    p.add_argument("--count",  type=int, default=1)
    p.add_argument("--verify", metavar="KEY", help="Verify a key (shorthand)")

    args = p.parse_args()
    secret = _get_secret()

    if args.verify:
        result = verify_key(args.verify, secret)
        if result["valid"]:
            print(f"✓ VALID  tier={result['label']}  id={result['key_id']}")
        else:
            print(f"✗ INVALID  reason={result['reason']}", file=sys.stderr)
            sys.exit(1)
        return

    tier = getattr(args, "tier", None)
    if not tier:
        p.print_help()
        sys.exit(0)

    count = getattr(args, "count", 1) or 1
    print(f"# EchoPose {_TIER_LABELS[tier]} license key(s)\n")
    for _ in range(count):
        key = generate_key(tier, secret)
        print(key)

    if count == 1:
        print(f"\n# Add to customer's .env:\n# ECHOPOSE_LICENSE_KEY={key}")


if __name__ == "__main__":
    main()
