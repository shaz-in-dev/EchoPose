from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


MAGIC = b"EPB1"


@dataclass(frozen=True)
class EchoPoseBundleMetadata:
    model_name: str
    model_format: str
    sensor_profile: str
    input_shape: str
    output_shape: str
    created_at: str
    version: str = "1.0"



def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat()



def generate_keypair() -> Tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    priv = Ed25519PrivateKey.generate()
    return priv, priv.public_key()



def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()



def pack_signed_bundle(
    model_bytes: bytes,
    metadata: EchoPoseBundleMetadata,
    private_key: Ed25519PrivateKey,
) -> bytes:
    payload_sha = _sha256(model_bytes)
    signature = private_key.sign(model_bytes).hex()

    header: Dict[str, Any] = {
        "metadata": asdict(metadata),
        "payload_sha256": payload_sha,
        "signature": {
            "algorithm": "ed25519",
            "signature_hex": signature,
        },
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    header_len = len(header_bytes).to_bytes(4, "big")
    return MAGIC + header_len + header_bytes + model_bytes



def unpack_signed_bundle(container: bytes) -> Tuple[Dict[str, Any], bytes]:
    if len(container) < 8 or container[:4] != MAGIC:
        raise ValueError("invalid EchoPose bundle magic")

    header_len = int.from_bytes(container[4:8], "big")
    header_start = 8
    header_end = header_start + header_len
    header = json.loads(container[header_start:header_end].decode("utf-8"))
    model_bytes = container[header_end:]
    return header, model_bytes



def verify_signed_bundle(container: bytes, public_key: Ed25519PublicKey) -> Tuple[bool, str]:
    header, model_bytes = unpack_signed_bundle(container)
    sig = header.get("signature", {})
    if sig.get("algorithm") != "ed25519":
        return False, "unsupported signature algorithm"

    expected_sha = header.get("payload_sha256")
    got_sha = _sha256(model_bytes)
    if expected_sha != got_sha:
        return False, "payload hash mismatch"

    try:
        public_key.verify(bytes.fromhex(sig.get("signature_hex", "")), model_bytes)
    except Exception:
        return False, "signature verification failed"

    return True, "bundle verification passed"
