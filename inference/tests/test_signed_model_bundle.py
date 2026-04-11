from __future__ import annotations

from research.signed_model_bundle import (
    EchoPoseBundleMetadata,
    generate_keypair,
    pack_signed_bundle,
    utc_now_iso,
    verify_signed_bundle,
)


def test_signed_bundle_roundtrip() -> None:
    priv, pub = generate_keypair()
    meta = EchoPoseBundleMetadata(
        model_name="pose-net",
        model_format="pt",
        sensor_profile="esp32-v1",
        input_shape="[1,3,64,16]",
        output_shape="[1,3,17,4]",
        created_at=utc_now_iso(),
    )
    blob = pack_signed_bundle(b"dummy-model-bytes", meta, priv)
    ok, msg = verify_signed_bundle(blob, pub)
    assert ok, msg


def test_signed_bundle_detects_tamper() -> None:
    priv, pub = generate_keypair()
    meta = EchoPoseBundleMetadata(
        model_name="pose-net",
        model_format="pt",
        sensor_profile="esp32-v1",
        input_shape="[1,3,64,16]",
        output_shape="[1,3,17,4]",
        created_at=utc_now_iso(),
    )
    blob = pack_signed_bundle(b"dummy-model-bytes", meta, priv)

    tampered = blob[:-1] + (b"X" if blob[-1:] != b"X" else b"Y")
    ok, _ = verify_signed_bundle(tampered, pub)
    assert not ok
