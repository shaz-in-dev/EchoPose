from pipeline.hardware_normalization import canonicalize_frame


def test_esp32_canonical_shape_is_64():
    frame = {
        "timestamp_us": 42,
        "node_id": 1,
        "amplitudes": [0.1] * 64,
        "phases": [0.0] * 64,
    }
    out = canonicalize_frame(frame, "esp32")
    assert out.source == "esp32"
    assert len(out.amplitudes) == 64
    assert len(out.phases) == 64


def test_intel5300_iq_canonical_shape_is_64():
    frame = {
        "timestamp_us": 43,
        "node_id": 2,
        "i": [1.0] * 30,
        "q": [0.5] * 30,
    }
    out = canonicalize_frame(frame, "intel5300")
    assert out.source == "intel5300"
    assert len(out.amplitudes) == 64
    assert len(out.phases) == 64
