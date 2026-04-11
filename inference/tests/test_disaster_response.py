from pipeline.disaster_response import DisasterResponseEngine


def test_normal_no_alerts():
    engine = DisasterResponseEngine()
    analytics = {
        "occupancy": {"num_people": 1},
        "activity": {"activity": "standing"},
        "fall": {"fall_detected": False},
    }
    tactical = {"anomalies": {"is_anomaly": False}}
    out = engine.evaluate(analytics, tactical)
    assert out["disaster_level"] == "NORMAL"
    assert out["alerts"] == []


def test_fall_is_critical():
    engine = DisasterResponseEngine()
    analytics = {
        "occupancy": {"num_people": 2},
        "activity": {"activity": "walking"},
        "fall": {"fall_detected": True},
    }
    tactical = {"anomalies": {"is_anomaly": False}}
    out = engine.evaluate(analytics, tactical)
    assert out["disaster_level"] == "CRITICAL"
    assert any(a["code"] == "FALL_EVENT" for a in out["alerts"])


def test_crowd_surge_warning():
    engine = DisasterResponseEngine()
    tactical = {"anomalies": {"is_anomaly": False}}

    for n in [1, 1, 2, 2, 7]:
        analytics = {
            "occupancy": {"num_people": n},
            "activity": {"activity": "standing"},
            "fall": {"fall_detected": False},
        }
        out = engine.evaluate(analytics, tactical)

    assert out["disaster_level"] == "WARNING"
    assert any(a["code"] == "CROWD_SURGE" for a in out["alerts"])
