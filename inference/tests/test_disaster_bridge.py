from pipeline.disaster_bridge import attach_disaster_context
from pipeline.disaster_response import DisasterResponseEngine


def test_attach_disaster_context_adds_disaster_block():
    engine = DisasterResponseEngine()
    analytics = {
        "occupancy": {"num_people": 1},
        "activity": {"activity": "standing"},
        "fall": {"fall_detected": False},
    }
    tactical = {"anomalies": {"is_anomaly": False}}

    enriched, disaster = attach_disaster_context(analytics, tactical, engine)

    assert "disaster" in enriched
    assert enriched["disaster"] == disaster
    assert disaster["disaster_level"] == "NORMAL"


def test_attach_disaster_context_escalates_on_fall():
    engine = DisasterResponseEngine()
    analytics = {
        "occupancy": {"num_people": 2},
        "activity": {"activity": "walking"},
        "fall": {"fall_detected": True},
    }
    tactical = {"anomalies": {"is_anomaly": False}}

    enriched, disaster = attach_disaster_context(analytics, tactical, engine)

    assert enriched["disaster"]["disaster_level"] == "CRITICAL"
    assert any(alert["code"] == "FALL_EVENT" for alert in disaster["alerts"])
