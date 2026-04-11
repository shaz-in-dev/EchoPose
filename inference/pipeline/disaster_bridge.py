from __future__ import annotations

from typing import Dict, Tuple

from .disaster_response import DisasterResponseEngine


def attach_disaster_context(
    analytics: Dict,
    tactical: Dict,
    engine: DisasterResponseEngine,
) -> Tuple[Dict, Dict]:
    """Evaluate and attach disaster-response context to analytics payloads."""
    disaster = engine.evaluate(analytics=analytics, tactical=tactical)
    enriched = dict(analytics)
    enriched["disaster"] = disaster
    return enriched, disaster
