"""
pipeline/tactical — Advanced tactical analytics for EchoPose

15 modules providing through-wall detection, indoor tracking,
gait biometrics, crowd analysis, behavioral profiling, sensor fusion,
and environmental mapping from WiFi CSI signals.

COMPLIANCE NOTICE:
  Deployment of these capabilities may be subject to export controls
  (ITAR/EAR), government certification (FIPS 140-2, TEMPEST, EMC),
  and legal review.  All outputs are ANALYTICAL — they do NOT
  constitute automated engagement or lethal-force recommendations.
  Life-critical decisions require independent validation systems.
"""

from .threat_tracker import TacticalTargetTracker
from .building_mapper import BuildingMapper
from .concealment import ConcealmentDetector
from .indoor_tracker import IndoorTracker
from .gait_biometrics import GaitBiometricIdentifier
from .weapon_detector import WeaponDetectionSystem
from .crowd_analyzer import CrowdDensityAnalyzer
from .tactical_activity import TacticalActivityClassifier
from .acoustic_detector import AcousticEventDetector
from .anomaly_scanner import AnomalyScanner
from .intent_predictor import BehavioralIntentPredictor
from .anti_jamming import AntiJammingDefense
from .coverage_planner import CoveragePlanner
from .sensor_fusion import MultiDomainFusion

__all__ = [
    "TacticalTargetTracker",
    "BuildingMapper",
    "ConcealmentDetector",
    "IndoorTracker",
    "GaitBiometricIdentifier",
    "WeaponDetectionSystem",
    "CrowdDensityAnalyzer",
    "TacticalActivityClassifier",
    "AcousticEventDetector",
    "AnomalyScanner",
    "BehavioralIntentPredictor",
    "AntiJammingDefense",
    "CoveragePlanner",
    "MultiDomainFusion",
]
