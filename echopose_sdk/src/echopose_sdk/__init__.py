"""echopose_sdk — WiFi CSI pose estimation toolkit.

Public API summary
------------------
csi         Signal utilities (normalise, Doppler, pilot interpolation, presence)
skeleton    COCO-17 keypoint helpers, body height, smoothing, normalisation
metrics     MPJPE, PCK, body-normalised PCK, PA-MPJPE, per-joint tables
streaming   BundleReader / BundleWriter JSONL context managers
quality     Confidence summary statistics
validation  CSI bundle schema validation
"""

__version__ = "0.2.0"
__author__ = "Muhammed Shazin Sadhik Kunhi Parambath"

# ── core utilities ─────────────────────────────────────────────────────────────
from .quality import summarize_confidence
from .validation import validate_bundle

# ── CSI signal utilities ───────────────────────────────────────────────────────
from .csi import (
    normalize_subcarriers,
    sanitize_csi_bundle,
    subcarrier_variance,
    extract_doppler_features,
    csi_correlation_matrix,
    estimate_human_presence,
    interpolate_pilot_subcarriers,
)

# ── skeleton helpers ───────────────────────────────────────────────────────────
from .skeleton import (
    JOINT_NAMES,
    COCO_BONES,
    keypoints_to_array,
    array_to_keypoints,
    bone_lengths,
    body_height,
    torso_height,
    filter_by_confidence,
    normalize_to_body_height,
    align_root,
    smooth_skeleton_sequence,
)

# ── evaluation metrics ─────────────────────────────────────────────────────────
from .metrics import (
    mpjpe,
    pck,
    body_normalized_pck,
    pa_mpjpe,
    mpjve,
    confidence_mae,
    per_joint_error_table,
    summary_report,
)

# ── streaming / I-O ────────────────────────────────────────────────────────────
from .streaming import (
    BundleReader,
    BundleWriter,
    stream_frames,
    filter_by_time,
    count_bundles,
    split_train_test,
)

__all__ = [
    # version
    "__version__",
    # quality / validation
    "summarize_confidence",
    "validate_bundle",
    # csi
    "normalize_subcarriers",
    "sanitize_csi_bundle",
    "subcarrier_variance",
    "extract_doppler_features",
    "csi_correlation_matrix",
    "estimate_human_presence",
    "interpolate_pilot_subcarriers",
    # skeleton
    "JOINT_NAMES",
    "COCO_BONES",
    "keypoints_to_array",
    "array_to_keypoints",
    "bone_lengths",
    "body_height",
    "torso_height",
    "filter_by_confidence",
    "normalize_to_body_height",
    "align_root",
    "smooth_skeleton_sequence",
    # metrics
    "mpjpe",
    "pck",
    "body_normalized_pck",
    "pa_mpjpe",
    "mpjve",
    "confidence_mae",
    "per_joint_error_table",
    "summary_report",
    # streaming
    "BundleReader",
    "BundleWriter",
    "stream_frames",
    "filter_by_time",
    "count_bundles",
    "split_train_test",
]

