# echopose-sdk

[![PyPI version](https://img.shields.io/pypi/v/echopose-sdk)](https://pypi.org/project/echopose-sdk/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://pypi.org/project/echopose-sdk/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

WiFi CSI pose-estimation toolkit — the Python companion to the [EchoPose](https://github.com/shaz-in-dev/EchoPose) system.

EchoPose turns commodity ESP32-S3 access points into a privacy-preserving human pose estimator that requires no cameras, no wearables, and works through walls. This SDK provides the tools you need to collect, process, evaluate, and replay CSI data and pose predictions.

## Features

| Module | What it does |
|---|---|
| `echopose_sdk.csi` | Subcarrier normalisation, Doppler feature extraction, pilot interpolation, human-presence detection |
| `echopose_sdk.skeleton` | COCO-17 keypoint helpers, bone lengths, body-height normalisation, temporal smoothing |
| `echopose_sdk.metrics` | MPJPE, PCK@t, body-normalised PCK, PA-MPJPE, per-joint error tables |
| `echopose_sdk.streaming` | `BundleReader`/`BundleWriter` JSONL context managers, time-windowed filtering, train/test split |
| `echopose_sdk.validation` | Bundle schema validation |
| `echopose_sdk.quality` | Confidence summary statistics |

## Installation

```bash
pip install echopose-sdk
```

Development install (editable):

```bash
pip install -e echopose_sdk/
```

## Quick start

### Validate a bundle

```python
from echopose_sdk import validate_bundle
ok, reason = validate_bundle(bundle_dict)
```

### Compute pose metrics

```python
import numpy as np
from echopose_sdk.metrics import summary_report

pred = np.load("pred_poses.npy")   # shape (N, 17, 3)
gt   = np.load("gt_poses.npy")

report = summary_report(pred, gt)
# {'mpjpe': 0.42, 'pck_01_abs': 0.31, 'body_pck_01': 0.68, 'pa_mpjpe': 0.29, ...}
```

### Normalise CSI amplitudes

```python
from echopose_sdk.csi import normalize_subcarriers
norm = normalize_subcarriers(raw_amplitudes, method="zscore")
```

### Smooth a skeleton sequence

```python
from echopose_sdk.skeleton import smooth_skeleton_sequence
smoothed = smooth_skeleton_sequence(seq, method="gaussian", window=7)
```

### Stream a session file

```python
from echopose_sdk.streaming import BundleReader

with BundleReader("session.jsonl") as reader:
    for bundle in reader:
        print(bundle["timestamp_ms"])
```

## CLI

```bash
echopose-sdk validate  <bundle.json>
echopose-sdk inspect   <bundle.json> [--node NODE] [--pretty]
echopose-sdk metrics   --pred pred.npy --gt gt.npy [--per-joint] [--pretty]
echopose-sdk stream    <session.jsonl> [--fps 20] [--limit 100]
```

## License

MIT © Muhammed Shazin Sadhik Kunhi Parambath
