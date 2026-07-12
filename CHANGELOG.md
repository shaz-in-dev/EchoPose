# Changelog

All notable changes to EchoPose are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [0.3.2] — 2026-07-12

### Fixed

- `echopose_sdk/README.md` (and therefore the PyPI project description) was
  still the original scaffold placeholder ("Initial features: validate CSI
  bundle shape...") despite the SDK having grown to 8 real modules (`csi`,
  `skeleton`, `metrics`, `streaming`, `validation`, `quality`, `alerts`,
  `cli`). Replaced with an accurate description covering every module,
  verified example-by-example against the actual function signatures.
- `echopose-sdk` version bump to 0.2.2 to ship the corrected PyPI listing.

## [0.3.1] — 2026-07-12

### Fixed — Pose Pipeline Correctness & Safety

- **`PoseNetV2` architecture bug**: a blanket `Sigmoid` on the regression head clamped
  world-space x/y/z coordinates to `[0,1]`, making accurate real-world pose regression
  architecturally impossible regardless of training data quality. Coordinates are now
  `tanh`-bounded to a configurable room-scale range; confidence remains sigmoid-bounded.
- **Shipped model checkpoint**: `inference/models/pose_net.pt` was trained on synthetic
  random data (noise-to-noise) and was committed to git despite a comment claiming it
  was gitignored — every download silently shipped a "model" performing at chance level
  while reporting `is_simulation: false`. Renamed to `pose_net.synthetic_smoketest.pt`,
  gitignored going forward, and the server now refuses to start in simulation mode in
  production deployments (`ECHOPOSE_ENV=production`) unless explicitly acknowledged via
  `ALLOW_SIMULATION_MODE=true`.
- **`pose.py`**: `is_simulation` was left uninitialized on the ONNX/optimized-inference
  success paths, causing an `AttributeError` on every frame if either backend loaded.
- **Aggregator `POST /calibrate/rssi`**: previously logged a reference measurement and
  told the operator to set a `RSSI_OFFSET_DB` env var that didn't exist anywhere in the
  codebase. Now computes and applies a live RSSI offset immediately, no restart needed.
- **`server.py`**: incoming CSI bundles are now validated against the existing
  `IncomingCSIBundle` schema (was defined but never invoked).
- **`train.py`**: refuses to train on synthetic random data without `--allow-synthetic`,
  and writes to a clearly-labeled filename instead of the production checkpoint path.
- **`train_with_splits.py`**: the LORO/LOSO evaluation harness now actually trains and
  evaluates a fresh model per fold; it previously always fell back to a zero-baseline
  regardless of the `--eval-only` flag.
- **`security.py`**: license tier enforcement now defaults to `enforced` when
  `ECHOPOSE_ENV=production`, matching the existing API-token behavior, instead of
  silently staying permissive.

### Added

- `inference/models/pose_net.sim_pretrained.pt` groundwork: a checkpoint pretrained on
  physics-simulated CSI driven by real CMU motion-capture data, intended as a
  `fast_adapt.py` starting point once real hardware data is collected. Not a validated
  production model — see `inference/models/pose_net.sim_pretrained.README.md`.
- `echopose-sdk` version bump to `0.2.1` (packaging/release alignment; no SDK API changes).

## [0.3.0] — 2026-06-10

### Added — Caregiver & Smart Home

- **Caregiver alert system** (`inference/integrations/webhooks.py`)
  - Instant phone alerts via **Pushover** (siren sound for falls)
  - **Telegram** bot notifications (free, works globally)
  - Generic **HTTP webhooks** for IFTTT, Zapier, n8n, Home Assistant
  - Events: `fall_detected`, `person_entered`, `person_left`, `inactivity_alert`, `vitals_critical`, `health_alert`
  - Configurable inactivity threshold (default 4 hours — alerts if no movement)
  - Per-event debounce windows prevent alert spam
  - HMAC-SHA256 signing for webhook payloads
  - `GET /status/caregiver` — public endpoint for mobile dashboards

- **Home Assistant MQTT** — complete rewrite
  - 13 auto-discovered sensors (presence, HR, RR, SpO2, activity, gait, gesture, sleep, stress, fall, anomaly, health alert)
  - Last Will Testament so HA marks entities offline on crash
  - Auto-reconnect with exponential backoff; re-publishes discovery on reconnect
  - `state_class: measurement` for history graphs; `expire_after` on transient alerts
  - JSON attributes topic for advanced HA templates

- **Matter protocol bridge** (`matter-bridge/`)
  - Full Node.js bridge using `@matter/main` — real Matter devices, not a stub
  - 4 Matter devices: Occupancy Sensor, Contact Sensor, Temperature Sensor, Humidity Sensor
  - `GET /matter/pairing` returns QR code data URL — scan once with Apple Home / Google Home / Alexa
  - Commissioning state persisted across restarts in `matter-storage/`
  - Graceful handling when already commissioned (no broken null QR code)
  - `MATTER_HOST` env var for correct Docker service networking

- **`CaregiverAlerts` Python SDK** (`echopose_sdk/alerts.py`)
  - High-level async API for sons/daughters monitoring elderly parents
  - `get_status()`, `add_pushover()`, `add_telegram()`, `add_webhook()`, `health_check()`
  - Async context manager support

- **`GET /status/caregiver`** — no-auth public endpoint for mobile app widgets
- **`GET /license`** — public endpoint showing active tier and feature availability
- **`GET /webhooks`**, **`POST /webhooks/register`**, **`DELETE /webhooks/{id}`** — webhook management API
- **`POST /adapt`** — LoRA fast domain adaptation endpoint (Professional license)
- **`GET /matter/pairing`**, **`GET /matter/status`** — Matter commissioning endpoints

### Added — ML / Inference

- **Micro-LoRA fast adaptation** (`inference/pipeline/fast_adapt.py`)
  - Injects low-rank A×B adapter matrices into PoseNetV2 pose head
  - Keeps BatchNorm layers in feature extractors trainable for room adaptation
  - Self-supervised NT-Xent contrastive loss — no pose labels needed
  - 30-second adaptation cycle via `POST /adapt?timeout=30`
  - Save/load adapter weights per environment

- **Self-supervised pretraining script** (`inference/scripts/pretrain.py`)
  - SimCLR-style contrastive pretraining on unlabeled CSI data
  - `--mock` flag for running without hardware
  - Saves best encoder checkpoint for downstream transfer

### Added — Infrastructure

- **License key system** (`inference/security.py`)
  - Tiers: Community (free), Professional, Enterprise, Defense
  - HMAC-SHA256 signed keys — `scripts/generate_license.py` for key minting
  - `ECHOPOSE_LICENSE_MODE=enforced` gates features; `permissive` mode (default) never blocks
  - `require_tier()` FastAPI dependency for clean endpoint gating

- **Docker Compose** — `matter-bridge` service added with named `matter-storage` volume
- **`matter-bridge/Dockerfile`** — Node.js 20 Alpine with healthcheck
- **`.env.example`** — completely rewritten for the elderly care market; plain English, organized by use case

### Fixed

- `inject_lora()` no longer freezes BatchNorm layers — fixes `RuntimeError: element 0 of tensors does not require grad` during contrastive adaptation
- LoRA frame buffer uses `EXPECTED_NODES` env var instead of hardcoded `3` — works with 2 or 4 node deployments
- `MATTER_HOST` env var prevents Docker networking failure (`127.0.0.1` → `matter-bridge` service name)
- Matter `/pairing` endpoint returns a clear message when already commissioned instead of crashing on null `pairingCodes`
- `server_v2.py` now starts/stops WebhookManager alongside HA and Matter bridges

### Tests

- `test_homeassistant.py` — 15 tests covering all 13 entities, LWT, reconnect, analytics extraction
- `test_matter.py` — 12 tests covering from_env, payload format, error resilience
- `test_fast_adapt.py` — 18 tests covering LoRALinear, inject_lora, FastAdapter save/load
- `test_pretrain.py` — 12 tests covering encoder, augmentation, NT-Xent loss, full training run
- `test_license.py` — 20 tests covering tier ordering, key generation, HMAC validation, enforcement
- `test_webhooks.py` — 35 tests covering all alert types, debounce, dispatch, HMAC signing

---

## [0.2.0] — 2026-05-16

### Added

- **PoseNetV2** — multi-scale 1D CNN + 2-layer LSTM + 8-head attention; 3-person simultaneous tracking
- **Tactical modules** — 14 classes: TacticalTargetTracker, WeaponDetectionSystem, GaitBiometricIdentifier, BuildingMapper, BehavioralIntentPredictor, AntiJammingDefense, and more
- **Analytics pipeline** — vitals (HR/RR/SpO2), activity classification, fall detection, sleep staging, gesture recognition, emotion/stress scoring, occupancy, gait analysis, health anomaly alerts
- **Research modules** — domain adaptation (MMD), adversarial certification, contrastive pretraining, PINN, continual personalization, cross-polarization fusion
- **Disaster response engine** with context bridge
- **PWA dashboard** with mobile and tablet layouts
- **Rust aggregator** — 5 published crates: echopose-types, echopose-denoise, echopose-sync, echopose-localize, echopose-wasm
- **`echopose-sdk`** published to PyPI; **`echopose-aggregator`** and **`echopose-inference`** Docker images on GHCR
- **Enterprise security** — rate limiting, API key validation, Fernet session encryption, payload validation
- **Prometheus metrics** + structured logging + rotating file handler
- **Docker Compose**, **Kubernetes manifests**, **systemd** service definitions
- **GitHub Actions** CI/CD — 5 workflows

### Changed

- Aggregator rewritten in Rust (was Python) for 10× throughput improvement

---

## [0.1.0] — 2026-04-01

### Added

- Initial ESP32-S3 firmware — 802.11n CSI capture at 20 Hz via binary UDP frames
- Python inference server — FusionPipeline → PoseEstimator (V1, 3-layer CNN) → Three.js dashboard
- Basic pose estimation — 17 COCO keypoints, single person
- Mock ESP32 mesh simulator for development without hardware
