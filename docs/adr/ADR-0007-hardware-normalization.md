# ADR-0007: Hardware-Agnostic CSI Normalization

- **Status:** Accepted
- **Date:** 2026-03-16
- **Decision Makers:** Core team

## Context

EchoPose must ingest CSI from multiple hardware families (ESP32-S3, Intel 5300,
future SDRs).  Each chip reports different subcarrier counts (52 vs 30 vs 114),
different amplitude scales, and different phase conventions.  Feeding raw
hardware-specific frames into the model would require per-hardware model
variants.

## Decision

All CSI enters the inference pipeline through `canonicalize_frame()` which
produces a **`CanonicalCSIFrame`** with exactly **64 subcarriers**, normalised
to unit peak amplitude.

Per-hardware normalizers:
- `normalize_esp32_frame()` — 52 float32 amplitudes → 64
- `normalize_intel5300_frame()` — 30 complex128 values → 64

Resampling uses `numpy.interp` (linear) for speed; higher-order interpolation
is available via flag.

## Consequences

- **Positive:** Single model checkpoint works across all hardware.
- **Positive:** New hardware support = one new normalizer function.
- **Negative:** Interpolation introduces small artefacts at band edges.
- **Negative:** Phase normalisation for complex-valued Intel CSI is approximate.

## Alternatives Considered

| Alternative             | Why rejected                                  |
|-------------------------|-----------------------------------------------|
| Per-hardware models     | N× training cost; deployment complexity       |
| Zero-pad to largest     | Introduces spectral leakage                  |
| Hardware-specific heads | Adds model branching; harder to maintain      |
