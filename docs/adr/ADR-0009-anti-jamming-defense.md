# ADR-0009: Anti-Jamming & Counter-Surveillance Defense

- **Status:** Accepted
- **Date:** 2026-03-16
- **Decision Makers:** Core team

## Context

WiFi sensing systems are vulnerable to deliberate interference: broadband
jamming, CSI spoofing, and frequency sweeps can degrade or deceive the
pipeline.  Failing silently is unacceptable in tactical deployments.

## Decision

We implement `AntiJammingDefense` with four detection methods:

1. **Broadband noise floor elevation** — Z-score > 4 σ above calibrated
   baseline triggers `ACTIVE_JAMMING` alert.
2. **Frame-to-frame cosine discontinuity** — similarity drop below 0.5
   indicates `CSI_SPOOFING`.
3. **Physics-violation checks** — zero-variance (synthetic) or negative
   amplitude (impossible) signals.
4. **Spectral flatness** — unnaturally flat high-energy spectrum indicates
   active `FREQUENCY_SWEEP` reconnaissance.

Alerts include severity levels (MEDIUM / HIGH / CRITICAL) and automated
recommendations (INVESTIGATE / INCREASE_MONITORING / SWITCH_TO_BACKUP_SENSORS).

## Consequences

- **Positive:** Real-time detection of RF attacks with actionable recommendations.
- **Positive:** Calibration-based — adapts to any deployment environment.
- **Negative:** Requires initial clean-room calibration phase.
- **Negative:** Legitimate environmental changes (e.g., heavy rain) may trigger
  false positives.

## Alternatives Considered

| Alternative                 | Why rejected                               |
|-----------------------------|--------------------------------------------|
| Hardware frequency hopping  | Requires firmware changes; not always viable|
| Ignore jamming              | Silent failure in adversarial environments |
| External RF monitor         | Extra hardware cost; integration complexity|
