# ADR-0005: Ed25519 Signed Model Bundles

- **Status:** Accepted
- **Date:** 2026-03-16
- **Decision Makers:** Core team

## Context

Deploying ML models in adversarial environments requires guarantees that a model
hasn't been tampered with.  Standard .pt / .onnx files have no built-in
integrity verification, meaning a compromised CI/CD pipeline or supply-chain
attack could silently swap in a backdoored model.

## Decision

We ship models inside an **EchoPose Bundle (EPB1)** format that wraps:

1. The raw model bytes.
2. A JSON metadata header (architecture, version, SHA-256 of weights).
3. An **Ed25519 digital signature** over (metadata ‖ model hash).

Only keys registered in the project keyring can produce valid bundles.
`verify_signed_bundle()` is called at load time, and inference is blocked if
the signature check fails.

## Consequences

- **Positive:** Tamper-evident model distribution; auditability.
- **Positive:** Lightweight — Ed25519 adds < 1 KB overhead per bundle.
- **Negative:** Requires key management (keypair generation, secure storage).
- **Negative:** Unsigned legacy .pt files bypass the check (migration path needed).

## Alternatives Considered

| Alternative           | Why rejected                              |
|-----------------------|-------------------------------------------|
| GPG-signed tarballs   | Heavy tooling, poor Python integration    |
| HMAC-SHA256           | Symmetric key — anyone with key can forge  |
| No signing            | Unacceptable for adversarial deployments  |
