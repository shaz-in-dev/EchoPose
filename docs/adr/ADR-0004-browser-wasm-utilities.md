# ADR-0004: Browser WASM Utilities

- Status: Accepted
- Date: 2026-04-11

## Context

UI-side normalization and rendering helpers can become a CPU bottleneck in browser runtime.

## Decision

Provide a small Rust WASM utility module at `ui/wasm/echopose_wasm/` for normalization and confidence mapping.

## Consequences

Positive:
- Better browser performance envelope.
- Foundation for future browser-side signal transforms.

Negative:
- Requires WASM toolchain in contributor environments.
