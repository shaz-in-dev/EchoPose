# EchoPose WASM Runtime

This folder contains browser-side WebAssembly utilities used by the UI.

## Current Module

- `echopose_wasm`: normalization and confidence utilities for rendering.

## Build

```bash
cd ui/wasm/echopose_wasm
wasm-pack build --target web
```

The generated package can be imported by the web UI to offload lightweight DSP helpers to WASM.
