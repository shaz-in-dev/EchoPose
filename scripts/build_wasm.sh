#!/usr/bin/env bash
# scripts/build_wasm.sh — Build the echopose_wasm crate for browser usage.
#
# Requires: wasm-pack (cargo install wasm-pack)
# Output:   ui/wasm/pkg/  (JS + WASM bindings)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
CRATE="$ROOT/ui/wasm/echopose_wasm"
OUT="$ROOT/ui/wasm/pkg"

echo "==> Building echopose_wasm for browser target..."
wasm-pack build "$CRATE" --target web --out-dir "$OUT" --release

echo "==> WASM build complete: $OUT"
ls -lh "$OUT"/*.wasm "$OUT"/*.js 2>/dev/null || true
