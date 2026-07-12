/**
 * wasm_bridge.js — Load EchoPose WASM utilities into the browser.
 *
 * Exposes the global `EchoPoseWasm` with:
 *   .ready              — Promise<boolean>  true when WASM loaded
 *   .normalizeCSI(arr)  — Float32Array → normalised Float32Array (sync after ready)
 *   .confAlpha(v)       — f32 → f32 opacity (sync after ready)
 *
 * Build the WASM module first:
 *   PowerShell:  scripts/build_wasm.ps1
 *   Bash:        scripts/build_wasm.sh
 */

'use strict';

/* global EchoPoseWasm */
window.EchoPoseWasm = (function () {
  let _wasm = null;
  let _ready = false;
  let _rejectReady = null;

  const ready = new Promise((resolve, reject) => {
    _rejectReady = reject;
    _load().then(ok => {
      if (ok) resolve(true);
      // If _load() returned false the rejection was already called
    });
  });

  async function _load() {
    try {
      // wasm-pack --target web emits an ES-module init default export.
      // We dynamic-import it so the rest of the page can stay non-module.
      const mod = await import('./wasm/pkg/echopose_wasm.js');
      await mod.default();              // fetch + compile .wasm
      _wasm = mod;
      _ready = true;
      console.log('[wasm_bridge] EchoPose WASM loaded');
      return true;
    } catch (e) {
      // WASM is an optional accelerator — JS fallbacks cover all functions.
      console.info('[EchoPose WASM] Not available (JS fallback active). Build with scripts/build_wasm.ps1 to enable.');
      _wasm = null;
      _ready = false;
      // Reject any pending .ready waiters
      if (_rejectReady) _rejectReady(e);
      return false;
    }
  }

  /** Returns true only when WASM loaded successfully. */
  function isReady() {
    return _ready;
  }

  /** Normalise subcarrier amplitudes to [0,1] (sync after await ready). */
  function normalizeCSI(amplitudes) {
    if (_wasm && _wasm.normalize_subcarriers) {
      return Array.from(_wasm.normalize_subcarriers(new Float32Array(amplitudes)));
    }
    // JS fallback — min-max normalisation
    const min = Math.min(...amplitudes);
    const max = Math.max(...amplitudes);
    const d = (max - min) || 1e-6;
    return amplitudes.map(function (v) { return (v - min) / d; });
  }

  /** Confidence → opacity helper (sync after await ready). */
  function confAlpha(conf) {
    if (_wasm && _wasm.confidence_to_alpha) {
      return _wasm.confidence_to_alpha(conf);
    }
    return Math.min(Math.max(conf, 0), 1);
  }

  return { ready: ready, isReady: isReady, normalizeCSI: normalizeCSI, confAlpha: confAlpha };
})();
