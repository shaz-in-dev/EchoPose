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

  async function _load() {
    try {
      // wasm-pack --target web emits an ES-module init default export.
      // We dynamic-import it so the rest of the page can stay non-module.
      const mod = await import('./wasm/pkg/echopose_wasm.js');
      await mod.default();              // fetch + compile .wasm
      _wasm = mod;
      console.log('[wasm_bridge] EchoPose WASM loaded');
      return true;
    } catch (e) {
      console.warn('[wasm_bridge] WASM not available — falling back to JS.', e.message);
      return false;
    }
  }

  const ready = _load();

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

  return { ready: ready, normalizeCSI: normalizeCSI, confAlpha: confAlpha };
})();
