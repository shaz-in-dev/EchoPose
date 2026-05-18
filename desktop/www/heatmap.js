/**
 * heatmap.js — CSI amplitude heatmap renderer
 *
 * Draws the per-subcarrier amplitude array for a selected node
 * as a colour-mapped 2D heatmap (subcarriers × time).
 * Uses a hot-cold colour scale (blue→cyan→green→yellow→red).
 */

class CsiHeatmap {
  constructor(canvasId, nodeSelectId, numSub = 64, timeSteps = 40) {
    this.canvas   = document.getElementById(canvasId);
    this.ctx      = this.canvas.getContext('2d');
    this.select   = document.getElementById(nodeSelectId);
    this.numSub   = numSub;
    this.steps    = timeSteps;
    this.nodeId   = 0;
    // Circular buffer [timeSteps][numSub]
    this.history  = Array.from({ length: timeSteps }, () => new Float32Array(numSub));
    this.head     = 0;

    this.select.addEventListener('change', () => { this.nodeId = +this.select.value; });
  }

  push(framesByNode) {
    // JSON keys are always strings; coerce nodeId to match
    const frame = framesByNode[String(this.nodeId)] ?? framesByNode[this.nodeId];
    if (!frame) return;
    const row = this.history[this.head % this.steps];
    frame.amplitudes.slice(0, this.numSub).forEach((v, i) => row[i] = v);
    this.head++;
    this._draw();
  }

  _heatColor(t) {
    // t in [0,1] — hot colour map
    const stops = [
      [0.00, [8,   12,  80 ]], // dark blue
      [0.25, [0,   180, 200]], // cyan
      [0.50, [52,  211, 153]], // green
      [0.75, [250, 200, 20 ]], // yellow
      [1.00, [248, 50,  50 ]], // red
    ];
    
    // Clamp t
    t = Math.max(0, Math.min(1, t));

    for (let i = 1; i < stops.length; i++) {
      const [t0, c0] = stops[i-1];
      const [t1, c1] = stops[i];
      if (t <= t1) {
        const f = (t - t0) / (t1 - t0);
        const r = Math.round(c0[0] + (c1[0] - c0[0]) * f);
        const g = Math.round(c0[1] + (c1[1] - c0[1]) * f);
        const b = Math.round(c0[2] + (c1[2] - c0[2]) * f);
        return [r, g, b];
      }
    }
    return stops[0][1];
  }

  _draw() {
    const { canvas, ctx, numSub, steps } = this;
    const W = canvas.width, H = canvas.height;
    const cellW = W / steps, cellH = H / numSub;

    // Find global max for normalisation in this frame
    let maxVal = 1e-6;
    this.history.forEach(row => row.forEach(v => { if (v > maxVal) maxVal = v; }));

    for (let t = 0; t < steps; t++) {
      const rowIdx = (this.head + t) % steps;
      const row    = this.history[rowIdx];
      for (let s = 0; s < numSub; s++) {
        const norm  = row[s] / maxVal;
        const [r,g,b] = this._heatColor(norm);
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(Math.round(t * cellW), Math.round(s * cellH),
                     Math.ceil(cellW) + 1, Math.ceil(cellH) + 1);
      }
    }

    // X-label
    ctx.fillStyle = 'rgba(124,143,168,0.9)';
    ctx.font = '9px JetBrains Mono';
    ctx.fillText('← time', 4, H - 4);
    ctx.fillText('subcarriers ↕', W - 70, H - 4);
  }
}
