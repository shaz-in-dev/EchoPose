/**
 * app.js — Main application controller
 *
 * Responsibilities:
 *  - WebSocket connection to inference server (ws://localhost:8765/ws/pose)
 *  - Dispatching incoming pose data to SkeletonRenderer + CsiHeatmap
 *  - Demo mode (synthetic walking animation when no server is connected)
 *  - FPS counter, node status, keypoints table
 */

'use strict';

// ── DOM refs ──────────────────────────────────────────────────────
const modal       = document.getElementById('connect-modal');
const wsUriInput  = document.getElementById('ws-uri');
const btnConnect  = document.getElementById('btn-connect');
const connectErr  = document.getElementById('connect-error');
const statusBadge = document.getElementById('status-badge');
const fpsDisplay  = document.getElementById('fps-display');
const nodeCount   = document.getElementById('node-count');
const kpList      = document.getElementById('keypoints-list');
const btnRotate   = document.getElementById('btn-rotate');
const btnReset    = document.getElementById('btn-reset');
const btnLocalize = document.getElementById('btn-localize');
const btnDemo     = document.getElementById('btn-demo');

// ── Renderers ─────────────────────────────────────────────────────
const skeleton = new SkeletonRenderer('skeleton-canvas');
const heatmap  = new CsiHeatmap('heatmap-canvas', 'heatmap-node');

// ── State ─────────────────────────────────────────────────────────
let ws         = null;
let demoMode   = false;
let demoTimer  = null;
let frameCount = 0;
let lastFpsTs  = performance.now();
let autoRotate = false;

// Reconnect vars
let reconnectTimer = null;
let reconnectAttempts = 0;
let intendedDisconnect = false;

// ── FPS counter ───────────────────────────────────────────────────
function tick() {
  frameCount++;
  const now = performance.now();
  if (now - lastFpsTs >= 1000) {
    fpsDisplay.textContent = frameCount;
    frameCount = 0;
    lastFpsTs  = now;
  }
}

// ── Handle a pose frame ───────────────────────────────────────────
function handleFrame(data) {
  if (typeof recordFrame === 'function') recordFrame(data);
  tick();
  
  // V3 format (Array of skeletons)
  if (data.skeletons) {
    skeleton.updateSkeletons(data.skeletons);
    if (data.skeletons[0]) updateKpTable(data.skeletons[0]);
  } 
  // V2 format (Backward compatibility for old recordings)
  else if (data.keypoints) {
    skeleton.updateSkeletons([data.keypoints]);
    updateKpTable(data.keypoints);
  }
}

// ── Node Health Polling ───────────────────────────────────────────
async function pollNodes() {
  if (demoMode) return;
  try {
    // Determine the aggregator HTTP URL from the WS URI
    const wsUrl = new URL(wsUriInput.value.trim());
    const httpUrl = `http://${wsUrl.hostname}:3000/nodes`;
    
    const res = await fetch(httpUrl);
    if (!res.ok) return;
    const nodes = await res.json();
    
    // Count nodes that sent data in the last 2000 milliseconds
    const now = Date.now();
    let active = 0;
    for (const [id, stats] of Object.entries(nodes)) {
      if (now - stats.last_seen_ms < 2000) active++;
    }
    nodeCount.textContent = active;
  } catch (e) {
    console.warn(`Node polling failed: ${e.message}`);
    statusBadge.textContent = 'Polling Error';
    statusBadge.className = 'badge badge--disconnected';
  }
}

// ── Automated Localization ────────────────────────────────────────
async function fetchLocalization() {
  try {
    const wsUrl = new URL(wsUriInput.value.trim());
    const httpUrl = `http://${wsUrl.hostname}:3000/localize`;
    const res = await fetch(httpUrl);
    if (!res.ok) return;
    const nodeCoords = await res.json();
    skeleton.updateNodes(nodeCoords);
    console.log("[V3] Node localization updated:", nodeCoords);
  } catch (e) {
    console.warn("Localization failed:", e);
  }
}

btnLocalize.addEventListener('click', () => fetchLocalization());

// Poll every 1 second
setInterval(pollNodes, 1000);

// ── Keypoints table ───────────────────────────────────────────────
const KP_NAMES = [
  'Nose','L Eye','R Eye','L Ear','R Ear',
  'L Shldr','R Shldr','L Elbow','R Elbow',
  'L Wrist','R Wrist','L Hip','R Hip',
  'L Knee','R Knee','L Ankle','R Ankle',
];
function updateKpTable(kps) {
  kpList.innerHTML = '';
  kps.forEach((kp, i) => {
    const div = document.createElement('div');
    div.className = 'kp-item';
    div.innerHTML = `
      <div class="kp-name">${KP_NAMES[i]}</div>
      <div class="kp-coords">
        ${kp.x.toFixed(2)}, ${kp.y.toFixed(2)}, ${kp.z.toFixed(2)}
        <span class="kp-conf">${(kp.confidence * 100).toFixed(0)}%</span>
      </div>`;
    kpList.appendChild(div);
  });
}

// ── WebSocket connection ──────────────────────────────────────────
const btnDemoModal = document.getElementById('btn-close-modal');

function connect(uri) {
  connectErr.textContent = '';
  if (ws) {
    intendedDisconnect = true;
    ws.close();
  }
  intendedDisconnect = false;
  clearTimeout(reconnectTimer);

  statusBadge.textContent = 'Connecting...';
  statusBadge.className   = 'badge';

  ws = new WebSocket(uri);

  ws.onopen = () => {
    reconnectAttempts = 0;
    modal.classList.add('hidden');
    statusBadge.textContent = 'Live';
    statusBadge.className   = 'badge badge--connected';
    if (demoMode) stopDemo();
  };

  ws.onmessage = (ev) => {
    try { handleFrame(JSON.parse(ev.data)); } catch (e) { /* skip */ }
  };

  ws.onerror = () => {
    connectErr.textContent = 'Connection failed. Is the inference server running?';
  };

  ws.onclose = () => {
    if (intendedDisconnect || demoMode) {
      statusBadge.textContent = 'Disconnected';
      statusBadge.className   = 'badge badge--disconnected';
      return;
    }

    // Exponential Backoff Reconnect
    const backoff = Math.min(1000 * Math.pow(1.5, reconnectAttempts), 10000);
    reconnectAttempts++;
    statusBadge.textContent = `Reconnecting in ${(backoff/1000).toFixed(1)}s...`;
    statusBadge.className   = 'badge badge--disconnected';
    
    reconnectTimer = setTimeout(() => connect(uri), backoff);
  };
}

btnConnect.addEventListener('click', () => connect(wsUriInput.value.trim()));
if (btnDemoModal) {
  btnDemoModal.addEventListener('click', () => {
    modal.classList.add('hidden');
    if (!demoMode) startDemo();
  });
}

// ── Demo mode (synthetic walking pose) ────────────────────────────
function syntheticPose(t) {
  const s = Math.sin, c = Math.cos;
  const walk = t * 0.04;
  const base = [
    { x:.50, y:.15, z:.50 }, // nose
    { x:.48, y:.13, z:.50 }, // l_eye
    { x:.52, y:.13, z:.50 }, // r_eye
    { x:.45, y:.14, z:.50 }, // l_ear
    { x:.55, y:.14, z:.50 }, // r_ear
    { x:.40, y:.28, z:.50 }, // l_shoulder
    { x:.60, y:.28, z:.50 }, // r_shoulder
    { x:.35 + s(walk)*.04, y:.42, z:.50 + s(walk)*.05 }, // l_elbow
    { x:.65 - s(walk)*.04, y:.42, z:.50 - s(walk)*.05 }, // r_elbow
    { x:.30 + s(walk)*.07, y:.56, z:.50 + s(walk)*.08 }, // l_wrist
    { x:.70 - s(walk)*.07, y:.56, z:.50 - s(walk)*.08 }, // r_wrist
    { x:.44, y:.58, z:.50 }, // l_hip
    { x:.56, y:.58, z:.50 }, // r_hip
    { x:.42 + s(walk+1)*.07, y:.73, z:.50 + s(walk+1)*.07 }, // l_knee
    { x:.58 - s(walk+1)*.07, y:.73, z:.50 - s(walk+1)*.07 }, // r_knee
    { x:.42 + s(walk+2)*.10, y:.88, z:.50 + s(walk+2)*.10 }, // l_ankle
    { x:.58 - s(walk+2)*.10, y:.88, z:.50 - s(walk+2)*.10 }, // r_ankle
  ].map(kp => ({ ...kp, confidence: 0.85 + Math.random() * 0.1 }));

  return base;
}

let demoT = 0;
function demoTick() {
  const kps = syntheticPose(demoT++);
  // V3 update asks for an array of skeletons
  skeleton.updateSkeletons([kps]);
  updateKpTable(kps);
  tick();
  // Fake heatmap data for demo
  heatmap.push({
    0: { amplitudes: Array.from({ length: 64 }, (_, i) => Math.abs(Math.sin(demoT * 0.1 + i * 0.2))) },
    1: { amplitudes: Array.from({ length: 64 }, (_, i) => Math.abs(Math.cos(demoT * 0.08 + i * 0.15))) },
    2: { amplitudes: Array.from({ length: 64 }, (_, i) => Math.abs(Math.sin(demoT * 0.12 + i * 0.25))) },
  });
  nodeCount.textContent = 3;
}

function startDemo() {
  demoMode  = true;
  demoTimer = setInterval(demoTick, 50);  // 20 Hz
  btnDemo.classList.add('active');
  statusBadge.textContent = 'Demo';
  statusBadge.className   = 'badge badge--connected';
  modal.classList.add('hidden');
}
function stopDemo() {
  clearInterval(demoTimer);
  demoMode = false;
  btnDemo.classList.remove('active');
}

btnDemo.addEventListener('click', () => { demoMode ? stopDemo() : startDemo(); });

// ── Camera controls ───────────────────────────────────────────────
btnRotate.addEventListener('click', () => {
  autoRotate = !autoRotate;
  skeleton.setAutoRotate(autoRotate);
  btnRotate.style.color = autoRotate ? 'var(--accent)' : '';
});
btnReset.addEventListener('click', () => skeleton.resetCamera());

// ── Entry point ───────────────────────────────────────────────────
// Auto-start demo so the user sees something immediately
startDemo();
