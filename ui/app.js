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

  // Show simulation mode indicator
  if (data.simulation) {
    statusBadge.textContent = 'Sim';
    statusBadge.className   = 'badge badge--sim';
  }
  
  // V3 format (Array of skeletons)
  if (data.skeletons) {
    skeleton.updateSkeletons(data.skeletons);
    if (data.skeletons[0]) updateKpTable(data.skeletons[0]);
  } 
  
  if (data.amplitudes) {
    // Use WASM normalisation if available (gracefully falls back to JS)
    if (window.EchoPoseWasm) {
      const nodes = data.amplitudes;
      for (const nid of Object.keys(nodes)) {
        if (nodes[nid] && nodes[nid].amplitudes) {
          nodes[nid].amplitudes = EchoPoseWasm.normalizeCSI(nodes[nid].amplitudes);
        }
      }
    }
    heatmap.push(data.amplitudes);
  }
  // V2 format (Backward compatibility for old recordings)
  else if (data.keypoints) {
    skeleton.updateSkeletons([data.keypoints]);
    updateKpTable(data.keypoints);
  }

  // Analytics dashboard
  if (data.analytics) {
    updateAnalytics(data.analytics);
  }

  // Tactical dashboard
  if (data.tactical) {
    updateTactical(data.tactical);
  }
}

// ── Analytics Dashboard ───────────────────────────────────────────
const analyticsCards = document.getElementById('analytics-cards');
const btnToggleAnalytics = document.getElementById('btn-toggle-analytics');

if (btnToggleAnalytics) {
  btnToggleAnalytics.addEventListener('click', () => {
    const hidden = analyticsCards.classList.toggle('collapsed');
    btnToggleAnalytics.textContent = hidden ? 'Show' : 'Hide';
  });
}

// ── Tactical Dashboard ────────────────────────────────────────────
const tacticalCards = document.getElementById('tactical-cards');
const btnToggleTactical = document.getElementById('btn-toggle-tactical');

if (btnToggleTactical) {
  btnToggleTactical.addEventListener('click', () => {
    const hidden = tacticalCards.classList.toggle('collapsed');
    btnToggleTactical.textContent = hidden ? 'Show' : 'Hide';
  });
}

function updateTactical(t) {
  // ── Threats ──
  const tgt = t.targets || {};
  const threatEl = document.getElementById('t-threat-level');
  if (threatEl) {
    const level = tgt.threat_level || 'GREEN';
    threatEl.textContent = level;
    threatEl.className = 'acard-big'
      + (level === 'RED' ? ' acard-big--critical' : '')
      + (level === 'YELLOW' ? ' acard-big--warning' : '');
  }
  setText('t-target-count', tgt.target_count || 0);
  const targets = tgt.targets || [];
  setText('t-target-type', targets.length ? targets[0].classification || '--' : '--');

  // ── Concealment ──
  const conc = t.concealment || {};
  setText('t-concealed-count', conc.concealed_targets || 0);
  setText('t-scan-quality', conc.scan_quality || '--');

  // ── Weapon ──
  const weap = t.weapon || {};
  const weapEl = document.getElementById('t-weapon-type');
  if (weapEl) {
    const wt = weap.weapon_type || 'UNARMED';
    weapEl.textContent = wt;
    weapEl.className = 'acard-big'
      + (wt === 'RIFLE' || wt === 'HEAVY_LOAD' ? ' acard-big--critical' : '')
      + (wt === 'HANDGUN' ? ' acard-big--warning' : '');
  }
  setText('t-weapon-conf', weap.confidence != null ? `${(weap.confidence * 100).toFixed(0)}%` : '--');
  setText('t-armor', weap.body_armor_likelihood != null ? `${(weap.body_armor_likelihood * 100).toFixed(0)}%` : '--');

  // ── Crowd ──
  const crowd = t.crowd || {};
  setText('t-crowd-count', crowd.estimated_count || 0);
  setText('t-crowd-density', crowd.density_per_m2 != null ? `${crowd.density_per_m2} /m²` : '--');
  setText('t-density-cat', crowd.density_category || 'SPARSE');

  // ── Tactical Activity ──
  const tac = t.tactical_activity || {};
  setText('t-tac-activity', tac.activity || '--');
  setText('t-tac-conf', tac.confidence != null ? `${(tac.confidence * 100).toFixed(0)}%` : '--');

  // ── Anomalies ──
  const anom = t.anomalies || {};
  const assessEl = document.getElementById('t-threat-assess');
  if (assessEl) {
    const assess = anom.threat_assessment || 'CLEAR';
    assessEl.textContent = assess;
    assessEl.className = 'acard-big'
      + (assess === 'DANGER' ? ' acard-big--critical' : '')
      + (assess === 'SUSPICIOUS' ? ' acard-big--warning' : '');
  }
  setText('t-anomaly-count', anom.anomalies_found || 0);
  setText('t-scan-coverage', anom.scan_coverage || '--');

  // ── Intent ──
  const intent = t.intent || {};
  const intentEl = document.getElementById('t-intent');
  if (intentEl) {
    const il = intent.intent || 'NORMAL';
    intentEl.textContent = il;
    intentEl.className = 'acard-big'
      + (il === 'ATTACK_IMMINENT' ? ' acard-big--critical' : '')
      + (il === 'ACCESS_WEAPON' || il === 'FLEE' ? ' acard-big--warning' : '');
  }
  const intentBar = document.getElementById('t-intent-bar');
  const intentScore = (intent.attack_probability || 0) * 100;
  if (intentBar) intentBar.style.width = `${intentScore}%`;
  setText('t-intent-score', `${intentScore.toFixed(0)}%`);

  // ── Anti-Jam ──
  const jam = t.anti_jam || {};
  const jamEl = document.getElementById('t-jam-status');
  if (jamEl) {
    const jamStatus = jam.under_attack ? 'UNDER ATTACK' : (jam.status || 'CLEAN');
    jamEl.textContent = jamStatus;
    jamEl.className = 'acard-big'
      + (jam.under_attack ? ' acard-big--critical' : '');
  }
  setText('t-jam-threats', jam.threats ? jam.threats.length : 0);

  // ── Fusion COP ──
  const fus = t.fusion || {};
  setText('t-fusion-tracks', fus.total_tracks || 0);
  const sources = fus.active_modalities || [];
  setText('t-fusion-sources', sources.length ? sources.join(', ') : '--');
}

function updateAnalytics(a) {
  // ── Vitals ──
  const v = a.vitals || {};
  const hr = v.heart_rate || {};
  const rr = v.respiratory_rate || {};
  const spo2 = v.spo2 || {};
  const temp = v.temperature || {};
  const bp = v.blood_pressure || {};

  setText('v-hr',   hr.heart_rate != null ? hr.heart_rate : '--');
  setText('v-rr',   rr.respiratory_rate != null ? rr.respiratory_rate : '--');
  setText('v-spo2', spo2.spo2 != null ? spo2.spo2 : '--');
  setText('v-temp', temp.temperature_c != null ? temp.temperature_c : '--');
  setText('v-bp',   bp.systolic_mmhg != null
    ? `${Math.round(bp.systolic_mmhg)}/${Math.round(bp.diastolic_mmhg)}`
    : '--/--');

  // ── Activity & Gait ──
  const act = a.activity || {};
  const gait = a.gait || {};
  setText('v-activity', capitalize(act.activity || '--'));
  setText('v-activity-conf', act.confidence != null ? `${(act.confidence * 100).toFixed(0)}% conf` : '');
  setText('v-gait-speed',   gait.walking_speed_ms != null ? `${gait.walking_speed_ms} m/s` : '--');
  setText('v-gait-cadence', gait.cadence_steps_min != null ? `${gait.cadence_steps_min} spm` : '--');
  setText('v-gait-stride',  gait.stride_length_m != null ? `${gait.stride_length_m} m` : '--');
  setText('v-gait-sym',     gait.gait_symmetry != null ? `${(gait.gait_symmetry * 100).toFixed(0)}%` : '--');

  // ── Fall ──
  const fall = a.fall || {};
  const fallEl = document.getElementById('v-fall-status');
  if (fallEl) {
    if (fall.fall_detected) {
      fallEl.textContent = 'FALL!';
      fallEl.className = 'acard-big acard-big--critical';
    } else {
      fallEl.textContent = 'Safe';
      fallEl.className = 'acard-big';
    }
  }
  setText('v-fall-risk', fall.fall_risk || '--');
  setText('v-fall-balance', fall.balance_score != null ? `${(fall.balance_score * 100).toFixed(0)}%` : '--');

  // ── Gesture ──
  const gest = a.gestures || {};
  setText('v-gest-left',  capitalize(gest.left_hand || 'idle'));
  setText('v-gest-right', capitalize(gest.right_hand || 'idle'));

  // ── Sleep ──
  const sleep = a.sleep || {};
  setText('v-sleep-stage', sleep.sleep_stage || '--');
  setText('v-sleep-conf',  sleep.confidence != null ? `${(sleep.confidence * 100).toFixed(0)}% conf` : '');

  // ── Occupancy ──
  const occ = a.occupancy || {};
  setText('v-occupancy', occ.occupied ? 'Occupied' : 'Empty');
  setText('v-occ-count', occ.num_people != null ? occ.num_people : '0');

  // ── Stress ──
  const emo = a.emotion || {};
  setText('v-stress-level', emo.stress_level || '--');
  setText('v-stress-score', emo.stress_score != null ? `${emo.stress_score.toFixed(0)} / 100` : '0 / 100');
  const bar = document.getElementById('v-stress-bar');
  if (bar) bar.style.width = `${emo.stress_score || 0}%`;

  // ── Health Alerts ──
  const alerts = a.health_alerts || {};
  const alertEl = document.getElementById('v-alert-level');
  if (alertEl) {
    alertEl.textContent = alerts.alert_level || 'NORMAL';
    alertEl.className = 'acard-big'
      + (alerts.alert_level === 'CRITICAL' ? ' acard-big--critical' : '')
      + (alerts.alert_level === 'WARNING'  ? ' acard-big--warning' : '');
  }
  const alertList = document.getElementById('v-alert-list');
  if (alertList) {
    if (alerts.anomalies && alerts.anomalies.length) {
      alertList.innerHTML = alerts.anomalies.map(a =>
        `<div class="acard-alert-item">${escapeHtml(a)}</div>`
      ).join('');
    } else {
      alertList.innerHTML = '<span class="acard-sub">No anomalies</span>';
    }
  }
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function capitalize(s) {
  return s && typeof s === 'string' ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

function escapeHtml(str) {
  const d = document.createElement('div');
  d.textContent = str;
  return d.innerHTML;
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
    
    // Count nodes — the aggregator sends last_seen_ms as a Unix ms timestamp.
    // We compare against Date.now() with a 5-second freshness window.
    const now = Date.now();
    let active = 0;
    for (const [id, stats] of Object.entries(nodes)) {
      const age = now - stats.last_seen_ms;
      if (age >= 0 && age < 5000) active++;
    }
    // If all timestamps look like they are in the past but recent, count them
    if (active === 0 && Object.keys(nodes).length > 0) {
      // Fallback: just count non-zero packet_count nodes
      for (const [id, stats] of Object.entries(nodes)) {
        if (stats.packet_count > 0) active++;
      }
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

  // Synthetic analytics for demo
  const hrBase = 72 + 5 * Math.sin(demoT * 0.02);
  const rrBase = 16 + 2 * Math.sin(demoT * 0.015);
  const activities = ['standing', 'walking', 'running', 'sitting'];
  const actIdx = Math.floor((demoT / 200) % activities.length);
  const stressVal = 15 + 20 * Math.abs(Math.sin(demoT * 0.01));

  updateAnalytics({
    vitals: {
      heart_rate:       { heart_rate: +hrBase.toFixed(1), confidence: 0.85 },
      respiratory_rate: { respiratory_rate: +rrBase.toFixed(1), confidence: 0.80 },
      spo2:             { spo2: +(97 + Math.random()).toFixed(1), confidence: 0.55 },
      temperature:      { temperature_c: +(36.8 + 0.3 * Math.sin(demoT * 0.005)).toFixed(1), confidence: 0.60 },
      blood_pressure:   { systolic_mmhg: 120, diastolic_mmhg: 80, confidence: 0.50 },
    },
    activity:  { activity: activities[actIdx], confidence: 0.82 },
    gait:      { walking_speed_ms: +(0.8 + 0.3 * Math.sin(demoT * 0.03)).toFixed(2), cadence_steps_min: 108, stride_length_m: 0.72, gait_symmetry: 0.94, step_count: Math.floor(demoT / 10) },
    fall:      { fall_detected: false, fall_risk: 'LOW', balance_score: 0.92 },
    gestures:  { left_hand: 'idle', right_hand: demoT % 200 < 30 ? 'wave' : 'idle', confidence: 0.80 },
    sleep:     { sleep_stage: 'AWAKE', confidence: 0.88 },
    occupancy: { occupied: true, num_people: 1, method: 'skeleton', confidence: 0.95 },
    emotion:   { stress_level: stressVal < 30 ? 'CALM' : 'MODERATE', stress_score: +stressVal.toFixed(1), hr_elevation_pct: 5.2, rr_elevation_pct: 3.1 },
    health_alerts: { anomalies_detected: false, anomalies: [], alert_level: 'NORMAL' },
  });

  // Synthetic tactical data for demo
  const tacActivities = ['STANDING', 'MOVING_TACTICAL', 'TAKING_AIM', 'CRAWLING'];
  const tacIdx = Math.floor((demoT / 300) % tacActivities.length);
  const atkProb = 0.05 + 0.1 * Math.abs(Math.sin(demoT * 0.008));

  updateTactical({
    targets: { threat_level: 'GREEN', target_count: 1, targets: [{ classification: 'HUMAN_WALKING', confidence: 0.82 }] },
    concealment: { concealed_targets: 0, scan_quality: 'PARTIAL' },
    weapon: { weapon_type: 'UNARMED', confidence: 0.85, body_armor_likelihood: 0.0, threat_level: 'LOW' },
    crowd: { estimated_count: 1, density_per_m2: 0.02, density_category: 'SPARSE', confidence: 0.95 },
    tactical_activity: { activity: tacActivities[tacIdx], confidence: 0.78 },
    anomalies: { threat_assessment: 'CLEAR', anomalies_found: 0, scan_coverage: 'FULL' },
    intent: { intent: 'NORMAL', attack_probability: atkProb },
    anti_jam: { under_attack: false, status: 'CLEAN', threats: [] },
    fusion: { total_tracks: 1, active_modalities: ['wifi_csi'] },
  });
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
// Auto-connect to the local inference server by default
connect("ws://localhost:8765/ws/pose");
