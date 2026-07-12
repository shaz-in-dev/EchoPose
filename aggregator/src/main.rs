// ============================================================
// main.rs — RF-Mesh Aggregator Server
//
// ┌──────────────┐  UDP:5005   ┌────────────┐
// │  ESP32 Node0 │ ──────────► │            │
// │  ESP32 Node1 │ ──────────► │ Aggregator │ WS:8080 ► Inference
// │  ESP32 Node2 │ ──────────► │            │ REST:3000 ► Status
// └──────────────┘             └────────────┘
//
// Architecture:
//   1. UDP listener task  → parses raw CSI frames
//   2. Sync task          → bundles frames per 50ms window
//   3. Broadcast channel  → fans synced bundles to all WS clients
//   4. Axum HTTP server   → /ws (streaming), /health, /config
// ============================================================

mod sync;
mod types;
mod denoise;
mod localize;

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
    routing::get,
    Json, Router,
};
use serde::Serialize;
use std::{collections::HashMap, net::SocketAddr, sync::Arc, time::{Instant, SystemTime, UNIX_EPOCH}};
use tokio::{net::UdpSocket, sync::{broadcast, RwLock}};
use tower_http::{cors::{Any, CorsLayer}, services::ServeFile};
use tracing::{error, info, warn};

use sync::NodeSynchronizer;
use types::{CsiFrame, RawCsiFrame, SyncedBundle};

use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[derive(Debug, Clone, Serialize)]
pub struct NodeStats {
    pub last_seen_ms: u64,
    pub packet_count: u64,
}

#[derive(Clone, Default)]
struct CalibrationState {
    is_calibrating: bool,
    /// Wall-clock end time kept only for the /calibrate response JSON; do NOT use for elapsed
    /// checking — use `started_at` + `duration_ms` instead to be immune to NTP jumps.
    end_ms: u64,
    started_at: Option<Instant>,
    duration_ms: u64,
    // [node_id] -> (summed_iq_data, sample_count)
    accumulators: HashMap<u8, (Vec<f32>, u32)>,
    // [node_id] -> averaged baseline
    baselines: HashMap<u8, Vec<f32>>,
}

type NodeTracker = Arc<RwLock<HashMap<u8, NodeStats>>>;

// AppState Definition
#[derive(Clone)]
struct AppState {
    tx: broadcast::Sender<SyncedBundle>,
    tracker: NodeTracker,
    calibration: Arc<RwLock<CalibrationState>>,
    localization: Arc<RwLock<localize::LocalizationSolver>>,
    udp_port: u16,
    http_port: u16,
    expected_nodes: usize,
    /// Amplitude-to-RSSI baseline offset (dB). Starts from RSSI_OFFSET_DB env var
    /// (default -50, the old hardcoded placeholder) and can be recalibrated live
    /// via POST /calibrate/rssi without restarting the process.
    rssi_offset_db: Arc<RwLock<f32>>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Attempt to load the central .env file from the project root
    let root_env = std::path::Path::new("..").join(".env");
    let _ = dotenvy::from_path(root_env);

    // ── Logging ───────────────────────────────────────────────────
    std::fs::create_dir_all("logs").ok();
    let file_appender = tracing_appender::rolling::daily("logs", "aggregator.json");
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    let env_filter = tracing_subscriber::EnvFilter::from_default_env()
        .add_directive(std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()).parse()?);

    let stdout_layer = tracing_subscriber::fmt::layer()
        .pretty();
    
    let file_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(non_blocking);

    tracing_subscriber::registry()
        .with(env_filter)
        .with(stdout_layer)
        .with(file_layer)
        .init();

    // ── Configuration ─────────────────────────────────────────────
    let udp_port: u16 = std::env::var("AGGREGATOR_UDP_PORT")
        .unwrap_or_else(|_| "5005".to_string())
        .parse()?;
    
    let http_port: u16 = std::env::var("AGGREGATOR_HTTP_PORT")
        .unwrap_or_else(|_| "3000".to_string())
        .parse()?;

    let expected_nodes: usize = std::env::var("EXPECTED_NODES")
        .unwrap_or_else(|_| "3".to_string())
        .parse()
        .unwrap_or(3);

    // Initialise OnceLock statics from env before any task reads them
    sync::init_window_us();

    let bcast_cap: usize = 256;

    let (tx, _rx) = broadcast::channel::<SyncedBundle>(bcast_cap);
    let tx_udp = tx.clone();

    let expected_nodes_clone = expected_nodes;
    let tracker = Arc::new(RwLock::new(HashMap::new()));
    let tracker_udp = tracker.clone();
    let calibration = Arc::new(RwLock::new(CalibrationState::default()));
    let calibration_udp = calibration.clone();
    let localization = Arc::new(RwLock::new(localize::LocalizationSolver::new()));
    let localization_udp = localization.clone();

    let initial_rssi_offset: f32 = std::env::var("RSSI_OFFSET_DB")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(-50.0);
    let rssi_offset_db = Arc::new(RwLock::new(initial_rssi_offset));
    let rssi_offset_db_udp = rssi_offset_db.clone();

    // ── Unified Denoising ─────────────────────────────────────────
    let mut rolling_denoiser = denoise::RollingDenoiser::new();
    
    // ── UDP listener + sync task ──────────────────────────────────
    tokio::spawn(async move {
        let sock = UdpSocket::bind(format!("0.0.0.0:{}", udp_port))
            .await
            .expect("Failed to bind UDP socket");
        info!("UDP listener on :{}", udp_port);

        let mut buf = vec![0u8; RawCsiFrame::FRAME_SIZE + 64];
        let mut syncer = NodeSynchronizer::new(expected_nodes_clone);

        loop {
            let (n, peer) = match sock.recv_from(&mut buf).await {
                Ok(v) => v,
                Err(e) => { error!("UDP recv error: {}", e); continue; }
            };

            let raw = match RawCsiFrame::from_bytes(&buf[..n]) {
                Some(r) => r,
                None => {
                    warn!("Invalid frame from {}", peer);
                    continue;
                }
            };

            let mut frame = CsiFrame::from(&raw);

            // Update tracker stats & apply calibration
            let now_ms = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64;
            
            // Critical Sync Fix: Overwrite unaligned device uptimes with host arrival time
            frame.timestamp_us = now_ms * 1000;

            {
                let mut tr = tracker_udp.write().await;
                let stats = tr.entry(frame.node_id).or_insert(NodeStats { last_seen_ms: 0, packet_count: 0 });
                stats.packet_count += 1;
                stats.last_seen_ms = now_ms; // Fix missing heartbeat
                
                // V3: Record RSSI for automated localization
                if !frame.amplitudes.is_empty() {
                    let mut loc = localization_udp.write().await;
                    // Use raw amplitude BEFORE denoising for RSSI estimation.
                    // Each node reports its own amplitudes, so the "seen_by" node IS frame.node_id.
                    // The offset starts from RSSI_OFFSET_DB (default -50, a placeholder) and can
                    // be recalibrated live via POST /calibrate/rssi.
                    let offset_db = *rssi_offset_db_udp.read().await;
                    let mean_amp = frame.amplitudes.iter().sum::<f32>() / frame.amplitudes.len() as f32;
                    let rssi = (20.0 * mean_amp.max(1e-6).log10() + offset_db) as i16;
                    let seen_by = frame.node_id; // correct: each node reports its own measurement
                    loc.record_rssi(frame.node_id, seen_by, rssi);
                    // Evict stale nodes to prevent rssi_matrix from growing unbounded
                    let active_ids: Vec<u8> = tr.keys().copied().collect();
                    loc.evict_stale_nodes(&active_ids);
                }
            }

            {
                let mut cal = calibration_udp.write().await;
                if cal.is_calibrating {
                    // Use monotonic Instant to be immune to NTP clock jumps
                    let still_calibrating = cal.started_at
                        .map(|t| t.elapsed().as_millis() < cal.duration_ms as u128)
                        .unwrap_or(false);
                    if still_calibrating {
                        // Accumulate samples for baseline average
                        const EXPECTED_SUBCARRIERS: usize = 64;
                        if frame.amplitudes.len() != EXPECTED_SUBCARRIERS {
                            warn!("Node {}: unexpected amplitude count {} (expected {}), skipping calibration frame",
                                  frame.node_id, frame.amplitudes.len(), EXPECTED_SUBCARRIERS);
                            // still update stats but skip calibration accumulation
                        } else {
                        let entry = cal.accumulators.entry(frame.node_id).or_insert((vec![0.0; frame.amplitudes.len()], 0));
                        for (i, &val) in frame.amplitudes.iter().enumerate() {
                            entry.0[i] += val;
                        }
                        entry.1 += 1;
                        }
                    } else {
                        // Time's up: finalize baselines
                        cal.is_calibrating = false;
                        cal.baselines.clear();
                        let accumulators: Vec<_> = cal.accumulators.drain().collect(); // Move out values before iterating
                        for (node_id, (sum, count)) in accumulators {
                            let baseline: Vec<f32> = sum.into_iter().map(|val| val / count as f32).collect();
                            cal.baselines.insert(node_id, baseline);
                        }
                        info!("Room Calibration Complete: Static noise floor baselines calculated.");
                    }
                }
                
                // Explicit baseline subtraction while holding the lock
                if !cal.is_calibrating {
                    if let Some(baseline) = cal.baselines.get(&frame.node_id) {
                        for i in 0..frame.amplitudes.len().min(baseline.len()) {
                            frame.amplitudes[i] = (frame.amplitudes[i] - baseline[i]).max(0.0);
                        }
                    }
                }
            }

            // Apply rolling median background subtraction (Unified Policy)
            rolling_denoiser.denoise(frame.node_id, &mut frame.amplitudes);

            if let Some(bundle) = syncer.push(frame) {
                match tx_udp.send(bundle) {
                    Ok(_) => {}
                    Err(e) => warn!("Bundle broadcast dropped (buffer full or no receivers): {e}"),
                }
            }
        }
    });

    // ── Axum HTTP + WebSocket server ──────────────────────────────
    let state = AppState {
        tx,
        tracker,
        calibration,
        localization,
        udp_port,
        http_port,
        expected_nodes,
        rssi_offset_db,
    };

    let origins: Vec<axum::http::HeaderValue> = std::env::var("ALLOWED_ORIGINS")
        .unwrap_or_else(|_| "http://localhost:8000,http://localhost:8080".to_string())
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();

    let cors = CorsLayer::new()
        .allow_origin(origins)
        .allow_methods(Any)
        .allow_headers(Any);

    use axum::routing::post;

    let app = Router::new()
        .route("/health", get(health))
        .route("/nodes",  get(nodes_handler))
        .route("/nodes/:id/offline", axum::routing::delete(mark_node_offline))
        .route("/ws",     get(ws_handler))
        .route("/config", get(config_handler))
        .route("/calibrate", post(calibrate_handler))
        .route("/calibrate/rssi", post(calibrate_rssi))
        .route("/localize",  get(localize_handler))
        .route_service("/firmware.bin", ServeFile::new("../firmware/build/firmware.bin"))
        .layer(cors)
        .with_state(Arc::new(state));

    let addr: SocketAddr = format!("0.0.0.0:{}", http_port).parse()?;
    info!("HTTP/WS server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// GET /health
async fn health() -> &'static str { "ok" }

// GET /nodes
async fn nodes_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let tr = state.tracker.read().await;
    Json(tr.clone())
}

// POST /calibrate — Start the 5-second static room calibration (token-gated)
async fn calibrate_handler(
    headers: axum::http::HeaderMap,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Simple bearer-token auth to prevent unauthorized calibration resets
    let expected = std::env::var("ECHOPOSE_API_TOKEN").unwrap_or_default();
    if !expected.is_empty() {
        let provided = headers
            .get("X-EchoPose-Token")
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default();
        if provided != expected {
            return Json(serde_json::json!({"error": "unauthorized"}));
        }
    }
    let mut cal = state.calibration.write().await;
    let duration_ms: u64 = 5000;
    cal.is_calibrating = true;
    cal.started_at = Some(Instant::now());
    cal.duration_ms = duration_ms;
    cal.end_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_millis() as u64 + duration_ms;
    cal.accumulators.clear();
    info!("Calibration Initiated for {}ms over WS/UDP", duration_ms);
    Json(serde_json::json!({"status": "calibrating", "duration_ms": duration_ms}))
}

// GET /localize — Run the solver and return estimated (x,y,z) coordinates
async fn localize_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    // Clone data under lock, then release before expensive compute
    let loc_snapshot = state.localization.read().await.clone();
    let node_ids: Vec<u8> = state.tracker.read().await.keys().cloned().collect();

    // Run the iterative solver off the async executor
    let result = tokio::task::spawn_blocking(move || loc_snapshot.solve(&node_ids))
        .await
        .unwrap_or_default();
    info!("Automated Localization: Estimated positions for {} nodes.", result.len());
    Json(result)
}

// GET /config
async fn config_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    use crate::sync::WINDOW_US;
    let window_us = *WINDOW_US.get().unwrap_or(&50_000);
    let rssi_offset_db = *state.rssi_offset_db.read().await;
    Json(serde_json::json!({
        "udp_port":       state.udp_port,
        "http_port":      state.http_port,
        "expected_nodes": state.expected_nodes,
        "window_ms":      window_us / 1_000,
        "rssi_offset_db": rssi_offset_db,
    }))
}

// GET /ws — upgrades to WebSocket; streams SyncedBundle JSON
async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

async fn handle_socket(mut socket: WebSocket, state: Arc<AppState>) {
    let mut rx = state.tx.subscribe();
    info!("WebSocket client connected");

    loop {
        match rx.recv().await {
            Ok(bundle) => {
                let json = match serde_json::to_string(&bundle) {
                    Ok(j) => j,
                    Err(e) => { error!("Serialize error: {}", e); continue; }
                };
                if socket.send(Message::Text(json)).await.is_err() {
                    break; // client disconnected
                }
            }
            Err(broadcast::error::RecvError::Lagged(n)) => {
                warn!("WS client lagged by {} frames", n);
            }
            Err(_) => break,
        }
    }
    info!("WebSocket client disconnected");
}

// DELETE /nodes/:id/offline — manually mark a node as offline and remove it from the tracker
async fn mark_node_offline(
    axum::extract::Path(node_id): axum::extract::Path<u8>,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let mut tracker = state.tracker.write().await;
    if tracker.remove(&node_id).is_some() {
        info!("Node {} manually marked offline and removed from tracker", node_id);
        Json(serde_json::json!({"status": "removed", "node_id": node_id}))
    } else {
        Json(serde_json::json!({"status": "not_found", "node_id": node_id}))
    }
}

// POST /calibrate/rssi — record a reference distance measurement for RSSI calibration
#[derive(serde::Deserialize)]
struct RssiCalibRequest {
    node_a: u8,
    node_b: u8,
    distance_m: f32,
}

async fn calibrate_rssi(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RssiCalibRequest>,
) -> impl IntoResponse {
    if req.distance_m <= 0.0 {
        return Json(serde_json::json!({
            "status": "error",
            "detail": "distance_m must be positive"
        }));
    }

    // Need an existing RSSI observation between the two nodes to calibrate against.
    // Try both directions since either node may have "seen" the other first.
    let observed_rssi = {
        let loc = state.localization.read().await;
        loc.get_rssi(req.node_b, req.node_a)
            .or_else(|| loc.get_rssi(req.node_a, req.node_b))
    };

    let Some(observed_rssi) = observed_rssi else {
        return Json(serde_json::json!({
            "status": "no_data",
            "detail": "No RSSI observed yet between these nodes — make sure both are streaming CSI, then retry."
        }));
    };

    // Same log-distance model as localize.rs's target_dist: target_dist = 10^((-40 - rssi) / 20)
    // Solve for the RSSI that would make target_dist equal the measured reference distance.
    let target_rssi = -40.0 - 20.0 * req.distance_m.log10();
    let delta = target_rssi - observed_rssi as f32;

    let new_offset = {
        let mut offset = state.rssi_offset_db.write().await;
        *offset += delta;
        *offset
    };

    info!(
        "RSSI calibrated: node {} <-> node {} = {:.2}m (observed {} dB, target {:.1} dB) -> offset now {:.1} dB — applied immediately",
        req.node_a, req.node_b, req.distance_m, observed_rssi, target_rssi, new_offset
    );

    Json(serde_json::json!({
        "status": "calibrated",
        "node_a": req.node_a,
        "node_b": req.node_b,
        "distance_m": req.distance_m,
        "observed_rssi_db": observed_rssi,
        "rssi_offset_db": new_offset,
        "note": "Applied immediately, no restart needed. Set RSSI_OFFSET_DB env var to this value to persist it across restarts."
    }))
}
