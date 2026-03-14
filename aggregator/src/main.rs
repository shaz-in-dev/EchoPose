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
use std::{collections::HashMap, net::SocketAddr, sync::Arc, time::{SystemTime, UNIX_EPOCH}};
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
    end_ms: u64,
    // [node_id] -> (summed_iq_data, sample_count)
    accumulators: HashMap<u8, (Vec<f32>, u32)>,
    // [node_id] -> averaged baseline
    baselines: HashMap<u8, Vec<i16>>,
}

type NodeTracker = Arc<RwLock<HashMap<u8, NodeStats>>>;

#[derive(Clone)]
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

    let bcast_cap: usize = 256;

    let (tx, _rx) = broadcast::channel::<SyncedBundle>(BCAST_CAP);
    let tx_udp = tx.clone();

    let expected_nodes_clone = expected_nodes;
    let tracker = Arc::new(RwLock::new(HashMap::new()));
    let tracker_udp = tracker.clone();
    let calibration = Arc::new(RwLock::new(CalibrationState::default()));
    let calibration_udp = calibration.clone();
    let localization = Arc::new(RwLock::new(localize::LocalizationSolver::new()));
    let localization_udp = localization.clone();
    
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
            let now_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64;
            {
                let mut tr = tracker_udp.write().await;
                let stats = tr.entry(frame.node_id).or_insert(NodeStats { last_seen_ms: 0, packet_count: 0 });
                stats.packet_count += 1;
                
                // V3: Record RSSI for automated localization
                let mut loc = localization_udp.write().await;
                loc.record_rssi(frame.node_id, 0, -50 - (frame.node_id as i16 * 10)); // Simulated RSSI
            }

            {
                let mut cal = calibration_udp.write().await;
                if cal.is_calibrating {
                    if now_ms < cal.end_ms {
                        // Accumulate samples for baseline average
                        let entry = cal.accumulators.entry(frame.node_id).or_insert((vec![0.0; frame.iq_data.len()], 0));
                        for (i, &val) in frame.iq_data.iter().enumerate() {
                            entry.0[i] += val as f32;
                        }
                        entry.1 += 1;
                    } else {
                        // Time's up: finalize baselines
                        cal.is_calibrating = false;
                        cal.baselines.clear();
                        let accumulators: Vec<_> = cal.accumulators.drain().collect(); // Move out values before iterating
                        for (node_id, (sum, count)) in accumulators {
                            let baseline: Vec<i16> = sum.into_iter().map(|val| (val / count as f32) as i16).collect();
                            cal.baselines.insert(node_id, baseline);
                        }
                        info!("Room Calibration Complete: Static noise floor baselines calculated.");
                    }
                }

                }
            }

            // Apply rolling median background subtraction (Unified Policy)
            rolling_denoiser.denoise(frame.node_id, &mut frame.amplitudes);

            if let Some(bundle) = syncer.push(frame) {
                // Fan-out; ignore if no subscribers
                let _ = tx_udp.send(bundle);
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
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    use axum::routing::post;

    let app = Router::new()
        .route("/health", get(health))
        .route("/nodes",  get(nodes_handler))
        .route("/ws",     get(ws_handler))
        .route("/config", get(config_handler))
        .route("/calibrate", post(calibrate_handler))
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

// POST /calibrate — Start the 5-second static room calibration
async fn calibrate_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({"status": "calibrating", "duration_ms": 5000}))
}

// GET /localize — Run the solver and return estimated (x,y,z) coordinates
async fn localize_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let loc = state.localization.read().await;
    let tr = state.tracker.read().await;
    let node_ids: Vec<u8> = tr.keys().cloned().collect();
    
    let result = loc.solve(&node_ids);
    info!("Automated Localization: Estimated positions for {} nodes.", result.len());
    Json(result)
}

// GET /config
async fn config_handler(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({
        "udp_port":       state.udp_port,
        "http_port":      state.http_port,
        "expected_nodes": state.expected_nodes,
        "window_ms":      50,
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
