use std::collections::HashMap;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct NodePosition {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Automated Node Localization solver.
/// Uses a simplified Multi-Dimensional Scaling (MDS) approach to estimate
/// relative (x, y) coordinates of ESP32 nodes based on their inter-node signal strengths.
#[derive(Clone)]
pub struct LocalizationSolver {
    // node_id -> { other_node_id -> rssi }
    rssi_matrix: HashMap<u8, HashMap<u8, i16>>,
}

impl LocalizationSolver {
    pub fn new() -> Self {
        Self {
            rssi_matrix: HashMap::new(),
        }
    }

    /// Record a signal strength measurement between two nodes.
    pub fn record_rssi(&mut self, from_node: u8, seen_by_node: u8, rssi: i16) {
        self.rssi_matrix
            .entry(seen_by_node)
            .or_insert_with(HashMap::new)
            .insert(from_node, rssi);
    }

    /// Look up the most recent RSSI reading `seen_by_node` recorded for `from_node`.
    pub fn get_rssi(&self, seen_by_node: u8, from_node: u8) -> Option<i16> {
        self.rssi_matrix.get(&seen_by_node)?.get(&from_node).copied()
    }

    /// Evict entries for nodes no longer in the active set, preventing unbounded growth.
    pub fn evict_stale_nodes(&mut self, active_node_ids: &[u8]) {
        let active: std::collections::HashSet<u8> = active_node_ids.iter().copied().collect();
        self.rssi_matrix.retain(|k, _| active.contains(k));
    }

    /// Solves for relative (x, y, z) coordinates.
    /// In this V3 prototype, we use a spring-embedded simulation to converge on positions.
    pub fn solve(&self, node_ids: &[u8]) -> HashMap<u8, NodePosition> {
        let n = node_ids.len();
        if n == 0 { return HashMap::new(); }

        let mut positions: HashMap<u8, NodePosition> = HashMap::new();
        
        // Initialise nodes in a circle
        for (i, &id) in node_ids.iter().enumerate() {
            let angle = (i as f32 / n as f32) * 2.0 * std::f32::consts::PI;
            positions.insert(id, NodePosition {
                x: angle.cos() * 2.0,
                y: -1.8, // Floor level
                z: angle.sin() * 2.0,
            });
        }

        // Anchor the first node at (0, -1.8, 2)
        let first_id = node_ids.first().copied();
        if let Some(fid) = first_id {
            positions.insert(fid, NodePosition { x: 0.0, y: -1.8, z: 2.0 });
        }

        // Simple iterative force-directed refinement (100 steps)
        // Nodes "push" each other based on RSSI (stronger = closer)
        for _ in 0..100 {
            let mut forces: HashMap<u8, (f32, f32)> = HashMap::new();
            let mut max_force_mag: f32 = 0.0;

            for &i_id in node_ids {
                let mut fx = 0.0;
                let mut fz = 0.0;
                let pi = positions[&i_id].clone();

                for &j_id in node_ids {
                    if i_id == j_id { continue; }
                    let pj = positions[&j_id].clone();

                    let dx = pj.x - pi.x;
                    let dz = pj.z - pi.z;
                    let dist = (dx*dx + dz*dz).sqrt().max(0.1);

                    // Target distance based on RSSI (approximate log model)
                    // RSSI -40 (close) -> ~0.5m, RSSI -80 (far) -> ~5.0m
                    let rssi = self.rssi_matrix.get(&i_id)
                        .and_then(|m| m.get(&j_id))
                        .copied()
                        .unwrap_or(-70);

                    let target_dist = 10.0f32.powf((-40.0 - rssi as f32) / 20.0);

                    let diff = dist - target_dist;
                    let strength = 0.05;
                    fx += (dx / dist) * diff * strength;
                    fz += (dz / dist) * diff * strength;
                }
                max_force_mag = max_force_mag.max(fx.abs() + fz.abs());
                forces.insert(i_id, (fx, fz));
            }

            // Apply forces — skip the anchor node so it never drifts
            for &id in node_ids {
                if Some(id) == first_id { continue; }  // anchor is fixed
                let f = forces[&id];
                if let Some(p) = positions.get_mut(&id) {
                    p.x += f.0;
                    p.z += f.1;
                }
            }

            // Convergence early exit
            if max_force_mag < 1e-3 {
                break;
            }
        }

        positions
    }
}
