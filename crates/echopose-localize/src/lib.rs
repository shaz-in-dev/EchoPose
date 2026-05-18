use std::collections::HashMap;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct NodePosition {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// RSSI-based node position estimation using spring-embedded MDS.
#[derive(Clone)]
pub struct LocalizationSolver {
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

    /// Solves for relative (x, y, z) coordinates via force-directed simulation.
    pub fn solve(&self, node_ids: &[u8]) -> HashMap<u8, NodePosition> {
        let n = node_ids.len();
        if n == 0 {
            return HashMap::new();
        }

        let mut positions: HashMap<u8, NodePosition> = HashMap::new();

        for (i, &id) in node_ids.iter().enumerate() {
            let angle = (i as f32 / n as f32) * 2.0 * std::f32::consts::PI;
            positions.insert(
                id,
                NodePosition {
                    x: angle.cos() * 2.0,
                    y: -1.8,
                    z: angle.sin() * 2.0,
                },
            );
        }

        if let Some(&first_id) = node_ids.first() {
            positions.insert(
                first_id,
                NodePosition {
                    x: 0.0,
                    y: -1.8,
                    z: 2.0,
                },
            );
        }

        for _ in 0..100 {
            let mut forces: HashMap<u8, (f32, f32)> = HashMap::new();

            for &i_id in node_ids {
                let mut fx = 0.0_f32;
                let mut fz = 0.0_f32;
                let pi = positions[&i_id].clone();

                for &j_id in node_ids {
                    if i_id == j_id {
                        continue;
                    }
                    let pj = positions[&j_id].clone();
                    let dx = pj.x - pi.x;
                    let dz = pj.z - pi.z;
                    let dist = (dx * dx + dz * dz).sqrt().max(0.1);

                    let rssi = self
                        .rssi_matrix
                        .get(&i_id)
                        .and_then(|m| m.get(&j_id))
                        .copied()
                        .unwrap_or(-70);

                    let target_dist = 10.0f32.powf((-40.0 - rssi as f32) / 20.0);
                    let diff = dist - target_dist;
                    let strength = 0.05;
                    fx += (dx / dist) * diff * strength;
                    fz += (dz / dist) * diff * strength;
                }
                forces.insert(i_id, (fx, fz));
            }

            for (idx, &id) in node_ids.iter().enumerate() {
                if idx == 0 {
                    continue;
                }
                let f = forces[&id];
                if let Some(p) = positions.get_mut(&id) {
                    p.x += f.0;
                    p.z += f.1;
                }
            }
        }

        positions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_solve() {
        let s = LocalizationSolver::new();
        assert!(s.solve(&[]).is_empty());
    }

    #[test]
    fn single_node_at_anchor() {
        let s = LocalizationSolver::new();
        let pos = s.solve(&[0]);
        assert!(pos.contains_key(&0));
        assert!((pos[&0].x - 0.0).abs() < 0.1);
    }

    #[test]
    fn two_nodes_separate() {
        let mut s = LocalizationSolver::new();
        s.record_rssi(0, 1, -60);
        s.record_rssi(1, 0, -60);
        let pos = s.solve(&[0, 1]);
        let d = ((pos[&0].x - pos[&1].x).powi(2) + (pos[&0].z - pos[&1].z).powi(2)).sqrt();
        assert!(d > 0.1);
    }
}
