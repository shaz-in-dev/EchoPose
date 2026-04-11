use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn normalize_subcarriers(values: Vec<f32>) -> Vec<f32> {
    if values.is_empty() {
        return values;
    }

    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;

    for v in &values {
        if *v < min_v {
            min_v = *v;
        }
        if *v > max_v {
            max_v = *v;
        }
    }

    let denom = (max_v - min_v).max(1e-6);
    values.into_iter().map(|x| (x - min_v) / denom).collect()
}

#[wasm_bindgen]
pub fn confidence_to_alpha(conf: f32) -> f32 {
    conf.clamp(0.0, 1.0)
}
