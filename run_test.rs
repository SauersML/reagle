fn select_top_k_heap(scores: &[f32], k: usize, require_positive: bool) -> Vec<(usize, f32)> {
    let mut ranked = Vec::new();
    for (idx, &score) in scores.iter().enumerate() {
        if !score.is_finite() || (require_positive && score <= 0.0) {
            continue;
        }
        ranked.push((idx, score));
    }
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if ranked.len() > k {
        ranked.truncate(k);
    }
    ranked
}
fn main() {
    let scores = vec![0.0, 0.0, 0.0, 0.0];
    let ranked = select_top_k_heap(&scores, 4, false);
    println!("{:?}", ranked);
}
