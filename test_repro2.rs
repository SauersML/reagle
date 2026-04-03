fn prescan_match_weight(freq: f32, min_freq: f32) -> f32 {
    let p = freq.clamp(min_freq, 1.0 - min_freq);
    ((1.0 - p) / p).ln().max(0.0)
}
fn main() {
    let scores = [
        (0, prescan_match_weight(50.0/100.0, 0.01)),
        (1, prescan_match_weight(50.0/100.0, 0.01))
    ];
    println!("scores: {:?}", scores);
}
