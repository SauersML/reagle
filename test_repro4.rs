fn prescan_match_weight(freq: f32, min_freq: f32) -> f32 {
    let p = freq.clamp(min_freq, 1.0 - min_freq);
    ((1.0 - p) / p).ln().max(0.0)
}
fn main() {
    let freq = 0.5;
    println!("weight for 0.5: {}", prescan_match_weight(freq, 0.01));
    let freq = 1.0;
    println!("weight for 1.0: {}", prescan_match_weight(freq, 0.01));
    let freq = 0.0;
    println!("weight for 0.0: {}", prescan_match_weight(freq, 0.01));
}
