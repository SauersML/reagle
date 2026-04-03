fn prescan_match_weight(freq: f32, min_freq: f32) -> f32 {
    let p = freq.clamp(min_freq, 1.0 - min_freq);
    let raw_weight = ((1.0 - p) / p).ln();
    raw_weight.abs() // Absolute log-odds makes symmetric weight
}
fn main() {
    println!("weight for 0.4: {}", prescan_match_weight(0.4, 0.01));
    println!("weight for 0.49: {}", prescan_match_weight(0.49, 0.01));
    println!("weight for 0.5: {}", prescan_match_weight(0.5, 0.01));
    println!("weight for 0.51: {}", prescan_match_weight(0.51, 0.01));
    println!("weight for 0.6: {}", prescan_match_weight(0.6, 0.01));
}
