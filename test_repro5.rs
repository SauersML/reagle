fn prescan_match_weight(freq: f32, min_freq: f32) -> f32 {
    let p = freq.clamp(min_freq, 1.0 - min_freq);
    ((1.0 - p) / p).ln().max(0.0)
}
fn main() {
    let p = 1.0;
    println!("p = {}, weight = {}", p, prescan_match_weight(p, 0.01));
}
