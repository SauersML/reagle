fn prescan_match_weight(freq: f32, min_freq: f32) -> f32 {
    let p = freq.clamp(min_freq, 1.0 - min_freq);
    ((1.0 - p) / p).ln().max(0.0)
}
fn main() {
    let min_freq = 1.0 / 100.0; // 100 haps
    let freq = 100.0 / 100.0; // present = 100, allele_counts = 100
    println!("weight: {}", prescan_match_weight(freq, min_freq));
    
    let freq_rare = 1.0 / 100.0;
    println!("weight_rare: {}", prescan_match_weight(freq_rare, min_freq));

    let freq_half = 50.0 / 100.0;
    println!("weight_half: {}", prescan_match_weight(freq_half, min_freq));
}
