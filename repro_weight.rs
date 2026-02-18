fn prescan_match_weight(freq: f32, min_freq: f32) -> f32 {
    let p = freq.clamp(min_freq, 1.0 - min_freq);
    ((1.0 - p) / p).ln().max(0.0)
}

fn prescan_match_weight_beagle(freq: f32, min_freq: f32) -> f32 {
    let p = freq.clamp(min_freq, 1.0 - min_freq);
    -p.ln()
}

fn main() {
    let min_freq = 1.0 / 200.0;
    
    let freqs = vec![0.005, 0.01, 0.1, 0.5, 0.9, 0.99, 0.995];
    
    println!("{:<10} {:<15} {:<15}", "Freq", "Reagle", "Beagle");
    for &f in &freqs {
        let r = prescan_match_weight(f, min_freq);
        let b = prescan_match_weight_beagle(f, min_freq);
        println!("{:<10.4} {:<15.4} {:<15.4}", f, r, b);
    }
}
