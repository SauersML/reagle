use bitvec::prelude::*;

fn main() {
    let n_ref_haps = 100;
    
    // Case where window_rank_hits == 0 for everything
    let mut best_window_scores = vec![1.0; n_ref_haps]; 
    let mut window_rank_hits = vec![0; n_ref_haps];
    
    let mut abyss = bitvec![u64, Lsb0; 0; n_ref_haps];
    let mut abyss_count = 0usize;
    for h in 0..n_ref_haps {
        let score = best_window_scores[h];
        if window_rank_hits[h] == 0 || !score.is_finite() || score <= 0.0 {
            abyss.set(h, true);
            abyss_count += 1;
        }
    }
    
    println!("Initial abyss_count: {}", abyss_count);
    
    let min_survivors = 25usize.min(n_ref_haps);
    let mut survivors = n_ref_haps.saturating_sub(abyss_count);
    println!("Initial survivors: {}", survivors);
}
