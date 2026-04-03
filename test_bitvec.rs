use bitvec::prelude::*;
fn main() {
    let n_ref_haps = 100;
    let window_top_k = 60; // Just some config value
    let mut best_window_scores = vec![vec![1.0; n_ref_haps]; 1]; // batch size 1
    let mut window_rank_hits = vec![vec![1; n_ref_haps]; 1];
    
    // Simulate abyss check
    for i in 0..1 {
        let mut abyss = bitvec![u64, Lsb0; 0; n_ref_haps];
        let mut abyss_count = 0usize;
        for h in 0..n_ref_haps {
            let score = best_window_scores[i][h];
            if window_rank_hits[i][h] == 0 || !score.is_finite() || score <= 0.0 {
                abyss.set(h, true);
                abyss_count += 1;
            }
        }
        
        let min_survivors = 25usize.min(n_ref_haps);
        let mut survivors = n_ref_haps.saturating_sub(abyss_count);
        println!("Initial survivors: {}", survivors);
        if survivors < min_survivors {
            let ranked = vec![(0, 0.0)]; // dummy
            for (h, _) in ranked {
                if survivors >= min_survivors {
                    break;
                }
                if abyss[h] {
                    abyss.set(h, false);
                    abyss_count = abyss_count.saturating_sub(1);
                    survivors += 1;
                }
            }
            if survivors < min_survivors {
                for h in 0..n_ref_haps {
                    if survivors >= min_survivors {
                        break;
                    }
                    if abyss[h] {
                        abyss.set(h, false);
                        abyss_count = abyss_count.saturating_sub(1);
                        survivors += 1;
                    }
                }
            }
        }
        println!("Final survivors: {}", survivors);
    }
}
