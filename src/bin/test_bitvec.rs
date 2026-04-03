fn compute_abyss_rank_cutoff(n_ref_haps: usize, window_top_k: usize) -> usize {
    if n_ref_haps == 0 {
        return 1;
    }
    window_top_k.max(1).min(n_ref_haps)
}
fn main() {
    println!("{}", compute_abyss_rank_cutoff(100, 20));
}
