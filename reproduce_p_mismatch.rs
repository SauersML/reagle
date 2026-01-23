fn main() {
    let n_haps = 1_000_000;
    let p = li_stephens_p_mismatch(n_haps);
    println!("n_haps: {}, p: {:.10e}", n_haps, p);

    let n_haps = 100;
    let p = li_stephens_p_mismatch(n_haps);
    println!("n_haps: {}, p: {:.10e}", n_haps, p);
}

fn li_stephens_p_mismatch(n_haps: usize) -> f32 {
    if n_haps <= 1 {
        return 0.0001;
    }
    let n = n_haps as f64;
    let theta = 1.0 / (n.ln() + 0.5);
    (theta / (2.0 * (theta + n))) as f32
}
