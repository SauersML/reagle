fn main() {
    let n_haps_list = vec![100, 1000, 10000, 100000];
    for &n_haps in &n_haps_list {
        let n = n_haps as f64;
        let theta = 1.0 / (n.ln() + 0.5);
        let p = theta / (2.0 * (theta + n));
        println!("n={}, p={:.2e}", n, p);
    }
}
