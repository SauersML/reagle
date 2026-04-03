fn main() {
    let mut scores = vec![0.0f32; 100];
    let min_freq = 0.01;
    // MAF is 50%, so weight = 0.
    // If weight == 0, no score is added.
    // Thus all scores are 0.
    // Then `if window_rank_hits[i][h] == 0 || !score.is_finite() || score <= 0.0` sets abyss = true
}
