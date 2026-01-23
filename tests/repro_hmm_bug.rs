use reagle::model::hmm::HmmUpdater;

#[test]
fn test_bwd_update_constant_consistency() {
    let n_states = 4;
    let emit_probs = [0.9f32, 0.1];
    let mismatches = vec![0u8, 0, 1, 1];
    let emissions: Vec<f32> = mismatches.iter().map(|&m| emit_probs[m as usize]).collect();
    let p_switch = 0.1f32;

    // 1. Calculate using bwd_update (known correct relative to Li-Stephens)
    let mut bwd_std = vec![1.0f32; n_states];
    HmmUpdater::bwd_update(&mut bwd_std, p_switch, &emit_probs, &mismatches, n_states);
    
    // Normalize bwd_std to compare shape
    let sum_std: f32 = bwd_std.iter().sum();
    for x in &mut bwd_std { *x /= sum_std; }

    // 2. Calculate using bwd_update_constant
    // constant_term C = sum(bwd_next[k] * emit[k])
    // Here bwd_next is all 1.0. So C = sum(emit)
    let constant_term: f32 = emissions.iter().sum();
    
    let mut bwd_const = vec![1.0f32; n_states];
    HmmUpdater::bwd_update_constant(&mut bwd_const, p_switch, &emissions, constant_term, n_states);

    // Normalize bwd_const
    let sum_const: f32 = bwd_const.iter().sum();
    for x in &mut bwd_const { *x /= sum_const; }

    // Compare
    println!("Standard: {:?}", bwd_std);
    println!("Constant: {:?}", bwd_const);

    for i in 0..n_states {
        let diff = (bwd_std[i] - bwd_const[i]).abs();
        assert!(diff < 1e-5, "Mismatch at index {}: std={}, const={}", i, bwd_std[i], bwd_const[i]);
    }
}
