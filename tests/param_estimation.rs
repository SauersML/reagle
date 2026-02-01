use reagle::model::parameters::ParamEstimates;

#[test]
fn test_param_estimates_recomb_intensity_scaling() {
    let mut estimates = ParamEstimates::new();
    
    // Simulate 1 Morgan of distance (100 cM) with 1 expected switch.
    // This corresponds to a recombination intensity of 1.0 per Morgan.
    // (Poisson rate 1.0 => expected 1 event in unit interval)
    
    let gen_dist_cm = 100.0;
    let expected_switches = 1.0;
    
    estimates.add_switch(gen_dist_cm, expected_switches);
    
    let estimated_intensity = estimates.recomb_intensity().unwrap();
    
    println!("Estimated intensity: {}", estimated_intensity);
    
    // If logic is correct (intensity in M^-1), result should be 1.0.
    // If logic is missing factor of 100 (intensity in cM^-1), result will be 0.01.
    assert!((estimated_intensity - 1.0).abs() < 0.1, "Estimated intensity {} should be close to 1.0 (switches per Morgan)", estimated_intensity);
}
