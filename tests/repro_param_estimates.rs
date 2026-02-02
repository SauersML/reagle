
use reagle::model::parameters::ParamEstimates;

#[test]
fn test_param_estimates_scaling_repro() {
    let mut estimates = ParamEstimates::new();
    // Suppose we have 100 cM distance (1 Morgan)
    // And we observe 1 switch.
    // The rate should be 1 switch / 1 Morgan = 1.0.
    
    estimates.add_switch(100.0, 1.0);
    
    let rate = estimates.recomb_intensity().unwrap();
    // Currently (before fix) this is 1.0 / 100.0 = 0.01.
    // With fix it should be 1.0.
    
    assert!((rate - 0.01).abs() < 1e-6, "Current behavior: no scaling");
}
