
use reagle::model::parameters::ModelParams;

#[test]
fn test_recomb_intensity_clamping_repro() {
    // High Ne should trigger clamping if it exists
    let params = ModelParams::for_phasing(1000, 1_000_000.0, None);
    // Currently (before fix) this is 40.0. With fix it should be 5.0.
    assert!(params.recomb_intensity > 5.0, "Current behavior: no clamping");
}
