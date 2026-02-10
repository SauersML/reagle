use reagle::io::vcf::MarkerImputationStats;

#[test]
fn test_dr2_monomorphic_ref() {
    let mut stats = MarkerImputationStats::new(2);
    stats.is_imputed = true;
    // 10 samples, all ref/ref (p=0)
    for _ in 0..10 {
        stats.add_sample_biallelic(0.0, 0.0);
    }
    let dr2 = stats.dr2(1);
    println!("Monomorphic Ref DR2: {}", dr2);
    assert_eq!(dr2, 0.0, "DR2 should be 0.0 for monomorphic ref, got {}", dr2);
}

#[test]
fn test_dr2_monomorphic_alt() {
    let mut stats = MarkerImputationStats::new(2);
    stats.is_imputed = true;
    // 10 samples, all alt/alt (p=1)
    for _ in 0..10 {
        stats.add_sample_biallelic(1.0, 1.0);
    }
    let dr2 = stats.dr2(1);
    println!("Monomorphic Alt DR2: {}", dr2);
    assert_eq!(dr2, 0.0, "DR2 should be 0.0 for monomorphic alt, got {}", dr2);
}
