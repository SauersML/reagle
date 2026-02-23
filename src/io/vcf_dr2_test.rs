#[cfg(test)]
mod tests {
    use crate::io::vcf::MarkerImputationStats;

    #[test]
    fn test_dr2_monomorphic_matches_beagle() {
        let mut stats = MarkerImputationStats::new(2);
        stats.is_imputed = true;
        // All ref (0.0)
        for _ in 0..10 {
            stats.add_sample_biallelic(0.0, 0.0);
        }
        
        let dr2 = stats.dr2(1);
        println!("Monomorphic Ref DR2: {}", dr2);
        // Beagle returns 0.0. Current Reagle returns 1.0.
        assert!(dr2 < 0.001, "DR2 should be 0.0 for monomorphic sites to match Beagle, got {}", dr2); 
    }
}
