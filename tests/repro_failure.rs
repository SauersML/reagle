
#[cfg(test)]
mod tests {
    use reagle::pipelines::phasing::*;
    use reagle::model::ibs2::Ibs2;
    use reagle::model::phase_ibs::BidirectionalPhaseIbs;
    use reagle::data::marker::AnyMarkerSpace;
    use reagle::utils::workspace::ThreadWorkspace;

    #[test]
    fn repro_dynamic_mcmc_symmetric() {
        let n_markers = 50;
        let n_target_haps = 2;
        let n_ref_haps = 2;
        let n_total_haps = n_target_haps + n_ref_haps;

        let alleles: Vec<Vec<u8>> = (0..n_markers)
            .map(|_| {
                let mut haps = vec![255u8; n_total_haps];
                haps[2] = 0; // hero
                haps[3] = 1; // anti-hero
                haps
            })
            .collect();
        let subset_to_global: Vec<usize> = (0..n_markers).collect();
        let phase_ibs = BidirectionalPhaseIbs::build_for_subset(
            alleles,
            n_total_haps,
            n_markers,
            &subset_to_global,
        );

        let ibs2 = Ibs2::empty(1);
        let seq1 = vec![0u8; n_markers];
        let seq2 = vec![1u8; n_markers];
        let conf = vec![1.0f32; n_markers];
        let p_recomb = vec![0.02f32; n_markers];
        let het_positions: Vec<usize> = (0..n_markers).collect();

        // Access private function via test-exposed wrapper if necessary?
        // Ah, sample_dynamic_mcmc is private.
        // But the failing test is in src/pipelines/phasing.rs under #[cfg(test)].
        // I cannot call it from outside the crate if it's private.
        
        // So I must run the existing test.
    }
}
