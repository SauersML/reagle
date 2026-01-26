
#[cfg(test)]
mod tests {
    use reagle::data::storage::GenotypeMatrix;
    use reagle::model::parameters::ModelParams;
    use reagle::model::hmm::BeagleHmm;
    use reagle::model::states::ThreadedHaps;
    use reagle::model::block_hash::types::GlobalId;
    use reagle::data::haplotype::Samples;
    use reagle::data::marker::{Allele, Marker, Markers, Nucleotide};
    use reagle::data::storage::GenotypeColumn;
    use reagle::data::ChromIdx;
    use std::sync::Arc;

    fn make_test_ref_panel(n_markers: usize) -> GenotypeMatrix {
        let samples = Arc::new(Samples::from_ids(vec![
            "R1".to_string(),
            "R2".to_string(),
        ]));
        let mut markers = Markers::new();
        markers.add_chrom("chr1");

        let mut columns = Vec::new();
        for i in 0..n_markers {
            let m = Marker::new(
                ChromIdx::new(0),
                (i * 1000 + 100) as u32,
                None,
                Allele::Base(Nucleotide::A),
                vec![Allele::Base(Nucleotide::C)],
            );
            markers.push(m);
            // Just alternating alleles
            let alleles = vec![0, 1, 0, 1];
            columns.push(GenotypeColumn::from_alleles(&alleles, 2));
        }

        GenotypeMatrix::new_unphased(markers, columns, samples)
    }

    #[test]
    fn test_long_sequence_underflow() {
        let n_markers = 2000; // Long enough to cause underflow if not normalized
        let ref_panel = make_test_ref_panel(n_markers);
        let params = ModelParams::for_phasing(2, 10000.0, None);
        // Small recombination probability
        let p_recomb = vec![0.001; n_markers];

        let n_states = 2;
        let mut threaded_haps = ThreadedHaps::new(n_states, n_states * 2, n_markers);
        threaded_haps.push_new(GlobalId::new(0));
        threaded_haps.push_new(GlobalId::new(1));

        let hmm = BeagleHmm::new(&ref_panel, &params, n_states, p_recomb);

        let target_alleles = vec![0; n_markers];
        let mut fwd = Vec::new();
        let mut bwd = Vec::new();

        hmm.conditioned_forward_backward(
            &target_alleles,
            &target_alleles,
            &target_alleles,
            None,
            None,
            None,
            None,
            &threaded_haps,
            &mut fwd,
            &mut bwd,
        );

        // Check the first marker's backward probabilities
        // If underflow occurred, these will be 0.0
        let first_row_bwd = &bwd[0..n_states];
        let sum: f32 = first_row_bwd.iter().sum();
        
        println!("Sum at first marker: {:e}", sum);
        assert!(sum > 1e-20, "Backward values underflowed! Sum: {:e}", sum);
    }
}
