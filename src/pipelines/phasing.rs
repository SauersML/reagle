    #[test]
    fn test_run_phase_auto_states() {
        use crate::data::ChromIdx;
        use crate::data::genetic_map::GeneticMaps;
        use crate::data::haplotype::Samples;
        use crate::data::marker::{Allele, Marker, Markers};
        use crate::data::storage::GenotypeColumn;
        use crate::data::storage::matrix::GenotypeMatrix;
        use std::sync::Arc;
        use crate::data::marker::Nucleotide;

        let n_markers = 10;
        let n_samples = 5;

        // Mock Markers
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");
        for i in 0..n_markers {
            let m = Marker::new(
                ChromIdx::new(0),
                i as u32 * 1000,
                Some(format!("m{}", i).into()),
                Allele::Base(Nucleotide::A),
                vec![Allele::Base(Nucleotide::T)],
            );
            markers.push(m);
        }

        // Mock Samples
        let samples = Arc::new(Samples::from_ids(
            (0..n_samples).map(|i| format!("s{}", i)).collect(),
        ));

        // Mock Genotypes
        let columns: Vec<GenotypeColumn> = (0..n_markers)
            .map(|_| {
                let bytes: Vec<u8> = (0..n_samples * 2).map(|i| (i % 2) as u8).collect();
                GenotypeColumn::from_alleles(&bytes, 2)
            })
            .collect();

        let gt = GenotypeMatrix::new_unphased(markers, columns, samples);
        let gen_maps = GeneticMaps::new();

        let config = Config {
            target: PathBuf::from("test.vcf"),
            r#ref: None,
            out: PathBuf::from("out"),
            map: None,
            chrom: None,
            excludesamples: None,
            excludemarkers: None,
            burnin: 1,
            iterations: 1,
            mcmc_burnin: 1,
            dynamic_mcmc: false,
            mcmc_steps: 1,
            mcmc_lr_samples: 1,
            phase_states: 0, // Auto
            rare: 0.002,
            impute: true,
            imp_states: 10,
            imp_segment: 6.0,
            imp_step: 0.1,
            imp_nsteps: 7,
            cluster: 0.005,
            pbwt_batch_mb: 256,
            ap: false,
            gp: false,
            ne: 10000.0,
            err: None,
            em: false,
            window: 40.0,
            window_markers: 100000,
            overlap: 2.0,
            seed: 12345,
            nthreads: Some(1),
            profile: false,
        };

        let mut pipeline = PhasingPipeline::<crate::data::AnyMarkerSpace>::new(config, None);
        let _ = pipeline.phase_in_memory_with_overlap(&gt, &gen_maps, None, None);

        // n_total_haps = 10 (target) + 0 (ref) = 10.
        // n_states = min(DEFAULT_PHASE_STATES, 10 - 2) = 8.
        assert_eq!(pipeline.params.n_states, 8);
    }
}
