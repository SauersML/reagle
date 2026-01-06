use reagle::config::Config;
use reagle::data::storage::{GenotypeMatrix, PhaseState};
use reagle::pipelines::imputation::ImputationPipeline;
use reagle::data::marker::{Marker, MarkerIdx, Allele};
use reagle::data::haplotype::Samples;
use reagle::data::ChromIdx;
use std::sync::Arc;
use reagle::data::storage::GenotypeColumn;
use reagle::data::storage::phase_state::{Phased, Unphased};

/// Create a GenotypeMatrix with specific pattern
fn create_synthetic_panel(
    n_markers: usize,
    n_samples: usize,
    pattern: &dyn Fn(usize, usize) -> u8
) -> GenotypeMatrix<Phased> {
    let mut markers = reagle::data::marker::Markers::new();
    let chrom_idx = markers.add_chrom("chr1");
    
    for m in 0..n_markers {
        markers.push(Marker::new(
            chrom_idx, 
            m as u32 * 1000 + 1, 
            Some(format!("rs{}", m).into()), 
            Allele::Base(0), // A
            vec![Allele::Base(3)] // T
        )); 
    }

    let n_haps = n_samples * 2;
    // Create columns
    let columns: Vec<GenotypeColumn> = (0..n_markers)
        .map(|m| {
            let col_alleles: Vec<u8> = (0..n_haps).map(|h| pattern(m, h)).collect();
            GenotypeColumn::from_alleles(&col_alleles, 2)
        })
        .collect();
        
    // Create samples
    let sample_ids: Vec<String> = (0..n_samples).map(|i| format!("Ref{}", i)).collect();
    let samples = Arc::new(Samples::from_ids(sample_ids));

    GenotypeMatrix::new_phased(markers, columns, samples)
}

fn create_unphased_target(
    n_markers: usize,
    pattern: &dyn Fn(usize) -> u8,
    mask_predicate: &dyn Fn(usize) -> bool
) -> GenotypeMatrix<Unphased> {
    let mut markers = reagle::data::marker::Markers::new();
    let chrom_idx = markers.add_chrom("chr1");

    for m in 0..n_markers {
        markers.push(Marker::new(
            chrom_idx, 
            m as u32 * 1000 + 1, 
            Some(format!("rs{}", m).into()), 
            Allele::Base(0), 
            vec![Allele::Base(3)]
        ));
    }

    // 1 sample = 2 haplotypes
    let n_samples = 1;
    let n_haps = 2;
    
    let columns: Vec<GenotypeColumn> = (0..n_markers)
        .map(|m| {
            let allele = if mask_predicate(m) { 255 } else { pattern(m) };
            let col_alleles = vec![allele, allele]; 
            GenotypeColumn::from_alleles(&col_alleles, 2)
        })
        .collect();
        
    let sample_ids = vec!["Target1".to_string()];
    let samples = Arc::new(Samples::from_ids(sample_ids));
        
    GenotypeMatrix::new_unphased(markers, columns, samples)
}

#[test]
fn test_synthetic_slam_dunk() {
    let n_markers = 50;
    
    // 1. Reference: 50 haps of 0s, 50 haps of 1s
    // 50 samples = 100 haplotypes.
    // Pattern: first 50 haplotypes are 0, next 50 are 1.
    let ref_gt = create_synthetic_panel(n_markers, 50, &|_, h| {
        if h < 50 { 0 } else { 1 }
    });
    
    // 2. Target: 1 sample. Matches "All 0s". Mask odd markers.
    let target_gt = create_unphased_target(n_markers, &|_| 0, &|m| m % 2 != 0);
    
    let mut config = Config::default();
    config.imp_states = 50;
    config.imp_nsteps = 10;
    config.ne = 10000.0;
    config.err = 0.0001;
    // VERY IMPORTANT: run_in_memory needs threads for rayon
    // In unit tests, global thread pool is used.
    
    let mut pipeline = ImputationPipeline::new(config);
    let result = pipeline.run_in_memory(&target_gt, &ref_gt).expect("Imputation success");
    
    // 3. Verification
    // Check missing markers (odd indices)
    let n_target_samples = 1;
    for m in (1..n_markers).step_by(2) {
        // Dosages are flat [marker][sample] -> index = m * n_samples + sample_idx
        let idx = m * n_target_samples + 0; 
        let ds = result.dosages[idx];
        assert!(ds < 0.05, "Marker {} should be imputed to 0 (Ref), got DS={}", m, ds);
        
        let stats = result.quality.get(m).unwrap();
        assert!(stats.dr2(1) > 0.9, "Marker {} should have high R2, got {}", m, stats.dr2(1));
    }
}

#[test]
fn test_synthetic_recombination() {
    let n_markers = 50;
    
    // 1. Reference: same 50/50 split
    let ref_gt = create_synthetic_panel(n_markers, 50, &|_, h| {
        if h < 50 { 0 } else { 1 }
    });
    
    // 2. Target: Recombinant. 0..20 are '0', 20..50 are '1'.
    let target_gt = create_unphased_target(n_markers, 
        &|m| if m < 20 { 0 } else { 1 }, 
        &|m| m == 15 || m == 25
    );
    
    let mut config = Config::default();
    config.imp_states = 50;
    config.ne = 10000.0;
    
    let mut pipeline = ImputationPipeline::new(config);
    let result = pipeline.run_in_memory(&target_gt, &ref_gt).expect("Imputation success");
    
    let n_target_samples = 1;
    
    let idx_15 = 15 * n_target_samples + 0;
    let ds_15 = result.dosages[idx_15];
    
    let idx_25 = 25 * n_target_samples + 0;
    let ds_25 = result.dosages[idx_25];
    
    assert!(ds_15 < 0.05, "Marker 15 (Pre-switch) should be 0, got {}", ds_15);
    assert!(ds_25 > 1.95, "Marker 25 (Post-switch) should be 2 (HomAlt), got {}", ds_25);
}
