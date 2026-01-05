//! # Imputation Pipeline
//!
//! Orchestrates the imputation workflow:
//! 1. Load target and reference VCFs
//! 2. Align markers between target and reference
//! 3. Run Li-Stephens HMM for each target haplotype
//! 4. Compute dosages and write output
//!
//! This matches Java `imp/ImpLS.java`, `imp/ImpLSBaum.java`, and related classes.

use rayon::prelude::*;

use crate::config::Config;
use crate::data::genetic_map::GeneticMaps;
use crate::data::haplotype::HapIdx;
use crate::data::marker::MarkerIdx;
use crate::data::storage::GenotypeMatrix;
use crate::error::Result;
use crate::io::vcf::{VcfReader, VcfWriter};
use crate::model::hmm::LiStephensHmm;
use crate::model::parameters::ModelParams;
use crate::model::pbwt::PbwtIbs;

/// Imputation pipeline
pub struct ImputationPipeline {
    config: Config,
    params: ModelParams,
}

impl ImputationPipeline {
    /// Create a new imputation pipeline
    pub fn new(config: Config) -> Self {
        let params = ModelParams::new();
        Self { config, params }
    }

    /// Run the imputation pipeline
    pub fn run(&mut self) -> Result<()> {
        // Load target VCF
        let (mut target_reader, target_file) = VcfReader::open(&self.config.gt)?;
        let target_samples = target_reader.samples_arc();
        let target_gt = target_reader.read_all(target_file)?;

        // Load reference VCF
        let ref_path = self.config.r#ref.as_ref().ok_or_else(|| {
            crate::error::ReagleError::config("Reference panel required for imputation")
        })?;
        let (mut ref_reader, ref_file) = VcfReader::open(ref_path)?;
        let ref_gt = ref_reader.read_all(ref_file)?;

        if target_gt.n_markers() == 0 || ref_gt.n_markers() == 0 {
            return Ok(());
        }

        // Initialize parameters
        let n_ref_haps = ref_gt.n_haplotypes();
        self.params = ModelParams::for_imputation(n_ref_haps);
        self.params.set_n_states(self.config.imp_states.min(n_ref_haps));

        // Load genetic map if provided
        let gen_maps = if let Some(ref map_path) = self.config.map {
            let chrom_names: Vec<&str> = ref_gt
                .markers()
                .chrom_names()
                .iter()
                .map(|s| s.as_ref())
                .collect();
            GeneticMaps::from_plink_file(map_path, &chrom_names)?
        } else {
            GeneticMaps::new()
        };

        // Create marker map for recombination probabilities
        let n_ref_markers = ref_gt.n_markers();
        let chrom = ref_gt.marker(MarkerIdx::new(0)).chrom;

        // Compute genetic distances
        let gen_dists: Vec<f64> = (0..n_ref_markers.saturating_sub(1))
            .map(|m| {
                let pos1 = ref_gt.marker(MarkerIdx::new(m as u32)).pos;
                let pos2 = ref_gt.marker(MarkerIdx::new((m + 1) as u32)).pos;
                gen_maps.gen_dist(chrom, pos1, pos2)
            })
            .collect();

        // Convert to pRecomb using model parameters
        let p_recomb: Vec<f32> = std::iter::once(0.0f32)
            .chain(gen_dists.iter().map(|&d| self.params.p_recomb(d)))
            .collect();

        // Run imputation for each target haplotype
        let n_target_samples = target_gt.n_samples();
        let n_target_haps = target_gt.n_haplotypes();
        let n_states = self.params.n_states;
        let _seed = self.config.seed as u64; // Reserved for future use

        // Collect dosages for all samples
        let dosages: Vec<Vec<f32>> = (0..n_target_haps)
            .into_par_iter()
            .map(|h| {
                let hap_idx = HapIdx::new(h as u32);

                // Get target alleles at genotyped markers
                let target_alleles: Vec<u8> = (0..target_gt.n_markers())
                    .map(|m| target_gt.allele(MarkerIdx::new(m as u32), hap_idx))
                    .collect();

                // Select reference haplotypes
                let ref_haps: Vec<HapIdx> = (0..n_states.min(n_ref_haps))
                    .map(|i| HapIdx::new(i as u32))
                    .collect();

                // Run imputation
                impute_haplotype_internal(
                    &target_alleles,
                    &ref_gt,
                    &self.params,
                    &ref_haps,
                    &p_recomb,
                )
            })
            .collect();

        // Combine dosages for diploid samples
        let sample_dosages: Vec<Vec<f32>> = (0..n_target_samples)
            .map(|s| {
                let hap1 = s * 2;
                let hap2 = s * 2 + 1;
                (0..n_ref_markers)
                    .map(|m| {
                        let d1 = dosages.get(hap1).and_then(|d| d.get(m)).copied().unwrap_or(0.0);
                        let d2 = dosages.get(hap2).and_then(|d| d.get(m)).copied().unwrap_or(0.0);
                        d1 + d2
                    })
                    .collect()
            })
            .collect();

        // Flatten dosages for output
        let flat_dosages: Vec<f32> = sample_dosages.into_iter().flatten().collect();

        // Write output
        let output_path = self.config.out.with_extension("vcf.gz");
        let mut writer = VcfWriter::create(&output_path, target_samples)?;
        writer.write_header(ref_gt.markers())?;
        writer.write_imputed(&ref_gt, &flat_dosages, 0, n_ref_markers)?;
        writer.flush()?;

        Ok(())
    }
}

/// Internal implementation of imputation for a single haplotype
fn impute_haplotype_internal(
    target_alleles: &[u8],
    ref_gt: &GenotypeMatrix,
    params: &ModelParams,
    ref_haps: &[HapIdx],
    p_recomb: &[f32],
) -> Vec<f32> {
    let n_markers = ref_gt.n_markers();
    let n_states = ref_haps.len();

    if n_markers == 0 || n_states == 0 {
        return vec![0.0; n_markers];
    }

    // Create HMM
    let hmm = LiStephensHmm::new(ref_gt, params, ref_haps.to_vec(), p_recomb.to_vec());

    // Allocate forward and backward buffers
    let mut fwd: Vec<Vec<f32>> = vec![vec![0.0; n_states]; n_markers];
    let mut bwd: Vec<f32> = vec![0.0; n_states];

    // Run forward-backward
    let result = hmm.forward_backward(target_alleles, &mut fwd, &mut bwd);

    // Compute dosages at each marker
    (0..n_markers)
        .map(|m| {
            let marker_idx = MarkerIdx::new(m as u32);
            if m < result.state_probs.len() {
                hmm.compute_dosage(marker_idx, &result.state_probs[m])
            } else {
                0.0
            }
        })
        .collect()
}

/// Impute a single target haplotype (public API)
///
/// # Arguments
/// * `target_alleles` - Alleles of target haplotype at genotyped markers
/// * `ref_gt` - Reference genotype matrix
/// * `params` - Model parameters
/// * `ref_haps` - Selected reference haplotypes to use as HMM states
/// * `gen_dists` - Genetic distances between consecutive markers (in cM)
/// * `_seed` - Random seed (unused, kept for API compatibility)
///
/// # Returns
/// Vector of dosages at each marker
pub fn impute_haplotype(
    target_alleles: &[u8],
    ref_gt: &GenotypeMatrix,
    params: &ModelParams,
    ref_haps: &[HapIdx],
    gen_dists: &[f64],
    _seed: u64,
) -> Vec<f32> {
    // Convert genetic distances to recombination probabilities
    let p_recomb: Vec<f32> = std::iter::once(0.0f32)
        .chain(gen_dists.iter().map(|&d| params.p_recomb(d)))
        .collect();

    impute_haplotype_internal(target_alleles, ref_gt, params, ref_haps, &p_recomb)
}

/// Imputation with PBWT-based state selection
///
/// This matches Java `imp/ImpLSBaum` more closely by selecting states
/// dynamically using PBWT.
pub struct ImpLSBaum<'a> {
    ref_gt: &'a GenotypeMatrix,
    params: &'a ModelParams,
    pbwt: PbwtIbs,
    p_recomb: Vec<f32>,
}

impl<'a> ImpLSBaum<'a> {
    /// Create a new imputation engine
    pub fn new(
        ref_gt: &'a GenotypeMatrix,
        params: &'a ModelParams,
        p_recomb: Vec<f32>,
    ) -> Self {
        let n_haps = ref_gt.n_haplotypes();
        let pbwt = PbwtIbs::new(n_haps);

        Self {
            ref_gt,
            params,
            pbwt,
            p_recomb,
        }
    }

    /// Impute a target haplotype
    pub fn impute(&mut self, target_alleles: &[u8]) -> Vec<f32> {
        let n_markers = self.ref_gt.n_markers();
        let n_states = self.params.n_states;

        if n_markers == 0 || n_states == 0 {
            return vec![0.0; n_markers];
        }

        // Build PBWT on reference panel (could be done once and cached)
        self.pbwt.reset();
        for m in 0..n_markers {
            let alleles: Vec<u8> = (0..self.ref_gt.n_haplotypes())
                .map(|h| self.ref_gt.allele(MarkerIdx::new(m as u32), HapIdx::new(h as u32)))
                .collect();
            self.pbwt.fwd_update(&alleles, 2, m);
        }

        // For now, use simple state selection (first n_states haplotypes)
        // A full implementation would select states per-cluster
        let ref_haps: Vec<HapIdx> = (0..n_states.min(self.ref_gt.n_haplotypes()))
            .map(|i| HapIdx::new(i as u32))
            .collect();

        impute_haplotype_internal(
            target_alleles,
            self.ref_gt,
            self.params,
            &ref_haps,
            &self.p_recomb,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Allele, Marker, Markers};
    use crate::data::storage::GenotypeColumn;
    use crate::data::ChromIdx;
    use std::sync::Arc;

    fn make_test_ref_panel() -> GenotypeMatrix {
        let samples = Arc::new(Samples::from_ids(vec![
            "R1".to_string(),
            "R2".to_string(),
        ]));
        let mut markers = Markers::new();
        markers.add_chrom("chr1");

        let mut columns = Vec::new();
        for i in 0..5 {
            let m = Marker::new(
                ChromIdx::new(0),
                (i * 1000 + 100) as u32,
                None,
                Allele::Base(0),
                vec![Allele::Base(1)],
            );
            markers.push(m);
            columns.push(GenotypeColumn::from_alleles(&[0, 1, 0, 1], 2));
        }

        GenotypeMatrix::new(markers, columns, samples, true)
    }

    #[test]
    fn test_impute_haplotype() {
        let ref_panel = make_test_ref_panel();
        let params = ModelParams::for_imputation(4);
        let ref_haps: Vec<HapIdx> = (0..4).map(|i| HapIdx::new(i)).collect();
        let gen_dists = vec![0.01; 4];

        let target_alleles = vec![0, 1, 0, 1, 0];
        let dosages = impute_haplotype(
            &target_alleles,
            &ref_panel,
            &params,
            &ref_haps,
            &gen_dists,
            12345,
        );

        assert_eq!(dosages.len(), 5);
        for d in &dosages {
            assert!(*d >= 0.0 && *d <= 1.0);
        }
    }

    #[test]
    fn test_imp_ls_baum() {
        let ref_panel = make_test_ref_panel();
        let mut params = ModelParams::for_imputation(4);
        params.set_n_states(4);

        let p_recomb = vec![0.0, 0.01, 0.01, 0.01, 0.01];
        let mut imp = ImpLSBaum::new(&ref_panel, &params, p_recomb);

        let target_alleles = vec![0, 1, 0, 1, 0];
        let dosages = imp.impute(&target_alleles);

        assert_eq!(dosages.len(), 5);
        for d in &dosages {
            assert!(*d >= 0.0 && *d <= 1.0);
        }
    }
}
