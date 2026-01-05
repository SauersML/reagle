//! # Li-Stephens Hidden Markov Model
//!
//! Implementation of the Li-Stephens HMM for haplotype phasing and imputation.
//! Uses the forward-backward algorithm with scaling for numerical stability.
//!
//! ## Key Concepts
//! - **States**: Reference haplotypes that the target could copy from
//! - **Transitions**: Probability of switching to a different reference haplotype
//! - **Emissions**: Probability of observing target allele given reference allele
//!
//! ## Reference
//! Li N, Stephens M. Genetics 2003 Dec;165(4):2213-33

use crate::data::{GenotypeMatrix, GenotypeView, HapIdx, MarkerIdx};
use crate::model::parameters::ModelParams;
use crate::utils::Workspace;

/// Result of HMM forward-backward computation
#[derive(Debug)]
pub struct HmmResult {
    /// Posterior state probabilities at each marker (n_markers x n_states)
    pub state_probs: Vec<Vec<f32>>,
    /// Sampled state path (one state per marker)
    pub sampled_path: Vec<usize>,
    /// Log-likelihood of the data
    pub log_likelihood: f64,
}

/// Li-Stephens HMM for a single target haplotype
pub struct LiStephensHmm<'a> {
    /// Reference panel genotypes
    ref_gt: GenotypeView<'a>,
    /// Model parameters
    params: &'a ModelParams,
    /// Selected reference haplotype indices (the HMM states)
    ref_haps: Vec<HapIdx>,
    /// Genetic distances between consecutive markers (cM)
    gen_dists: Vec<f64>,
}

impl<'a> LiStephensHmm<'a> {
    /// Create a new HMM for the given reference panel
    pub fn new(
        ref_gt: impl Into<GenotypeView<'a>>,
        params: &'a ModelParams,
        ref_haps: Vec<HapIdx>,
        gen_dists: Vec<f64>,
    ) -> Self {
        Self {
            ref_gt: ref_gt.into(),
            params,
            ref_haps,
            gen_dists,
        }
    }

    /// Number of HMM states
    pub fn n_states(&self) -> usize {
        self.ref_haps.len()
    }

    /// Number of markers
    pub fn n_markers(&self) -> usize {
        self.ref_gt.n_markers()
    }

    /// Run forward-backward algorithm for a target haplotype
    ///
    /// # Arguments
    /// * `target_alleles` - Alleles of the target haplotype at each marker
    /// * `workspace` - Pre-allocated workspace buffers
    ///
    /// # Returns
    /// HMM result with posterior probabilities and sampled path
    pub fn forward_backward(
        &self,
        target_alleles: &[u8],
        workspace: &mut Workspace,
    ) -> HmmResult {
        let n_markers = self.n_markers();
        let n_states = self.n_states();

        if n_markers == 0 || n_states == 0 {
            return HmmResult {
                state_probs: Vec::new(),
                sampled_path: Vec::new(),
                log_likelihood: 0.0,
            };
        }

        // Ensure workspace is sized correctly
        workspace.resize(n_states, n_markers, 0);

        // Forward pass with scaling
        let scale_factors = self.forward_pass(target_alleles, workspace);

        // Backward pass
        self.backward_pass(target_alleles, &scale_factors, workspace);

        // Compute posteriors
        self.compute_posteriors(workspace);

        // Sample path
        let sampled_path = self.sample_path(workspace);

        // Compute log-likelihood from scale factors
        let log_likelihood: f64 = scale_factors.iter().map(|&s| (s as f64).ln()).sum();

        HmmResult {
            state_probs: workspace.state_probs.clone(),
            sampled_path,
            log_likelihood,
        }
    }

    /// Forward pass with scaling
    fn forward_pass(&self, target_alleles: &[u8], workspace: &mut Workspace) -> Vec<f32> {
        let n_markers = self.n_markers();
        let n_states = self.n_states();
        let mut scale_factors = Vec::with_capacity(n_markers);

        // Initialize at first marker
        let init_prob = 1.0 / n_states as f32;
        for (s, &ref_hap) in self.ref_haps.iter().enumerate() {
            let ref_allele = self.ref_gt.allele(MarkerIdx::new(0), ref_hap);
            let emit = self.emission_prob(target_alleles[0], ref_allele);
            workspace.fwd[s] = init_prob * emit;
        }

        // Scale first marker
        let scale = self.scale_vector(&mut workspace.fwd[..n_states]);
        scale_factors.push(scale);

        // Forward recursion
        for m in 1..n_markers {
            let gen_dist = self.gen_dists[m - 1];
            let p_switch = self.params.switch_prob(gen_dist);
            let p_stay = 1.0 - p_switch;
            let p_switch_to = p_switch / n_states as f32;

            // Sum of previous forward probs
            let fwd_sum: f32 = workspace.fwd[..n_states].iter().sum();

            // Update forward probabilities
            for s in 0..n_states {
                let ref_hap = self.ref_haps[s];
                let ref_allele = self.ref_gt.allele(MarkerIdx::new(m as u32), ref_hap);
                let emit = self.emission_prob(target_alleles[m], ref_allele);

                // Transition: stay in same state or switch from any state
                let trans = p_stay * workspace.fwd[s] + p_switch_to * fwd_sum;
                workspace.tmp[s] = trans * emit;
            }

            // Copy tmp to fwd
            workspace.fwd[..n_states].copy_from_slice(&workspace.tmp[..n_states]);

            // Scale
            let scale = self.scale_vector(&mut workspace.fwd[..n_states]);
            scale_factors.push(scale);
        }

        scale_factors
    }

    /// Backward pass
    fn backward_pass(
        &self,
        target_alleles: &[u8],
        scale_factors: &[f32],
        workspace: &mut Workspace,
    ) {
        let n_markers = self.n_markers();
        let n_states = self.n_states();

        // Initialize at last marker
        workspace.bwd[..n_states].fill(1.0);

        // Backward recursion
        for m in (0..n_markers - 1).rev() {
            let gen_dist = self.gen_dists[m];
            let p_switch = self.params.switch_prob(gen_dist);
            let p_stay = 1.0 - p_switch;
            let p_switch_to = p_switch / n_states as f32;

            // Compute emission * backward for next marker
            for s in 0..n_states {
                let ref_hap = self.ref_haps[s];
                let ref_allele = self.ref_gt.allele(MarkerIdx::new((m + 1) as u32), ref_hap);
                let emit = self.emission_prob(target_alleles[m + 1], ref_allele);
                workspace.tmp[s] = emit * workspace.bwd[s];
            }

            // Sum for switch transitions
            let tmp_sum: f32 = workspace.tmp[..n_states].iter().sum();

            // Update backward probabilities
            for s in 0..n_states {
                workspace.bwd[s] = p_stay * workspace.tmp[s] + p_switch_to * tmp_sum;
            }

            // Scale using same factor as forward pass
            let scale = scale_factors[m + 1];
            if scale > 0.0 {
                for b in &mut workspace.bwd[..n_states] {
                    *b /= scale;
                }
            }
        }
    }

    /// Compute posterior state probabilities
    fn compute_posteriors(&self, workspace: &mut Workspace) {
        let n_markers = self.n_markers();
        let n_states = self.n_states();

        for m in 0..n_markers {
            // Posterior = forward * backward (normalized)
            let mut sum = 0.0f32;
            for s in 0..n_states {
                let prob = workspace.fwd[s] * workspace.bwd[s];
                workspace.state_probs[m][s] = prob;
                sum += prob;
            }

            // Normalize
            if sum > 0.0 {
                for s in 0..n_states {
                    workspace.state_probs[m][s] /= sum;
                }
            }
        }
    }

    /// Sample a state path from posterior
    fn sample_path(&self, workspace: &mut Workspace) -> Vec<usize> {
        let n_markers = self.n_markers();
        let mut path = Vec::with_capacity(n_markers);

        for m in 0..n_markers {
            // Copy probabilities to avoid borrow conflict
            let probs: Vec<f32> = workspace.state_probs[m].clone();
            let state = workspace.sample_index(&probs);
            path.push(state);
        }

        path
    }

    /// Compute emission probability
    #[inline]
    fn emission_prob(&self, target_allele: u8, ref_allele: u8) -> f32 {
        if target_allele == ref_allele {
            self.params.emit_match()
        } else {
            self.params.emit_mismatch()
        }
    }

    /// Scale a probability vector to sum to 1, return the scale factor
    fn scale_vector(&self, probs: &mut [f32]) -> f32 {
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }
        sum
    }

    /// Get the allele for a state at a marker
    pub fn state_allele(&self, marker: MarkerIdx, state: usize) -> u8 {
        self.ref_gt.allele(marker, self.ref_haps[state])
    }

    /// Compute dosage at a marker given state probabilities
    pub fn compute_dosage(&self, marker: MarkerIdx, state_probs: &[f32]) -> f32 {
        let mut dosage = 0.0f32;
        for (s, &prob) in state_probs.iter().enumerate() {
            let allele = self.state_allele(marker, s);
            dosage += prob * allele as f32;
        }
        dosage
    }
}

/// Phasing HMM that considers both haplotypes of a diploid sample
pub struct PhasingHmm<'a> {
    /// Reference genotypes (other samples' haplotypes)
    ref_gt: GenotypeView<'a>,
    /// Model parameters
    params: &'a ModelParams,
}

impl<'a> PhasingHmm<'a> {
    /// Create a new phasing HMM
    pub fn new(ref_gt: impl Into<GenotypeView<'a>>, params: &'a ModelParams) -> Self {
        Self {
            ref_gt: ref_gt.into(),
            params,
        }
    }

    /// Phase a single sample
    ///
    /// # Arguments
    /// * `alleles1` - Current alleles on haplotype 1
    /// * `alleles2` - Current alleles on haplotype 2
    /// * `het_markers` - Indices of heterozygous markers
    /// * `ref_haps` - Selected reference haplotypes for HMM states
    /// * `gen_dists` - Genetic distances between consecutive markers
    /// * `workspace` - Pre-allocated workspace
    ///
    /// # Returns
    /// Updated alleles for both haplotypes
    pub fn phase_sample(
        &self,
        alleles1: &[u8],
        alleles2: &[u8],
        het_markers: &[usize],
        ref_haps: &[HapIdx],
        gen_dists: &[f64],
        workspace: &mut Workspace,
    ) -> (Vec<u8>, Vec<u8>) {
        if het_markers.is_empty() {
            return (alleles1.to_vec(), alleles2.to_vec());
        }

        let _n_markers = alleles1.len();
        let _n_states = ref_haps.len();

        // Create HMM for haplotype 1
        let hmm = LiStephensHmm::new(self.ref_gt, self.params, ref_haps.to_vec(), gen_dists.to_vec());

        // Run forward-backward for haplotype 1
        let result1 = hmm.forward_backward(alleles1, workspace);

        // Run forward-backward for haplotype 2
        let result2 = hmm.forward_backward(alleles2, workspace);

        // Decide phase at each het marker based on which assignment has higher probability
        let mut new_alleles1 = alleles1.to_vec();
        let mut new_alleles2 = alleles2.to_vec();

        for &m in het_markers {
            // Get most likely state for each haplotype
            let state1 = result1.sampled_path.get(m).copied().unwrap_or(0);
            let state2 = result2.sampled_path.get(m).copied().unwrap_or(0);

            // Get reference alleles for these states
            let ref_allele1 = hmm.state_allele(MarkerIdx::new(m as u32), state1);
            let ref_allele2 = hmm.state_allele(MarkerIdx::new(m as u32), state2);

            // Current assignment
            let curr1 = alleles1[m];
            let curr2 = alleles2[m];

            // Check if swapping improves match
            let curr_match = (curr1 == ref_allele1) as i32 + (curr2 == ref_allele2) as i32;
            let swap_match = (curr2 == ref_allele1) as i32 + (curr1 == ref_allele2) as i32;

            if swap_match > curr_match {
                new_alleles1[m] = curr2;
                new_alleles2[m] = curr1;
            }
        }

        (new_alleles1, new_alleles2)
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
            "R3".to_string(),
        ]));
        let mut markers = Markers::new();
        markers.add_chrom("chr1");

        // Create 5 markers
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

            // 6 haplotypes with different patterns
            let alleles = match i {
                0 => vec![0, 0, 1, 1, 0, 1],
                1 => vec![0, 1, 1, 0, 0, 1],
                2 => vec![0, 0, 1, 1, 1, 0],
                3 => vec![1, 0, 1, 0, 1, 0],
                4 => vec![0, 1, 0, 1, 0, 1],
                _ => vec![0; 6],
            };
            columns.push(GenotypeColumn::from_alleles(&alleles, 2));
        }

        GenotypeMatrix::new(markers, columns, samples, true)
    }

    #[test]
    fn test_hmm_forward_backward() {
        let ref_panel = make_test_ref_panel();
        let params = ModelParams::for_phasing(6);
        let ref_haps: Vec<HapIdx> = (0..6).map(|i| HapIdx::new(i)).collect();
        let gen_dists = vec![0.01; 4]; // 0.01 cM between markers

        let hmm = LiStephensHmm::new(&ref_panel, &params, ref_haps, gen_dists);

        let target_alleles = vec![0, 0, 0, 1, 0]; // Should match haplotype 0 or 4
        let mut workspace = Workspace::new(6, 5, 6);
        workspace.set_seed(12345);

        let result = hmm.forward_backward(&target_alleles, &mut workspace);

        assert_eq!(result.state_probs.len(), 5);
        assert_eq!(result.sampled_path.len(), 5);
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_emission_prob() {
        let ref_panel = make_test_ref_panel();
        let params = ModelParams::new();
        let hmm = LiStephensHmm::new(&ref_panel, &params, vec![], vec![]);

        assert!(hmm.emission_prob(0, 0) > 0.99);
        assert!(hmm.emission_prob(0, 1) < 0.01);
    }
}