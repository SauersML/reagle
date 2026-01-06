//! # Li-Stephens Hidden Markov Model
//!
//! Implementation of the Li-Stephens HMM for haplotype phasing and imputation.
//! Uses the forward-backward algorithm with scaling for numerical stability.
//!
//! ## Key Concepts
//! - `States`: Reference haplotypes that the target could copy from
//! - `Transitions`: Probability of switching to a different reference haplotype
//! - `Emissions`: Probability of observing target allele given reference allele
//!
//! ## Reference
//! Li N, Stephens M. Genetics 2003 Dec;165(4):2213-33
//!
//! This implementation follows the Beagle Java code (HmmUpdater.java) closely.

use crate::data::storage::GenotypeView;
use crate::data::{HapIdx, MarkerIdx};
use crate::model::parameters::ModelParams;

/// Result of HMM forward-backward computation
#[derive(Debug, Clone)]
pub struct HmmResult {
    /// Posterior state probabilities at each marker (flattened: n_markers * n_states)
    pub state_probs: Vec<f32>,
    /// Number of states (to reconstruct dimensions)
    pub n_states: usize,
    /// Sampled state path (one state per marker)
    pub sampled_path: Vec<usize>,
    /// Log-likelihood of the data
    pub log_likelihood: f64,
}

/// Static HMM update functions matching Java HmmUpdater
pub struct HmmUpdater;

impl HmmUpdater {
    /// Forward update matching Java HmmUpdater.fwdUpdate exactly.
    ///
    /// Updates forward values and returns the sum of updated forward values.
    ///
    /// # Arguments
    /// * `fwd` - Forward values array that will be updated in place
    /// * `fwd_sum` - Sum of forward values before update
    /// * `p_switch` - Probability of jumping to a random HMM state
    /// * `emit_probs` - Two-element array: [p_match, p_mismatch]
    /// * `mismatches` - Number of mismatches (0 or 1) for each state
    /// * `n_states` - Number of states to process
    ///
    /// # Returns
    /// Sum of updated forward values
    #[inline]
    pub fn fwd_update(
        fwd: &mut [f32],
        fwd_sum: f32,
        p_switch: f32,
        emit_probs: &[f32; 2],
        mismatches: &[u8],
        n_states: usize,
    ) -> f32 {
        let shift = p_switch / n_states as f32;
        let scale = (1.0 - p_switch) / fwd_sum;

        let mut new_sum = 0.0f32;
        for k in 0..n_states {
            let emit = emit_probs[mismatches[k] as usize];
            fwd[k] = emit * (scale * fwd[k] + shift);
            new_sum += fwd[k];
        }
        new_sum
    }

    /// Backward update matching Java HmmUpdater.bwdUpdate exactly.
    ///
    /// Updates backward values in place.
    ///
    /// # Arguments
    /// * `bwd` - Backward values array that will be updated in place
    /// * `p_switch` - Probability of jumping to a random HMM state
    /// * `emit_probs` - Two-element array: [p_match, p_mismatch]
    /// * `mismatches` - Number of mismatches (0 or 1) for each state
    /// * `n_states` - Number of states to process
    #[inline]
    pub fn bwd_update(
        bwd: &mut [f32],
        p_switch: f32,
        emit_probs: &[f32; 2],
        mismatches: &[u8],
        n_states: usize,
    ) {
        // First: multiply by emission and compute sum
        let mut sum = 0.0f32;
        for k in 0..n_states {
            bwd[k] *= emit_probs[mismatches[k] as usize];
            sum += bwd[k];
        }

        // Then: apply transition
        let shift = p_switch / n_states as f32;
        let scale = (1.0 - p_switch) / sum;

        for k in 0..n_states {
            bwd[k] = scale * bwd[k] + shift;
        }
    }
}

/// Li-Stephens HMM for a single target haplotype
pub struct LiStephensHmm<'a> {
    /// Reference panel genotypes (can be GenotypeMatrix or MutableGenotypes via GenotypeView)
    ref_gt: GenotypeView<'a>,
    /// Model parameters
    params: &'a ModelParams,
    /// Selected reference haplotype indices (the HMM states)
    ref_haps: Vec<HapIdx>,
    /// Recombination probabilities between consecutive markers
    p_recomb: Vec<f32>,
}

impl<'a> LiStephensHmm<'a> {
    /// Create a new HMM for the given reference panel
    ///
    /// # Arguments
    /// * `ref_gt` - Reference genotype view (can be Matrix or Mutable)
    /// * `params` - Model parameters
    /// * `ref_haps` - Selected reference haplotypes to use as HMM states
    /// * `p_recomb` - Recombination probabilities for each marker (first element is 0)
    pub fn new(
        ref_gt: impl Into<GenotypeView<'a>>,
        params: &'a ModelParams,
        ref_haps: Vec<HapIdx>,
        p_recomb: Vec<f32>,
    ) -> Self {
        Self {
            ref_gt: ref_gt.into(),
            params,
            ref_haps,
            p_recomb,
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

    /// Get reference allele at marker for state
    #[inline]
    pub fn state_allele(&self, marker: MarkerIdx, state: usize) -> u8 {
        self.ref_gt.allele(marker, self.ref_haps[state])
    }

    /// Get reference alleles at marker for all states
    pub fn state_alleles(&self, marker: MarkerIdx) -> Vec<u8> {
        self.ref_haps
            .iter()
            .map(|&h| self.ref_gt.allele(marker, h))
            .collect()
    }

    /// Run forward-backward algorithm for a target haplotype
    ///
    /// This implements the Baum-Welch style forward-backward following
    /// the Java ImpLSBaum implementation.
    ///
    /// # Arguments
    /// * `target_alleles` - Alleles of the target haplotype at each marker
    /// * `fwd` - Pre-allocated forward buffer [n_markers * n_states]
    /// * `bwd` - Pre-allocated backward buffer [n_states]
    ///
    /// # Returns
    /// HMM result with posterior probabilities
    pub fn forward_backward(
        &self,
        target_alleles: &[u8],
        fwd: &mut Vec<f32>,
        bwd: &mut Vec<f32>,
    ) -> HmmResult {
        let n_markers = self.n_markers();
        let n_states = self.n_states();

        if n_markers == 0 || n_states == 0 {
            return HmmResult {
                state_probs: Vec::new(),
                n_states,
                sampled_path: Vec::new(),
                log_likelihood: 0.0,
            };
        }

        // Ensure buffers are sized correctly
        let total_size = n_markers * n_states;
        fwd.resize(total_size, 0.0);
        bwd.resize(n_states, 0.0);

        let p_err = self.params.p_mismatch;
        let p_no_err = 1.0 - p_err;
        let emit_probs = [p_no_err, p_err];

        // Forward pass
        let fwd_sum = self.forward_pass(target_alleles, fwd, &emit_probs);

        // Backward pass and compute posteriors
        // Initialize backward values
        let init_bwd = 1.0 / n_states as f32;
        bwd.fill(init_bwd);
        let mut bwd_sum = 1.0f32;

        // Process from last marker to first
        for m in (0..n_markers).rev() {
            let marker_idx = MarkerIdx::new(m as u32);
            let targ_al = target_alleles[m];
            let row_offset = m * n_states;

            // Compute mismatches for this marker
            let ref_alleles = self.state_alleles(marker_idx);
            let mismatches: Vec<u8> = ref_alleles
                .iter()
                .map(|&r| if r == targ_al { 0 } else { 1 })
                .collect();

            // Finish backward transition, combine with forward to get posteriors
            if m < n_markers - 1 {
                let p_recomb = self.p_recomb.get(m + 1).copied().unwrap_or(0.0);
                let shift = p_recomb / n_states as f32;
                let scale = (1.0 - p_recomb) / bwd_sum;

                for k in 0..n_states {
                    bwd[k] = scale * bwd[k] + shift;
                }
            }

            // Compute state probabilities: fwd * bwd
            let mut state_sum = 0.0f32;
            for k in 0..n_states {
                let idx = row_offset + k;
                fwd[idx] *= bwd[k];
                state_sum += fwd[idx];
            }

            // Normalize
            if state_sum > 0.0 {
                let inv_sum = 1.0 / state_sum;
                for k in 0..n_states {
                    fwd[row_offset + k] *= inv_sum;
                }
            }

            // Apply emission for backward (for next iteration)
            if m > 0 {
                bwd_sum = 0.0;
                for k in 0..n_states {
                    bwd[k] *= emit_probs[mismatches[k] as usize];
                    bwd_sum += bwd[k];
                }
            }
        }

        // Sample path from posteriors
        let sampled_path = self.sample_path(fwd, n_markers, n_states);

        // Compute log-likelihood (approximate from forward sums)
        let log_likelihood = fwd_sum.ln() as f64;

        HmmResult {
            state_probs: fwd.clone(),
            n_states,
            sampled_path,
            log_likelihood,
        }
    }

    /// Forward pass following Java ImpLSBaum.setFwdValues
    /// Forward pass following Java ImpLSBaum.setFwdValues
    fn forward_pass(
        &self,
        target_alleles: &[u8],
        fwd: &mut Vec<f32>,
        emit_probs: &[f32; 2],
    ) -> f32 {
        let n_markers = self.n_markers();
        let n_states = self.n_states();

        let mut fwd_sum = 1.0f32;

        for m in 0..n_markers {
            let marker_idx = MarkerIdx::new(m as u32);
            let targ_al = target_alleles[m];
            let p_recomb = self.p_recomb.get(m).copied().unwrap_or(0.0);

            let shift = p_recomb / n_states as f32;
            let scale = (1.0 - p_recomb) / fwd_sum;

            let mut new_sum = 0.0f32;
            let row_offset = m * n_states;
            let prev_row_offset = if m > 0 { (m - 1) * n_states } else { 0 };

            for k in 0..n_states {
                let ref_al = self.state_allele(marker_idx, k);
                let emit = if ref_al == targ_al {
                    emit_probs[0]
                } else {
                    emit_probs[1]
                };

                let val = if m == 0 {
                    emit / n_states as f32
                } else {
                    emit * (scale * fwd[prev_row_offset + k] + shift)
                };
                
                fwd[row_offset + k] = val;
                new_sum += val;
            }
            fwd_sum = new_sum;
        }

        fwd_sum
    }

    /// Sample a state path from posterior probabilities
    fn sample_path(
        &self,
        state_probs: &[f32],
        n_markers: usize,
        n_states: usize,
    ) -> Vec<usize> {
        let mut path = Vec::with_capacity(n_markers);

        for m in 0..n_markers {
            let row_offset = m * n_states;
            // Find max probability state (Viterbi-style for deterministic output)
            let mut max_state = 0;
            let mut max_prob = state_probs[row_offset];
            for k in 1..n_states {
                let prob = state_probs[row_offset + k];
                if prob > max_prob {
                    max_prob = prob;
                    max_state = k;
                }
            }
            path.push(max_state);
        }

        path
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ChromIdx;
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Allele, Marker, Markers};
    use crate::data::storage::{GenotypeColumn, GenotypeMatrix};
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
    fn test_hmm_updater_fwd() {
        let mut fwd = vec![0.25f32; 4];
        let emit_probs = [0.99f32, 0.01];
        let mismatches = vec![0u8, 0, 1, 0];

        let sum = HmmUpdater::fwd_update(&mut fwd, 1.0, 0.01, &emit_probs, &mismatches, 4);

        assert!(sum > 0.0);
        assert!(sum < 2.0);
    }

    #[test]
    fn test_hmm_updater_bwd() {
        let mut bwd = vec![0.25f32; 4];
        let emit_probs = [0.99f32, 0.01];
        let mismatches = vec![0u8, 0, 1, 0];

        HmmUpdater::bwd_update(&mut bwd, 0.01, &emit_probs, &mismatches, 4);

        let sum: f32 = bwd.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_hmm_forward_backward() {
        let ref_panel = make_test_ref_panel();
        let params = ModelParams::for_phasing(6);
        let ref_haps: Vec<HapIdx> = (0..6).map(|i| HapIdx::new(i)).collect();
        let p_recomb = vec![0.0, 0.01, 0.01, 0.01, 0.01];

        let hmm = LiStephensHmm::new(&ref_panel, &params, ref_haps, p_recomb);

        let target_alleles = vec![0, 0, 0, 1, 0]; // Should match haplotype 0 or 4
        let mut fwd = Vec::new();
        let mut bwd = Vec::new();

        let result = hmm.forward_backward(&target_alleles, &mut fwd, &mut bwd);

        assert_eq!(result.state_probs.len(), 5 * 6); // 5 markers * 6 states
        assert_eq!(result.sampled_path.len(), 5);
        assert!(result.log_likelihood.is_finite());

        // Check probabilities sum to ~1 at each marker
        for m in 0..5 {
            let start = m * 6;
            let end = start + 6;
            let sum: f32 = result.state_probs[start..end].iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "Sum at marker {} was {}", m, sum);
        }
    }

    #[test]
    fn test_dosage_computation() {
        let ref_panel = make_test_ref_panel();
        let params = ModelParams::for_phasing(6);
        let ref_haps: Vec<HapIdx> = (0..6).map(|i| HapIdx::new(i)).collect();
        let p_recomb = vec![0.0, 0.01, 0.01, 0.01, 0.01];

        let hmm = LiStephensHmm::new(&ref_panel, &params, ref_haps, p_recomb);

        // Uniform state probs
        let state_probs = vec![1.0 / 6.0; 6];
        let dosage = hmm.compute_dosage(MarkerIdx::new(0), &state_probs);

        // Marker 0 alleles: [0, 0, 1, 1, 0, 1] -> mean = 0.5
        assert!((dosage - 0.5).abs() < 0.01);
    }
}
