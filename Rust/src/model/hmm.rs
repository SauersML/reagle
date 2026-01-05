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

use crate::data::storage::GenotypeMatrix;
use crate::data::{HapIdx, MarkerIdx};
use crate::model::parameters::ModelParams;

/// Result of HMM forward-backward computation
#[derive(Debug, Clone)]
pub struct HmmResult {
    /// Posterior state probabilities at each marker (n_markers x n_states)
    pub state_probs: Vec<Vec<f32>>,
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

    /// Forward update with allele comparison (convenience wrapper)
    /// 
    /// Missing data (allele 255) is treated as uninformative (no emission penalty).
    #[inline]
    pub fn fwd_update_alleles(
        fwd: &mut [f32],
        fwd_sum: f32,
        p_switch: f32,
        p_mismatch: f32,
        target_allele: u8,
        ref_alleles: &[u8],
        n_states: usize,
    ) -> f32 {
        let emit_probs = [1.0 - p_mismatch, p_mismatch];
        let shift = p_switch / n_states as f32;
        let scale = (1.0 - p_switch) / fwd_sum;

        let mut new_sum = 0.0f32;
        for k in 0..n_states {
            // Missing data (255) - no penalty, use match emission
            let mismatch = if target_allele == 255 || ref_alleles[k] == 255 {
                0
            } else if ref_alleles[k] == target_allele {
                0
            } else {
                1
            };
            let emit = emit_probs[mismatch];
            fwd[k] = emit * (scale * fwd[k] + shift);
            new_sum += fwd[k];
        }
        new_sum
    }

    /// Backward update with allele comparison (convenience wrapper)
    /// 
    /// Missing data (allele 255) is treated as uninformative (no emission penalty).
    #[inline]
    pub fn bwd_update_alleles(
        bwd: &mut [f32],
        p_switch: f32,
        p_mismatch: f32,
        target_allele: u8,
        ref_alleles: &[u8],
        n_states: usize,
    ) {
        let emit_probs = [1.0 - p_mismatch, p_mismatch];

        // First: multiply by emission and compute sum
        let mut sum = 0.0f32;
        for k in 0..n_states {
            // Missing data (255) - no penalty, use match emission
            let mismatch = if target_allele == 255 || ref_alleles[k] == 255 {
                0
            } else if ref_alleles[k] == target_allele {
                0
            } else {
                1
            };
            bwd[k] *= emit_probs[mismatch];
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
    /// Reference panel genotypes
    ref_gt: &'a GenotypeMatrix,
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
    /// * `ref_gt` - Reference genotype matrix
    /// * `params` - Model parameters
    /// * `ref_haps` - Selected reference haplotypes to use as HMM states
    /// * `p_recomb` - Recombination probabilities for each marker (first element is 0)
    pub fn new(
        ref_gt: &'a GenotypeMatrix,
        params: &'a ModelParams,
        ref_haps: Vec<HapIdx>,
        p_recomb: Vec<f32>,
    ) -> Self {
        Self {
            ref_gt,
            params,
            ref_haps,
            p_recomb,
        }
    }

    /// Create HMM from genetic distances (converts to pRecomb internally)
    pub fn from_gen_dists(
        ref_gt: &'a GenotypeMatrix,
        params: &'a ModelParams,
        ref_haps: Vec<HapIdx>,
        gen_dists: &[f64],
    ) -> Self {
        let p_recomb = Self::gen_dists_to_p_recomb(gen_dists, params.recomb_intensity);
        Self::new(ref_gt, params, ref_haps, p_recomb)
    }

    /// Convert genetic distances to recombination probabilities
    ///
    /// Formula: pRecomb = 1 - exp(-recombIntensity * genDist)
    /// This matches Java MarkerMap.pRecomb
    fn gen_dists_to_p_recomb(gen_dists: &[f64], recomb_intensity: f32) -> Vec<f32> {
        let mut p_recomb = Vec::with_capacity(gen_dists.len() + 1);
        p_recomb.push(0.0); // First marker has no preceding marker

        let c = -(recomb_intensity as f64);
        for &dist in gen_dists {
            // -expm1(x) = 1 - exp(x) but more numerically stable
            let p = -f64::exp_m1(c * dist);
            p_recomb.push(p as f32);
        }
        p_recomb
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
    /// * `fwd` - Pre-allocated forward buffer [n_markers][n_states]
    /// * `bwd` - Pre-allocated backward buffer [n_states]
    ///
    /// # Returns
    /// HMM result with posterior probabilities
    pub fn forward_backward(
        &self,
        target_alleles: &[u8],
        fwd: &mut Vec<Vec<f32>>,
        bwd: &mut Vec<f32>,
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

        // Ensure buffers are sized correctly
        fwd.resize(n_markers, vec![0.0; n_states]);
        for row in fwd.iter_mut() {
            row.resize(n_states, 0.0);
        }
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
                fwd[m][k] *= bwd[k];
                state_sum += fwd[m][k];
            }

            // Normalize
            if state_sum > 0.0 {
                for k in 0..n_states {
                    fwd[m][k] /= state_sum;
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
            sampled_path,
            log_likelihood,
        }
    }

    /// Forward pass following Java ImpLSBaum.setFwdValues
    fn forward_pass(
        &self,
        target_alleles: &[u8],
        fwd: &mut Vec<Vec<f32>>,
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
            for k in 0..n_states {
                let ref_al = self.state_allele(marker_idx, k);
                let emit = if ref_al == targ_al {
                    emit_probs[0]
                } else {
                    emit_probs[1]
                };

                fwd[m][k] = if m == 0 {
                    emit / n_states as f32
                } else {
                    emit * (scale * fwd[m - 1][k] + shift)
                };
                new_sum += fwd[m][k];
            }
            fwd_sum = new_sum;
        }

        fwd_sum
    }

    /// Sample a state path from posterior probabilities
    fn sample_path(&self, state_probs: &[Vec<f32>], n_markers: usize, n_states: usize) -> Vec<usize> {
        let mut path = Vec::with_capacity(n_markers);

        for m in 0..n_markers {
            // Find max probability state (Viterbi-style for deterministic output)
            let mut max_state = 0;
            let mut max_prob = state_probs[m][0];
            for k in 1..n_states {
                if state_probs[m][k] > max_prob {
                    max_prob = state_probs[m][k];
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

/// Phasing HMM that considers both haplotypes of a diploid sample
///
/// This follows the approach in Java PhaseBaum2 where we track
/// three sets of forward/backward values:
/// - fwd[0]/bwd[0]: Combined (used at homozygous sites)
/// - fwd[1]/bwd[1]: For haplotype 1
/// - fwd[2]/bwd[2]: For haplotype 2
pub struct PhasingHmm<'a> {
    /// Reference genotypes (other samples' haplotypes)
    ref_gt: &'a GenotypeMatrix,
    /// Model parameters
    params: &'a ModelParams,
}

impl<'a> PhasingHmm<'a> {
    /// Create a new phasing HMM
    pub fn new(ref_gt: &'a GenotypeMatrix, params: &'a ModelParams) -> Self {
        Self { ref_gt, params }
    }

    /// Phase a single sample using the diploid HMM approach
    ///
    /// This follows Java PhaseBaum2's phaseHet logic.
    ///
    /// # Arguments
    /// * `alleles1` - Current alleles on haplotype 1
    /// * `alleles2` - Current alleles on haplotype 2
    /// * `het_markers` - Indices of heterozygous markers
    /// * `ref_haps` - Selected reference haplotypes for HMM states
    /// * `p_recomb` - Recombination probabilities
    ///
    /// # Returns
    /// Vector of marker indices where phase should be switched
    pub fn phase_sample(
        &self,
        alleles1: &[u8],
        alleles2: &[u8],
        het_markers: &[usize],
        ref_haps: &[HapIdx],
        p_recomb: &[f32],
    ) -> Vec<usize> {
        if het_markers.is_empty() || ref_haps.is_empty() {
            return Vec::new();
        }

        let n_markers = alleles1.len();
        let n_states = ref_haps.len();

        let p_err = self.params.p_mismatch;
        let p_no_err = 1.0 - p_err;

        // Allocate buffers for three haplotype views
        let mut fwd = vec![vec![0.0f32; n_states]; 3];
        let mut bwd = vec![vec![0.0f32; n_states]; 3];
        let mut fwd_sums = [1.0f32; 3];

        // Initialize forward values uniformly
        let init_prob = 1.0 / n_states as f32;
        for i in 0..3 {
            fwd[i].fill(init_prob);
        }

        // Storage for backward values at het sites
        let mut bwd_het1: Vec<Vec<f32>> = Vec::with_capacity(het_markers.len());
        let mut bwd_het2: Vec<Vec<f32>> = Vec::with_capacity(het_markers.len());

        // Backward pass: save values at het markers
        for i in 0..3 {
            bwd[i].fill(init_prob);
        }

        let mut het_idx = het_markers.len();
        for m in (0..n_markers).rev() {
            if m < n_markers - 1 {
                let p_switch = p_recomb.get(m + 1).copied().unwrap_or(0.0);

                for i in 0..3 {
                    let sum: f32 = bwd[i].iter().sum();
                    if sum > 0.0 {
                        let shift = p_switch / n_states as f32;
                        let scale = (1.0 - p_switch) / sum;
                        for k in 0..n_states {
                            bwd[i][k] = scale * bwd[i][k] + shift;
                        }
                    }
                }
            }

            // Check if this is a het marker
            if het_idx > 0 && het_markers[het_idx - 1] == m {
                het_idx -= 1;
                bwd_het1.push(bwd[1].clone());
                bwd_het2.push(bwd[2].clone());

                // After het, reset hap-specific to combined
                let combined = bwd[0].clone();
                bwd[1].copy_from_slice(&combined);
                bwd[2].copy_from_slice(&combined);
            }

            // Apply emission
            for k in 0..n_states {
                let ref_al = self.ref_gt.allele(MarkerIdx::new(m as u32), ref_haps[k]);

                let emit0 = if ref_al == alleles1[m] && ref_al == alleles2[m] {
                    p_no_err
                } else {
                    p_err
                };
                let emit1 = if ref_al == alleles1[m] { p_no_err } else { p_err };
                let emit2 = if ref_al == alleles2[m] { p_no_err } else { p_err };

                bwd[0][k] *= emit0;
                bwd[1][k] *= emit1;
                bwd[2][k] *= emit2;
            }
        }

        // Reverse the stored backward values
        bwd_het1.reverse();
        bwd_het2.reverse();

        // Forward pass with phase decisions
        let mut switch_markers = Vec::new();
        let mut swap_haps = false;
        let mut het_idx = 0;

        // Reset forward values
        for i in 0..3 {
            fwd[i].fill(init_prob);
            fwd_sums[i] = 1.0;
        }

        for m in 0..n_markers {
            // Check if this is a het marker
            if het_idx < het_markers.len() && het_markers[het_idx] == m {
                // Phase decision
                let b1 = &bwd_het1[het_idx];
                let b2 = &bwd_het2[het_idx];

                let mut p11 = 0.0f32;
                let mut p12 = 0.0f32;
                let mut p21 = 0.0f32;
                let mut p22 = 0.0f32;

                for k in 0..n_states {
                    p11 += fwd[1][k] * b1[k];
                    p12 += fwd[1][k] * b2[k];
                    p21 += fwd[2][k] * b1[k];
                    p22 += fwd[2][k] * b2[k];
                }

                let num = p11 * p22;
                let den = p12 * p21;

                let should_swap = num < den;
                if should_swap != swap_haps {
                    switch_markers.push(m);
                    swap_haps = should_swap;
                }

                // Reset hap-specific to combined
                let combined = fwd[0].clone();
                fwd[1].copy_from_slice(&combined);
                fwd[2].copy_from_slice(&combined);
                fwd_sums[1] = fwd_sums[0];
                fwd_sums[2] = fwd_sums[0];

                het_idx += 1;
            }

            // Forward update
            let p_switch = p_recomb.get(m).copied().unwrap_or(0.0);

            for i in 0..3 {
                let shift = p_switch / n_states as f32;
                let scale = (1.0 - p_switch) / fwd_sums[i];

                let mut new_sum = 0.0f32;
                for k in 0..n_states {
                    let ref_al = self.ref_gt.allele(MarkerIdx::new(m as u32), ref_haps[k]);

                    let emit = match i {
                        0 => {
                            if ref_al == alleles1[m] && ref_al == alleles2[m] {
                                p_no_err
                            } else {
                                p_err
                            }
                        }
                        1 => if ref_al == alleles1[m] { p_no_err } else { p_err },
                        2 => if ref_al == alleles2[m] { p_no_err } else { p_err },
                        _ => unreachable!(),
                    };

                    fwd[i][k] = if m == 0 {
                        emit / n_states as f32
                    } else {
                        emit * (scale * fwd[i][k] + shift)
                    };
                    new_sum += fwd[i][k];
                }
                fwd_sums[i] = new_sum;
            }
        }

        switch_markers
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

        assert_eq!(result.state_probs.len(), 5);
        assert_eq!(result.sampled_path.len(), 5);
        assert!(result.log_likelihood.is_finite());

        // Check probabilities sum to ~1 at each marker
        for probs in &result.state_probs {
            let sum: f32 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "Sum was {}", sum);
        }
    }

    #[test]
    fn test_gen_dist_to_p_recomb() {
        let gen_dists = vec![0.01, 0.02, 0.05]; // cM
        let p_recomb = LiStephensHmm::gen_dists_to_p_recomb(&gen_dists, 1.0);

        assert_eq!(p_recomb.len(), 4);
        assert_eq!(p_recomb[0], 0.0); // First marker
        assert!(p_recomb[1] > 0.0);
        assert!(p_recomb[2] > p_recomb[1]); // Larger distance = higher recomb prob
        assert!(p_recomb[3] > p_recomb[2]);
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
