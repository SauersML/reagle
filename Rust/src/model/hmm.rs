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
use wide::f32x8;

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

        let shift_vec = f32x8::splat(shift);
        let scale_vec = f32x8::splat(scale);
        let mut sum_vec = f32x8::splat(0.0);

        let p0 = emit_probs[0];
        let p1 = emit_probs[1];
        let diff = p1 - p0; // optimization: emit = p0 + mismatch * diff

        let mut k = 0;
        
        // Vectorized loop (process 8 items at a time)
        while k + 8 <= n_states {
            // Load fwd chunk
            let mut fwd_arr = [0.0f32; 8];
            fwd_arr.copy_from_slice(&fwd[k..k+8]);
            let fwd_chunk = f32x8::from(fwd_arr);
            
            // Construct emission vector branchlessly
            let m_chunk = &mismatches[k..k+8];
            let emit_arr = [
                p0 + (m_chunk[0] as f32) * diff,
                p0 + (m_chunk[1] as f32) * diff,
                p0 + (m_chunk[2] as f32) * diff,
                p0 + (m_chunk[3] as f32) * diff,
                p0 + (m_chunk[4] as f32) * diff,
                p0 + (m_chunk[5] as f32) * diff,
                p0 + (m_chunk[6] as f32) * diff,
                p0 + (m_chunk[7] as f32) * diff,
            ];
            let emit_vec = f32x8::from(emit_arr);
            
            // Compute
            // fwd[k] = emit * (scale * fwd[k] + shift)
            let res = emit_vec * (scale_vec * fwd_chunk + shift_vec);
            
            // Store result
            let res_arr: [f32; 8] = res.into();
            fwd[k..k+8].copy_from_slice(&res_arr);
            
            sum_vec += res;
            k += 8;
        }

        let mut new_sum = sum_vec.reduce_add();

        // Scalar tail loop
        for i in k..n_states {
            let emit = emit_probs[mismatches[i] as usize];
            fwd[i] = emit * (scale * fwd[i] + shift);
            new_sum += fwd[i];
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
        let mut sum_vec = f32x8::splat(0.0);
        let p0 = emit_probs[0];
        let p1 = emit_probs[1];
        let diff = p1 - p0;
        
        let mut k = 0;
        while k + 8 <= n_states {
            let mut bwd_arr = [0.0f32; 8];
            bwd_arr.copy_from_slice(&bwd[k..k+8]);
            let bwd_chunk = f32x8::from(bwd_arr);
            
            let m_chunk = &mismatches[k..k+8];
            let emit_arr = [
                p0 + (m_chunk[0] as f32) * diff,
                p0 + (m_chunk[1] as f32) * diff,
                p0 + (m_chunk[2] as f32) * diff,
                p0 + (m_chunk[3] as f32) * diff,
                p0 + (m_chunk[4] as f32) * diff,
                p0 + (m_chunk[5] as f32) * diff,
                p0 + (m_chunk[6] as f32) * diff,
                p0 + (m_chunk[7] as f32) * diff,
            ];
            let emit_vec = f32x8::from(emit_arr);
            
            let res = bwd_chunk * emit_vec;
            let res_arr: [f32; 8] = res.into();
            bwd[k..k+8].copy_from_slice(&res_arr);
            
            sum_vec += res;
            k += 8;
        }
        
        let mut sum = sum_vec.reduce_add();
        
        // Tail loop 1
        for i in k..n_states {
            bwd[i] *= emit_probs[mismatches[i] as usize];
            sum += bwd[i];
        }

        // Then: apply transition
        let shift = p_switch / n_states as f32;
        let scale = (1.0 - p_switch) / sum;
        
        let shift_vec = f32x8::splat(shift);
        let scale_vec = f32x8::splat(scale);
        
        k = 0;
        while k + 8 <= n_states {
            let mut bwd_arr = [0.0f32; 8];
            bwd_arr.copy_from_slice(&bwd[k..k+8]);
            let bwd_chunk = f32x8::from(bwd_arr);
            
            let res = scale_vec * bwd_chunk + shift_vec;
            let res_arr: [f32; 8] = res.into();
            bwd[k..k+8].copy_from_slice(&res_arr);
            
            k += 8;
        }

        for i in k..n_states {
            bwd[i] = scale * bwd[i] + shift;
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

    /// Compute expected allele dosage at a marker given state probabilities
    ///
    /// Returns the expected ALT allele count (0.0 to 1.0 for a single haplotype)
    pub fn compute_dosage(&self, marker: MarkerIdx, state_probs: &[f32]) -> f32 {
        let mut dosage = 0.0f32;
        for (k, &prob) in state_probs.iter().enumerate() {
            let allele = self.state_allele(marker, k);
            if allele != 255 {
                dosage += prob * allele as f32;
            }
        }
        dosage
    }

    /// Run forward-backward algorithm and return raw Forward and Backward tables
    ///
    /// This separates the forward and backward passes to allow cross-haplotype
    /// calculations (e.g. Fwd1 * Bwd2) needed for phasing.
    ///
    /// # Arguments
    /// * `target_alleles` - Alleles of the target haplotype
    /// * `fwd` - Pre-allocated forward buffer [n_markers * n_states]
    /// * `bwd` - Pre-allocated backward buffer [n_markers * n_states]
    ///
    /// # Returns
    /// Log-likelihood of the data
    pub fn forward_backward_raw(
        &self,
        target_alleles: &[u8],
        fwd: &mut Vec<f32>,
        bwd: &mut Vec<f32>,
    ) -> f64 {
        let n_markers = self.n_markers();
        let n_states = self.n_states();
        let total_size = n_markers * n_states;

        if n_markers == 0 || n_states == 0 {
            return 0.0;
        }

        // Resize buffers
        fwd.resize(total_size, 0.0);
        bwd.resize(total_size, 0.0);

        let p_err = self.params.p_mismatch;
        let p_no_err = 1.0 - p_err;
        let emit_probs = [p_no_err, p_err];

        // 1. Forward Pass
        let fwd_sum = self.forward_pass(target_alleles, fwd, &emit_probs);

        // 2. Backward Pass
        // Initialize last marker of backward table
        let last_row = (n_markers - 1) * n_states;
        let init_bwd = 1.0 / n_states as f32;
        for k in 0..n_states {
            bwd[last_row + k] = init_bwd;
        }

        // Iterate backwards from M-2 to 0
        // (The last marker M-1 is already initialized)
        for m in (0..n_markers - 1).rev() {
            // Setup for step FROM m+1 TO m
            // But bwd_update computes values at m based on m+1
            let m_next = m + 1;
            let marker_next_idx = MarkerIdx::new(m_next as u32);
            let targ_al_next = target_alleles[m_next];
            
            // Recombination from m to m+1
            let p_recomb = self.p_recomb.get(m_next).copied().unwrap_or(0.0);
            
            // Compute mismatches at m+1
            let ref_alleles = self.state_alleles(marker_next_idx);
            let mismatches: Vec<u8> = ref_alleles
                .iter()
                .map(|&r| if r == targ_al_next { 0 } else { 1 })
                .collect();

            // Calculate Bwd[m] based on Bwd[m+1]
            // We need to copy Bwd[m+1] to Bwd[m] then update in place
            let curr_row = m * n_states;
            let next_row = m_next * n_states;
            
            // Copy next row values to current row to prepare for update
            for k in 0..n_states {
                bwd[curr_row + k] = bwd[next_row + k];
            }
            
            // Update in place (this matches HmmUpdater::bwd_update logic)
            // It multiplies by emission at m+1, then applies transition from m to m+1
            // Result is Prob(Tail | State at m)
            
            // Apply bwd_update on the slice
            let bwd_slice = &mut bwd[curr_row..curr_row + n_states];
            
            // HmmUpdater::bwd_update normalizes using bwd_sum? 
            // Wait, HmmUpdater::bwd_update calculates its OWN scale based on sum.
            // But we need to be careful about scaling consistency with Forward?
            // Usually we just normalize Bwd to sum to 1 to prevent underflow.
            
            HmmUpdater::bwd_update(bwd_slice, p_recomb, &emit_probs, &mismatches, n_states);
        }

        fwd_sum.ln() as f64
    }

    /// Run forward-backward algorithm for a target haplotype
    ///
    /// This implements the Baum-Welch style forward-backward following
    /// the Java ImpLSBaum implementation.
    ///
    /// # Arguments
    /// * `target_alleles` - Alleles of the target haplotype at each marker
    /// * `fwd` - Pre-allocated forward buffer [n_markers * n_states]
    /// * `bwd` - Pre-allocated backward buffer [n_states] (scratch)
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

    /// Collect statistics for EM parameter estimation
    ///
    /// # Arguments
    /// * `target_alleles` - Alleles of the target haplotype
    /// * `gen_dists` - Genetic distances between markers (in cM)
    /// * `estimates` - Estimates structure to accumulate results
    #[allow(dead_code)]
    pub fn collect_stats(
        &self,
        target_alleles: &[u8],
        gen_dists: &[f64],
        estimates: &mut crate::model::parameters::ParamEstimates,
    ) {
        let n_markers = self.n_markers();
        let n_states = self.n_states();
        if n_markers < 2 || n_states <= 1 {
            return;
        }

        let p_err = self.params.p_mismatch;
        let p_no_err = 1.0 - p_err;
        let emit_probs = [p_no_err, p_err];

        // 1. Backward pass: compute all backward values
        let mut saved_bwd = vec![0.0f32; n_markers * n_states];
        let mut bwd = vec![1.0f32; n_states];
        
        // Initialize last row
        let last_row_start = (n_markers - 1) * n_states;
        saved_bwd[last_row_start..last_row_start + n_states].fill(1.0);

        for m in (0..n_markers - 1).rev() {
            let m_next = m + 1;
            let marker_next_idx = MarkerIdx::new(m_next as u32);
            let targ_al_next = target_alleles[m_next];
            let p_recomb = self.p_recomb.get(m_next).copied().unwrap_or(0.0);
            
            let ref_alleles = self.state_alleles(marker_next_idx);
            let mismatches: Vec<u8> = ref_alleles
                .iter()
                .map(|&r| if r == targ_al_next { 0 } else { 1 })
                .collect();

            HmmUpdater::bwd_update(&mut bwd, p_recomb, &emit_probs, &mismatches, n_states);
            let row_start = m * n_states;
            saved_bwd[row_start..row_start + n_states].copy_from_slice(&bwd);
        }

        // 2. Forward pass and accumulate stats
        let h_factor = n_states as f32 / (n_states - 1) as f32;
        let mut fwd = vec![1.0f32 / n_states as f32; n_states];
        let mut last_fwd_sum = 1.0f32;

        for m in 0..n_markers {
            let marker_idx = MarkerIdx::new(m as u32);
            let targ_al = target_alleles[m];
            let p_switch = self.p_recomb.get(m).copied().unwrap_or(0.0);
            let shift = p_switch / n_states as f32;
            let scale = (1.0 - p_switch) / last_fwd_sum;
            let no_switch_scale = ((1.0 - p_switch) + shift) / last_fwd_sum;

            let mut joint_state_sum = 0.0f32;
            let mut state_sum = 0.0f32;
            let mut mismatch_sum = 0.0f32;
            let mut next_fwd_sum = 0.0f32;

            let bwd_m = &saved_bwd[m * n_states .. (m + 1) * n_states];
            let ref_alleles = self.state_alleles(marker_idx);

            for k in 0..n_states {
                let is_mismatch = ref_alleles[k] != targ_al;
                let em = if is_mismatch { p_err } else { p_no_err };
                
                // P(State_m = k, State_{m-1} = k, Data_m)
                joint_state_sum += bwd_m[k] * em * no_switch_scale * fwd[k];
                
                // Update fwd for next marker
                fwd[k] = em * (scale * fwd[k] + shift);
                next_fwd_sum += fwd[k];
                
                // Posterior P(State_m = k | Data) * P(Data)
                let state_prob = fwd[k] * bwd_m[k];
                state_sum += state_prob;
                if is_mismatch {
                    mismatch_sum += state_prob;
                }
            }

            // Accumulate mismatch stats
            if state_sum > 0.0 {
                estimates.add_emission(
                    (1.0 - mismatch_sum / state_sum) as f64,
                    (mismatch_sum / state_sum) as f64
                );
            }

            // Accumulate switch stats
            if m > 0 && state_sum > 0.0 {
                let switch_prob = h_factor * (1.0 - joint_state_sum / state_sum);
                if switch_prob > 0.0 {
                    let gen_dist = gen_dists.get(m).copied().unwrap_or(0.0);
                    estimates.add_switch(gen_dist, switch_prob as f64);
                }
            }

            last_fwd_sum = next_fwd_sum;
        }
    }
}

/// Li-Stephens HMM with composite (mosaic) reference states
///
/// Unlike the standard HMM where each state is a single reference haplotype,
/// composite states can switch which reference haplotype they point to at
/// different markers, following IBS sharing patterns.
pub struct CompositeHmm<'a> {
    /// Reference panel genotypes
    ref_gt: GenotypeView<'a>,
    /// Model parameters
    params: &'a ModelParams,
    /// State-to-haplotype mapping: state_map[marker * n_states + state] = ref_hap
    state_map: Vec<HapIdx>,
    /// Number of states
    n_states: usize,
    /// Recombination probabilities between consecutive markers
    p_recomb: Vec<f32>,
}

impl<'a> CompositeHmm<'a> {
    pub fn new(
        ref_gt: impl Into<GenotypeView<'a>>,
        params: &'a ModelParams,
        state_map: Vec<HapIdx>,
        n_states: usize,
        p_recomb: Vec<f32>,
    ) -> Self {
        Self {
            ref_gt: ref_gt.into(),
            params,
            state_map,
            n_states,
            p_recomb,
        }
    }

    pub fn n_states(&self) -> usize {
        self.n_states
    }

    pub fn n_markers(&self) -> usize {
        self.ref_gt.n_markers()
    }

    /// Get reference allele at marker for composite state
    /// Unlike static HMM, this looks up which ref hap the state points to at this marker
    #[inline]
    pub fn state_allele(&self, marker: MarkerIdx, state: usize) -> u8 {
        let idx = marker.as_usize() * self.n_states + state;
        let ref_hap = self.state_map[idx];
        self.ref_gt.allele(marker, ref_hap)
    }

    /// Get all reference alleles for all states at a marker
    pub fn state_alleles(&self, marker: MarkerIdx) -> Vec<u8> {
        let m = marker.as_usize();
        let base = m * self.n_states;
        (0..self.n_states)
            .map(|s| {
                let ref_hap = self.state_map[base + s];
                self.ref_gt.allele(marker, ref_hap)
            })
            .collect()
    }

    /// Compute expected allele dosage at a marker given state probabilities
    pub fn compute_dosage(&self, marker: MarkerIdx, state_probs: &[f32]) -> f32 {
        let mut dosage = 0.0f32;
        for (k, &prob) in state_probs.iter().enumerate() {
            let allele = self.state_allele(marker, k);
            if allele != 255 {
                dosage += prob * allele as f32;
            }
        }
        dosage
    }

    /// Run forward-backward algorithm and return raw Forward and Backward tables
    pub fn forward_backward_raw(
        &self,
        target_alleles: &[u8],
        fwd: &mut Vec<f32>,
        bwd: &mut Vec<f32>,
    ) -> f64 {
        let n_markers = self.n_markers();
        let n_states = self.n_states;
        let total_size = n_markers * n_states;

        if n_markers == 0 || n_states == 0 {
            return 0.0;
        }

        fwd.resize(total_size, 0.0);
        bwd.resize(total_size, 0.0);

        let p_err = self.params.p_mismatch;
        let p_no_err = 1.0 - p_err;
        let emit_probs = [p_no_err, p_err];

        let fwd_sum = self.forward_pass(target_alleles, fwd, &emit_probs);

        let last_row = (n_markers - 1) * n_states;
        let init_bwd = 1.0 / n_states as f32;
        for k in 0..n_states {
            bwd[last_row + k] = init_bwd;
        }

        for m in (0..n_markers - 1).rev() {
            let m_next = m + 1;
            let marker_next_idx = MarkerIdx::new(m_next as u32);
            let targ_al_next = target_alleles[m_next];

            let p_recomb = self.p_recomb.get(m_next).copied().unwrap_or(0.0);

            let ref_alleles = self.state_alleles(marker_next_idx);
            let mismatches: Vec<u8> = ref_alleles
                .iter()
                .map(|&r| if r == targ_al_next { 0 } else { 1 })
                .collect();

            let curr_row = m * n_states;
            let next_row = m_next * n_states;

            for k in 0..n_states {
                bwd[curr_row + k] = bwd[next_row + k];
            }

            let bwd_slice = &mut bwd[curr_row..curr_row + n_states];
            HmmUpdater::bwd_update(bwd_slice, p_recomb, &emit_probs, &mismatches, n_states);
        }

        fwd_sum.ln() as f64
    }

    /// Run forward-backward algorithm for a target haplotype
    pub fn forward_backward(
        &self,
        target_alleles: &[u8],
        fwd: &mut Vec<f32>,
        bwd: &mut Vec<f32>,
    ) -> HmmResult {
        let n_markers = self.n_markers();
        let n_states = self.n_states;

        if n_markers == 0 || n_states == 0 {
            return HmmResult {
                state_probs: Vec::new(),
                n_states,
                sampled_path: Vec::new(),
                log_likelihood: 0.0,
            };
        }

        let total_size = n_markers * n_states;
        fwd.resize(total_size, 0.0);
        bwd.resize(n_states, 0.0);

        let p_err = self.params.p_mismatch;
        let p_no_err = 1.0 - p_err;
        let emit_probs = [p_no_err, p_err];

        let fwd_sum = self.forward_pass(target_alleles, fwd, &emit_probs);

        let init_bwd = 1.0 / n_states as f32;
        bwd.fill(init_bwd);
        let mut bwd_sum = 1.0f32;

        for m in (0..n_markers).rev() {
            let marker_idx = MarkerIdx::new(m as u32);
            let targ_al = target_alleles[m];
            let row_offset = m * n_states;

            let ref_alleles = self.state_alleles(marker_idx);
            let mismatches: Vec<u8> = ref_alleles
                .iter()
                .map(|&r| if r == targ_al { 0 } else { 1 })
                .collect();

            if m < n_markers - 1 {
                let p_recomb = self.p_recomb.get(m + 1).copied().unwrap_or(0.0);
                let shift = p_recomb / n_states as f32;
                let scale = (1.0 - p_recomb) / bwd_sum;

                for k in 0..n_states {
                    bwd[k] = scale * bwd[k] + shift;
                }
            }

            let mut state_sum = 0.0f32;
            for k in 0..n_states {
                let idx = row_offset + k;
                fwd[idx] *= bwd[k];
                state_sum += fwd[idx];
            }

            if state_sum > 0.0 {
                let inv_sum = 1.0 / state_sum;
                for k in 0..n_states {
                    fwd[row_offset + k] *= inv_sum;
                }
            }

            if m > 0 {
                bwd_sum = 0.0;
                for k in 0..n_states {
                    bwd[k] *= emit_probs[mismatches[k] as usize];
                    bwd_sum += bwd[k];
                }
            }
        }

        let sampled_path = self.sample_path(fwd, n_markers, n_states);
        let log_likelihood = fwd_sum.ln() as f64;

        HmmResult {
            state_probs: fwd.clone(),
            n_states,
            sampled_path,
            log_likelihood,
        }
    }

    fn forward_pass(
        &self,
        target_alleles: &[u8],
        fwd: &mut Vec<f32>,
        emit_probs: &[f32; 2],
    ) -> f32 {
        let n_markers = self.n_markers();
        let n_states = self.n_states;

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

    fn sample_path(
        &self,
        state_probs: &[f32],
        n_markers: usize,
        n_states: usize,
    ) -> Vec<usize> {
        let mut path = Vec::with_capacity(n_markers);

        for m in 0..n_markers {
            let row_offset = m * n_states;
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

        GenotypeMatrix::new_unphased(markers, columns, samples)
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
