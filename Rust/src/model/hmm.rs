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

// ============================================================================
// BeagleHmm: Memory-Efficient Mosaic HMM with A-B-C Loop Pattern
// ============================================================================

use crate::model::states::{AlleleScratch, MosaicCursor, ThreadedHaps};

/// High-performance Li-Stephens HMM using mosaic states with A-B-C loop pattern.
///
/// This implementation achieves:
/// - **Memory efficiency**: O(K * segments) instead of O(M * K) for state map
/// - **SIMD friendliness**: Separates state maintenance from math kernel
/// - **Java parity**: Matches Beagle's composite state approach
///
/// ## The A-B-C Loop Pattern
/// - **Phase A**: State maintenance (integer logic, branch-predictable)
/// - **Phase B**: Allele materialization (memory fetch into contiguous scratch)
/// - **Phase C**: Math kernel (SIMD-vectorizable on flat data)
pub struct BeagleHmm<'a> {
    /// Reference panel genotypes
    ref_gt: GenotypeView<'a>,
    /// Model parameters
    params: &'a ModelParams,
    /// Number of HMM states
    n_states: usize,
    /// Recombination probabilities between consecutive markers
    p_recomb: Vec<f32>,
}

impl<'a> BeagleHmm<'a> {
    /// Create a new BeagleHmm
    pub fn new(
        ref_gt: impl Into<GenotypeView<'a>>,
        params: &'a ModelParams,
        n_states: usize,
        p_recomb: Vec<f32>,
    ) -> Self {
        Self {
            ref_gt: ref_gt.into(),
            params,
            n_states,
            p_recomb,
        }
    }

    /// Number of HMM states
    pub fn n_states(&self) -> usize {
        self.n_states
    }

    /// Number of markers
    pub fn n_markers(&self) -> usize {
        self.ref_gt.n_markers()
    }

    /// Run forward-backward with mosaic cursor using A-B-C loop pattern
    ///
    /// This is the high-performance implementation that:
    /// 1. Uses MosaicCursor for O(K*segments) memory
    /// 2. Pre-materializes alleles into scratch buffer for SIMD
    /// 3. Runs vectorizable math on contiguous data
    ///
    /// Returns (log_likelihood, fwd_buffer, bwd_buffer)
    pub fn forward_backward_raw(
        &self,
        target_alleles: &[u8],
        threaded_haps: &ThreadedHaps,
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

        // Allocate scratch buffer for allele materialization
        let mut scratch = AlleleScratch::new(n_states);
        let mut mismatches = vec![0u8; n_states];

        // Create cursor for traversal
        let mut cursor = MosaicCursor::from_threaded(threaded_haps);

        // =====================================================================
        // FORWARD PASS with A-B-C Loop
        // =====================================================================
        let mut fwd_sum = 1.0f32;

        // Store active haps at each marker for backward pass
        let mut marker_haps: Vec<Vec<u32>> = Vec::with_capacity(n_markers);

        for m in 0..n_markers {
            let targ_al = target_alleles[m];
            let p_recomb_m = self.p_recomb.get(m).copied().unwrap_or(0.0);

            // Phase A: Advance cursor (state maintenance)
            cursor.advance_to_marker(m, threaded_haps);
            
            // Store snapshot of active haps for backward pass
            marker_haps.push(cursor.active_haps().to_vec());

            // Phase B: Materialize alleles into scratch buffer
            scratch.materialize(&cursor, m, |marker, hap| {
                self.ref_gt.allele(MarkerIdx::new(marker as u32), HapIdx::new(hap))
            });

            // Compute mismatches
            for k in 0..n_states {
                mismatches[k] = if scratch.alleles[k] == targ_al { 0 } else { 1 };
            }

            // Phase C: Math kernel (SIMD-friendly on contiguous data)
            let row_offset = m * n_states;

            if m == 0 {
                // Initialize
                let init_val = 1.0 / n_states as f32;
                for k in 0..n_states {
                    fwd[row_offset + k] = init_val * emit_probs[mismatches[k] as usize];
                }
                fwd_sum = 1.0;
            } else {
                let shift = p_recomb_m / n_states as f32;
                let stay = 1.0 - p_recomb_m;

                fwd_sum = HmmUpdater::fwd_update(
                    &mut fwd[row_offset..row_offset + n_states],
                    fwd_sum,
                    stay + shift,
                    &emit_probs,
                    &mismatches,
                    n_states,
                );
            }
        }

        // =====================================================================
        // BACKWARD PASS using stored marker haplotypes
        // =====================================================================

        // Initialize last row
        let last_row = (n_markers - 1) * n_states;
        let init_bwd = 1.0 / n_states as f32;
        for k in 0..n_states {
            bwd[last_row + k] = init_bwd;
        }

        // Backward sweep using stored haplotypes
        for m in (0..n_markers - 1).rev() {
            let m_next = m + 1;
            let marker_next_idx = MarkerIdx::new(m_next as u32);
            let targ_al_next = target_alleles[m_next];

            let p_recomb_next = self.p_recomb.get(m_next).copied().unwrap_or(0.0);

            // Use stored haplotypes for marker m_next
            for k in 0..n_states {
                let hap = marker_haps[m_next][k];
                scratch.alleles[k] = self.ref_gt.allele(marker_next_idx, HapIdx::new(hap));
                mismatches[k] = if scratch.alleles[k] == targ_al_next { 0 } else { 1 };
            }

            // Copy backward values from next row
            let next_row = m_next * n_states;
            let curr_row = m * n_states;
            for k in 0..n_states {
                bwd[curr_row + k] = bwd[next_row + k];
            }

            // Apply backward update
            let shift = p_recomb_next / n_states as f32;
            let stay = 1.0 - p_recomb_next;

            HmmUpdater::bwd_update(
                &mut bwd[curr_row..curr_row + n_states],
                stay + shift,
                &emit_probs,
                &mismatches,
                n_states,
            );
        }

        fwd_sum.ln() as f64
    }

    /// Compute expected allele dosage at a marker given state probabilities
    pub fn compute_dosage(
        &self,
        marker: MarkerIdx,
        state_probs: &[f32],
        marker_haps: &[u32],
    ) -> f32 {
        let mut dosage = 0.0f32;
        for (k, &prob) in state_probs.iter().enumerate() {
            let hap = marker_haps[k];
            let allele = self.ref_gt.allele(marker, HapIdx::new(hap));
            if allele != 255 {
                dosage += prob * allele as f32;
            }
        }
        dosage
    }

    /// Collect statistics for EM parameter estimation
    pub fn collect_stats(
        &self,
        target_alleles: &[u8],
        threaded_haps: &ThreadedHaps,
        gen_dists: &[f64],
        estimates: &mut crate::model::parameters::ParamEstimates,
    ) {
        let n_markers = self.n_markers();
        let n_states = self.n_states;
        if n_markers < 2 || n_states <= 1 {
            return;
        }

        let p_err = self.params.p_mismatch;
        let p_no_err = 1.0 - p_err;
        let emit_probs = [p_no_err, p_err];

        // Create cursor and store haps at each marker
        let mut cursor = MosaicCursor::from_threaded(threaded_haps);
        let mut marker_haps: Vec<Vec<u32>> = Vec::with_capacity(n_markers);
        for m in 0..n_markers {
            cursor.advance_to_marker(m, threaded_haps);
            marker_haps.push(cursor.active_haps().to_vec());
        }

        // 1. Backward pass: compute all backward values
        let mut saved_bwd = vec![0.0f32; n_markers * n_states];
        let mut bwd = vec![1.0f32; n_states];
        
        let last_row_start = (n_markers - 1) * n_states;
        saved_bwd[last_row_start..last_row_start + n_states].fill(1.0);

        for m in (0..n_markers - 1).rev() {
            let m_next = m + 1;
            let marker_next_idx = MarkerIdx::new(m_next as u32);
            let targ_al_next = target_alleles[m_next];
            let p_recomb = self.p_recomb.get(m_next).copied().unwrap_or(0.0);
            
            let mismatches: Vec<u8> = marker_haps[m_next].iter()
                .map(|&h| {
                    let r = self.ref_gt.allele(marker_next_idx, HapIdx::new(h));
                    if r == targ_al_next { 0 } else { 1 }
                })
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

            for k in 0..n_states {
                let ref_al = self.ref_gt.allele(marker_idx, HapIdx::new(marker_haps[m][k]));
                let is_mismatch = ref_al != targ_al;
                let em = if is_mismatch { p_err } else { p_no_err };
                
                joint_state_sum += bwd_m[k] * em * no_switch_scale * fwd[k];
                
                fwd[k] = em * (scale * fwd[k] + shift);
                next_fwd_sum += fwd[k];
                
                let state_prob = fwd[k] * bwd_m[k];
                state_sum += state_prob;
                if is_mismatch {
                    mismatch_sum += state_prob;
                }
            }

            if state_sum > 0.0 {
                estimates.add_emission(
                    (1.0 - mismatch_sum / state_sum) as f64,
                    (mismatch_sum / state_sum) as f64
                );
            }

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

// ============================================================================
// LiStephensHmm: Static-State Wrapper for BeagleHmm
// ============================================================================

/// Static-state Li-Stephens HMM that wraps BeagleHmm for backwards compatibility.
///
/// This provides the same API as the original LiStephensHmm but uses the unified
/// BeagleHmm internally with a static `ThreadedHaps` (single segment per state).
///
/// Use this when states are selected once via PBWT and remain constant for the
/// entire forward-backward run (typical in phasing).
pub struct LiStephensHmm<'a> {
    /// Reference panel genotypes
    ref_gt: GenotypeView<'a>,
    /// Model parameters
    params: &'a ModelParams,
    /// Static haplotype indices (one per state)
    ref_haps: Vec<HapIdx>,
    /// Recombination probabilities between consecutive markers
    p_recomb: Vec<f32>,
}

impl<'a> LiStephensHmm<'a> {
    /// Create a new static Li-Stephens HMM
    ///
    /// # Arguments
    /// * `ref_gt` - Reference panel genotypes
    /// * `params` - Model parameters
    /// * `ref_haps` - Reference haplotype indices for each state (constant across markers)
    /// * `p_recomb` - Recombination probabilities
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

    /// Run forward-backward algorithm and return full result with posteriors
    ///
    /// This is the high-level API that returns normalized posteriors and sampled path.
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
                n_states: 0,
                sampled_path: Vec::new(),
                log_likelihood: 0.0,
            };
        }

        // Create static ThreadedHaps (single segment per state)
        let threaded_haps = ThreadedHaps::from_static_haps(&self.ref_haps, n_markers);

        // Create BeagleHmm and run forward-backward
        let beagle_hmm = BeagleHmm::new(
            self.ref_gt,
            self.params,
            n_states,
            self.p_recomb.clone(),
        );

        let log_likelihood = beagle_hmm.forward_backward_raw(
            target_alleles,
            &threaded_haps,
            fwd,
            bwd,
        );

        // Compute normalized posteriors: state_probs = fwd * bwd (normalized per marker)
        let total_size = n_markers * n_states;
        let mut state_probs = vec![0.0f32; total_size];

        for m in 0..n_markers {
            let row_start = m * n_states;
            let row_end = row_start + n_states;

            let mut sum = 0.0f32;
            for k in 0..n_states {
                let p = fwd[row_start + k] * bwd[row_start + k];
                state_probs[row_start + k] = p;
                sum += p;
            }

            // Normalize
            if sum > 0.0 {
                let inv_sum = 1.0 / sum;
                for k in 0..n_states {
                    state_probs[row_start + k] *= inv_sum;
                }
            }
        }

        // Sample path (simple argmax for now)
        let sampled_path: Vec<usize> = (0..n_markers)
            .map(|m| {
                let row_start = m * n_states;
                (0..n_states)
                    .max_by(|&a, &b| {
                        state_probs[row_start + a]
                            .partial_cmp(&state_probs[row_start + b])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0)
            })
            .collect();

        HmmResult {
            state_probs,
            n_states,
            sampled_path,
            log_likelihood,
        }
    }

    /// Run forward-backward and return raw buffers (for phasing pipeline)
    ///
    /// Returns log-likelihood. The fwd/bwd buffers contain scaled (but not normalized)
    /// forward/backward values.
    pub fn forward_backward_raw(
        &self,
        target_alleles: &[u8],
        fwd: &mut Vec<f32>,
        bwd: &mut Vec<f32>,
    ) -> f64 {
        let n_markers = self.n_markers();
        let n_states = self.n_states();

        if n_markers == 0 || n_states == 0 {
            return 0.0;
        }

        // Create static ThreadedHaps
        let threaded_haps = ThreadedHaps::from_static_haps(&self.ref_haps, n_markers);

        // Create BeagleHmm and delegate
        let beagle_hmm = BeagleHmm::new(
            self.ref_gt,
            self.params,
            n_states,
            self.p_recomb.clone(),
        );

        beagle_hmm.forward_backward_raw(target_alleles, &threaded_haps, fwd, bwd)
    }

    /// Compute expected allele dosage at a marker given state probabilities
    pub fn compute_dosage(&self, marker: MarkerIdx, state_probs: &[f32]) -> f32 {
        let mut dosage = 0.0f32;
        for (k, &prob) in state_probs.iter().enumerate() {
            if k < self.ref_haps.len() {
                let hap = self.ref_haps[k];
                let allele = self.ref_gt.allele(marker, hap);
                if allele != 255 {
                    dosage += prob * allele as f32;
                }
            }
        }
        dosage
    }

    /// Collect statistics for EM parameter estimation
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

        // Create static ThreadedHaps
        let threaded_haps = ThreadedHaps::from_static_haps(&self.ref_haps, n_markers);

        // Create BeagleHmm and delegate
        let beagle_hmm = BeagleHmm::new(
            self.ref_gt,
            self.params,
            n_states,
            self.p_recomb.clone(),
        );

        beagle_hmm.collect_stats(target_alleles, &threaded_haps, gen_dists, estimates)
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
        let params = ModelParams::for_phasing(6, 10000.0, None);
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
        let params = ModelParams::for_phasing(6, 10000.0, None);
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
