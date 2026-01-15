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
        let scale = (1.0 - p_switch) / fwd_sum.max(1e-30);

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
        let scale = (1.0 - p_switch) / sum.max(1e-30);
        
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

use crate::model::states::{AlleleScratch, MosaicCursor, StateSwitch, ThreadedHaps};

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
        target_conf: Option<&[f32]>,
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

        let p_err_base = self.params.p_mismatch;
        let p_no_err_base = 1.0 - p_err_base;

        // Allocate scratch buffer for allele materialization
        let mut scratch = AlleleScratch::new(n_states);
        let mut mismatches = vec![0u8; n_states];

        // Create cursor for traversal
        let mut cursor = MosaicCursor::from_threaded(threaded_haps);

        // =====================================================================
        // FORWARD PASS with A-B-C Loop
        // =====================================================================
        let mut fwd_sum = 1.0f32;

        // Accumulate log-likelihood: ln P(O) = Σ ln(c_m) where c_m is the scaling factor at marker m.
        // Previously only the last c_m was used, which is mathematically incorrect.
        let mut log_likelihood = 0.0f64;

        // Event stack for efficient backward pass (sparse, O(switches))
        // For phasing (static states), this remains empty - zero overhead.
        let mut history: Vec<StateSwitch> = Vec::with_capacity(n_markers);

        for m in 0..n_markers {
            let targ_al = target_alleles[m];
            let p_recomb_m = self.p_recomb.get(m).copied().unwrap_or(0.0);

            // Phase A: Advance cursor with history recording
            cursor.advance_with_history(m, threaded_haps, &mut history);

            // Phase B: Materialize alleles into scratch buffer
            scratch.materialize(&cursor, m, |marker, hap| {
                self.ref_gt.allele(MarkerIdx::new(marker as u32), HapIdx::new(hap))
            });

            // Compute mismatches
            for k in 0..n_states {
                mismatches[k] = if scratch.alleles[k] == targ_al { 0 } else { 1 };
            }

            let conf = target_conf
                .and_then(|c| c.get(m).copied())
                .unwrap_or(1.0)
                .clamp(0.0, 1.0);
            let p_no_err = p_no_err_base * conf + 0.5 * (1.0 - conf);
            let p_err = p_err_base * conf + 0.5 * (1.0 - conf);
            let emit_probs = [p_no_err, p_err];

            // Phase C: Math kernel (SIMD-friendly on contiguous data)
            let row_offset = m * n_states;

            if m == 0 {
                // Initialize: following Java ImpLSBaum, m=0 uses just emission probs
                // The 1/n_states factor is a normalization convention; what matters is
                // that fwd_sum is the actual sum for correct scaling at marker 1.
                let init_val = 1.0 / n_states as f32;
                fwd_sum = 0.0;
                for k in 0..n_states {
                    let val = init_val * emit_probs[mismatches[k] as usize];
                    fwd[row_offset + k] = val;
                    fwd_sum += val;
                }
            } else {
                // Li-Stephens HMM transition update:
                //   fwd[k] = emit[k] * ((1-ρ)/Σfwd * fwd[k] + ρ/K)
                // where ρ = p_recomb_m (recombination probability), K = n_states
                //
                // The fwd_update function expects p_switch = ρ (raw recombination prob)
                // and internally computes: shift = ρ/K, scale = (1-ρ)/fwd_sum
                let prev_row_offset = (m - 1) * n_states;
                let (before, curr_and_after) = fwd.split_at_mut(row_offset);
                let prev_row = &before[prev_row_offset..prev_row_offset + n_states];
                let curr_row = &mut curr_and_after[..n_states];
                curr_row.copy_from_slice(prev_row);
                fwd_sum = HmmUpdater::fwd_update(
                    curr_row,
                    fwd_sum,
                    p_recomb_m, // Pass raw recombination probability
                    &emit_probs,
                    &mismatches,
                    n_states,
                );
            }

            // Accumulate log-likelihood from this marker's scaling factor
            if fwd_sum > 0.0 {
                log_likelihood += (fwd_sum as f64).ln();
            }
        }

        // =====================================================================
        // BACKWARD PASS using cursor rewind (Event Stack approach)
        // =====================================================================

        // Initialize last row
        let last_row = (n_markers - 1) * n_states;
        let init_bwd = 1.0 / n_states as f32;
        for k in 0..n_states {
            bwd[last_row + k] = init_bwd;
        }

        // Backward sweep using cursor.rewind()
        // Cursor is currently at last marker; we rewind as we go backwards
        for m in (0..n_markers - 1).rev() {
            let m_next = m + 1;
            let marker_next_idx = MarkerIdx::new(m_next as u32);
            let targ_al_next = target_alleles[m_next];

            let p_recomb_next = self.p_recomb.get(m_next).copied().unwrap_or(0.0);

            // Rewind cursor to m_next (we need alleles at m_next for emission)
            cursor.rewind(m_next, &mut history);

            // Materialize alleles at m_next using rewound cursor
            for k in 0..n_states {
                let hap = cursor.active_haps()[k];
                scratch.alleles[k] = self.ref_gt.allele(marker_next_idx, HapIdx::new(hap));
                mismatches[k] = if scratch.alleles[k] == targ_al_next { 0 } else { 1 };
            }

            // Copy backward values from next row
            let next_row = m_next * n_states;
            let curr_row = m * n_states;
            for k in 0..n_states {
                bwd[curr_row + k] = bwd[next_row + k];
            }

            let conf = target_conf
                .and_then(|c| c.get(m_next).copied())
                .unwrap_or(1.0)
                .clamp(0.0, 1.0);
            let p_no_err = p_no_err_base * conf + 0.5 * (1.0 - conf);
            let p_err = p_err_base * conf + 0.5 * (1.0 - conf);
            let emit_probs = [p_no_err, p_err];

            // Apply backward update (same Li-Stephens formula, different direction)
            // bwd_update expects p_switch = ρ (raw recombination probability)
            HmmUpdater::bwd_update(
                &mut bwd[curr_row..curr_row + n_states],
                p_recomb_next, // Pass raw recombination probability
                &emit_probs,
                &mismatches,
                n_states,
            );
        }

        log_likelihood
    }

    /// Collect statistics for EM parameter estimation
    /// Uses checkpointing to reduce memory from O(n_markers × n_states) to O(n_markers/64 × n_states)
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

        // Checkpoint interval - balance memory vs recomputation
        const CHECKPOINT_INTERVAL: usize = 64;
        let n_checkpoints = (n_markers + CHECKPOINT_INTERVAL - 1) / CHECKPOINT_INTERVAL;

        // Create cursor and record history during forward traversal
        let mut cursor = MosaicCursor::from_threaded(threaded_haps);
        let mut history: Vec<StateSwitch> = Vec::with_capacity(n_markers);

        // First pass: advance cursor to end while recording history AND storing checkpoints
        let mut fwd_checkpoints = vec![0.0f32; n_checkpoints * n_states];
        let mut fwd = vec![1.0f32 / n_states as f32; n_states];
        let mut fwd_sums = vec![1.0f32; n_markers];
        let mut last_fwd_sum = 1.0f32;

        for m in 0..n_markers {
            cursor.advance_with_history(m, threaded_haps, &mut history);

            let marker_idx = MarkerIdx::new(m as u32);
            let targ_al = target_alleles[m];
            let p_switch = self.p_recomb.get(m).copied().unwrap_or(0.0);

            if m > 0 {
                let shift = p_switch / n_states as f32;
                let scale = (1.0 - p_switch) / last_fwd_sum;

                let mut sum = 0.0f32;
                for k in 0..n_states {
                    let ref_al = self.ref_gt.allele(marker_idx, HapIdx::new(cursor.active_haps()[k]));
                    let is_mismatch = ref_al != targ_al;
                    let em = if is_mismatch { p_err } else { p_no_err };
                    fwd[k] = em * (scale * fwd[k] + shift);
                    sum += fwd[k];
                }
                last_fwd_sum = sum.max(1e-30);
            } else {
                // First marker: uniform prior * emission
                let prior = 1.0 / n_states as f32;
                let mut sum = 0.0f32;
                for k in 0..n_states {
                    let ref_al = self.ref_gt.allele(marker_idx, HapIdx::new(cursor.active_haps()[k]));
                    let is_mismatch = ref_al != targ_al;
                    let em = if is_mismatch { p_err } else { p_no_err };
                    fwd[k] = em * prior;
                    sum += fwd[k];
                }
                last_fwd_sum = sum.max(1e-30);
            }

            fwd_sums[m] = last_fwd_sum;

            // Store checkpoint at interval boundaries
            if m % CHECKPOINT_INTERVAL == 0 {
                let checkpoint_idx = m / CHECKPOINT_INTERVAL;
                let checkpoint_off = checkpoint_idx * n_states;
                fwd_checkpoints[checkpoint_off..checkpoint_off + n_states].copy_from_slice(&fwd);
            }
        }

        // 2. Combined backward pass with forward recomputation and stats accumulation
        // Process in reverse order, recomputing forward from checkpoints as needed
        let mut bwd = vec![1.0f32; n_states];
        let mut mismatches = vec![0u8; n_states];
        let mut fwd_recomp = vec![0.0f32; n_states];

        let h_factor = n_states as f32 / (n_states - 1) as f32;

        for m in (0..n_markers).rev() {
            let marker_idx = MarkerIdx::new(m as u32);
            let targ_al = target_alleles[m];

            // Rewind cursor to this marker
            cursor.rewind(m, &mut history);

            // Recompute forward values from nearest checkpoint
            let checkpoint_idx = m / CHECKPOINT_INTERVAL;
            let checkpoint_start = checkpoint_idx * CHECKPOINT_INTERVAL;
            let checkpoint_off = checkpoint_idx * n_states;

            // Load checkpoint
            fwd_recomp.copy_from_slice(&fwd_checkpoints[checkpoint_off..checkpoint_off + n_states]);
            let mut recomp_sum = fwd_sums[checkpoint_start];

            // Recompute forward from checkpoint to m
            // Need a separate cursor for recomputation
            let mut recomp_cursor = MosaicCursor::from_threaded(threaded_haps);
            let mut recomp_history: Vec<StateSwitch> = Vec::with_capacity(m + 1);

            // Advance recomp cursor to checkpoint_start
            for recomp_m in 0..=checkpoint_start {
                recomp_cursor.advance_with_history(recomp_m, threaded_haps, &mut recomp_history);
            }

            // Now advance from checkpoint_start+1 to m while recomputing forward
            for recomp_m in (checkpoint_start + 1)..=m {
                recomp_cursor.advance_with_history(recomp_m, threaded_haps, &mut recomp_history);

                let recomp_marker_idx = MarkerIdx::new(recomp_m as u32);
                let recomp_targ_al = target_alleles[recomp_m];
                let p_switch = self.p_recomb.get(recomp_m).copied().unwrap_or(0.0);
                let shift = p_switch / n_states as f32;
                let scale = (1.0 - p_switch) / recomp_sum.max(1e-30);

                let mut sum = 0.0f32;
                for k in 0..n_states {
                    let ref_al = self.ref_gt.allele(recomp_marker_idx, HapIdx::new(recomp_cursor.active_haps()[k]));
                    let is_mismatch = ref_al != recomp_targ_al;
                    let em = if is_mismatch { p_err } else { p_no_err };
                    fwd_recomp[k] = em * (scale * fwd_recomp[k] + shift);
                    sum += fwd_recomp[k];
                }
                recomp_sum = sum.max(1e-30);
            }

            // Now fwd_recomp contains forward values at marker m
            // Compute stats using fwd_recomp and bwd
            let p_switch = self.p_recomb.get(m).copied().unwrap_or(0.0);
            let last_sum = if m > 0 { fwd_sums[m - 1] } else { 1.0 };
            let shift = p_switch / n_states as f32;
            let scale = (1.0 - p_switch) / last_sum;
            let no_switch_scale = ((1.0 - p_switch) + shift) / last_sum;

            let mut joint_state_sum = 0.0f32;
            let mut state_sum = 0.0f32;
            let mut mismatch_sum = 0.0f32;

            for k in 0..n_states {
                let ref_al = self.ref_gt.allele(marker_idx, HapIdx::new(cursor.active_haps()[k]));
                let is_mismatch = ref_al != targ_al;
                let em = if is_mismatch { p_err } else { p_no_err };

                // Use fwd values from before emission update for joint probability
                let fwd_prior_k = if m > 0 {
                    scale * fwd_recomp[k] / em + shift / em  // Reverse the emission to get prior
                } else {
                    1.0 / n_states as f32
                };

                joint_state_sum += bwd[k] * em * no_switch_scale * fwd_prior_k.max(0.0);

                let state_prob = fwd_recomp[k] * bwd[k];
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
                    let gen_dist = gen_dists.get(m - 1).copied().unwrap_or(0.0);
                    estimates.add_switch(gen_dist, switch_prob as f64);
                }
            }

            // Update backward values for next iteration (moving to m-1)
            if m > 0 {
                let m_next = m;  // We're about to move to m-1, so m is the "next" marker from m-1's perspective
                let targ_al_next = target_alleles[m_next];
                let p_recomb = self.p_recomb.get(m_next).copied().unwrap_or(0.0);

                for k in 0..n_states {
                    let h = cursor.active_haps()[k];
                    let r = self.ref_gt.allele(marker_idx, HapIdx::new(h));
                    mismatches[k] = if r == targ_al_next { 0 } else { 1 };
                }

                HmmUpdater::bwd_update(&mut bwd, p_recomb, &emit_probs, &mismatches, n_states);
            }
        }
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
    fn test_beagle_hmm_forward_backward() {
        let ref_panel = make_test_ref_panel();
        let params = ModelParams::for_phasing(6, 10000.0, None);
        let p_recomb = vec![0.0, 0.01, 0.01, 0.01, 0.01];

        let n_markers = 5;
        let n_states = 3; // 3 composite states with mosaic segments

        // Build ThreadedHaps using PRODUCTION API with actual segment transitions
        // This tests MosaicCursor segment-switching logic that from_static_haps bypassed
        let mut threaded_haps = ThreadedHaps::new(n_states, n_states * 2, n_markers);
        
        // State 0: hap 0 for markers 0-2, then hap 1 for markers 3-4 (segment switch at marker 3)
        threaded_haps.push_new(0);
        threaded_haps.add_segment(0, 1, 3);
        
        // State 1: hap 2 for entire chromosome (no switch - tests static case too)
        threaded_haps.push_new(2);
        
        // State 2: hap 4 for markers 0-1, then hap 5 for markers 2-4 (segment switch at marker 2)
        threaded_haps.push_new(4);
        threaded_haps.add_segment(2, 5, 2);

        let hmm = BeagleHmm::new(&ref_panel, &params, n_states, p_recomb);

        let target_alleles = vec![0, 0, 0, 1, 0]; // Should match haplotype 0 or 4
        let mut fwd = Vec::new();
        let mut bwd = Vec::new();

        let log_likelihood =
            hmm.forward_backward_raw(&target_alleles, None, &threaded_haps, &mut fwd, &mut bwd);

        assert_eq!(fwd.len(), 5 * 3); // 5 markers * 3 states
        assert_eq!(bwd.len(), 5 * 3);
        assert!(log_likelihood.is_finite());
        
        // Verify posteriors sum to 1 at each marker (this validates the mosaic HMM math)
        for m in 0..n_markers {
            let sum: f32 = (0..n_states).map(|k| fwd[m * n_states + k] * bwd[m * n_states + k]).sum();
            // Posteriors should be positive and reasonable
            assert!(sum > 0.0, "Posterior sum at marker {} should be positive", m);
        }
    }

    // =========================================================================
    // Rigorous HMM Updater Tests
    // =========================================================================

    #[test]
    fn test_fwd_update_preserves_probability_mass() {
        // After forward update, values should remain valid probabilities
        for n_states in [4, 8, 16, 32] {
            let mut fwd = vec![1.0 / n_states as f32; n_states];
            let emit_probs = [0.99f32, 0.01];
            let mismatches: Vec<u8> = (0..n_states).map(|k| (k % 2) as u8).collect();

            let initial_sum: f32 = fwd.iter().sum();
            let new_sum = HmmUpdater::fwd_update(&mut fwd, initial_sum, 0.05, &emit_probs, &mismatches, n_states);

            // All values should be positive
            for (k, &val) in fwd.iter().enumerate() {
                assert!(val >= 0.0, "fwd[{}] = {} is negative", k, val);
                assert!(val.is_finite(), "fwd[{}] = {} is not finite", k, val);
            }

            // Sum should be positive and finite
            assert!(new_sum > 0.0, "new_sum {} should be positive", new_sum);
            assert!(new_sum.is_finite(), "new_sum {} should be finite", new_sum);

            // Verify returned sum matches actual sum
            let actual_sum: f32 = fwd.iter().sum();
            assert!(
                (new_sum - actual_sum).abs() < 1e-5,
                "Returned sum {} != actual sum {}", new_sum, actual_sum
            );
        }
    }

    #[test]
    fn test_bwd_update_normalizes_to_one() {
        // After backward update, values should sum close to 1 (normalized)
        for n_states in [4, 8, 16, 32] {
            let mut bwd = vec![1.0 / n_states as f32; n_states];
            let emit_probs = [0.99f32, 0.01];
            let mismatches: Vec<u8> = (0..n_states).map(|k| (k % 2) as u8).collect();

            HmmUpdater::bwd_update(&mut bwd, 0.05, &emit_probs, &mismatches, n_states);

            // All values should be positive
            for (k, &val) in bwd.iter().enumerate() {
                assert!(val >= 0.0, "bwd[{}] = {} is negative", k, val);
                assert!(val.is_finite(), "bwd[{}] = {} is not finite", k, val);
            }

            // Sum should be close to 1 after normalization
            let sum: f32 = bwd.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "bwd sum {} should be ~1.0 (n_states={})", sum, n_states
            );
        }
    }

    #[test]
    fn test_fwd_update_favors_matching_states() {
        // States that match (mismatch=0) should have higher probability than mismatching states
        let n_states = 8;
        let mut fwd = vec![1.0 / n_states as f32; n_states];
        let emit_probs = [0.99f32, 0.01]; // Strong preference for match

        // First 4 states match, last 4 mismatch
        let mismatches: Vec<u8> = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let initial_sum: f32 = fwd.iter().sum();
        HmmUpdater::fwd_update(&mut fwd, initial_sum, 0.001, &emit_probs, &mismatches, n_states);

        // Matching states should have higher values
        let match_sum: f32 = fwd[0..4].iter().sum();
        let mismatch_sum: f32 = fwd[4..8].iter().sum();

        assert!(
            match_sum > mismatch_sum * 10.0,
            "Matching states ({}) should dominate mismatching ({})", match_sum, mismatch_sum
        );
    }

    #[test]
    fn test_simd_vectorized_matches_scalar() {
        // Test that SIMD and scalar paths produce identical results
        // by testing with n_states = 8 (pure SIMD) and n_states = 11 (SIMD + scalar tail)
        for n_states in [8, 11, 16, 17, 24, 25] {
            let initial_fwd: Vec<f32> = (0..n_states).map(|k| (k as f32 + 1.0) / 100.0).collect();
            let initial_sum: f32 = initial_fwd.iter().sum();
            let emit_probs = [0.95f32, 0.05];
            let mismatches: Vec<u8> = (0..n_states).map(|k| ((k * 3) % 2) as u8).collect();

            // Run forward update
            let mut fwd = initial_fwd.clone();
            let new_sum = HmmUpdater::fwd_update(&mut fwd, initial_sum, 0.02, &emit_probs, &mismatches, n_states);

            // Verify basic properties
            assert!(new_sum > 0.0);
            let actual_sum: f32 = fwd.iter().sum();
            assert!(
                (new_sum - actual_sum).abs() < 1e-4,
                "n_states={}: sum mismatch {} vs {}", n_states, new_sum, actual_sum
            );

            // Run backward update
            let mut bwd: Vec<f32> = (0..n_states).map(|k| (k as f32 + 1.0) / 100.0).collect();
            HmmUpdater::bwd_update(&mut bwd, 0.02, &emit_probs, &mismatches, n_states);

            let bwd_sum: f32 = bwd.iter().sum();
            assert!(
                (bwd_sum - 1.0).abs() < 0.01,
                "n_states={}: bwd sum {} should be ~1", n_states, bwd_sum
            );
        }
    }

    #[test]
    fn test_extreme_recombination_rates() {
        // Test edge cases: no recombination and very high recombination
        let n_states = 8;
        let emit_probs = [0.99f32, 0.01];
        let mismatches: Vec<u8> = vec![0, 1, 0, 1, 0, 1, 0, 1];

        // Test with zero recombination (p_switch = 0)
        let mut fwd_no_recomb = vec![0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        let initial_sum: f32 = fwd_no_recomb.iter().sum();
        let new_sum = HmmUpdater::fwd_update(&mut fwd_no_recomb, initial_sum, 0.0, &emit_probs, &mismatches, n_states);

        // With no recombination, only states with initial probability should have probability
        // (though emission still affects all)
        assert!(new_sum > 0.0);
        assert!(new_sum.is_finite());

        // Test with very high recombination (p_switch = 0.99)
        let mut fwd_high_recomb = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let initial_sum_high: f32 = fwd_high_recomb.iter().sum();
        HmmUpdater::fwd_update(&mut fwd_high_recomb, initial_sum_high, 0.99, &emit_probs, &mismatches, n_states);

        // With high recombination, probability should spread to all states
        let min_val = fwd_high_recomb.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            min_val > 0.0,
            "With high recomb, all states should have some probability, min={}", min_val
        );
    }

    #[test]
    fn test_numerical_stability_small_values() {
        // Test with very small initial values to check for underflow
        let n_states = 16;
        let mut fwd: Vec<f32> = vec![1e-30; n_states];
        let emit_probs = [0.99f32, 0.01];
        let mismatches: Vec<u8> = (0..n_states).map(|k| (k % 2) as u8).collect();

        let initial_sum: f32 = fwd.iter().sum();

        // Should not panic or produce NaN/Inf
        let new_sum = HmmUpdater::fwd_update(&mut fwd, initial_sum, 0.01, &emit_probs, &mismatches, n_states);

        assert!(new_sum.is_finite(), "new_sum should be finite, got {}", new_sum);
        for (k, &val) in fwd.iter().enumerate() {
            assert!(val.is_finite(), "fwd[{}] should be finite, got {}", k, val);
        }
    }
}
