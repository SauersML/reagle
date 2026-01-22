//! # HMM Kernel: Reuse Existing AVX-512 Optimized Code
//!
//! This module integrates the block-hash HMM with Reagle's existing
//! SIMD-optimized HMM kernels instead of writing new scalar loops.

use super::compressed_block::CompressedBlock;
use super::workspace::BlockHmmWorkspace;
use super::types::PatternId;
use super::weighted_kernel::WeightedHmmUpdater;
use crate::pipelines::imputation::AllelePosteriors;

/// Run forward pass within a single block using existing SIMD kernel
///
/// # Arguments
/// * `block` - The CompressedBlock (immutable reference data)
/// * `target_genotypes` - Target genotypes for this block [marker_in_window]
/// * `error_rate` - Genotyping error rate
/// * `ws` - Mutable workspace
pub fn forward_within_block(
    block: &CompressedBlock,
    target_genotypes: &[u8],
    error_rate: f32,
    ws: &mut BlockHmmWorkspace,
) {
    let n_patterns = block.n_patterns();
    let window_size = block.window_size();

    assert_eq!(
        target_genotypes.len(),
        window_size,
        "Target genotypes must match block size"
    );

    // For each marker in the window
    for marker_in_window in 0..window_size {
        let target_allele = target_genotypes[marker_in_window];
        // Fix: Marker 0 uses 0.0 rate (prior already transitioned by Bridge).
        // Markers > 0 use rate[m-1] (transition from m-1 to m).
        let recomb_rate = if marker_in_window == 0 {
            0.0
        } else {
            block.local_recomb_rates[marker_in_window - 1]
        };

        // Compute emission probabilities for all patterns
        // Use ws.emissions as temp buffer
        let emissions = &mut ws.emissions;
        
        for pattern_idx in 0..n_patterns {
            let pattern_id = PatternId::new(pattern_idx as u32);
            emissions[pattern_idx] = emission_prob(
                block,
                pattern_id,
                marker_in_window,
                target_allele,
                error_rate,
            );
        }

        // REUSE: Use new weighted kernel
        let fwd_sum = ws.fwd[..n_patterns].iter().sum::<f32>() + ws.reservoir_prob_fwd;

        WeightedHmmUpdater::fwd_update_weighted(
            &mut ws.fwd,
            fwd_sum,
            recomb_rate,
            block.n_ref_haps(),
            &block.pattern_counts,
            emissions,
            n_patterns,
        );

        // Handle reservoir separately (not part of SIMD kernel)
        if block.reservoir_count > 0 {
            let reservoir_emission = emission_prob(
                block,
                PatternId::RESERVOIR,
                marker_in_window,
                target_allele,
                error_rate,
            );

            let total_mass = fwd_sum;
            let background = total_mass * recomb_rate / block.n_ref_haps() as f32;
            let stay = ws.reservoir_prob_fwd * (1.0 - recomb_rate);

            ws.reservoir_prob_fwd =
                reservoir_emission * (stay + background * block.reservoir_count as f32);
        }

        // Normalize to prevent underflow
        ws.normalize_forward(n_patterns);
    }
}

/// Run forward pass up to a specific marker within a block
///
/// Used for extracting state at arbitrary positions (e.g. for priors handoff).
/// Returns state after observing `stop_marker_in_window`.
pub fn forward_to_marker_in_block(
    block: &CompressedBlock,
    target_genotypes: &[u8],
    error_rate: f32,
    ws: &mut BlockHmmWorkspace,
    stop_marker_in_window: usize,
) {
    let n_patterns = block.n_patterns();
    let window_size = block.window_size();

    assert!(stop_marker_in_window < window_size);
    assert_eq!(target_genotypes.len(), window_size);

    for marker_in_window in 0..=stop_marker_in_window {
        let target_allele = target_genotypes[marker_in_window];
        let recomb_rate = if marker_in_window == 0 {
            0.0
        } else {
            block.local_recomb_rates[marker_in_window - 1]
        };

        let emissions = &mut ws.emissions;
        
        for pattern_idx in 0..n_patterns {
            let pattern_id = PatternId::new(pattern_idx as u32);
            emissions[pattern_idx] = emission_prob(
                block,
                pattern_id,
                marker_in_window,
                target_allele,
                error_rate,
            );
        }

        let fwd_sum = ws.fwd[..n_patterns].iter().sum::<f32>() + ws.reservoir_prob_fwd;

        WeightedHmmUpdater::fwd_update_weighted(
            &mut ws.fwd,
            fwd_sum,
            recomb_rate,
            block.n_ref_haps(),
            &block.pattern_counts,
            emissions,
            n_patterns,
        );

        if block.reservoir_count > 0 {
            let reservoir_emission = emission_prob(
                block,
                PatternId::RESERVOIR,
                marker_in_window,
                target_allele,
                error_rate,
            );

            let total_mass = fwd_sum;
            let background = total_mass * recomb_rate / block.n_ref_haps() as f32;
            let stay = ws.reservoir_prob_fwd * (1.0 - recomb_rate);

            ws.reservoir_prob_fwd =
                reservoir_emission * (stay + background * block.reservoir_count as f32);
        }

        ws.normalize_forward(n_patterns);
    }
}

/// Backward pass within block AND emit posteriors
///
/// Combines forward probabilities (from checkpoint) with backward probabilities
/// to compute posterior probabilities for each allele.
///
/// # Arguments
/// * `output` - Mutable slice to write posteriors into (must be same length as window_size)
pub fn backward_and_emit_block(
    block: &CompressedBlock,
    target_genotypes: &[u8],
    error_rate: f32,
    ws: &mut BlockHmmWorkspace,
    output: &mut [AllelePosteriors],
) {
    let n_patterns = block.n_patterns();
    let window_size = block.window_size();
    
    assert_eq!(output.len(), window_size, "Output slice size mismatch");
    
    // We compute posteriors in reverse order (because backward pass is reverse).
    // The `output` slice corresponds to markers [start..end]. 
    // We will write into `output[marker_idx]` directly.
    // Care needed: backward loop is `(0..window_size).rev()`.

    // Re-run Forward and store history into pre-allocated workspace buffer
    for marker_idx in 0..window_size {
        let target_allele = target_genotypes[marker_idx];
        let recomb_rate = if marker_idx == 0 {
            0.0
        } else {
            block.local_recomb_rates[marker_idx - 1]
        };
        let n_alleles = block.n_alleles(marker_idx);
        
        let emissions = &mut ws.emissions;
        for pattern_idx in 0..n_patterns {
            emissions[pattern_idx] = emission_prob(
                block,
                PatternId::new(pattern_idx as u32),
                marker_idx,
                target_allele,
                error_rate,
                n_alleles,
            );
        }
        
        let fwd_sum = ws.fwd[..n_patterns].iter().sum::<f32>() + ws.reservoir_prob_fwd;
        
        WeightedHmmUpdater::fwd_update_weighted(
            &mut ws.fwd,
            fwd_sum,
            recomb_rate,
            block.n_ref_haps(),
            &block.pattern_counts,
            emissions,
            n_patterns,
        );

        if block.reservoir_count > 0 {
             let reservoir_emission = emission_prob(
                block,
                PatternId::RESERVOIR,
                marker_idx,
                target_allele,
                error_rate,
                n_alleles,
            );
            let total_mass = fwd_sum;
            let background = total_mass * recomb_rate / block.n_ref_haps() as f32;
            let stay = ws.reservoir_prob_fwd * (1.0 - recomb_rate);
            ws.reservoir_prob_fwd = reservoir_emission * (stay + background * block.reservoir_count as f32);
        }
        
        ws.normalize_forward(n_patterns);
        
        // Save state to flattened history
        // Access fields directly to avoid borrowing `self` entirely
        let stride = ws.max_states + 1;
        let start = marker_idx * stride;
        let history = &mut ws.fwd_history[start..start + stride];
        
        history[..n_patterns].copy_from_slice(&ws.fwd[..n_patterns]);
        history[n_patterns] = ws.reservoir_prob_fwd;
    }
    
    // Now Backward (reverse)
    for marker_idx in (0..window_size).rev() {
        let target_allele = target_genotypes[marker_idx];
        let recomb_rate = if marker_idx == 0 {
            0.0
        } else {
            block.local_recomb_rates[marker_idx - 1]
        };
        let n_alleles = block.n_alleles(marker_idx);
        
        // Accumulate allele probabilities
        let mut allele_probs = vec![0.0f32; n_alleles];
        let mut total_prob = 0.0;
        
        let stride = ws.max_states + 1;
        let start = marker_idx * stride;
        let current_fwd = &ws.fwd_history[start..start + stride];
        
        // Pattern states
        for pattern_idx in 0..n_patterns {
            let p = current_fwd[pattern_idx] * ws.bwd[pattern_idx];
            if p > 0.0 {
                total_prob += p;
                let allele = block.pattern_allele(pattern_idx_to_id(pattern_idx), marker_idx) as usize;
                if allele < n_alleles {
                    allele_probs[allele] += p;
                }
            }
        }
        
        // Reservoir state (Mixed alleles)
        let res_p = current_fwd[n_patterns] * ws.reservoir_prob_bwd;
        if res_p > 0.0 {
            total_prob += res_p;
            // Iterate all alleles to distribute probability based on frequency
            for allele in 0..n_alleles {
                let freq = block.reservoir_freq(marker_idx, allele as u8);
                if freq > 0.0 {
                    allele_probs[allele] += res_p * freq;
                }
            }
        }
        
        // Normalize
        if total_prob > 0.0 {
            let scale = 1.0 / total_prob;
            for p in &mut allele_probs {
                *p *= scale;
            }
        }
        
        if n_alleles == 2 {
            output[marker_idx] = AllelePosteriors::Biallelic(allele_probs[1]);
        } else {
            output[marker_idx] = AllelePosteriors::Multiallelic(allele_probs);
        }
        
        // Update bwd to t-1
        let emissions = &mut ws.emissions;
        for pattern_idx in 0..n_patterns {
            emissions[pattern_idx] = emission_prob(
                block,
                PatternId::new(pattern_idx as u32),
                marker_idx,
                target_allele,
                error_rate,
                n_alleles,
            );
        }
        
        // beta = beta * emit
        for i in 0..n_patterns {
            ws.bwd[i] *= emissions[i];
        }
        
        let reservoir_emission = emission_prob(
            block,
            PatternId::RESERVOIR,
            marker_idx,
            target_allele,
            error_rate,
            n_alleles,
        );
        ws.reservoir_prob_bwd *= reservoir_emission;
        
        // Pure transition step
        emissions.fill(1.0);
        let bwd_sum = ws.bwd[..n_patterns].iter().sum::<f32>() + ws.reservoir_prob_bwd;
        WeightedHmmUpdater::fwd_update_weighted(
            &mut ws.bwd,
            bwd_sum,
            recomb_rate,
            block.n_ref_haps(),
            &block.pattern_counts,
            emissions,
            n_patterns,
        );
        
        let total_mass = bwd_sum;
        let background = total_mass * recomb_rate / block.n_ref_haps() as f32;
        let stay = ws.reservoir_prob_bwd * (1.0 - recomb_rate);
        ws.reservoir_prob_bwd = stay + background * block.reservoir_count as f32;
        
        ws.normalize_bwd(n_patterns);
    }
}

/// Compute emission probability for a pattern at a marker
///
/// Implements Split Error Model: Mismatch probability is split among all non-matching alleles.
#[inline]
fn emission_prob(
    block: &CompressedBlock,
    pattern_id: PatternId,
    marker_in_window: usize,
    target_allele: u8,
    error_rate: f32,
    n_alleles: usize,
) -> f32 {
    // Missing data - neutral (1.0)
    if target_allele == 255 {
        return 1.0;
    }

    // Split error model: epsilon / (K - 1)
    // If K=1 (monomorphic), mismatch is impossible (or probability 0).
    // We clamp divisor to 1.0 to avoid NaN, but logically K=1 should match always or error if observed is different.
    let mismatch_prob = if n_alleles > 1 {
        error_rate / (n_alleles - 1) as f32
    } else {
        error_rate // Fallback, though K=1 implies no variation
    };

    let match_prob = 1.0 - error_rate;

    if pattern_id.is_reservoir() {
        // Reservoir uses allele frequency
        // P(obs | Res) = sum_k [ P(obs | k) * P(k | Res) ]
        //              = P(match) * freq[obs] + sum_{k != obs} [ P(obs | k) * freq[k] ]
        //              = match_prob * freq[obs] + mismatch_prob * (1 - freq[obs])
        
        let freq_obs = block.reservoir_freq(marker_in_window, target_allele);
        match_prob * freq_obs + mismatch_prob * (1.0 - freq_obs)
    } else {
        // Pattern uses exact allele matching
        // pattern_allele returns f32, convert to u8. 
        // Note: pattern_allele call is safe for non-reservoir patterns.
        let ref_allele = block.pattern_allele(pattern_id, marker_in_window) as u8;

        if target_allele == ref_allele {
            match_prob
        } else {
            mismatch_prob
        }
    }
}

fn pattern_idx_to_id(idx: usize) -> PatternId {
    PatternId::new(idx as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::block_hash::compression::build_compressed_block;
    use crate::data::storage::{GenotypeColumn, GenotypeMatrix};
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Marker, Allele, Markers};
    use std::sync::Arc;

    /// Ported/Adapted from deleted `test_compute_cluster_mismatches_accumulation`
    #[test]
    fn test_hmm_accumulate_penalties_weighted() {
        // Scenario:
        // Haplotype 0: (0, 0) -> Matches target (0, 0)
        // Haplotype 1: (0, 1) -> Mismatch at pos 1
        // Haplotype 2: (1, 1) -> Mismatch at pos 0 and 1
        //
        // Target: (0, 0)
        // Error rate: 0.01
        
        let n_haps = 3;
        // let n_markers = 2; // Unused
        
        let col0 = GenotypeColumn::from_alleles(&[0, 0, 1], 2);
        let col1 = GenotypeColumn::from_alleles(&[0, 1, 1], 2);
        
        let mut markers = Markers::new();
        let chr = markers.add_chrom("1");
        markers.push(Marker::new(chr, 100, None, Allele::Base(0), vec![Allele::Base(1)]));
        markers.push(Marker::new(chr, 200, None, Allele::Base(0), vec![Allele::Base(1)]));

        let samples = Arc::new(Samples::from_ids(vec!["H0".to_string(), "H1".to_string()])); // Dummy
        let gt = GenotypeMatrix::new_phased(markers, vec![col0, col1], samples);
        
        let rates = vec![0.0, 0.0]; // No recombination to isolate emissions
        let block = build_compressed_block(&gt, 0..2, 0, &rates);
        
        // Haps are distinct, so we expect 3 patterns?
        // 0,0 -> P0 (count 1)
        // 0,1 -> P1 (count 1)
        // 1,1 -> P2 (count 1)
        assert_eq!(block.n_patterns(), 3);
        let p0 = block.pattern_for_haplotype(crate::model::block_hash::GlobalId::new(0)); // 0,0
        let p1 = block.pattern_for_haplotype(crate::model::block_hash::GlobalId::new(1)); // 0,1
        let p2 = block.pattern_for_haplotype(crate::model::block_hash::GlobalId::new(2)); // 1,1
        
        let mut ws = BlockHmmWorkspace::new(10);
        // Start uniform
        ws.fwd.fill(0.0);
        ws.fwd[p0.as_usize()] = 1.0/3.0;
        ws.fwd[p1.as_usize()] = 1.0/3.0;
        ws.fwd[p2.as_usize()] = 1.0/3.0;
        
        let target_genotypes = vec![0, 0];
        let error = 0.01;
        let match_prob = 1.0 - error;
        let mismatch_prob = error;
        
        // Run forward pass
        forward_within_block(&block, &target_genotypes, error, &mut ws);
        
        // Expected probs (ignoring normalization for a moment, or rather checking ratios)
        // P0 (0 mismatches): Init * match * match
        // P1 (1 mismatch):   Init * match * mismatch
        // P2 (2 mismatches): Init * mismatch * mismatch
        
        let prob0 = ws.fwd[p0.as_usize()];
        let prob1 = ws.fwd[p1.as_usize()];
        let prob2 = ws.fwd[p2.as_usize()];
        
        // Ratios should reflect penalty accumulation
        // prob1 / prob0 ~ mismatch/match
        let ratio1 = prob1 / prob0;
        let expected1 = mismatch_prob / match_prob;
        
        assert!((ratio1 - expected1).abs() < 1e-4, "Hap 1 should be penalized once. Got ratio {}, expected {}", ratio1, expected1);
        
        // prob2 / prob0 ~ (mismatch/match)^2
        let ratio2 = prob2 / prob0;
        let expected2 = (mismatch_prob / match_prob).powi(2);
        assert!((ratio2 - expected2).abs() < 1e-5, "Hap 2 should be penalized twice. Got ratio {}, expected {}", ratio2, expected2);
    }
}