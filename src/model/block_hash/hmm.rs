//! # HMM Kernel: Reuse Existing AVX-512 Optimized Code
//!
//! This module integrates the block-hash HMM with Reagle's existing
//! SIMD-optimized HMM kernels instead of writing new scalar loops.

use super::compressed_block::CompressedBlock;
use super::workspace::BlockHmmWorkspace;
use super::types::PatternId;
use super::weighted_kernel::WeightedHmmUpdater;
use crate::pipelines::imputation::AllelePosteriors;
use crate::data::storage::GenotypeMatrix;
use crate::data::storage::phase_state::PhaseState;
use crate::data::marker::MarkerIdx;
use crate::data::alignment::MarkerAlignment;
use crate::model::pl_emission::{allele_probs_uncond_from_pl, emit_from_allele_probs};

/// Provides per-marker emission probabilities
pub struct EmissionProvider<'a, S: PhaseState> {
    pub gt: &'a GenotypeMatrix<S>,
    pub sample: usize,
    pub alignment: &'a MarkerAlignment,
}

impl<'a, S: PhaseState> EmissionProvider<'a, S> {}

/// Helper for pre-computing emissions
fn compute_allele_emission_cache<S: PhaseState>(
    provider: &EmissionProvider<S>,
    global_marker: usize,
    target_allele: u8,
    n_alleles: usize,
    error_rate: f32,
    allele_probs_buffer: &mut Vec<f32>,
) -> Option<Vec<f32>> {
    let mut per_allele_emissions = vec![0.0; n_alleles];
    let mut using_provider = false;

    let target_marker_idx = provider.alignment.target_marker(global_marker);
    if let Some(t_idx) = target_marker_idx {
        let pl = provider.gt.sample_pl(MarkerIdx::new(t_idx as u32), provider.sample);
        if let Some(pl_values) = pl {
             if !pl_values.is_empty() {
                 if allele_probs_uncond_from_pl(pl_values, allele_probs_buffer).is_some() {
                     let p_no_err = 1.0 - error_rate;
                     let p_err_other = if allele_probs_buffer.len() > 1 {
                         error_rate / (allele_probs_buffer.len() as f32 - 1.0)
                     } else {
                         0.0
                     };

                     for r in 0..n_alleles {
                         let target_a = provider.alignment.reverse_map_allele(t_idx, r as u8);
                         per_allele_emissions[r] = emit_from_allele_probs(target_a, allele_probs_buffer, p_no_err, p_err_other);
                     }
                     using_provider = true;
                 }
             }
        }

        if !using_provider {
            // Try confidence
            let conf = provider.gt.sample_confidence_f32(MarkerIdx::new(t_idx as u32), provider.sample);
            if conf < 0.999 {
                let random_prob = if n_alleles > 0 { 1.0 / n_alleles as f32 } else { 0.5 };
                let mismatch_prob = if n_alleles > 1 { error_rate / (n_alleles - 1) as f32 } else { error_rate };

                for r in 0..n_alleles {
                    let base_prob = if r as u8 == target_allele { 1.0 - error_rate } else { mismatch_prob };
                    per_allele_emissions[r] = base_prob * conf + random_prob * (1.0 - conf);
                }
                using_provider = true;
            }
        }
    }

    if using_provider {
        Some(per_allele_emissions)
    } else {
        None
    }
}

/// Run forward pass within a single block using existing SIMD kernel
///
/// # Arguments
/// * `block` - The CompressedBlock (immutable reference data)
/// * `target_genotypes` - Target genotypes for this block (Hard calls, Ref-encoded)
/// * `error_rate` - Genotyping error rate
/// * `ws` - Mutable workspace
pub fn forward_within_block<S: PhaseState>(
    block: &CompressedBlock,
    target_genotypes: &[u8],
    error_rate: f32,
    ws: &mut BlockHmmWorkspace,
    emission_provider: Option<&EmissionProvider<S>>,
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
        let recomb_rate = if marker_in_window == 0 {
            0.0
        } else {
            block.local_recomb_rates[marker_in_window - 1]
        };

        let emissions = &mut ws.emissions;
        
        let n_alleles = block.n_alleles(marker_in_window);
        
        let allele_emission_cache = if let Some(ep) = emission_provider {
            compute_allele_emission_cache(
                ep,
                block.start_marker + marker_in_window,
                target_allele,
                n_alleles,
                error_rate,
                &mut ws.allele_probs,
            )
        } else {
            None
        };
        let emission_cache = allele_emission_cache.as_deref();

        for pattern_idx in 0..n_patterns {
            let pattern_id = PatternId::new(pattern_idx as u32);
            emissions[pattern_idx] = emission_prob(
                block,
                pattern_id,
                marker_in_window,
                target_allele,
                error_rate,
                n_alleles,
                emission_cache
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
                n_alleles,
                emission_cache
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

/// Run forward pass up to a specific marker within a block
pub fn forward_to_marker_in_block<S: PhaseState>(
    block: &CompressedBlock,
    target_genotypes: &[u8],
    error_rate: f32,
    ws: &mut BlockHmmWorkspace,
    stop_marker_in_window: usize,
    emission_provider: Option<&EmissionProvider<S>>,
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
        
        let n_alleles = block.n_alleles(marker_in_window);
        
        let allele_emission_cache = if let Some(ep) = emission_provider {
            compute_allele_emission_cache(
                ep,
                block.start_marker + marker_in_window,
                target_allele,
                n_alleles,
                error_rate,
                &mut ws.allele_probs,
            )
        } else {
            None
        };
        let emission_cache = allele_emission_cache.as_deref();

        for pattern_idx in 0..n_patterns {
            let pattern_id = PatternId::new(pattern_idx as u32);
            emissions[pattern_idx] = emission_prob(
                block,
                pattern_id,
                marker_in_window,
                target_allele,
                error_rate,
                n_alleles,
                emission_cache
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
                n_alleles,
                emission_cache
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
pub fn backward_and_emit_block<S: PhaseState>(
    block: &CompressedBlock,
    target_genotypes: &[u8],
    error_rate: f32,
    ws: &mut BlockHmmWorkspace,
    output: &mut [AllelePosteriors],
    emission_provider: Option<&EmissionProvider<S>>,
) {
    let n_patterns = block.n_patterns();
    let window_size = block.window_size();
    
    assert_eq!(output.len(), window_size, "Output slice size mismatch");
    
    // Re-run Forward
    for marker_idx in 0..window_size {
        let target_allele = target_genotypes[marker_idx];
        let recomb_rate = if marker_idx == 0 { 0.0 } else { block.local_recomb_rates[marker_idx - 1] };
        let n_alleles = block.n_alleles(marker_idx);
        
        let allele_emission_cache = if let Some(ep) = emission_provider {
            compute_allele_emission_cache(
                ep,
                block.start_marker + marker_idx,
                target_allele,
                n_alleles,
                error_rate,
                &mut ws.allele_probs,
            )
        } else {
            None
        };
        let emission_cache = allele_emission_cache.as_deref();

        let emissions = &mut ws.emissions;
        for pattern_idx in 0..n_patterns {
            emissions[pattern_idx] = emission_prob(
                block,
                PatternId::new(pattern_idx as u32),
                marker_idx,
                target_allele,
                error_rate,
                n_alleles,
                emission_cache
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
                emission_cache
            );
            let total_mass = fwd_sum;
            let background = total_mass * recomb_rate / block.n_ref_haps() as f32;
            let stay = ws.reservoir_prob_fwd * (1.0 - recomb_rate);
            ws.reservoir_prob_fwd = reservoir_emission * (stay + background * block.reservoir_count as f32);
        }
        
        ws.normalize_forward(n_patterns);
        
        // Save state to flattened history
        let stride = ws.max_states + 1;
        let start = marker_idx * stride;
        let history = &mut ws.fwd_history[start..start + stride];
        history[..n_patterns].copy_from_slice(&ws.fwd[..n_patterns]);
        history[n_patterns] = ws.reservoir_prob_fwd;
    }
    
    // Now Backward (reverse)
    for marker_idx in (0..window_size).rev() {
        let target_allele = target_genotypes[marker_idx];
        let recomb_rate = if marker_idx == 0 { 0.0 } else { block.local_recomb_rates[marker_idx - 1] };
        let n_alleles = block.n_alleles(marker_idx);
        
        // Compute emission cache again
        let allele_emission_cache = if let Some(ep) = emission_provider {
            compute_allele_emission_cache(
                ep,
                block.start_marker + marker_idx,
                target_allele,
                n_alleles,
                error_rate,
                &mut ws.allele_probs,
            )
        } else {
            None
        };
        let emission_cache = allele_emission_cache.as_deref();

        // Accumulate allele probabilities
        let mut allele_probs = vec![0.0f32; n_alleles];
        let mut total_prob = 0.0;
        
        let stride = ws.max_states + 1;
        let start = marker_idx * stride;
        let current_fwd = &ws.fwd_history[start..start + stride];
        
        for pattern_idx in 0..n_patterns {
            let p = current_fwd[pattern_idx] * ws.bwd[pattern_idx];
            if p > 0.0 {
                total_prob += p;
                let allele = block.get_pattern_allele(pattern_idx_to_id(pattern_idx), marker_idx) as usize;
                if allele < n_alleles {
                    allele_probs[allele] += p;
                }
            }
        }
        
        let res_p = current_fwd[n_patterns] * ws.reservoir_prob_bwd;
        if res_p > 0.0 {
            total_prob += res_p;
            for allele in 0..n_alleles {
                let freq = block.reservoir_freq(marker_idx, allele as u8);
                if freq > 0.0 {
                    allele_probs[allele] += res_p * freq;
                }
            }
        }
        
        if total_prob > 0.0 {
            let scale = 1.0 / total_prob;
            for p in &mut allele_probs { *p *= scale; }
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
                emission_cache
            );
        }
        
        for i in 0..n_patterns { ws.bwd[i] *= emissions[i]; }
        
        let reservoir_emission = emission_prob(
            block,
            PatternId::RESERVOIR,
            marker_idx,
            target_allele,
            error_rate,
            n_alleles,
            emission_cache
        );
        ws.reservoir_prob_bwd *= reservoir_emission;
        
        {
            let n_ref = block.n_ref_haps() as f32;
            let mut weighted_sum = 0.0f32;
            for i in 0..n_patterns {
                weighted_sum += ws.bwd[i] * block.pattern_counts[i];
            }
            if block.reservoir_count > 0 {
                weighted_sum += ws.reservoir_prob_bwd * block.reservoir_count as f32;
            }
            let constant_term = weighted_sum / n_ref;

            let stay_prob = 1.0 - recomb_rate;
            let recomb_prob = recomb_rate;
            let common_add = recomb_prob * constant_term;
            
            for i in 0..n_patterns {
                ws.bwd[i] = ws.bwd[i] * stay_prob + common_add;
            }
            ws.reservoir_prob_bwd = ws.reservoir_prob_bwd * stay_prob + common_add;
        }
        
        ws.normalize_bwd(n_patterns);
    }
}

/// Compute emission probability for a pattern at a marker
#[inline]
fn emission_prob(
    block: &CompressedBlock,
    pattern_id: PatternId,
    marker_in_window: usize,
    target_allele: u8,
    error_rate: f32,
    n_alleles: usize,
    emission_cache: Option<&[f32]>,
) -> f32 {
    // If we have pre-computed emissions (from PL or Confidence), use them
    if let Some(cache) = emission_cache {
        if pattern_id.is_reservoir() {
            // Weighted average based on reservoir frequencies
            // P(obs | Res) = sum_k P(obs | k) * P(k | Res)
            // = sum_k cache[k] * freq[k]

            let obs_fraction = block.get_reservoir_obs_fraction(marker_in_window);
            if obs_fraction > 0.0 {
                let mut p = 0.0f32;
                for a in 0..n_alleles {
                    let freq = block.reservoir_freq(marker_in_window, a as u8);
                    if freq > 0.0 {
                        p += cache.get(a).copied().unwrap_or(0.0) * freq;
                    }
                }
                // Account for missingness: P(miss|Res) = 1.0?
                // cache[k] is P(obs | k).
                // If reservoir has missing data, we assume missing matches (1.0).
                // Mixed: obs_fraction * (sum...) + (1-obs_fraction) * 1.0
                return p * obs_fraction + 1.0 * (1.0 - obs_fraction);
            } else {
                return 1.0;
            }
        } else {
            let ref_allele = block.get_pattern_allele(pattern_id, marker_in_window);
            if ref_allele == 255 {
                return 1.0;
            }
            return cache.get(ref_allele as usize).copied().unwrap_or(error_rate);
        }
    }

    // Fallback to Hard-Call logic
    // Missing data - neutral (1.0)
    if target_allele == 255 {
        return 1.0;
    }

    let mismatch_prob = if n_alleles > 1 {
        error_rate / (n_alleles - 1) as f32
    } else {
        error_rate
    };

    let match_prob = 1.0 - error_rate;

    if pattern_id.is_reservoir() {
        let obs_fraction = block.get_reservoir_obs_fraction(marker_in_window);
        
        if obs_fraction > 0.0 {
            let freq_obs = block.reservoir_freq(marker_in_window, target_allele);
            let p_given_observed = match_prob * freq_obs + mismatch_prob * (1.0 - freq_obs);
            
            p_given_observed * obs_fraction + 1.0 * (1.0 - obs_fraction)
        } else {
            1.0
        }
    } else {
        let ref_allele = block.get_pattern_allele(pattern_id, marker_in_window);

        if ref_allele == 255 {
            return 1.0;
        }

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
    use crate::data::storage::phase_state::Phased;
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

        let col0 = GenotypeColumn::from_alleles(&[0, 0, 1], 2);
        let col1 = GenotypeColumn::from_alleles(&[0, 1, 1], 2);
        
        let mut markers = Markers::new();
        let chr = markers.add_chrom("1");
        markers.push(Marker::new(chr, 100, None, Allele::Base(0), vec![Allele::Base(1)]));
        markers.push(Marker::new(chr, 200, None, Allele::Base(0), vec![Allele::Base(1)]));

        let samples = Arc::new(Samples::from_ids(vec!["H0".to_string(), "H1".to_string()])); // Dummy
        let gt = GenotypeMatrix::new_phased(markers, vec![col0, col1], samples);
        
        let rates = vec![0.0]; // No recombination to isolate emissions
        let block = build_compressed_block(&gt, 0..2, 0, &rates);
        
        // Haps are distinct, so we expect 3 patterns?
        // 0,0 -> P0 (count 1)
        // 0,1 -> P1 (count 1)
        // 1,1 -> P2 (count 1)
        assert_eq!(block.n_patterns(), 3);
        let p0 = block.pattern_for_haplotype(crate::model::block_hash::types::GlobalId::new(0)); // 0,0
        let p1 = block.pattern_for_haplotype(crate::model::block_hash::types::GlobalId::new(1)); // 0,1
        let p2 = block.pattern_for_haplotype(crate::model::block_hash::types::GlobalId::new(2)); // 1,1
        
        let mut ws = BlockHmmWorkspace::new(10, 1, 2);
        // Start uniform
        ws.fwd.fill(0.0);
        ws.fwd[p0.as_usize()] = 1.0/3.0;
        ws.fwd[p1.as_usize()] = 1.0/3.0;
        ws.fwd[p2.as_usize()] = 1.0/3.0;
        
        let target_genotypes = vec![0, 0];
        let error = 0.01;
        let match_prob = 1.0 - error;
        let mismatch_prob = error;
        
        // Run forward pass with NO emission provider (Hard Call logic)
        let provider: Option<&EmissionProvider<Phased>> = None;
        forward_within_block(&block, &target_genotypes, error, &mut ws, provider);
        
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
