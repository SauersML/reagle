//! # HMM Kernel: Reuse Existing AVX-512 Optimized Code
//!
//! This module integrates the block-hash HMM with Reagle's existing
//! SIMD-optimized HMM kernels instead of writing new scalar loops.

use super::compressed_block::CompressedBlock;
use super::types::PatternId;
use super::weighted_kernel::WeightedHmmUpdater;
use super::workspace::BlockHmmWorkspace;
use crate::pipelines::imputation::AllelePosteriors;

/// Per-marker allele probability distributions for a single haplotype.
pub struct TargetAlleleProbs {
    offsets: Vec<usize>,
    probs: Vec<f32>,
}

impl TargetAlleleProbs {
    pub fn new(offsets: Vec<usize>, probs: Vec<f32>) -> Self {
        Self { offsets, probs }
    }

    #[inline]
    pub fn probs_for_marker(&self, marker_idx: usize) -> &[f32] {
        let start = self.offsets[marker_idx];
        let end = self.offsets[marker_idx + 1];
        &self.probs[start..end]
    }

    #[inline]
    pub fn n_markers(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }
}

/// View into TargetAlleleProbs for a block-local marker range.
pub struct TargetAlleleProbsView<'a> {
    probs: &'a TargetAlleleProbs,
    start_marker: usize,
}

impl<'a> TargetAlleleProbsView<'a> {
    pub fn new(probs: &'a TargetAlleleProbs, start_marker: usize) -> Self {
        Self {
            probs,
            start_marker,
        }
    }

    #[inline]
    pub fn probs_for_marker(&self, marker_in_window: usize) -> &[f32] {
        self.probs
            .probs_for_marker(self.start_marker + marker_in_window)
    }
}

/// Run forward pass within a single block using soft allele probabilities.
pub fn forward_within_block_probs(
    block: &CompressedBlock,
    target_probs: &TargetAlleleProbsView<'_>,
    error_rate: f32,
    ws: &mut BlockHmmWorkspace,
    initial_recomb_rate: f32,
) {
    let n_patterns = block.n_patterns();
    let window_size = block.window_size();

    // For each marker in the window
    let mut prev_probs: &[f32] = &[];
    for marker_in_window in 0..window_size {
        let probs = target_probs.probs_for_marker(marker_in_window);
        let recomb_rate = if marker_in_window == 0 {
            initial_recomb_rate
        } else {
            block.local_recomb_rates[marker_in_window - 1]
        };

        let emissions = &mut ws.emissions;
        fill_emissions_for_marker_probs(block, marker_in_window, probs, error_rate, emissions);

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
            let reservoir_emission = emission_prob_soft(
                block,
                PatternId::RESERVOIR,
                marker_in_window,
                probs,
                error_rate,
                block.n_alleles(marker_in_window),
            );

            let total_mass = fwd_sum;
            let background = total_mass * recomb_rate / block.n_ref_haps() as f32;
            let coherence = reservoir_coherence(block, marker_in_window, prev_probs, probs);
            let stay = ws.reservoir_prob_fwd * (1.0 - recomb_rate) * coherence;
            ws.reservoir_prob_fwd =
                reservoir_emission * (stay + background * block.reservoir_count as f32);
        }

        ws.normalize_forward(n_patterns);
        prev_probs = probs;
    }
}

/// Run forward pass to a marker within a block using soft allele probabilities.
pub fn forward_to_marker_in_block_probs(
    block: &CompressedBlock,
    target_probs: &TargetAlleleProbsView<'_>,
    error_rate: f32,
    ws: &mut BlockHmmWorkspace,
    stop_marker_in_window: usize,
    initial_recomb_rate: f32,
) {
    let n_patterns = block.n_patterns();
    let window_size = block.window_size();

    assert!(stop_marker_in_window < window_size);

    let mut prev_probs: &[f32] = &[];
    for marker_in_window in 0..=stop_marker_in_window {
        let probs = target_probs.probs_for_marker(marker_in_window);
        let recomb_rate = if marker_in_window == 0 {
            initial_recomb_rate
        } else {
            block.local_recomb_rates[marker_in_window - 1]
        };

        let emissions = &mut ws.emissions;
        fill_emissions_for_marker_probs(block, marker_in_window, probs, error_rate, emissions);

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
            let reservoir_emission = emission_prob_soft(
                block,
                PatternId::RESERVOIR,
                marker_in_window,
                probs,
                error_rate,
                block.n_alleles(marker_in_window),
            );

            let total_mass = fwd_sum;
            let background = total_mass * recomb_rate / block.n_ref_haps() as f32;
            let coherence = reservoir_coherence(block, marker_in_window, prev_probs, probs);
            let stay = ws.reservoir_prob_fwd * (1.0 - recomb_rate) * coherence;
            ws.reservoir_prob_fwd =
                reservoir_emission * (stay + background * block.reservoir_count as f32);
        }

        ws.normalize_forward(n_patterns);
        prev_probs = probs;
    }
}

/// Backward pass within block AND emit posteriors (soft allele probabilities).
pub fn backward_and_emit_block_probs(
    block: &CompressedBlock,
    target_probs: &TargetAlleleProbsView<'_>,
    error_rate: f32,
    ws: &mut BlockHmmWorkspace,
    output: &mut [AllelePosteriors],
    initial_recomb_rate: f32,
) {
    let n_patterns = block.n_patterns();
    let window_size = block.window_size();

    assert_eq!(output.len(), window_size, "Output slice size mismatch");

    let mut prev_probs: &[f32] = &[];
    for marker_idx in 0..window_size {
        let probs = target_probs.probs_for_marker(marker_idx);
        let recomb_rate = if marker_idx == 0 {
            initial_recomb_rate
        } else {
            block.local_recomb_rates[marker_idx - 1]
        };
        let n_alleles = block.n_alleles(marker_idx);

        let emissions = &mut ws.emissions;
        fill_emissions_for_marker_probs(block, marker_idx, probs, error_rate, emissions);

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
            let reservoir_emission = emission_prob_soft(
                block,
                PatternId::RESERVOIR,
                marker_idx,
                probs,
                error_rate,
                n_alleles,
            );
            let total_mass = fwd_sum;
            let background = total_mass * recomb_rate / block.n_ref_haps() as f32;
            let coherence = reservoir_coherence(block, marker_idx, prev_probs, probs);
            let stay = ws.reservoir_prob_fwd * (1.0 - recomb_rate) * coherence;
            ws.reservoir_prob_fwd =
                reservoir_emission * (stay + background * block.reservoir_count as f32);
        }

        ws.normalize_forward(n_patterns);

        let stride = ws.max_states + 1;
        let start = marker_idx * stride;
        let history = &mut ws.fwd_history[start..start + stride];

        history[..n_patterns].copy_from_slice(&ws.fwd[..n_patterns]);
        history[n_patterns] = ws.reservoir_prob_fwd;
        prev_probs = probs;
    }

    let mut next_probs: &[f32] = &[];
    for marker_idx in (0..window_size).rev() {
        let probs = target_probs.probs_for_marker(marker_idx);
        let recomb_rate = if marker_idx == 0 {
            initial_recomb_rate
        } else {
            block.local_recomb_rates[marker_idx - 1]
        };
        let n_alleles = block.n_alleles(marker_idx);

        let mut allele_probs = vec![0.0f32; n_alleles];
        let mut observed_mass = 0.0f32;

        let stride = ws.max_states + 1;
        let start = marker_idx * stride;
        let current_fwd = &ws.fwd_history[start..start + stride];

        for pattern_idx in 0..n_patterns {
            let p = current_fwd[pattern_idx] * ws.bwd[pattern_idx];
            if p > 0.0 {
                let allele =
                    block.get_pattern_allele(pattern_idx_to_id(pattern_idx), marker_idx) as usize;
                if allele < n_alleles {
                    allele_probs[allele] += p;
                    observed_mass += p;
                }
            }
        }

        let res_p = current_fwd[n_patterns] * ws.reservoir_prob_bwd;
        if res_p > 0.0 {
            let match_prob = 1.0 - error_rate;
            let mismatch_prob = if n_alleles > 1 {
                error_rate / (n_alleles - 1) as f32
            } else {
                error_rate
            };
            let obs_fraction = block.get_reservoir_obs_fraction(marker_idx);
            let mut denom = 0.0f32;
            if probs.is_empty() {
                for allele in 0..n_alleles {
                    let freq = block.reservoir_freq(marker_idx, allele as u8);
                    denom += freq;
                }
            } else {
                for allele in 0..n_alleles {
                    let freq = block.reservoir_freq(marker_idx, allele as u8);
                    if freq <= 0.0 {
                        continue;
                    }
                    let p_obs = probs.get(allele).copied().unwrap_or(0.0);
                    let emit_obs = mismatch_prob + (match_prob - mismatch_prob) * p_obs;
                    denom += freq * (obs_fraction * emit_obs + (1.0 - obs_fraction));
                }
            }
            let mut res_weight_sum = 0.0f32;
            if denom > 0.0 {
                for allele in 0..n_alleles {
                    let freq = block.reservoir_freq(marker_idx, allele as u8);
                    if freq <= 0.0 {
                        continue;
                    }
                    let weight = if probs.is_empty() {
                        freq / denom
                    } else {
                        let p_obs = probs.get(allele).copied().unwrap_or(0.0);
                        let emit_obs = mismatch_prob + (match_prob - mismatch_prob) * p_obs;
                        freq * (obs_fraction * emit_obs + (1.0 - obs_fraction)) / denom
                    };
                    let w = res_p * weight;
                    allele_probs[allele] += w;
                    res_weight_sum += w;
                }
            } else if n_alleles > 0 {
                let uniform = res_p / n_alleles as f32;
                for allele in 0..n_alleles {
                    allele_probs[allele] += uniform;
                }
                res_weight_sum = res_p;
            }
            observed_mass += res_weight_sum;
        }

        if observed_mass > 0.0 {
            let scale = 1.0 / observed_mass;
            for p in &mut allele_probs {
                *p *= scale;
            }
        } else if n_alleles > 0 {
            if block.reservoir_count > 0 {
                let mut sum = 0.0f32;
                for allele in 0..n_alleles {
                    let freq = block.reservoir_freq(marker_idx, allele as u8);
                    allele_probs[allele] = freq;
                    sum += freq;
                }
                if sum > 0.0 {
                    let scale = 1.0 / sum;
                    for p in &mut allele_probs {
                        *p *= scale;
                    }
                } else {
                    let uniform = 1.0 / n_alleles as f32;
                    allele_probs.fill(uniform);
                }
            } else {
                let uniform = 1.0 / n_alleles as f32;
                allele_probs.fill(uniform);
            }
        }

        if n_alleles == 2 {
            output[marker_idx] = AllelePosteriors::Biallelic(allele_probs[1]);
        } else {
            output[marker_idx] = AllelePosteriors::Multiallelic(allele_probs);
        }

        let emissions = &mut ws.emissions;
        fill_emissions_for_marker_probs(block, marker_idx, probs, error_rate, emissions);

        for i in 0..n_patterns {
            ws.bwd[i] *= emissions[i];
        }

        let reservoir_emission = emission_prob_soft(
            block,
            PatternId::RESERVOIR,
            marker_idx,
            probs,
            error_rate,
            n_alleles,
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
            let coherence = reservoir_coherence_backward(block, marker_idx, probs, next_probs);
            ws.reservoir_prob_bwd = ws.reservoir_prob_bwd * stay_prob * coherence + common_add;
        }

        ws.normalize_bwd(n_patterns);
        next_probs = probs;
    }
}

#[inline]
fn fill_emissions_for_marker_probs(
    block: &CompressedBlock,
    marker_in_window: usize,
    target_probs: &[f32],
    error_rate: f32,
    emissions: &mut [f32],
) {
    let n_patterns = block.n_patterns();
    if n_patterns == 0 {
        return;
    }
    if target_probs.is_empty() {
        emissions[..n_patterns].fill(1.0);
        return;
    }

    let n_alleles = block.n_alleles(marker_in_window);
    let mismatch_prob = if n_alleles > 1 {
        error_rate / (n_alleles - 1) as f32
    } else {
        error_rate
    };
    let match_prob = 1.0 - error_rate;

    let window_size = block.window_size();
    for pattern_idx in 0..n_patterns {
        let ref_allele = block.unpacked_alleles[pattern_idx * window_size + marker_in_window];
        let emit = if ref_allele == 255 {
            1.0
        } else {
            let p_match = target_probs
                .get(ref_allele as usize)
                .copied()
                .unwrap_or(0.0);
            mismatch_prob + (match_prob - mismatch_prob) * p_match
        };
        emissions[pattern_idx] = emit;
    }
}

#[inline]
fn emission_prob_soft(
    block: &CompressedBlock,
    pattern_id: PatternId,
    marker_in_window: usize,
    target_probs: &[f32],
    error_rate: f32,
    n_alleles: usize,
) -> f32 {
    if target_probs.is_empty() {
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
            let mut expected_match = 0.0f32;
            for allele in 0..n_alleles {
                let p = target_probs.get(allele).copied().unwrap_or(0.0);
                let freq = block.reservoir_freq(marker_in_window, allele as u8);
                expected_match += p * freq;
            }
            let p_given_observed = mismatch_prob + (match_prob - mismatch_prob) * expected_match;
            p_given_observed * obs_fraction + 1.0 * (1.0 - obs_fraction)
        } else {
            1.0
        }
    } else {
        let ref_allele = block.get_pattern_allele(pattern_id, marker_in_window);
        if ref_allele == 255 {
            return 1.0;
        }
        let p_match = target_probs
            .get(ref_allele as usize)
            .copied()
            .unwrap_or(0.0);
        mismatch_prob + (match_prob - mismatch_prob) * p_match
    }
}

fn reservoir_coherence(
    block: &CompressedBlock,
    marker_in_window: usize,
    prev_probs: &[f32],
    curr_probs: &[f32],
) -> f32 {
    if marker_in_window == 0 || block.reservoir_ld.is_empty() {
        return 1.0;
    }
    if prev_probs.len() < 2 || curr_probs.len() < 2 {
        return 1.0;
    }
    let idx = marker_in_window - 1;
    let ld = match block.reservoir_ld.get(idx) {
        Some(v) => v,
        None => return 1.0,
    };
    let p0_prev = prev_probs.get(0).copied().unwrap_or(0.0);
    let p1_prev = prev_probs.get(1).copied().unwrap_or(0.0);
    let p0_curr = curr_probs.get(0).copied().unwrap_or(0.0);
    let p1_curr = curr_probs.get(1).copied().unwrap_or(0.0);
    let coherence = p0_prev * (p0_curr * ld[0] + p1_curr * ld[1])
        + p1_prev * (p0_curr * ld[2] + p1_curr * ld[3]);
    if coherence.is_finite() {
        coherence.clamp(0.0, 10.0)
    } else {
        1.0
    }
}

fn reservoir_coherence_backward(
    block: &CompressedBlock,
    marker_in_window: usize,
    curr_probs: &[f32],
    next_probs: &[f32],
) -> f32 {
    if next_probs.is_empty() || block.reservoir_ld.is_empty() {
        return 1.0;
    }
    if curr_probs.len() < 2 || next_probs.len() < 2 {
        return 1.0;
    }
    let ld = match block.reservoir_ld.get(marker_in_window) {
        Some(v) => v,
        None => return 1.0,
    };
    let p0_curr = curr_probs.get(0).copied().unwrap_or(0.0);
    let p1_curr = curr_probs.get(1).copied().unwrap_or(0.0);
    let p0_next = next_probs.get(0).copied().unwrap_or(0.0);
    let p1_next = next_probs.get(1).copied().unwrap_or(0.0);
    let coherence = p0_curr * (p0_next * ld[0] + p1_next * ld[1])
        + p1_curr * (p0_next * ld[2] + p1_next * ld[3]);
    if coherence.is_finite() {
        coherence.clamp(0.0, 10.0)
    } else {
        1.0
    }
}

fn pattern_idx_to_id(idx: usize) -> PatternId {
    PatternId::new(idx as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Allele, Marker, MarkerIdx, Markers};
    use crate::data::storage::{GenotypeColumn, GenotypeMatrix};
    use crate::model::block_hash::compression::build_compressed_block_from_columns;
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
        markers.push(Marker::new(
            chr,
            100,
            None,
            Allele::Base(0),
            vec![Allele::Base(1)],
        ));
        markers.push(Marker::new(
            chr,
            200,
            None,
            Allele::Base(0),
            vec![Allele::Base(1)],
        ));

        let samples = Arc::new(Samples::from_ids(vec!["H0".to_string(), "H1".to_string()])); // Dummy
        let gt = GenotypeMatrix::new_phased(markers, vec![col0, col1], samples);

        let rates = vec![0.0]; // No recombination to isolate emissions
        let marker_vec: Vec<Marker> = (0..gt.n_markers())
            .map(|i| gt.marker(MarkerIdx::new(i as u32)).clone())
            .collect();
        let columns: Vec<GenotypeColumn> = (0..gt.n_markers())
            .map(|i| gt.column(MarkerIdx::new(i as u32)).clone())
            .collect();
        let block = build_compressed_block_from_columns(&marker_vec, &columns, 0, 0, &rates);

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
        ws.fwd[p0.as_usize()] = 1.0 / 3.0;
        ws.fwd[p1.as_usize()] = 1.0 / 3.0;
        ws.fwd[p2.as_usize()] = 1.0 / 3.0;

        let target_probs = TargetAlleleProbs::new(vec![0, 2, 4], vec![1.0, 0.0, 1.0, 0.0]);
        let error = 0.01;
        let match_prob = 1.0 - error;
        let mismatch_prob = error;

        // Run forward pass
        let view = TargetAlleleProbsView::new(&target_probs, 0);
        forward_within_block_probs(&block, &view, error, &mut ws, 0.0);

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

        assert!(
            (ratio1 - expected1).abs() < 1e-4,
            "Hap 1 should be penalized once. Got ratio {}, expected {}",
            ratio1,
            expected1
        );

        // prob2 / prob0 ~ (mismatch/match)^2
        let ratio2 = prob2 / prob0;
        let expected2 = (mismatch_prob / match_prob).powi(2);
        assert!(
            (ratio2 - expected2).abs() < 1e-5,
            "Hap 2 should be penalized twice. Got ratio {}, expected {}",
            ratio2,
            expected2
        );
    }
}
