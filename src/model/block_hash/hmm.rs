//! # HMM Kernel: Reuse Existing AVX-512 Optimized Code
//!
//! This module integrates the block-hash HMM with Reagle's existing
//! SIMD-optimized HMM kernels instead of writing new scalar loops.

use super::compressed_block::CompressedBlock;
use super::workspace::BlockHmmWorkspace;
use super::types::PatternId;
use crate::model::hmm::HmmUpdater;
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
        let recomb_rate = block.local_recomb_rates[marker_in_window];

        // Compute emission probabilities for all patterns
        // Use ws.emissions as temp buffer
        let emissions = &mut ws.emissions;
        
        for pattern_idx in 0..n_patterns {
            let pattern_id = PatternId::new(pattern_idx as u16);
            emissions[pattern_idx] = emission_prob(
                block,
                pattern_id,
                marker_in_window,
                target_allele,
                error_rate,
            );
        }

        // REUSE: Call existing AVX-512 optimized HmmUpdater
        let fwd_sum = ws.fwd[..n_patterns].iter().sum::<f32>() + ws.reservoir_prob_fwd;

        HmmUpdater::fwd_update_emissions(
            &mut ws.fwd,
            fwd_sum,
            recomb_rate,
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

/// Backward pass within block AND emit posteriors
///
/// Combines forward probabilities (from checkpoint) with backward probabilities
/// to compute posterior probabilities for each allele.
///
/// # Returns
/// Allele posteriors for markers in this block (in genomic order)
pub fn backward_and_emit_block(
    block: &CompressedBlock,
    target_genotypes: &[u8],
    error_rate: f32,
    ws: &mut BlockHmmWorkspace,
) -> Vec<AllelePosteriors> {
    let n_patterns = block.n_patterns();
    let window_size = block.window_size();
    
    // We compute posteriors in reverse order (because backward pass is reverse),
    // but return them in genomic order. We will reverse at end or insert at front.
    // Insert at front is O(N^2). Pre-allocate and fill in reverse?
    // Vec doesn't support filling from back easily.
    // We'll collect in reverse and then reverse the vector.
    let mut posteriors_rev = Vec::with_capacity(window_size);

    // Re-run Forward and store history into pre-allocated workspace buffer
    for marker_idx in 0..window_size {
        let target_allele = target_genotypes[marker_idx];
        let recomb_rate = block.local_recomb_rates[marker_idx];
        
        let emissions = &mut ws.emissions;
        for pattern_idx in 0..n_patterns {
            emissions[pattern_idx] = emission_prob(
                block,
                PatternId::new(pattern_idx as u16),
                marker_idx,
                target_allele,
                error_rate,
            );
        }
        
        let fwd_sum = ws.fwd[..n_patterns].iter().sum::<f32>() + ws.reservoir_prob_fwd;
        
        HmmUpdater::fwd_update_emissions(
            &mut ws.fwd,
            fwd_sum,
            recomb_rate,
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
        let recomb_rate = block.local_recomb_rates[marker_idx];
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
        
        // Reservoir state
        let res_p = current_fwd[n_patterns] * ws.reservoir_prob_bwd;
        if res_p > 0.0 {
            total_prob += res_p;
            // Reservoir contributes based on allele frequency
            // Assume biallelic distribution for reservoir if multiallelic?
            // block.reservoir_allele_freqs[marker_idx] is mean value.
            let mean = block.pattern_allele(PatternId::RESERVOIR, marker_idx);
            // Distribute mean between 0 and 1 (clamped)
            let p1 = mean.clamp(0.0, 1.0);
            let p0 = 1.0 - p1;
            
            if 0 < n_alleles { allele_probs[0] += res_p * p0; }
            if 1 < n_alleles { allele_probs[1] += res_p * p1; }
            // For >2 alleles, we ignore reservoir contribution to alleles >1 
            // (limitation of current reservoir compression)
        }
        
        // Normalize
        if total_prob > 0.0 {
            let scale = 1.0 / total_prob;
            for p in &mut allele_probs {
                *p *= scale;
            }
        }
        
        if n_alleles == 2 {
            posteriors_rev.push(AllelePosteriors::Biallelic(allele_probs[1]));
        } else {
            posteriors_rev.push(AllelePosteriors::Multiallelic(allele_probs));
        }
        
        // Update bwd to t-1
        let emissions = &mut ws.emissions;
        for pattern_idx in 0..n_patterns {
            emissions[pattern_idx] = emission_prob(
                block,
                PatternId::new(pattern_idx as u16),
                marker_idx,
                target_allele,
                error_rate,
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
        );
        ws.reservoir_prob_bwd *= reservoir_emission;
        
        // Pure transition step
        emissions.fill(1.0);
        let bwd_sum = ws.bwd[..n_patterns].iter().sum::<f32>() + ws.reservoir_prob_bwd;
        HmmUpdater::fwd_update_emissions(
            &mut ws.bwd,
            bwd_sum,
            recomb_rate,
            emissions,
            n_patterns,
        );
        
        let total_mass = bwd_sum;
        let background = total_mass * recomb_rate / block.n_ref_haps() as f32;
        let stay = ws.reservoir_prob_bwd * (1.0 - recomb_rate);
        ws.reservoir_prob_bwd = stay + background * block.reservoir_count as f32;
        
        ws.normalize_bwd(n_patterns);
    }
    
    posteriors_rev.reverse();
    posteriors_rev
}

fn pattern_idx_to_id(idx: usize) -> PatternId {
    PatternId::new(idx as u16)
}

/// Compute emission probability for a pattern at a marker
#[inline]
fn emission_prob(
    block: &CompressedBlock,
    pattern_id: PatternId,
    marker_in_window: usize,
    target_allele: u8,
    error_rate: f32,
) -> f32 {
    // Missing data - neutral (1.0)
    if target_allele == 255 {
        return 1.0;
    }

    let ref_allele = block.pattern_allele(pattern_id, marker_in_window);

    if pattern_id.is_reservoir() {
        // Reservoir uses allele frequency
        if target_allele == 0 {
            (1.0 - ref_allele) * (1.0 - error_rate) + ref_allele * error_rate
        } else {
            ref_allele * (1.0 - error_rate) + (1.0 - ref_allele) * error_rate
        }
    } else {
        // Pattern uses exact allele matching (handles multiallelic)
        // pattern_allele returns f32 (0.0, 1.0, 2.0...), convert to u8
        let ref_allele_int = ref_allele as u8;

        if target_allele == ref_allele_int {
            1.0 - error_rate
        } else {
            error_rate
        }
    }
}
