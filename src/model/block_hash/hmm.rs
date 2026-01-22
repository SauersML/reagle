//! # HMM Kernel: Reuse Existing AVX-512 Optimized Code
//!
//! This module integrates the block-hash HMM with Reagle's existing
//! SIMD-optimized HMM kernels instead of writing new scalar loops.

use super::compressed_block::CompressedBlock;
use super::workspace::BlockHmmWorkspace;
use super::types::PatternId;
use crate::model::hmm::HmmUpdater;

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

/// Backward pass within block AND emit dosages
///
/// Combines forward probabilities (from checkpoint) with backward probabilities
/// to compute posterior dosages.
///
/// # Returns
/// Dosages for markers in this block (in genomic order)
pub fn backward_and_emit_block(
    block: &CompressedBlock,
    target_genotypes: &[u8],
    error_rate: f32,
    ws: &mut BlockHmmWorkspace,
) -> Vec<f32> {
    let n_patterns = block.n_patterns();
    let window_size = block.window_size();
    
    // We compute dosages in reverse order (because backward pass is reverse),
    // but return them in genomic order.
    let mut dosages = vec![0.0; window_size];

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
        let history = ws.fwd_history_at_mut(marker_idx);
        history[..n_patterns].copy_from_slice(&ws.fwd[..n_patterns]);
        history[n_patterns] = ws.reservoir_prob_fwd;
    }
    
    // Now Backward (reverse)
    for marker_idx in (0..window_size).rev() {
        let target_allele = target_genotypes[marker_idx];
        let recomb_rate = block.local_recomb_rates[marker_idx];
        
        let mut total_prob = 0.0;
        let mut dosage_sum = 0.0;
        
        let current_fwd = ws.fwd_history_at(marker_idx);
        
        // Pattern states
        for pattern_idx in 0..n_patterns {
            let p = current_fwd[pattern_idx] * ws.bwd[pattern_idx];
            if p > 0.0 {
                total_prob += p;
                let allele = block.pattern_allele(PatternId::new(pattern_idx as u16), marker_idx);
                dosage_sum += p * allele;
            }
        }
        
        // Reservoir state
        let res_p = current_fwd[n_patterns] * ws.reservoir_prob_bwd;
        if res_p > 0.0 {
            total_prob += res_p;
            let allele_exp = block.pattern_allele(PatternId::RESERVOIR, marker_idx);
            dosage_sum += res_p * allele_exp;
        }
        
        if total_prob > 0.0 {
            dosages[marker_idx] = dosage_sum / total_prob;
        } else {
            dosages[marker_idx] = 0.0;
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
    
    dosages
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
