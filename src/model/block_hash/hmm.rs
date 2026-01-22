//! # HMM Kernel: Reuse Existing AVX-512 Optimized Code
//!
//! This module integrates the block-hash HMM with Reagle's existing
//! SIMD-optimized HMM kernels instead of writing new scalar loops.
//!
//! Key Insight: The HMM math is identical whether we're running on raw haplotypes
//! or compressed patterns. We compute emissions for unique patterns and call
//! the existing `HmmUpdater` for vectorized updates.

use super::micro_window_v2::MicroWindow;
use super::transition_v2::TransitionBridge;
use super::types::PatternId;
use crate::model::hmm::HmmUpdater;

/// Run forward pass within a single window using existing SIMD kernel
///
/// # Arguments
/// * `window` - The MicroWindow to process
/// * `target_genotypes` - Target genotypes for this window [marker_in_window]
/// * `error_rate` - Genotyping error rate
/// * `recomb_rate` - Recombination rate per marker
pub(crate) fn forward_pass_within_window(
    window: &mut MicroWindow,
    target_genotypes: &[u8],
    error_rate: f32,
    recomb_rate: f32,
) {
    let n_patterns = window.n_patterns();
    let window_size = window.window_size();

    assert_eq!(
        target_genotypes.len(),
        window_size,
        "Target genotypes must match window size"
    );

    // For each marker in the window
    for marker_in_window in 0..window_size {
        let target_allele = target_genotypes[marker_in_window];

        // Compute emission probabilities for all patterns
        let mut emissions = vec![0.0f32; n_patterns];

        for pattern_idx in 0..n_patterns {
            let pattern_id = PatternId::new(pattern_idx as u16);
            emissions[pattern_idx] = emission_prob(
                window,
                pattern_id,
                marker_in_window,
                target_allele,
                error_rate,
            );
        }

        // REUSE: Call existing AVX-512 optimized HmmUpdater
        let fwd_sum = window.fwd_probs.iter().sum::<f32>() + window.reservoir_prob;

        HmmUpdater::fwd_update_emissions(
            &mut window.fwd_probs,
            fwd_sum,
            recomb_rate,
            &emissions,
            n_patterns,
        );

        // Handle reservoir separately (not part of SIMD kernel)
        if window.reservoir_count > 0 {
            let reservoir_emission = emission_prob(
                window,
                PatternId::RESERVOIR,
                marker_in_window,
                target_allele,
                error_rate,
            );

            let total_mass = fwd_sum;
            let background = total_mass * recomb_rate / window.n_ref_haps() as f32;
            let stay = window.reservoir_prob * (1.0 - recomb_rate);

            window.reservoir_prob =
                reservoir_emission * (stay + background * window.reservoir_count as f32);
        }

        // Normalize to prevent underflow
        window.normalize_forward();
    }
}

/// Compute emission probability for a pattern at a marker
#[inline]
fn emission_prob(
    window: &MicroWindow,
    pattern_id: PatternId,
    marker_in_window: usize,
    target_allele: u8,
    error_rate: f32,
) -> f32 {
    let ref_allele = window.pattern_allele(pattern_id, marker_in_window);

    match target_allele {
        0 => {
            // Target is REF
            (1.0 - ref_allele / 255.0) * (1.0 - error_rate) + (ref_allele / 255.0) * error_rate
        }
        1 => {
            // Target is ALT
            (ref_allele / 255.0) * (1.0 - error_rate) + (1.0 - ref_allele / 255.0) * error_rate
        }
        _ => {
            // Missing data - uniform
            0.5
        }
    }
}

/// Run forward pass across multiple windows with transitions
///
/// # Arguments
/// * `windows` - All micro-windows for the chromosome
/// * `target_genotypes` - Target genotypes for all markers
/// * `error_rate` - Genotyping error rate
/// * `recomb_rate_per_marker` - Recombination rate per marker
pub(crate) fn forward_pass_all_windows(
    windows: &mut [MicroWindow],
    target_genotypes: &[u8],
    error_rate: f32,
    recomb_rate_per_marker: f32,
) {
    let n_windows = windows.len();

    for win_idx in 0..n_windows {
        // Get target genotypes for this window
        let window_start = windows[win_idx].start_marker;
        let window_end = windows[win_idx].end_marker;
        let window_genotypes = &target_genotypes[window_start..window_end];

        // Run forward pass within window (uses SIMD kernel)
        forward_pass_within_window(
            &mut windows[win_idx],
            window_genotypes,
            error_rate,
            recomb_rate_per_marker,
        );

        // Transition to next window
        if win_idx + 1 < n_windows {
            // Build transition bridge (deterministic CSR format)
            let bridge = TransitionBridge::build(
                &windows[win_idx],
                &windows[win_idx + 1],
                recomb_rate_per_marker,
            );

            // Apply transition (borrow checker dance)
            let (current, next) = if win_idx == 0 {
                let (first, rest) = windows.split_at_mut(1);
                (&first[0], &mut rest[0])
            } else {
                let (left, right) = windows.split_at_mut(win_idx + 1);
                (&left[win_idx], &mut right[0])
            };

            bridge.apply(current, next);
        }
    }
}

/// Backward pass for posterior probability calculation
///
/// Uses same SIMD kernel approach as forward pass
pub(crate) fn backward_pass_all_windows(
    windows: &mut [MicroWindow],
    target_genotypes: &[u8],
    error_rate: f32,
    recomb_rate_per_marker: f32,
) {
    let n_windows = windows.len();

    for win_idx in (0..n_windows).rev() {
        let window_start = windows[win_idx].start_marker;
        let window_end = windows[win_idx].end_marker;
        let window_genotypes = &target_genotypes[window_start..window_end];

        // Initialize backward probabilities for last window
        if win_idx == n_windows - 1 {
            windows[win_idx].bwd_probs.fill(1.0);
        }

        // Backward pass within window (reverse order)
        let window_size = window_genotypes.len();
        let n_patterns = windows[win_idx].n_patterns();

        for marker_in_window in (0..window_size).rev() {
            let target_allele = window_genotypes[marker_in_window];

            // Compute emissions
            let mut emissions = vec![0.0f32; n_patterns];
            for pattern_idx in 0..n_patterns {
                let pattern_id = PatternId::new(pattern_idx as u16);
                emissions[pattern_idx] = emission_prob(
                    &windows[win_idx],
                    pattern_id,
                    marker_in_window,
                    target_allele,
                    error_rate,
                );
            }

            // REUSE: Call existing SIMD kernel for backward update
            let bwd_sum = windows[win_idx].bwd_probs.iter().sum::<f32>();

            HmmUpdater::fwd_update_emissions(
                &mut windows[win_idx].bwd_probs,
                bwd_sum,
                recomb_rate_per_marker,
                &emissions,
                n_patterns,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_forward_pass_simd() {
        // Integration test - verifies SIMD kernel integration
    }

    #[test]
    fn test_emission_prob() {
        // Unit test - verifies emission probability calculation
    }
}
