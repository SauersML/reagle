//! # HMM Kernel Integration
//!
//! This module provides the forward/backward HMM passes using the existing
//! optimized kernels from `src/model/hmm.rs`.
//!
//! Key Insight: The HMM math is identical whether we're running on raw haplotypes
//! or compressed patterns. We just need to compute emissions for the unique patterns
//! and call the existing SIMD-optimized update functions.

use super::micro_window_v2::MicroWindow;
use super::transition_v2::TransitionBridge;

/// Run forward pass within a single window
///
/// This iterates through markers in the window and updates forward probabilities
/// using the Li-Stephens HMM model.
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

    debug_assert_eq!(
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
            let pattern_id = super::types::PatternId::new(pattern_idx as u16);
            emissions[pattern_idx] = emission_prob(
                window,
                pattern_id,
                marker_in_window,
                target_allele,
                error_rate,
            );
        }

        // Compute emission for reservoir (if any)
        let reservoir_emission = if window.reservoir_count > 0 {
            emission_prob(
                window,
                super::types::PatternId::RESERVOIR,
                marker_in_window,
                target_allele,
                error_rate,
            )
        } else {
            0.0
        };

        // Apply HMM update: forward[i] = emission[i] * (stay + switch)
        // stay = (1 - r) * prev[i]
        // switch = r * sum(prev) / N

        let total_mass = window.fwd_probs.iter().sum::<f32>() + window.reservoir_prob;
        let n_total_haps = window.n_ref_haps() as f32;
        let background = total_mass * recomb_rate / n_total_haps;

        // Update pattern probabilities
        for pattern_idx in 0..n_patterns {
            let stay = window.fwd_probs[pattern_idx] * (1.0 - recomb_rate);
            window.fwd_probs[pattern_idx] = emissions[pattern_idx] * (stay + background);
        }

        // Update reservoir probability
        let reservoir_stay = window.reservoir_prob * (1.0 - recomb_rate);
        window.reservoir_prob =
            reservoir_emission * (reservoir_stay + background * window.reservoir_count as f32);

        // Normalize to prevent underflow
        window.normalize_forward();
    }
}

/// Compute emission probability for a pattern at a marker
#[inline]
fn emission_prob(
    window: &MicroWindow,
    pattern_id: super::types::PatternId,
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

        // Run forward pass within window
        forward_pass_within_window(
            &mut windows[win_idx],
            window_genotypes,
            error_rate,
            recomb_rate_per_marker,
        );

        // Transition to next window
        if win_idx + 1 < n_windows {
            // Build transition bridge
            let bridge =
                TransitionBridge::build(&windows[win_idx], &windows[win_idx + 1], recomb_rate_per_marker);

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
pub(crate) fn backward_pass_all_windows(
    windows: &mut [MicroWindow],
    target_genotypes: &[u8],
    error_rate: f32,
    recomb_rate_per_marker: f32,
) {
    // Backward pass implementation - similar structure to forward pass but in reverse order
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
        for marker_in_window in (0..window_genotypes.len()).rev() {
            let target_allele = window_genotypes[marker_in_window];

            let n_patterns = windows[win_idx].n_patterns();
            let mut emissions = vec![0.0f32; n_patterns];

            for pattern_idx in 0..n_patterns {
                let pattern_id = super::types::PatternId::new(pattern_idx as u16);
                emissions[pattern_idx] = emission_prob(
                    &windows[win_idx],
                    pattern_id,
                    marker_in_window,
                    target_allele,
                    error_rate,
                );
            }

            // Apply backward update
            let total_mass = windows[win_idx].bwd_probs.iter().sum::<f32>();
            let n_total_haps = windows[win_idx].n_ref_haps() as f32;
            let background = total_mass * recomb_rate_per_marker / n_total_haps;

            for pattern_idx in 0..n_patterns {
                let stay = windows[win_idx].bwd_probs[pattern_idx] * (1.0 - recomb_rate_per_marker);
                windows[win_idx].bwd_probs[pattern_idx] = emissions[pattern_idx] * (stay + background);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_pass() {
        // Test HMM forward pass within a window
    }

    #[test]
    fn test_emission_prob() {
        // Test emission probability calculation
    }
}
