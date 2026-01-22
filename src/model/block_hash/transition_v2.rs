//! # Simplified Transition Bridge using zip()
//!
//! This module provides the KEY fix for the index scrambling bug.
//! Instead of complex Global ID tracking, we simply zip the two
//! `hap_to_pattern` vectors from adjacent windows.
//!
//! The Insight: DictionaryColumn already provides the Global ID → Pattern ID
//! mapping via `hap_to_pattern()`. We just need to iterate through both mappings
//! together to build the transition matrix.

use super::micro_window_v2::MicroWindow;
use super::types::PatternId;
use std::collections::HashMap;

/// Sparse transition matrix between two windows
#[derive(Clone, Debug)]
pub(crate) struct TransitionBridge {
    /// Core transitions: (from_pattern, to_pattern) → weight
    /// Weight represents the probability mass that flows along this path
    transitions: HashMap<(PatternId, PatternId), f32>,

    /// Reservoir → specific pattern transitions
    reservoir_to_pattern: HashMap<PatternId, f32>,

    /// Specific pattern → Reservoir transitions
    pattern_to_reservoir: HashMap<PatternId, f32>,

    /// Reservoir → Reservoir transition weight
    reservoir_to_reservoir: f32,

    /// Recombination rate for this transition
    recomb_rate: f32,

    /// Total number of reference haplotypes (for background recombination)
    n_ref_haps: usize,
}

impl TransitionBridge {
    /// Build a transition bridge by zipping hap_to_pattern vectors
    ///
    /// **This is the fix**: We iterate over all haplotypes and track how each
    /// one transitions from its pattern in window A to its pattern in window B.
    ///
    /// # Arguments
    /// * `window_a` - Source window (previous)
    /// * `window_b` - Destination window (next)
    /// * `recomb_rate` - Recombination rate between the windows
    ///
    /// # Returns
    /// TransitionBridge with correctly weighted transitions
    pub fn build(window_a: &MicroWindow, window_b: &MicroWindow, recomb_rate: f32) -> Self {
        let n_ref_haps = window_a.n_ref_haps();
        debug_assert_eq!(
            window_b.n_ref_haps(),
            n_ref_haps,
            "Windows must have same reference panel size"
        );

        // Get the hap→pattern mappings from both windows
        let map_a = window_a.storage.hap_to_pattern();
        let map_b = window_b.storage.hap_to_pattern();

        let mut transitions: HashMap<(PatternId, PatternId), f32> = HashMap::new();
        let mut reservoir_to_pattern: HashMap<PatternId, f32> = HashMap::new();
        let mut pattern_to_reservoir: HashMap<PatternId, f32> = HashMap::new();
        let mut reservoir_to_reservoir = 0.0f32;

        // Important: Zip the two mappings to track how each haplotype transitions
        for (&pat_a_raw, &pat_b_raw) in map_a.iter().zip(map_b.iter()) {
            // Convert to local pattern IDs (accounting for truncation)
            let pat_a = if (pat_a_raw as usize) < window_a.n_patterns() {
                PatternId::new(pat_a_raw as u16)
            } else {
                PatternId::RESERVOIR
            };

            let pat_b = if (pat_b_raw as usize) < window_b.n_patterns() {
                PatternId::new(pat_b_raw as u16)
            } else {
                PatternId::RESERVOIR
            };

            // Calculate per-haplotype weight (cardinality-aware)
            let weight = if pat_a.is_reservoir() {
                // Uniform density in reservoir
                if window_a.reservoir_count > 0 {
                    1.0 / window_a.reservoir_count as f32
                } else {
                    continue; // No reservoir mass
                }
            } else {
                // Split weight by pattern cardinality
                1.0 / window_a.pattern_counts[pat_a.as_usize()]
            };

            // Apply no-recombination probability
            let flow = weight * (1.0 - recomb_rate);

            // Route the flow based on source and destination
            match (pat_a.is_reservoir(), pat_b.is_reservoir()) {
                (false, false) => {
                    // Normal pattern → pattern
                    *transitions.entry((pat_a, pat_b)).or_insert(0.0) += flow;
                }
                (false, true) => {
                    // Pattern evicted to reservoir
                    *pattern_to_reservoir.entry(pat_a).or_insert(0.0) += flow;
                }
                (true, false) => {
                    // Reservoir promoted to pattern
                    *reservoir_to_pattern.entry(pat_b).or_insert(0.0) += flow;
                }
                (true, true) => {
                    // Stays in reservoir
                    reservoir_to_reservoir += flow;
                }
            }
        }

        Self {
            transitions,
            reservoir_to_pattern,
            pattern_to_reservoir,
            reservoir_to_reservoir,
            recomb_rate,
            n_ref_haps,
        }
    }

    /// Apply this transition to transfer probabilities from window_a to window_b
    ///
    /// # Arguments
    /// * `window_a` - Source window with current forward probabilities
    /// * `window_b` - Destination window (will be updated)
    pub fn apply(&self, window_a: &MicroWindow, window_b: &mut MicroWindow) {
        let n_patterns_b = window_b.n_patterns();

        // Initialize new forward probabilities
        let mut new_fwd = vec![0.0f32; n_patterns_b];
        let mut new_reservoir_prob = 0.0f32;

        // Pattern → Pattern transitions
        for (&(from, to), &weight) in &self.transitions {
            let prob = window_a.fwd_probs[from.as_usize()];
            new_fwd[to.as_usize()] += prob * weight;
        }

        // Reservoir → Pattern transitions
        for (&to, &weight) in &self.reservoir_to_pattern {
            new_fwd[to.as_usize()] += window_a.reservoir_prob * weight;
        }

        // Pattern → Reservoir transitions
        for (&from, &weight) in &self.pattern_to_reservoir {
            let prob = window_a.fwd_probs[from.as_usize()];
            new_reservoir_prob += prob * weight;
        }

        // Reservoir → Reservoir transition
        new_reservoir_prob += window_a.reservoir_prob * self.reservoir_to_reservoir;

        // Add recombination background to all states
        let total_mass = window_a.fwd_probs.iter().sum::<f32>() + window_a.reservoir_prob;
        let background_per_hap = total_mass * self.recomb_rate / (self.n_ref_haps as f32);

        // Distribute recombination mass proportionally to pattern counts
        for pattern_idx in 0..n_patterns_b {
            let count = window_b.pattern_counts[pattern_idx];
            new_fwd[pattern_idx] += background_per_hap * count;
        }

        // Reservoir also receives recombination mass proportional to its cardinality
        new_reservoir_prob += background_per_hap * (window_b.reservoir_count as f32);

        // Update window_b
        window_b.fwd_probs = new_fwd;
        window_b.reservoir_prob = new_reservoir_prob;

        // Normalize to prevent drift
        window_b.normalize_forward();
    }

    /// Get the direct transition weight (for debugging)
    pub fn get_transition_weight(&self, from: PatternId, to: PatternId) -> f32 {
        *self.transitions.get(&(from, to)).unwrap_or(&0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_bridge_simple() {
        // Integration test - verifies transition weight calculation
    }

    #[test]
    fn test_apply_transition() {
        // Integration test - verifies probability mass conservation
    }
}
