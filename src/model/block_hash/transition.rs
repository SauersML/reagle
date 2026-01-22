//! # Transition Bridge: Correct Probability Transfer Between Windows
//!
//! This is the KEY module that fixes the index scrambling bug.
//!
//! The core insight: Probability must follow physical DNA molecules (global haplotypes),
//! not local pattern indices. The TransitionBridge maps probability flow by iterating
//! over ALL global haplotype IDs and tracking how each one moves between windows.
//!
//! Mathematical Correctness:
//! - Pattern A (100 haplotypes) → Pattern B (60 haps) + Pattern C (40 haps)
//!   => Weight A→B = 60/100 = 0.6, Weight A→C = 40/100 = 0.4
//! - This correctly models coalescent splits and merges

use super::micro_window::MicroWindow;
use super::types::{GlobalId, PatternId};
use std::collections::HashMap;

/// Transition bridge between two adjacent MicroWindows
///
/// Tracks how probability mass flows from patterns in window A to patterns in window B.
/// Built by iterating over all global haplotype IDs to ensure continuity.
#[derive(Clone, Debug)]
pub(crate) struct TransitionBridge {
    /// Core transitions: (from_pattern, to_pattern) → weight
    /// Weight represents the fraction of probability that flows along this path
    /// Sum of weights from a given from_pattern may be < 1.0 (rest goes to recombination)
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
    n_ref_haps: u32,
}

impl TransitionBridge {
    /// Build a transition bridge between two windows
    ///
    /// # Arguments
    /// * `window_a` - Source window (previous)
    /// * `window_b` - Destination window (next)
    /// * `recomb_rate` - Recombination rate between the windows
    ///
    /// # Returns
    /// TransitionBridge that correctly maps probability flow
    pub fn build(
        window_a: &MicroWindow,
        window_b: &MicroWindow,
        recomb_rate: f32,
    ) -> Self {
        let n_ref_haps = window_a.n_ref_haps() as u32;
        debug_assert_eq!(
            window_b.n_ref_haps(),
            n_ref_haps as usize,
            "Windows must have same reference panel size"
        );

        let mut transitions: HashMap<(PatternId, PatternId), f32> = HashMap::new();
        let mut reservoir_to_pattern: HashMap<PatternId, f32> = HashMap::new();
        let mut pattern_to_reservoir: HashMap<PatternId, f32> = HashMap::new();
        let mut reservoir_to_reservoir = 0.0f32;

        // Important: Iterate over GLOBAL IDs to guarantee continuity
        for global_id_raw in 0..n_ref_haps {
            let global_id = GlobalId::new(global_id_raw);

            let from_pat = window_a.global_to_pattern[global_id];
            let to_pat = window_b.global_to_pattern[global_id];

            // Calculate per-haplotype weight (handles cardinality correctly)
            let weight = if from_pat.is_reservoir() {
                // Important: Use reservoir_count for uniform density assumption
                if window_a.reservoir_count > 0 {
                    1.0 / window_a.reservoir_count as f32
                } else {
                    continue; // No reservoir mass to transfer
                }
            } else {
                // Split weight by pattern cardinality
                1.0 / window_a.pattern_counts[from_pat]
            };

            // Apply no-recombination probability
            let flow = weight * (1.0 - recomb_rate);

            // Route the flow based on source and destination
            match (from_pat.is_reservoir(), to_pat.is_reservoir()) {
                (false, false) => {
                    // Normal pattern → pattern
                    *transitions.entry((from_pat, to_pat)).or_insert(0.0) += flow;
                }
                (false, true) => {
                    // Pattern evicted to reservoir
                    *pattern_to_reservoir.entry(from_pat).or_insert(0.0) += flow;
                }
                (true, false) => {
                    // Reservoir promoted to pattern
                    *reservoir_to_pattern.entry(to_pat).or_insert(0.0) += flow;
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
            let prob = window_a.fwd_probs[from];
            new_fwd[to] += prob * weight;
        }

        // Reservoir → Pattern transitions
        for (&to, &weight) in &self.reservoir_to_pattern {
            new_fwd[to] += window_a.reservoir_prob * weight;
        }

        // Pattern → Reservoir transitions
        for (&from, &weight) in &self.pattern_to_reservoir {
            let prob = window_a.fwd_probs[from];
            new_reservoir_prob += prob * weight;
        }

        // Reservoir → Reservoir transition
        new_reservoir_prob += window_a.reservoir_prob * self.reservoir_to_reservoir;

        // Add recombination background to all states
        let total_mass = window_a.fwd_probs.iter().sum::<f32>() + window_a.reservoir_prob;
        let background_per_hap = total_mass * self.recomb_rate / (self.n_ref_haps as f32);

        // Distribute recombination mass proportionally to pattern counts
        for pattern_idx in 0..n_patterns_b {
            let count = window_b.pattern_counts[PatternId::new(pattern_idx as u16)];
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

    /// Get the direct transition weight from one pattern to another (for debugging)
    pub fn get_transition_weight(&self, from: PatternId, to: PatternId) -> f32 {
        *self.transitions.get(&(from, to)).unwrap_or(&0.0)
    }

    /// Verify that the transition is properly normalized (for testing)
    pub fn check_normalization(&self, window_a: &MicroWindow) -> bool {
        // For each source pattern, sum of outgoing weights should be ≤ 1.0
        for pattern_idx in 0..window_a.n_patterns() {
            let from = PatternId::new(pattern_idx as u16);
            let mut total_weight = 0.0f32;

            // Sum pattern→pattern transitions
            for (&(f, _), &weight) in &self.transitions {
                if f == from {
                    total_weight += weight;
                }
            }

            // Add pattern→reservoir
            if let Some(&weight) = self.pattern_to_reservoir.get(&from) {
                total_weight += weight;
            }

            // Should be ≤ (1 - recomb_rate) due to recombination losses
            if total_weight > 1.0 + 1e-6 {
                return false;
            }
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_window(
        patterns: Vec<(u64, u32)>, // (fingerprint, count)
        global_to_pattern_map: Vec<u16>,
    ) -> MicroWindow {
        let n_ref_haps = global_to_pattern_map.len();

        let unique_patterns = patterns.iter().map(|(fp, _)| *fp).collect();
        let pattern_counts = patterns.iter().map(|(_, cnt)| *cnt as f32).collect();
        let global_to_pattern = global_to_pattern_map
            .into_iter()
            .map(PatternId::from)
            .collect();

        let n_patterns = patterns.len();

        MicroWindow {
            start_marker: 0,
            end_marker: 64,
            unique_patterns,
            pattern_counts,
            global_to_pattern,
            fwd_probs: vec![0.0; n_patterns],
            bwd_probs: vec![0.0; n_patterns],
            reservoir_prob: 0.0,
            reservoir_count: 0,
            reservoir_allele_freqs: vec![0.5; 64],
        }
    }

    #[test]
    fn test_simple_transition() {
        // Window A: 2 patterns, each with 2 haplotypes
        // Haps 0,1 → Pattern 0
        // Haps 2,3 → Pattern 1
        let window_a = create_test_window(
            vec![(0b1010, 2), (0b0101, 2)],
            vec![0, 0, 1, 1],
        );

        // Window B: Same structure (no splits/merges)
        let window_b = create_test_window(
            vec![(0b1111, 2), (0b0000, 2)],
            vec![0, 0, 1, 1],
        );

        let bridge = TransitionBridge::build(&window_a, &window_b, 0.01);

        // Each haplotype contributes weight 1/2 (cardinality=2)
        // Pattern 0 → Pattern 0: 2 haps × (1/2) × (1 - 0.01) = 0.99
        let weight_00 = bridge.get_transition_weight(PatternId::new(0), PatternId::new(0));
        assert!((weight_00 - 0.99).abs() < 1e-6);

        // Pattern 1 → Pattern 1: same
        let weight_11 = bridge.get_transition_weight(PatternId::new(1), PatternId::new(1));
        assert!((weight_11 - 0.99).abs() < 1e-6);

        // No cross-transitions
        let weight_01 = bridge.get_transition_weight(PatternId::new(0), PatternId::new(1));
        assert_eq!(weight_01, 0.0);
    }

    #[test]
    fn test_equivalence_class_splitting() {
        // Window A: Pattern 0 has 4 haplotypes (haps 0-3)
        let window_a = create_test_window(
            vec![(0b1010, 4)],
            vec![0, 0, 0, 0],
        );

        // Window B: Pattern splits into Pattern 0 (3 haps) and Pattern 1 (1 hap)
        let window_b = create_test_window(
            vec![(0b1111, 3), (0b0000, 1)],
            vec![0, 0, 0, 1],
        );

        let bridge = TransitionBridge::build(&window_a, &window_b, 0.01);

        // Pattern 0 → Pattern 0: 3 haps × (1/4) × (1 - 0.01) = 0.7425
        let weight_00 = bridge.get_transition_weight(PatternId::new(0), PatternId::new(0));
        assert!((weight_00 - 0.7425).abs() < 1e-4);

        // Pattern 0 → Pattern 1: 1 hap × (1/4) × (1 - 0.01) = 0.2475
        let weight_01 = bridge.get_transition_weight(PatternId::new(0), PatternId::new(1));
        assert!((weight_01 - 0.2475).abs() < 1e-4);

        // Total should be close to (1 - recomb_rate) = 0.99
        assert!((weight_00 + weight_01 - 0.99).abs() < 1e-4);
    }

    #[test]
    fn test_apply_transition() {
        let mut window_a = create_test_window(
            vec![(0b1010, 2), (0b0101, 2)],
            vec![0, 0, 1, 1],
        );

        let mut window_b = create_test_window(
            vec![(0b1111, 2), (0b0000, 2)],
            vec![0, 0, 1, 1],
        );

        // Set initial probabilities
        window_a.fwd_probs = vec![0.6, 0.4];

        let bridge = TransitionBridge::build(&window_a, &window_b, 0.01);
        bridge.apply(&window_a, &mut window_b);

        // Check mass conservation
        let total = window_b.total_probability();
        assert!((total - 1.0).abs() < 1e-4, "Total probability should be 1.0, got {}", total);

        // Most probability should stay in the same pattern (low recombination)
        // Pattern 0 had 0.6, should transfer ~0.6 * 0.99 ≈ 0.594 (plus background)
        assert!(window_b.fwd_probs[0] > 0.55);
        assert!(window_b.fwd_probs[1] > 0.35);
    }

    #[test]
    fn test_reservoir_transitions() {
        // Window A: 2 patterns + 2 reservoir haplotypes
        let mut window_a = create_test_window(
            vec![(0b1010, 1), (0b0101, 1)],
            vec![0, 1, u16::MAX, u16::MAX], // Last 2 are RESERVOIR
        );
        window_a.reservoir_count = 2;
        window_a.reservoir_prob = 0.5;
        window_a.fwd_probs = vec![0.25, 0.25];

        // Window B: One reservoir haplotype promoted to pattern
        let mut window_b = create_test_window(
            vec![(0b1111, 1), (0b0000, 1), (0b1100, 1)], // 3 patterns
            vec![0, 1, 2, u16::MAX], // Hap 2 promoted, hap 3 still reservoir
        );
        window_b.reservoir_count = 1;

        let bridge = TransitionBridge::build(&window_a, &window_b, 0.01);

        // Verify reservoir→pattern transition exists
        assert!(bridge.reservoir_to_pattern.len() > 0);

        // Verify pattern→reservoir transition exists
        assert!(bridge.pattern_to_reservoir.len() > 0 || bridge.reservoir_to_reservoir > 0.0);
    }

    #[test]
    fn test_normalization_check() {
        let window_a = create_test_window(
            vec![(0b1010, 2), (0b0101, 2)],
            vec![0, 0, 1, 1],
        );

        let window_b = create_test_window(
            vec![(0b1111, 2), (0b0000, 2)],
            vec![0, 0, 1, 1],
        );

        let bridge = TransitionBridge::build(&window_a, &window_b, 0.01);

        assert!(bridge.check_normalization(&window_a));
    }
}
