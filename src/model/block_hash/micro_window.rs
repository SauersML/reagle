//! # MicroWindow: Compressed HMM State for a 64-Marker Block
//!
//! This module defines the core data structure for the Block-Hash HMM.
//! Each MicroWindow represents a 64-marker block with:
//! - Unique haplotype patterns (compressed via fingerprinting)
//! - Mapping from global haplotype IDs to local pattern IDs
//! - Forward and backward probabilities for the HMM
//! - Reservoir state for truncated patterns (if needed)

use super::types::{GlobalId, PatternId};
use serde::{Deserialize, Serialize};

/// A micro-window of 64 markers with compressed haplotype patterns
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct MicroWindow {
    /// Genomic position range
    pub start_marker: usize,
    pub end_marker: usize,

    // ========================================================================
    // Unique Pattern Compression
    // ========================================================================
    /// Unique 64-bit fingerprints (one bit per marker)
    /// Length: U (number of unique patterns in this window)
    pub unique_patterns: Vec<u64>,

    /// Cardinality of each pattern (how many haplotypes have this pattern)
    /// Length: U
    /// This is Important for correct transition weight calculations
    pub pattern_counts: Vec<f32>,

    /// Mapping from global haplotype ID to local pattern ID
    /// Length: N (total reference panel size)
    /// Value: PatternId or PatternId::RESERVOIR if truncated
    pub global_to_pattern: Vec<PatternId>,

    // ========================================================================
    // HMM State
    // ========================================================================
    /// Forward probabilities (Li-Stephens HMM)
    /// Length: U
    pub fwd_probs: Vec<f32>,

    /// Backward probabilities (for sampling and confidence)
    /// Length: U
    pub bwd_probs: Vec<f32>,

    // ========================================================================
    // Reservoir State (for truncation if U > max_states)
    // ========================================================================
    /// Probability mass of patterns in the reservoir (truncated)
    pub reservoir_prob: f32,

    /// Important: Number of haplotypes in the reservoir
    /// This is required for correct transition calculations
    /// (not just the probability mass, but also the cardinality)
    pub reservoir_count: u32,

    /// Allele frequencies of reservoir haplotypes [marker_in_window]
    /// Used for emission probabilities when a haplotype is in the reservoir
    /// Length: window_size (typically 64)
    pub reservoir_allele_freqs: Vec<f32>,
}

impl MicroWindow {
    /// Create a new empty MicroWindow
    pub fn new(start_marker: usize, end_marker: usize, n_ref_haps: usize) -> Self {
        let window_size = end_marker - start_marker;
        Self {
            start_marker,
            end_marker,
            unique_patterns: Vec::new(),
            pattern_counts: Vec::new(),
            global_to_pattern: vec![PatternId::RESERVOIR; n_ref_haps],
            fwd_probs: Vec::new(),
            bwd_probs: Vec::new(),
            reservoir_prob: 0.0,
            reservoir_count: 0,
            reservoir_allele_freqs: vec![0.5; window_size],
        }
    }

    /// Number of unique patterns (excluding reservoir)
    #[inline]
    pub fn n_patterns(&self) -> usize {
        self.unique_patterns.len()
    }

    /// Window size in markers
    #[inline]
    pub fn window_size(&self) -> usize {
        self.end_marker - self.start_marker
    }

    /// Total number of reference haplotypes
    #[inline]
    pub fn n_ref_haps(&self) -> usize {
        self.global_to_pattern.len()
    }

    /// Get the pattern ID for a global haplotype
    #[inline]
    pub fn pattern_for_haplotype(&self, global_id: GlobalId) -> PatternId {
        self.global_to_pattern[global_id]
    }

    /// Get the allele at a specific marker within the window for a pattern
    ///
    /// # Arguments
    /// * `pattern_id` - The pattern to query
    /// * `marker_in_window` - Marker offset within this window (0..64)
    ///
    /// # Returns
    /// The allele (0 or 1) at that position, or the reservoir frequency if pattern is RESERVOIR
    #[inline]
    pub fn pattern_allele(&self, pattern_id: PatternId, marker_in_window: usize) -> f32 {
        if pattern_id.is_reservoir() {
            self.reservoir_allele_freqs[marker_in_window]
        } else {
            let fingerprint = self.unique_patterns[pattern_id];
            ((fingerprint >> marker_in_window) & 1) as f32
        }
    }

    /// Initialize forward probabilities to uniform prior
    pub fn initialize_uniform_prior(&mut self) {
        let n_patterns = self.n_patterns();
        if n_patterns == 0 {
            return;
        }

        // Uniform distribution over all patterns
        let uniform_prob = 1.0 / (n_patterns as f32 + self.reservoir_count as f32);

        self.fwd_probs = vec![uniform_prob; n_patterns];
        self.reservoir_prob = uniform_prob * self.reservoir_count as f32;
    }

    /// Normalize forward probabilities to sum to 1.0
    pub fn normalize_forward(&mut self) {
        let total: f32 = self.fwd_probs.iter().sum::<f32>() + self.reservoir_prob;

        if total > 1e-30 {
            let scale = 1.0 / total;
            for prob in &mut self.fwd_probs {
                *prob *= scale;
            }
            self.reservoir_prob *= scale;
        } else {
            // Underflow - reinitialize to uniform
            self.initialize_uniform_prior();
        }
    }

    /// Get emission probability for a target allele at a marker
    ///
    /// # Arguments
    /// * `pattern_id` - The pattern (HMM state)
    /// * `marker_in_window` - Marker offset within window
    /// * `target_allele` - Target genotype (0, 1, or 2 for missing)
    /// * `error_rate` - Genotyping error rate
    ///
    /// # Returns
    /// P(observed | hidden state)
    #[inline]
    pub fn emission_prob(
        &self,
        pattern_id: PatternId,
        marker_in_window: usize,
        target_allele: u8,
        error_rate: f32,
    ) -> f32 {
        let ref_allele = self.pattern_allele(pattern_id, marker_in_window);

        match target_allele {
            0 => {
                // Target is REF
                (1.0 - ref_allele) * (1.0 - error_rate) + ref_allele * error_rate
            }
            1 => {
                // Target is ALT
                ref_allele * (1.0 - error_rate) + (1.0 - ref_allele) * error_rate
            }
            _ => {
                // Missing data - uniform
                0.5
            }
        }
    }

    /// Get total probability mass (should always be ~1.0 after normalization)
    pub fn total_probability(&self) -> f32 {
        self.fwd_probs.iter().sum::<f32>() + self.reservoir_prob
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_micro_window_creation() {
        let window = MicroWindow::new(0, 64, 1000);
        assert_eq!(window.start_marker, 0);
        assert_eq!(window.end_marker, 64);
        assert_eq!(window.window_size(), 64);
        assert_eq!(window.n_ref_haps(), 1000);
        assert_eq!(window.n_patterns(), 0);
    }

    #[test]
    fn test_pattern_allele() {
        let mut window = MicroWindow::new(0, 64, 100);

        // Add a pattern: 0b1010...
        window.unique_patterns.push(0b1010);
        window.pattern_counts.push(1.0);

        let pattern = PatternId::new(0);

        // Check bits
        assert_eq!(window.pattern_allele(pattern, 0), 0.0); // LSB
        assert_eq!(window.pattern_allele(pattern, 1), 1.0);
        assert_eq!(window.pattern_allele(pattern, 2), 0.0);
        assert_eq!(window.pattern_allele(pattern, 3), 1.0);
    }

    #[test]
    fn test_reservoir_allele() {
        let mut window = MicroWindow::new(0, 64, 100);
        window.reservoir_allele_freqs[5] = 0.75;

        let allele = window.pattern_allele(PatternId::RESERVOIR, 5);
        assert_eq!(allele, 0.75);
    }

    #[test]
    fn test_uniform_prior() {
        let mut window = MicroWindow::new(0, 64, 100);

        // Add 3 patterns
        window.unique_patterns = vec![0b001, 0b010, 0b100];
        window.pattern_counts = vec![10.0, 20.0, 30.0];
        window.reservoir_count = 40;

        window.initialize_uniform_prior();

        // Total: 3 patterns + 40 reservoir = 100 haplotypes
        // Each should get 1/100 = 0.01 probability
        let expected = 1.0 / 100.0;

        for &prob in &window.fwd_probs {
            assert!((prob - expected).abs() < 1e-6);
        }

        // Reservoir should have 40 * 0.01 = 0.4
        assert!((window.reservoir_prob - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_normalize() {
        let mut window = MicroWindow::new(0, 64, 100);

        window.unique_patterns = vec![0, 1, 2];
        window.pattern_counts = vec![1.0, 1.0, 1.0];
        window.fwd_probs = vec![2.0, 3.0, 5.0];
        window.reservoir_prob = 10.0;

        // Total = 2 + 3 + 5 + 10 = 20
        window.normalize_forward();

        // Should now sum to 1.0
        let total = window.total_probability();
        assert!((total - 1.0).abs() < 1e-6);

        // Check proportions maintained
        assert!((window.fwd_probs[0] - 0.1).abs() < 1e-6);
        assert!((window.fwd_probs[1] - 0.15).abs() < 1e-6);
        assert!((window.fwd_probs[2] - 0.25).abs() < 1e-6);
        assert!((window.reservoir_prob - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_emission_prob() {
        let mut window = MicroWindow::new(0, 64, 100);

        // Pattern with allele=1 at position 0
        window.unique_patterns.push(0b1);
        window.pattern_counts.push(1.0);

        let pattern = PatternId::new(0);
        let error_rate = 0.01;

        // Target matches (both are 1)
        let prob_match = window.emission_prob(pattern, 0, 1, error_rate);
        assert!((prob_match - 0.99).abs() < 1e-6);

        // Target doesn't match (target=0, ref=1)
        let prob_mismatch = window.emission_prob(pattern, 0, 0, error_rate);
        assert!((prob_mismatch - 0.01).abs() < 1e-6);

        // Missing data
        let prob_missing = window.emission_prob(pattern, 0, 2, error_rate);
        assert_eq!(prob_missing, 0.5);
    }
}
