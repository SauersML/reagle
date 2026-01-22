//! # Compressed Block: Immutable Reference Data
//!
//! This module defines the immutable, Arc-shareable compressed reference data.
//! It is built ONCE per window and shared across all target samples.

use super::types::{GlobalId, PatternId};
use super::types::{GlobalId, PatternId};
// storage removed
use std::sync::Arc;

/// Immutable compressed reference data for a window
///
/// This is Arc-wrapped and shared across all target samples.
/// Contains only the reference panel compression, NOT per-sample HMM state.
#[derive(Clone, Debug)]
pub struct CompressedBlock {
    /// Genomic position range
    pub start_marker: usize,
    pub end_marker: usize,

    /// Final mapping from global haplotype ID to state ID (PatternId or RESERVOIR)
    /// This resolves the remapping issues by pre-calculating the exact state for each hap.
    pub hap_to_state: Vec<PatternId>,

    /// Cardinality of each pattern (how many haplotypes have this pattern)
    /// Required for correct transition weight calculations
    pub pattern_counts: Vec<f32>,

    /// Reverse mapping: for each pattern, which global haplotypes have it
    /// Required for MCMC sampling in phasing pipeline
    pub pattern_to_globals: Vec<Vec<GlobalId>>,

    /// Number of haplotypes in the reservoir (truncated patterns)
    pub reservoir_count: u32,

    /// Global IDs of haplotypes in the reservoir
    pub reservoir_globals: Vec<GlobalId>,

    /// Allele frequencies of reservoir haplotypes [marker_in_window]
    /// Used for emission probabilities
    pub reservoir_allele_freqs: Vec<f32>,

    /// Unpacked alleles for fast emission calculation [pattern_idx * window_size + marker_in_window]
    /// Avoids bit-unpacking overhead in hot loops
    pub unpacked_alleles: Vec<u8>,

    /// Recombination rates for each marker interval in the window [marker_in_window]
    /// Rate at index i is the probability of recombination between i and i+1?
    /// Or between i-1 and i?
    /// Standard HMM: Transition from t-1 to t uses rate[t-1]?
    /// We will store rate[i] = P(recomb between i and i+1).
    /// The last marker's rate is used for transition to next block.
    /// Wait, `TransitionBridge` handles block-to-block.
    /// Inside block, we use rates for transitions i -> i+1.
    /// So size is window_size (last one unused? or used for bridge?).
    /// Bridge uses its own rate.
    /// So we need rates for 0..window_size-1.
    /// Let's store full vector size `window_size` for simplicity, last element might be unused or passed to bridge.
    pub local_recomb_rates: Vec<f32>,

    /// Number of alleles at each marker (2 for biallelic, >2 for multiallelic)
    pub marker_n_alleles: Vec<u8>,
}

impl CompressedBlock {
    /// Number of unique patterns (excluding reservoir)
    #[inline]
    pub fn n_patterns(&self) -> usize {
        self.pattern_counts.len()
    }

    /// Number of alleles at a specific marker
    #[inline]
    pub fn n_alleles(&self, marker_in_window: usize) -> usize {
        self.marker_n_alleles[marker_in_window] as usize
    }

    /// Window size in markers
    #[inline]
    pub fn window_size(&self) -> usize {
        self.end_marker - self.start_marker
    }

    /// Total number of reference haplotypes
    #[inline]
    pub fn n_ref_haps(&self) -> usize {
        self.hap_to_state.len()
    }

    /// Get pattern ID for a global haplotype
    #[inline]
    pub fn pattern_for_haplotype(&self, global_id: GlobalId) -> PatternId {
        self.hap_to_state[global_id.as_usize()]
    }

    /// Get allele at a specific marker for a pattern
    #[inline]
    pub fn pattern_allele(&self, pattern_id: PatternId, marker_in_window: usize) -> f32 {
        if pattern_id.is_reservoir() {
            self.reservoir_allele_freqs[marker_in_window]
        } else {
            // Fast lookup from unpacked buffer
            let idx = pattern_id.as_usize() * self.window_size() + marker_in_window;
            self.unpacked_alleles[idx] as f32
        }
    }

    /// Sample a global haplotype ID from a pattern (for MCMC)
    pub fn sample_global_from_pattern<R: rand::Rng>(
        &self,
        pattern_id: PatternId,
        rng: &mut R,
    ) -> GlobalId {
        if pattern_id.is_reservoir() {
            let idx = rng.random_range(0..self.reservoir_globals.len());
            self.reservoir_globals[idx]
        } else {
            let globals = &self.pattern_to_globals[pattern_id.as_usize()];
            let idx = rng.random_range(0..globals.len());
            globals[idx]
        }
    }
}
