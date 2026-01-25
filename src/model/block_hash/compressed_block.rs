//! # Compressed Block: Immutable Reference Data
//!
//! This module defines the immutable, Arc-shareable compressed reference data.
//! It is built ONCE per window and shared across all target samples.

use super::types::{GlobalId, PatternId};
// storage removed

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

    /// Flattened reverse mapping: concatenated global haplotype IDs for all patterns
    pub pattern_globals: Vec<GlobalId>,

    /// Offsets into pattern_globals per pattern (len = n_patterns + 1)
    /// Globals for pattern i are pattern_globals[offsets[i]..offsets[i+1]]
    pub pattern_globals_offsets: Vec<usize>,

    /// Number of haplotypes in the reservoir (truncated patterns)
    pub reservoir_count: u32,

    /// Global IDs of haplotypes in the reservoir
    pub reservoir_globals: Vec<GlobalId>,

    /// Allele frequencies of reservoir haplotypes [flattened for all markers]
    /// Layout: [marker0_allele0, marker0_allele1, ..., marker1_allele0...]
    pub reservoir_freqs: Vec<f32>,

    /// Offsets into reservoir_freqs for each marker [marker_in_window]
    pub reservoir_freq_offsets: Vec<usize>,

    /// Reservoir LD coherence factors for adjacent markers (biallelic only).
    /// Layout per interval: [00, 01, 10, 11] for allele pairs (t, t+1).
    pub reservoir_ld: Vec<[f32; 4]>,

    /// Fraction of reservoir haplotypes that are NOT missing at each marker [marker_in_window]
    /// Needed for proper missingness handling in emission probabilities.
    pub reservoir_obs_fractions: Vec<f32>,

    /// Unpacked alleles for fast emission calculation [pattern_idx * window_size + marker_in_window]
    /// Avoids bit-unpacking overhead in hot loops
    pub unpacked_alleles: Vec<u8>,

    /// Recombination rates for each marker interval in the window [marker_in_window]
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

    /// Get global haplotypes for a pattern
    #[inline]
    pub fn pattern_globals(&self, pattern_idx: usize) -> &[GlobalId] {
        let start = self.pattern_globals_offsets[pattern_idx];
        let end = self.pattern_globals_offsets[pattern_idx + 1];
        &self.pattern_globals[start..end]
    }

    /// Get frequency of a specific allele in the reservoir
    #[inline]
    pub fn reservoir_freq(&self, marker_in_window: usize, allele: u8) -> f32 {
        let offset = self.reservoir_freq_offsets[marker_in_window];
        // If allele is out of bounds (shouldn't happen with valid data), return 0.0
        // But for safety/speed we assume caller passes valid allele < n_alleles
        // We can add a debug_assert.
        let n_alleles = self.marker_n_alleles[marker_in_window] as usize;
        if (allele as usize) < n_alleles {
            self.reservoir_freqs[offset + allele as usize]
        } else {
            0.0
        }
    }

    /// Get allele at a specific marker for a pattern (Non-Reservoir Only)
    ///
    /// # Panics
    /// Panics if called on a reservoir pattern ID.
    #[inline]
    pub fn get_pattern_allele(&self, pattern_id: PatternId, marker_in_window: usize) -> u8 {
        assert!(!pattern_id.is_reservoir(), "get_pattern_allele called on reservoir");
        // Fast lookup from unpacked buffer
        let idx = pattern_id.as_usize() * self.window_size() + marker_in_window;
        self.unpacked_alleles[idx]
    }

    /// Get fraction of reservoir haplotypes that are observed (not missing) at this marker
    #[inline]
    pub fn get_reservoir_obs_fraction(&self, marker_in_window: usize) -> f32 {
        if marker_in_window < self.reservoir_obs_fractions.len() {
             self.reservoir_obs_fractions[marker_in_window]
        } else {
             0.0
        }
    }
}
