//! # Compressed Block: Immutable Reference Data
//!
//! This module defines the immutable, Arc-shareable compressed reference data.
//! It is built ONCE per window and shared across all target samples.

use super::types::{GlobalId, PatternId};
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

    /// Reverse mapping: for each pattern, which global haplotypes have it
    /// Required for MCMC sampling in phasing pipeline
    pub pattern_to_globals: Vec<Vec<GlobalId>>,

    /// Number of haplotypes in the reservoir (truncated patterns)
    pub reservoir_count: u32,

    /// Global IDs of haplotypes in the reservoir
    pub reservoir_globals: Vec<GlobalId>,

    /// Allele frequencies of reservoir haplotypes [flattened for all markers]
    /// Layout: [marker0_allele0, marker0_allele1, ..., marker1_allele0...]
    pub reservoir_freqs: Vec<f32>,

    /// Offsets into reservoir_freqs for each marker [marker_in_window]
    pub reservoir_freq_offsets: Vec<usize>,

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

    /// Get allele at a specific marker for a pattern
    /// Note: For reservoir, this returns the frequency of allele 1 (for compatibility with existing logic?)
    /// WARNING: This old API returned f32. If strictly biallelic, it returned freq(1).
    /// But for multiallelic, "allele" isn't a single scalar.
    /// This method is deprecated/dangerous for multiallelic reservoir.
    /// We should probably remove it or only use it for non-reservoir.
    /// However, `hmm.rs` might rely on it. Let's see...
    /// `hmm.rs` uses `pattern_allele` to get REF allele.
    /// If reservoir, it was using it to get "ref_allele" which was actually frequency.
    /// The `emission_prob` updates will fix the usage site to use `reservoir_freq` directly.
    /// So this method should only be used for NON-reservoir patterns where it returns the exact allele (as f32).
    #[inline]
    pub fn pattern_allele(&self, pattern_id: PatternId, marker_in_window: usize) -> f32 {
        if pattern_id.is_reservoir() {
            // Fallback for legacy calls (biallelic logic): return freq of allele 1
            // This preserves behavior for K=2 but is meaningless for K>2
            self.reservoir_freq(marker_in_window, 1)
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
