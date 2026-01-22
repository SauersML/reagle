//! # Compressed Block: Immutable Reference Data
//!
//! This module defines the immutable, Arc-shareable compressed reference data.
//! It is built ONCE per window and shared across all target samples.

use super::types::{GlobalId, PatternId};
use crate::data::storage::dictionary::DictionaryColumn;
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

    /// Compressed storage with unique patterns and hap→pattern mapping
    /// Arc-wrapped for zero-cost sharing across threads
    pub storage: Arc<DictionaryColumn>,

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
}

impl CompressedBlock {
    /// Number of unique patterns (excluding reservoir)
    #[inline]
    pub fn n_patterns(&self) -> usize {
        self.pattern_counts.len()
    }

    /// Window size in markers
    #[inline]
    pub fn window_size(&self) -> usize {
        self.end_marker - self.start_marker
    }

    /// Total number of reference haplotypes
    #[inline]
    pub fn n_ref_haps(&self) -> usize {
        self.storage.n_haplotypes()
    }

    /// Get pattern ID for a global haplotype
    pub fn pattern_for_haplotype(&self, global_id: GlobalId) -> PatternId {
        let hap_to_pattern = self.storage.hap_to_pattern();
        let storage_pattern_id = hap_to_pattern[global_id.as_usize()];

        if (storage_pattern_id as usize) < self.n_patterns() {
            PatternId::new(storage_pattern_id as u16)
        } else {
            PatternId::RESERVOIR
        }
    }

    /// Get allele at a specific marker for a pattern
    #[inline]
    pub fn pattern_allele(&self, pattern_id: PatternId, marker_in_window: usize) -> f32 {
        if pattern_id.is_reservoir() {
            self.reservoir_allele_freqs[marker_in_window]
        } else {
            self.storage
                .pattern_allele(marker_in_window, pattern_id.as_usize()) as f32
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
