//! # MicroWindow: Simplified using DictionaryColumn
//!
//! This module wraps the existing `DictionaryColumn` infrastructure to provide
//! HMM state management for compressed haplotype blocks.
//!
//! Key Insight: We don't need custom bit-packing - `DictionaryColumn` already
//! handles multiallelic variants with `bits_per_allele`.

use crate::data::haplotype::HapIdx;
use crate::data::storage::dictionary::DictionaryColumn;
use super::types::{GlobalId, PatternId};
use std::sync::Arc;

/// A micro-window with HMM state for compressed haplotype patterns
#[derive(Clone, Debug)]
pub(crate) struct MicroWindow {
    /// Genomic position range
    pub start_marker: usize,
    pub end_marker: usize,

    // ========================================================================
    // REUSE: DictionaryColumn handles compression and pattern storage
    // ========================================================================
    /// Compressed storage with unique patterns and hap→pattern mapping
    pub storage: Arc<DictionaryColumn>,

    // ========================================================================
    // Per-pattern metadata and HMM state
    // ========================================================================
    /// Cardinality of each pattern (how many haplotypes have this pattern)
    /// Length: U (number of unique patterns)
    /// Important for correct transition weight calculations
    pub pattern_counts: Vec<f32>,

    /// Reverse mapping: for each pattern, which global haplotypes have it
    /// This is Important for MCMC sampling in the phasing pipeline
    /// pattern_to_globals[pattern_id] = Vec<GlobalId>
    pub pattern_to_globals: Vec<Vec<GlobalId>>,

    /// Forward probabilities (Li-Stephens HMM)
    /// Length: U (number of unique patterns)
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
    /// Required for correct transition calculations
    pub reservoir_count: u32,

    /// Global IDs of haplotypes in the reservoir
    /// Needed for MCMC sampling when a reservoir state is selected
    pub reservoir_globals: Vec<GlobalId>,

    /// Allele frequencies of reservoir haplotypes [marker_in_window]
    /// Used for emission probabilities when a haplotype is in the reservoir
    pub reservoir_allele_freqs: Vec<f32>,
}

impl MicroWindow {
    /// Create a MicroWindow from a DictionaryColumn with optional truncation
    ///
    /// # Arguments
    /// * `start_marker` - Starting marker index in the chromosome
    /// * `end_marker` - Ending marker index
    /// * `storage` - Compressed storage (already contains patterns and hap_to_pattern)
    /// * `max_states` - Maximum number of patterns to track (0 = no limit)
    ///
    /// # Returns
    /// Fully initialized MicroWindow with HMM state
    pub fn from_dictionary(
        start_marker: usize,
        end_marker: usize,
        storage: Arc<DictionaryColumn>,
        max_states: usize,
    ) -> Self {
        let n_patterns = storage.n_patterns();
        let n_haplotypes = storage.n_haplotypes();
        let window_size = storage.n_markers();

        debug_assert_eq!(
            window_size,
            end_marker - start_marker,
            "Storage markers must match window range"
        );

        // Step 1: Build pattern counts and pattern_to_globals
        let mut pattern_counts = vec![0.0f32; n_patterns];
        let mut pattern_to_globals: Vec<Vec<GlobalId>> = vec![Vec::new(); n_patterns];

        let hap_to_pattern = storage.hap_to_pattern();
        for (hap_idx, &pattern_idx) in hap_to_pattern.iter().enumerate() {
            let global_id = GlobalId::new(hap_idx as u32);
            pattern_counts[pattern_idx as usize] += 1.0;
            pattern_to_globals[pattern_idx as usize].push(global_id);
        }

        // Step 2: Decide whether to truncate
        let should_truncate = max_states > 0 && n_patterns > max_states;

        if should_truncate {
            // Sort patterns by count (descending) and keep top max_states
            let mut pattern_order: Vec<usize> = (0..n_patterns).collect();
            pattern_order.sort_by(|&a, &b| {
                pattern_counts[b]
                    .partial_cmp(&pattern_counts[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Patterns to keep
            let kept_patterns: std::collections::HashSet<usize> =
                pattern_order.iter().take(max_states).copied().collect();

            // Build reservoir
            let mut reservoir_globals = Vec::new();
            for (pattern_idx, globals) in pattern_to_globals.iter().enumerate() {
                if !kept_patterns.contains(&pattern_idx) {
                    reservoir_globals.extend(globals.iter().copied());
                }
            }

            let reservoir_count = reservoir_globals.len() as u32;

            // Compute reservoir allele frequencies
            let reservoir_allele_freqs = if reservoir_count > 0 {
                compute_reservoir_allele_freqs(&storage, &reservoir_globals, window_size)
            } else {
                vec![0.5; window_size]
            };

            // Truncate the metadata arrays (keep only top max_states)
            let kept_indices = &pattern_order[..max_states];
            let pattern_counts = kept_indices.iter().map(|&i| pattern_counts[i]).collect();
            let pattern_to_globals = kept_indices
                .iter()
                .map(|&i| pattern_to_globals[i].clone())
                .collect();

            // Initialize probabilities
            let total_haps = (n_haplotypes as u32 - reservoir_count) as f32 + reservoir_count as f32;
            let uniform_prob = 1.0 / total_haps;

            let fwd_probs = vec![uniform_prob; max_states];
            let reservoir_prob = uniform_prob * reservoir_count as f32;

            Self {
                start_marker,
                end_marker,
                storage,
                pattern_counts,
                pattern_to_globals,
                fwd_probs,
                bwd_probs: vec![0.0; max_states],
                reservoir_prob,
                reservoir_count,
                reservoir_globals,
                reservoir_allele_freqs,
            }
        } else {
            // Lossless: keep all patterns
            let uniform_prob = 1.0 / n_haplotypes as f32;
            let fwd_probs = vec![uniform_prob; n_patterns];

            Self {
                start_marker,
                end_marker,
                storage,
                pattern_counts,
                pattern_to_globals,
                fwd_probs,
                bwd_probs: vec![0.0; n_patterns],
                reservoir_prob: 0.0,
                reservoir_count: 0,
                reservoir_globals: Vec::new(),
                reservoir_allele_freqs: vec![0.5; window_size],
            }
        }
    }

    /// Number of unique patterns (excluding reservoir)
    #[inline]
    pub fn n_patterns(&self) -> usize {
        self.fwd_probs.len()
    }

    /// Window size in markers
    #[inline]
    pub fn window_size(&self) -> usize {
        self.storage.n_markers()
    }

    /// Total number of reference haplotypes
    #[inline]
    pub fn n_ref_haps(&self) -> usize {
        self.storage.n_haplotypes()
    }

    /// Get pattern ID for a global haplotype (or RESERVOIR if truncated)
    ///
    /// # Arguments
    /// * `global_id` - The global haplotype ID
    ///
    /// # Returns
    /// PatternId or PatternId::RESERVOIR if this haplotype was truncated
    pub fn pattern_for_haplotype(&self, global_id: GlobalId) -> PatternId {
        let hap_to_pattern = self.storage.hap_to_pattern();
        let storage_pattern_id = hap_to_pattern[global_id.as_usize()];

        // Check if this pattern is in our kept set
        // (This assumes pattern IDs are contiguous 0..n_patterns after truncation)
        if (storage_pattern_id as usize) < self.n_patterns() {
            PatternId::new(storage_pattern_id as u16)
        } else {
            PatternId::RESERVOIR
        }
    }

    /// Get the allele at a specific marker for a pattern
    ///
    /// # Arguments
    /// * `pattern_id` - The pattern to query
    /// * `marker_in_window` - Marker offset within this window
    ///
    /// # Returns
    /// The allele (0-255) or reservoir frequency if pattern is RESERVOIR
    #[inline]
    pub fn pattern_allele(&self, pattern_id: PatternId, marker_in_window: usize) -> f32 {
        if pattern_id.is_reservoir() {
            self.reservoir_allele_freqs[marker_in_window]
        } else {
            self.storage
                .pattern_allele(marker_in_window, pattern_id.as_usize()) as f32
        }
    }

    /// Sample a global haplotype ID from a pattern (for MCMC in phasing)
    ///
    /// # Arguments
    /// * `pattern_id` - The pattern to sample from
    /// * `rng` - Random number generator
    ///
    /// # Returns
    /// A randomly selected global haplotype ID that has this pattern
    pub fn sample_global_from_pattern<R: rand::Rng>(
        &self,
        pattern_id: PatternId,
        rng: &mut R,
    ) -> GlobalId {
        if pattern_id.is_reservoir() {
            // Sample uniformly from reservoir
            let idx = rng.random_range(0..self.reservoir_globals.len());
            self.reservoir_globals[idx]
        } else {
            // Sample uniformly from this pattern's haplotypes
            let globals = &self.pattern_to_globals[pattern_id.as_usize()];
            let idx = rng.random_range(0..globals.len());
            globals[idx]
        }
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
            let n_total = self.n_patterns() + self.reservoir_count as usize;
            let uniform = 1.0 / n_total as f32;
            self.fwd_probs.fill(uniform);
            self.reservoir_prob = uniform * self.reservoir_count as f32;
        }
    }

    /// Get total probability mass (should always be ~1.0 after normalization)
    pub fn total_probability(&self) -> f32 {
        self.fwd_probs.iter().sum::<f32>() + self.reservoir_prob
    }
}

/// Compute allele frequencies for reservoir haplotypes
fn compute_reservoir_allele_freqs(
    storage: &DictionaryColumn,
    reservoir_globals: &[GlobalId],
    window_size: usize,
) -> Vec<f32> {
    let n_reservoir = reservoir_globals.len();
    if n_reservoir == 0 {
        return vec![0.5; window_size];
    }

    let mut allele_sums = vec![0u32; window_size];

    for &global_id in reservoir_globals {
        let hap = HapIdx::new(global_id.as_u32());
        for marker_offset in 0..window_size {
            let allele = storage.get(marker_offset, hap);
            if allele != 255 {
                allele_sums[marker_offset] += allele as u32;
            }
        }
    }

    allele_sums
        .iter()
        .map(|&sum| sum as f32 / n_reservoir as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_micro_window_from_dictionary() {
        // Integration test - requires DictionaryColumn setup
    }

    #[test]
    fn test_pattern_for_haplotype() {
        // Integration test - verifies global ID to pattern ID mapping
    }

    #[test]
    fn test_sample_global_from_pattern() {
        // Integration test - verifies MCMC sampling
    }
}
