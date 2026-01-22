//! # Compression Wrapper for Building MicroWindows
//!
//! This module provides functions to build `DictionaryColumn` and `MicroWindow`
//! from ranges of markers in a `GenotypeMatrix`.

use crate::data::haplotype::HapIdx;
use crate::data::marker::MarkerIdx;
use crate::data::storage::dictionary::DictionaryColumn;
use crate::data::storage::matrix::GenotypeMatrix;
use crate::data::storage::phase_state::Phased;
use super::compressed_block::CompressedBlock;
use super::types::GlobalId;

use std::ops::Range;
use std::sync::Arc;

/// Default window size (can be adjusted based on LD patterns)
#[allow(unused)]
pub const DEFAULT_WINDOW_SIZE: usize = 32;

/// Maximum states before truncation (4096 unique patterns should cover most scenarios)
#[allow(unused)]
pub const DEFAULT_MAX_STATES: usize = 4096;

/// Build a CompressedBlock from a range of markers (Recommended API)
///
/// This creates immutable, Arc-shareable reference data.
/// Use this instead of build_micro_window for parallel processing.
///
/// # Arguments
/// * `ref_data` - Reference panel genotype matrix
/// * `marker_range` - Range of markers to include in this window
/// * `max_states` - Maximum number of unique patterns to track (0 = no limit)
/// * `recomb_rates` - Vector of recombination rates [marker_in_window]. Size should match marker_range.len().
///
/// # Returns
/// CompressedBlock with only immutable reference data (no per-sample HMM state)
pub fn build_compressed_block(
    ref_data: &GenotypeMatrix<Phased>,
    marker_range: Range<usize>,
    max_states: usize,
    recomb_rates: &[f32],
) -> CompressedBlock {
    let start_marker = marker_range.start;
    let end_marker = marker_range.end;
    let n_markers = marker_range.len();
    let n_haplotypes = ref_data.n_haplotypes();

    assert_eq!(recomb_rates.len(), n_markers, "Recombination rates must match window size");
    let local_recomb_rates = recomb_rates.to_vec();

    // Build column access closures for DictionaryColumn::compress
    let columns: Vec<Box<dyn Fn(HapIdx) -> u8>> = marker_range.clone()
        .map(|marker_idx| {
            let marker = MarkerIdx::new(marker_idx as u32);

            // Pre-fetch alleles into a vector
            let alleles: Vec<u8> = (0..n_haplotypes)
                .map(|hap_idx| ref_data.allele(marker, HapIdx::new(hap_idx as u32)))
                .collect();

            Box::new(move |hap: HapIdx| alleles[hap.as_usize()]) as Box<dyn Fn(HapIdx) -> u8>
        })
        .collect();

    // Determine max allele to set bits_per_allele dynamically
    // Also compute number of alleles per marker
    let mut max_allele_overall = 0u8;
    let mut marker_n_alleles = Vec::with_capacity(n_markers);
    
    for marker_idx in marker_range {
         let marker = MarkerIdx::new(marker_idx as u32);
         let mut max_allele_at_marker = 0u8;
         for hap_idx in 0..n_haplotypes {
             let allele = ref_data.allele(marker, HapIdx::new(hap_idx as u32));
             if allele != 255 {
                 max_allele_at_marker = max_allele_at_marker.max(allele);
             }
         }
         max_allele_overall = max_allele_overall.max(max_allele_at_marker);
         marker_n_alleles.push(max_allele_at_marker + 1);
    }
    
    let bits_per_allele = if max_allele_overall < 2 {
        1
    } else if max_allele_overall < 4 {
        2
    } else if max_allele_overall < 16 {
        4
    } else {
        8
    };

    // Compress using DictionaryColumn
    let dict_column = DictionaryColumn::compress(
        &columns.iter().map(|f| |h| f(h)).collect::<Vec<_>>(),
        n_markers,
        n_haplotypes,
        bits_per_allele,
    );

    let storage = Arc::new(dict_column);

    // Extract pattern metadata from DictionaryColumn
    let hap_to_pattern = storage.hap_to_pattern();
    let n_unique_patterns = storage.n_patterns();

    // Build pattern counts and pattern_to_globals
    let mut pattern_counts = vec![0.0f32; n_unique_patterns];
    let mut pattern_to_globals: Vec<Vec<GlobalId>> = vec![Vec::new(); n_unique_patterns];

    for (hap_idx, &pattern_idx) in hap_to_pattern.iter().enumerate() {
        let global_id = GlobalId::new(hap_idx as u32);
        pattern_counts[pattern_idx as usize] += 1.0;
        pattern_to_globals[pattern_idx as usize].push(global_id);
    }

    // Handle truncation if needed
    let (pattern_counts, pattern_to_globals, reservoir_count, reservoir_globals, reservoir_allele_freqs) =
        if max_states > 0 && n_unique_patterns > max_states {
            // Truncate to top max_states by count
            let mut pattern_order: Vec<usize> = (0..n_unique_patterns).collect();
            pattern_order.sort_by(|&a, &b| {
                pattern_counts[b]
                    .partial_cmp(&pattern_counts[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

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
                compute_reservoir_freqs(&storage, &reservoir_globals, n_markers)
            } else {
                vec![0.5; n_markers]
            };

            // Keep only top max_states
            let kept_indices = &pattern_order[..max_states];
            let pattern_counts = kept_indices.iter().map(|&i| pattern_counts[i]).collect();
            let pattern_to_globals = kept_indices
                .iter()
                .map(|&i| pattern_to_globals[i].clone())
                .collect();

            (
                pattern_counts,
                pattern_to_globals,
                reservoir_count,
                reservoir_globals,
                reservoir_allele_freqs,
            )
        } else {
            // No truncation needed
            (
                pattern_counts,
                pattern_to_globals,
                0,
                Vec::new(),
                vec![0.5; n_markers],
            )
        };

    // Pre-unpack alleles for all kept patterns to avoid bit-unpacking in hot loops
    // Flattened: [pattern_idx * n_markers + marker_idx]
    let n_kept_patterns = pattern_counts.len();
    let mut unpacked_alleles = vec![0u8; n_kept_patterns * n_markers];

    for (kept_idx, globals) in pattern_to_globals.iter().enumerate() {
        if let Some(first_global) = globals.first() {
             let hap = HapIdx::new(first_global.as_u32());
             for m in 0..n_markers {
                 let allele = storage.get(m, hap);
                 unpacked_alleles[kept_idx * n_markers + m] = allele;
             }
        }
    }

    CompressedBlock {
        start_marker,
        end_marker,
        storage,
        pattern_counts,
        pattern_to_globals,
        reservoir_count,
        reservoir_globals,
        reservoir_allele_freqs,
        unpacked_alleles,
        local_recomb_rates,
        marker_n_alleles,
    }
}

/// Compute allele frequencies for reservoir haplotypes
fn compute_reservoir_freqs(
    storage: &Arc<DictionaryColumn>,
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
            if allele != 255 && allele > 0 {
                allele_sums[marker_offset] += 1;
            }
        }
    }

    allele_sums
        .iter()
        .map(|&sum| sum as f32 / n_reservoir as f32)
        .collect()
}

/// Compression statistics for logging/debugging
#[derive(Debug, Clone)]
#[allow(unused)]
pub struct CompressionStats {
    pub n_windows: usize,
    pub total_patterns: usize,
    pub total_ref_haps: usize,
    pub avg_compression_ratio: f64,
    pub max_patterns: usize,
    pub min_patterns: usize,
}

#[allow(unused)]
impl CompressionStats {
    pub fn from_blocks(blocks: &[Arc<CompressedBlock>]) -> Self {
        let n_windows = blocks.len();
        let total_patterns: usize = blocks.iter().map(|w| w.n_patterns()).sum();
        let total_ref_haps = blocks.first().map(|w| w.n_ref_haps()).unwrap_or(0);

        let avg_compression_ratio = if n_windows > 0 {
            total_ref_haps as f64 / (total_patterns as f64 / n_windows as f64)
        } else {
            0.0
        };

        let max_patterns = blocks.iter().map(|w| w.n_patterns()).max().unwrap_or(0);
        let min_patterns = blocks.iter().map(|w| w.n_patterns()).min().unwrap_or(0);

        Self {
            n_windows,
            total_patterns,
            total_ref_haps,
            avg_compression_ratio,
            max_patterns,
            min_patterns,
        }
    }

    pub fn print_summary(&self) {
        println!("Compression Statistics:");
        println!("  Windows: {}", self.n_windows);
        println!("  Total patterns: {}", self.total_patterns);
        println!("  Reference haplotypes: {}", self.total_ref_haps);
        println!(
            "  Avg compression ratio: {:.2}x",
            self.avg_compression_ratio
        );
        println!("  Pattern range: {} - {}", self.min_patterns, self.max_patterns);
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;

    #[test]
    fn test_build_compressed_block() {
        // Integration test - requires full GenotypeMatrix setup
    }
}
