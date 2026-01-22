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
use super::types::{GlobalId, PatternId};

use std::ops::Range;
use std::sync::Arc;

/// Default window size (can be adjusted based on LD patterns)
pub const DEFAULT_WINDOW_SIZE: usize = 32;

/// Maximum states before truncation (0 = no limit)
pub const DEFAULT_MAX_STATES: usize = 0;

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

    // Handle truncation if needed
    // max_states=0 means no limit (use usize::MAX)
    let limit = if max_states == 0 { usize::MAX } else { max_states };

    let (pattern_counts, pattern_to_globals, reservoir_count, reservoir_globals, reservoir_allele_freqs, hap_to_state) =
        if n_unique_patterns > limit {
            // Truncate to top max_states by count
            let mut pattern_order: Vec<usize> = (0..n_unique_patterns).collect();
            pattern_order.sort_by(|&a, &b| {
                pattern_counts[b]
                    .partial_cmp(&pattern_counts[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Keep top limit
            let kept_indices = &pattern_order[..limit];
            let kept_patterns_set: std::collections::HashSet<usize> =
                kept_indices.iter().copied().collect();
            
            // Build old_id -> new_id map
            let mut old_to_new = vec![PatternId::RESERVOIR; n_unique_patterns];
            for (new_idx, &old_idx) in kept_indices.iter().enumerate() {
                old_to_new[old_idx] = PatternId::new(new_idx as u32);
            }

            // Build hap_to_state
            let hap_to_state: Vec<PatternId> = hap_to_pattern
                .iter()
                .map(|&old_idx| old_to_new[old_idx as usize])
                .collect();

            // Build reservoir globals
            let mut reservoir_globals = Vec::new();
            for (pattern_idx, globals) in pattern_to_globals.iter().enumerate() {
                if !kept_patterns_set.contains(&pattern_idx) {
                    reservoir_globals.extend(globals.iter().copied());
                }
            }

            let reservoir_count = reservoir_globals.len() as u32;

            // Compute reservoir allele frequencies
            let (reservoir_freqs, reservoir_freq_offsets) = if reservoir_count > 0 {
                compute_reservoir_freqs(&storage, &reservoir_globals, n_markers, &marker_n_alleles)
            } else {
                // Empty placeholders
                (Vec::new(), vec![0; n_markers])
            };

            // Filter/Reorder pattern_counts and pattern_to_globals
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
                reservoir_freqs,
                reservoir_freq_offsets,
                hap_to_state,
            )
        } else {
            // No truncation needed
            let hap_to_state: Vec<PatternId> = hap_to_pattern
                .iter()
                .map(|&idx| PatternId::new(idx as u32))
                .collect();

            (
                pattern_counts,
                pattern_to_globals,
                0,
                Vec::new(),
                Vec::new(),
                vec![0; n_markers],
                hap_to_state,
            )
        };

    // Pre-unpack alleles for all kept patterns
    // Flattened: [pattern_idx * n_markers + marker_idx]
    let n_kept_patterns = pattern_counts.len();
    // Pre-fill with 255 (missing) or 0? 0 is safer default, but data should overwrite.
    let mut unpacked_alleles = vec![0u8; n_kept_patterns * n_markers];

    // Note: pattern_to_globals is now indexed by usage rank
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
        hap_to_state, // Replaces storage
        pattern_counts,
        pattern_to_globals,
        reservoir_count,
        reservoir_globals,
        reservoir_freqs,
        reservoir_freq_offsets,
        unpacked_alleles,
        local_recomb_rates,
        marker_n_alleles,
    }
}

/// Compute allele frequencies for reservoir haplotypes (Multiallelic)
fn compute_reservoir_freqs(
    storage: &Arc<DictionaryColumn>,
    reservoir_globals: &[GlobalId],
    window_size: usize,
    marker_n_alleles: &[u8],
) -> (Vec<f32>, Vec<usize>) {
    let n_reservoir = reservoir_globals.len();
    if n_reservoir == 0 {
        return (Vec::new(), vec![0; window_size]);
    }

    // Calculate total size and offsets
    let mut offsets = Vec::with_capacity(window_size);
    let mut current_offset = 0;
    for &n in marker_n_alleles {
        offsets.push(current_offset);
        current_offset += n as usize;
    }
    let total_slots = current_offset;
    
    let mut counts = vec![0u32; total_slots];

    for &global_id in reservoir_globals {
        let hap = HapIdx::new(global_id.as_u32());
        for marker_idx in 0..window_size {
            let allele = storage.get(marker_idx, hap);
            // Skip missing data in frequency calculation? 
            // Or treat as "no info"?
            // Usually we only count observed alleles.
            // If allele is 255, we ignore it.
            if allele != 255 {
                let n_alleles = marker_n_alleles[marker_idx] as usize;
                if (allele as usize) < n_alleles {
                    let offset = offsets[marker_idx];
                    counts[offset + allele as usize] += 1;
                }
            }
        }
    }

    let freqs: Vec<f32> = counts
        .iter()
        .map(|&c| {
            if n_reservoir > 0 {
                c as f32 / n_reservoir as f32
            } else {
                0.0
            }
        })
        .collect();

    (freqs, offsets)
}

/// Compression statistics for logging/debugging
#[derive(Debug, Clone)]
pub struct CompressionStats {
    pub n_windows: usize,
    pub total_patterns: usize,
    pub total_ref_haps: usize,
    pub avg_compression_ratio: f64,
    pub max_patterns: usize,
    pub min_patterns: usize,
}

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
mod tests {
    use super::*;

    use crate::data::marker::Markers;
    use crate::data::storage::GenotypeColumn;
    use crate::data::storage::phase_state::Phased;

    #[test]
    fn test_build_compressed_block() {
        // Setup simple scenario: 4 haplotypes, 2 markers
        // Hap 0: (0, 0)
        // Hap 1: (0, 0)
        // Hap 2: (1, 1)
        // Hap 3: (1, 1)
        
        let col0 = GenotypeColumn::from_alleles(&[0, 0, 1, 1], 2);
        let col1 = GenotypeColumn::from_alleles(&[0, 0, 1, 1], 2);
        
        // Mock GenotypeMatrix wrapper equivalent
        // Since GenotypeMatrix is complex to mock, we'll implement a minimal verified fake or use public API if feasible.
        // The public API requires Samples, Markers etc. 
        // Let's create a minimal valid GenotypeMatrix.
        use crate::data::haplotype::Samples;
        use crate::data::marker::{Marker, Allele};
        use crate::data::ChromIdx;
        let samples = Arc::new(Samples::from_ids(vec!["S1".to_string(), "S2".to_string()]));
        let mut m = Markers::new();
        let chr = m.add_chrom("chr1");
        m.push(Marker::new(chr, 100, None, Allele::Base(0), vec![Allele::Base(1)]));
         m.push(Marker::new(chr, 200, None, Allele::Base(0), vec![Allele::Base(1)]));

        let gt = GenotypeMatrix::new_phased(
            m,
            vec![col0, col1],
            samples,
        );

        let recomb_rates = vec![0.01, 0.01];
        let block = build_compressed_block(&gt, 0..2, 0, &recomb_rates);

        // Verify compression
        // Haps 0,1 should be Pattern A
        // Haps 2,3 should be Pattern B
        assert_eq!(block.n_patterns(), 2);
        assert_eq!(block.n_ref_haps(), 4);
        
        let p0 = block.pattern_for_haplotype(GlobalId::new(0));
        let p1 = block.pattern_for_haplotype(GlobalId::new(1));
        let p2 = block.pattern_for_haplotype(GlobalId::new(2));
        let p3 = block.pattern_for_haplotype(GlobalId::new(3));

        assert_eq!(p0, p1);
        assert_eq!(p2, p3);
        assert_ne!(p0, p2);

        // Check counts
        assert_eq!(block.pattern_counts[p0.as_usize()], 2.0);
        assert_eq!(block.pattern_counts[p2.as_usize()], 2.0);

        // Check unpacked alleles
        // Pattern 0: (0, 0)
        assert_eq!(block.pattern_allele(p0, 0), 0.0);
        assert_eq!(block.pattern_allele(p0, 1), 0.0);
        // Pattern 1: (1, 1)
        assert_eq!(block.pattern_allele(p2, 0), 1.0);
        assert_eq!(block.pattern_allele(p2, 1), 1.0);
    }
}
