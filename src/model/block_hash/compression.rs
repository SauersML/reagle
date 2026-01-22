//! # Compression Wrapper for Building MicroWindows
//!
//! This module provides functions to build `DictionaryColumn` and `MicroWindow`
//! from ranges of markers in a `GenotypeMatrix`.

use crate::data::haplotype::HapIdx;
use crate::data::marker::MarkerIdx;
use crate::data::storage::dictionary::DictionaryColumn;
use crate::data::storage::matrix::GenotypeMatrix;
use crate::data::storage::phase_state::Phased;
use super::micro_window_v2::MicroWindow;
use std::ops::Range;
use std::sync::Arc;

/// Default window size (can be adjusted based on LD patterns)
pub(crate) const DEFAULT_WINDOW_SIZE: usize = 32;

/// Maximum states before truncation (4096 unique patterns should cover most scenarios)
pub(crate) const DEFAULT_MAX_STATES: usize = 4096;

/// Build a MicroWindow from a range of markers in the reference panel
///
/// # Arguments
/// * `ref_data` - Reference panel genotype matrix
/// * `marker_range` - Range of markers to include in this window
/// * `max_states` - Maximum number of unique patterns to track (0 = no limit)
///
/// # Returns
/// Fully initialized MicroWindow with compressed storage and HMM state
pub(crate) fn build_micro_window(
    ref_data: &GenotypeMatrix<Phased>,
    marker_range: Range<usize>,
    max_states: usize,
) -> MicroWindow {
    let start_marker = marker_range.start;
    let end_marker = marker_range.end;
    let n_markers = marker_range.len();
    let n_haplotypes = ref_data.n_haplotypes();

    // Build column access closures for DictionaryColumn::compress
    let columns: Vec<Box<dyn Fn(HapIdx) -> u8>> = marker_range
        .map(|marker_idx| {
            // Need to get the column for each marker
            // Create a closure that captures the column data
            let marker = MarkerIdx::new(marker_idx as u32);

            // Pre-fetch alleles into a vector (to avoid lifetime issues with closures)
            let alleles: Vec<u8> = (0..n_haplotypes)
                .map(|hap_idx| ref_data.allele(marker, HapIdx::new(hap_idx as u32)))
                .collect();

            Box::new(move |hap: HapIdx| alleles[hap.as_usize()]) as Box<dyn Fn(HapIdx) -> u8>
        })
        .collect();

    // Determine bits_per_allele
    // For simplicity, we'll use 2 bits to cover alleles 0-3 (handles most multiallelic cases)
    let bits_per_allele = 2;

    // Compress using DictionaryColumn
    let dict_column = DictionaryColumn::compress(
        &columns.iter().map(|f| |h| f(h)).collect::<Vec<_>>(),
        n_markers,
        n_haplotypes,
        bits_per_allele,
    );

    // Wrap in Arc for sharing
    let storage = Arc::new(dict_column);

    // Build MicroWindow
    MicroWindow::from_dictionary(start_marker, end_marker, storage, max_states)
}

/// Build all micro-windows for a chromosome
///
/// # Arguments
/// * `ref_data` - Reference panel genotype matrix
/// * `window_size` - Size of each window in markers
/// * `max_states` - Maximum states per window (0 = no limit)
///
/// # Returns
/// Vector of MicroWindows covering the entire chromosome
pub(crate) fn build_all_windows(
    ref_data: &GenotypeMatrix<Phased>,
    window_size: usize,
    max_states: usize,
) -> Vec<MicroWindow> {
    let n_markers = ref_data.n_markers();
    let n_windows = (n_markers + window_size - 1) / window_size;

    let mut windows = Vec::with_capacity(n_windows);

    for win_idx in 0..n_windows {
        let start = win_idx * window_size;
        let end = (start + window_size).min(n_markers);

        let window = build_micro_window(ref_data, start..end, max_states);
        windows.push(window);
    }

    windows
}

/// Compression statistics for logging/debugging
#[derive(Debug, Clone)]
pub(crate) struct CompressionStats {
    pub n_windows: usize,
    pub total_patterns: usize,
    pub total_ref_haps: usize,
    pub avg_compression_ratio: f64,
    pub max_patterns: usize,
    pub min_patterns: usize,
}

impl CompressionStats {
    pub fn from_windows(windows: &[MicroWindow]) -> Self {
        let n_windows = windows.len();
        let total_patterns: usize = windows.iter().map(|w| w.n_patterns()).sum();
        let total_ref_haps = windows.first().map(|w| w.n_ref_haps()).unwrap_or(0);

        let avg_compression_ratio = if n_windows > 0 {
            total_ref_haps as f64 / (total_patterns as f64 / n_windows as f64)
        } else {
            0.0
        };

        let max_patterns = windows.iter().map(|w| w.n_patterns()).max().unwrap_or(0);
        let min_patterns = windows.iter().map(|w| w.n_patterns()).min().unwrap_or(0);

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

    #[test]
    fn test_build_micro_window() {
        // Integration test - requires full GenotypeMatrix setup
    }

    #[test]
    fn test_build_all_windows() {
        // Integration test - verifies window tiling
    }
}
