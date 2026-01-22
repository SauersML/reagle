//! # Haplotype Fingerprinting with Blocked Transposition
//!
//! This module implements cache-efficient fingerprinting of reference haplotypes.
//! The key challenge: BREF3 is stored row-major (marker-by-marker), but fingerprinting
//! requires column-major access (haplotype-by-haplotype).
//!
//! Solution: Process in blocks of 256 haplotypes at a time, transposing within
//! L2 cache to avoid cache thrashing.
//!
//! Uses 2-bit packing to support multiallelic variants (alleles 0-3).
//! Window size reduced to 32 markers to fit in u64 with 2 bits per marker.

use crate::data::haplotype::HapIdx;
use crate::data::marker::MarkerIdx;
use crate::data::storage::matrix::GenotypeMatrix;
use crate::data::storage::phase_state::Phased;
use std::ops::Range;

/// Block size for transposition (256 haplotypes × 32 markers = 8KB, fits in L1 cache)
const BLOCK_SIZE: usize = 256;

/// Window size in markers (reduced to 32 for 2-bit packing in u64)
pub(crate) const WINDOW_SIZE: usize = 32;

/// Fingerprint a window of exactly 32 markers for all haplotypes
///
/// Uses blocked transposition to maintain cache locality despite row-major storage.
/// Uses 2-bit packing to support multiallelic variants (alleles 0-3).
///
/// # Arguments
/// * `ref_data` - Reference panel genotype matrix
/// * `marker_range` - Must be exactly 32 markers
///
/// # Returns
/// Vector of 64-bit fingerprints (one per haplotype, 2 bits per marker)
/// Bits [1:0] = marker 0, bits [3:2] = marker 1, etc.
pub(crate) fn fingerprint_window(
    ref_data: &GenotypeMatrix<Phased>,
    marker_range: Range<usize>,
) -> Vec<u64> {
    assert_eq!(
        marker_range.len(),
        WINDOW_SIZE,
        "Window must be exactly {} markers for u64 2-bit fingerprinting",
        WINDOW_SIZE
    );

    let n_haps = ref_data.n_haplotypes();
    let mut fingerprints = vec![0u64; n_haps];

    // Process in blocks to maintain cache locality
    for hap_block_start in (0..n_haps).step_by(BLOCK_SIZE) {
        let hap_block_end = (hap_block_start + BLOCK_SIZE).min(n_haps);
        let block_width = hap_block_end - hap_block_start;

        // Step 1: Load a transposed block into L1 cache
        // block[marker_in_window][hap_in_block] -> allele (u8)
        let mut block = vec![vec![0u8; block_width]; WINDOW_SIZE];

        // Load from row-major storage
        for (m_idx, marker) in marker_range.clone().enumerate() {
            let ref_col = ref_data.column(MarkerIdx::new(marker as u32));

            for hap_local in 0..block_width {
                let hap_global = hap_block_start + hap_local;
                let allele = ref_col.get(HapIdx::new(hap_global as u32));
                // Clamp to 0-3 range for 2-bit encoding
                // Missing (255) becomes 3, multiallelic >3 becomes 3
                let clamped = if allele >= 4 && allele != 255 { 3 } else { allele.min(3) };
                block[m_idx][hap_local] = clamped;
            }
        }

        // Step 2: Transpose in cache (column-major access now)
        // Pack 2 bits per marker into u64
        for hap_local in 0..block_width {
            let hap_global = hap_block_start + hap_local;
            let mut fp = 0u64;

            // Sequential access through transposed block
            // Each marker uses 2 bits: bits [1:0] for marker 0, [3:2] for marker 1, etc.
            for m_idx in 0..WINDOW_SIZE {
                let allele_2bit = (block[m_idx][hap_local] & 0b11) as u64;
                fp |= allele_2bit << (m_idx * 2);
            }

            fingerprints[hap_global] = fp;
        }
    }

    fingerprints
}


#[cfg(test)]
mod tests {
    // Integration tests with GenotypeMatrix will be in tests/ directory
}
