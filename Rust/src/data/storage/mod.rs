//! # Genotype Storage Backends
//!
//! Polymorphic storage for genotype data. Replaces Java's `GTRec` class hierarchy
//! with a single Rust enum.

pub mod coded_steps;
pub mod dense;
pub mod dictionary;
pub mod matrix;
pub mod mutable;
pub mod phase_state;
pub mod sample_phase;
pub mod seq_coded;
pub mod sparse;
pub mod view;

pub use dense::DenseColumn;
pub use dictionary::DictionaryColumn;
pub use matrix::GenotypeMatrix;
pub use mutable::MutableGenotypes;
pub use phase_state::PhaseState;
pub use seq_coded::{SeqCodedBlock, SeqCodedColumn};
pub use sparse::SparseColumn;
pub use view::GenotypeView;

use crate::data::HapIdx;
use std::sync::Arc;

/// The core genotype storage enum - replaces Java's class hierarchy
#[derive(Clone, Debug)]
pub enum GenotypeColumn {
    /// High-frequency variants (MAF > 0.01).
    /// Bit-packed: 1 bit per haplotype for biallelic sites.
    Dense(DenseColumn),

    /// Rare variants (MAF < 0.01).
    /// Store only indices of ALT allele carriers.
    Sparse(SparseColumn),

    /// Dictionary-compressed blocks.
    /// For runs of similar haplotype patterns.
    /// Stores (Shared Dictionary, Marker Offset in Dictionary)
    Dictionary(Arc<DictionaryColumn>, usize),

    /// Sequence-coded blocks (BREF3 native format).
    /// Preserves compact hap->seq->allele mapping without expansion.
    /// ~6x memory savings vs Dense for typical reference panels.
    SeqCoded(SeqCodedColumn),
}

impl GenotypeColumn {
    /// Get allele for a specific haplotype (0 = REF, 1+ = ALT)
    #[inline]
    pub fn get(&self, hap: HapIdx) -> u8 {
        match self {
            Self::Dense(col) => col.get(hap),
            Self::Sparse(col) => col.get(hap),
            Self::Dictionary(col, offset) => col.get(*offset, hap),
            Self::SeqCoded(col) => col.get(hap),
        }
    }

    /// Number of haplotypes in this column
    pub fn n_haplotypes(&self) -> usize {
        match self {
            Self::Dense(col) => col.n_haplotypes(),
            Self::Sparse(col) => col.n_haplotypes(),
            Self::Dictionary(col, _) => col.n_haplotypes(),
            Self::SeqCoded(col) => col.n_haplotypes(),
        }
    }

    /// Count of ALT allele carriers
    pub fn alt_count(&self) -> usize {
        match self {
            Self::Dense(col) => col.alt_count(),
            Self::Sparse(col) => col.n_carriers(),
            Self::Dictionary(col, offset) => col.alt_count(*offset),
            Self::SeqCoded(col) => col.alt_count(),
        }
    }

    /// Minor allele frequency
    pub fn maf(&self) -> f64 {
        let n = self.n_haplotypes();
        if n == 0 {
            return 0.0;
        }
        let alt = self.alt_count();
        let freq = alt as f64 / n as f64;
        freq.min(1.0 - freq)
    }

    /// Memory usage in bytes (approximate)
    pub fn size_bytes(&self) -> usize {
        match self {
            Self::Dense(col) => col.size_bytes(),
            Self::Sparse(col) => col.size_bytes(),
            Self::Dictionary(col, _) => col.size_bytes() / col.n_markers().max(1),
            Self::SeqCoded(col) => col.size_bytes(),
        }
    }

    /// Create from allele slice, automatically choosing storage type
    pub fn from_alleles(alleles: &[u8], n_alleles: usize) -> Self {
        let n_haps = alleles.len();
        if n_haps == 0 {
            return Self::Dense(DenseColumn::new(0, 1));
        }

        // Count ALT carriers for MAF calculation (ignore missing = 255)
        let alt_count = alleles.iter().filter(|&&a| a > 0 && a != 255).count();
        let present_count = alleles.iter().filter(|&&a| a != 255).count();
        
        let maf = if present_count > 0 {
            let freq = alt_count as f64 / present_count as f64;
            freq.min(1.0 - freq)
        } else {
            0.0
        };

        // Use sparse storage for rare variants (MAF < 1%)
        // Note: SparseColumn currently doesn't support missing data explicitly,
        // so we only use it if there are no missing alleles or if we are okay
        // with missing being treated as REF/ALT in sparse storage (usually rare variants don't have many missing).
        // For correctness, if there's missing data, use Dense storage.
        let has_missing = alleles.iter().any(|&a| a == 255);

        if maf < 0.01 && n_alleles == 2 && !has_missing {
            // Determine if we should store ALT or REF carriers (whichever is fewer)
            if alt_count <= n_haps / 2 {
                let carriers: Vec<HapIdx> = alleles
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| **a > 0)
                    .map(|(i, _)| HapIdx::new(i as u32))
                    .collect();
                Self::Sparse(SparseColumn::from_carriers(carriers, n_haps as u32, false))
            } else {
                let carriers: Vec<HapIdx> = alleles
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| **a == 0)
                    .map(|(i, _)| HapIdx::new(i as u32))
                    .collect();
                Self::Sparse(SparseColumn::from_carriers(carriers, n_haps as u32, true))
            }
        } else {
            Self::Dense(DenseColumn::from_alleles(
                alleles.iter().copied(),
                n_alleles,
            ))
        }
    }

}

impl Default for GenotypeColumn {
    fn default() -> Self {
        Self::Dense(DenseColumn::new(0, 1))
    }
}

/// Compress a block of markers using dictionary encoding when beneficial
///
/// Dictionary compression groups haplotypes by their allele patterns across
/// multiple markers. This is effective when many haplotypes share the same
/// local sequence (common in reference panels with LD structure).
///
/// Returns `Some(DictionaryColumn)` if compression ratio is favorable (< 0.5),
/// otherwise returns `None`.
pub fn compress_block<F>(
    get_allele: F,
    n_markers: usize,
    n_haplotypes: usize,
    bits_per_allele: u8,
) -> Option<DictionaryColumn>
where
    F: Fn(usize, HapIdx) -> u8,
{
    // Only compress if block has enough markers
    if n_markers < 4 || n_haplotypes == 0 {
        return None;
    }

    // Build closures for each marker
    let columns: Vec<Box<dyn Fn(HapIdx) -> u8>> = (0..n_markers)
        .map(|m| {
            let get_allele_ref = &get_allele;
            Box::new(move |h: HapIdx| get_allele_ref(m, h)) as Box<dyn Fn(HapIdx) -> u8>
        })
        .collect();

    // Create wrapper closures that the compress function can use
    let column_fns: Vec<_> = columns.iter().map(|f| |h: HapIdx| f(h)).collect();

    let dict = DictionaryColumn::compress(&column_fns, n_markers, n_haplotypes, bits_per_allele);

    // Only use if compression ratio is favorable (< 0.5 = 2x compression)
    if dict.compression_ratio() < 0.5 {
        Some(dict)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_selection() {
        // Common variant should use dense
        let common: Vec<u8> = (0..100).map(|i| if i % 2 == 0 { 0 } else { 1 }).collect();
        let col = GenotypeColumn::from_alleles(&common, 2);
        assert!(matches!(col, GenotypeColumn::Dense(_)));

        // Rare variant should use sparse
        let mut rare = vec![0u8; 1000];
        rare[0] = 1;
        rare[1] = 1;
        let col = GenotypeColumn::from_alleles(&rare, 2);
        assert!(matches!(col, GenotypeColumn::Sparse(_)));
    }

    #[test]
    fn test_maf_calculation() {
        let alleles = vec![0, 0, 0, 0, 1, 1, 0, 0, 0, 0]; // 20% ALT
        let col = GenotypeColumn::from_alleles(&alleles, 2);
        assert!((col.maf() - 0.2).abs() < 0.001);
    }
}
