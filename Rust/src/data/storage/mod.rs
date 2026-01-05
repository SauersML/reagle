//! # Genotype Storage Backends
//!
//! Polymorphic storage for genotype data. Replaces Java's `GTRec` class hierarchy
//! with a single Rust enum.

pub mod dense;
pub mod dictionary;
pub mod matrix;
pub mod sparse;

pub use dense::DenseColumn;
pub use dictionary::DictionaryColumn;
pub use matrix::GenotypeMatrix;
pub use sparse::SparseColumn;

use crate::data::HapIdx;

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
    Dictionary(DictionaryColumn),
}

impl GenotypeColumn {
    /// Get allele for a specific haplotype (0 = REF, 1+ = ALT)
    #[inline]
    pub fn get(&self, hap: HapIdx) -> u8 {
        match self {
            Self::Dense(col) => col.get(hap),
            Self::Sparse(col) => col.get(hap),
            Self::Dictionary(col) => col.get(0, hap), // Single marker
        }
    }

    /// Number of haplotypes in this column
    pub fn n_haplotypes(&self) -> usize {
        match self {
            Self::Dense(col) => col.n_haplotypes(),
            Self::Sparse(col) => col.n_haplotypes(),
            Self::Dictionary(col) => col.n_haplotypes(),
        }
    }

    /// Count of ALT allele carriers
    pub fn alt_count(&self) -> usize {
        match self {
            Self::Dense(col) => col.alt_count(),
            Self::Sparse(col) => col.n_carriers(),
            Self::Dictionary(col) => col.alt_count(0),
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
            Self::Dictionary(col) => col.size_bytes(),
        }
    }

    /// Create from allele slice, automatically choosing storage type
    pub fn from_alleles(alleles: &[u8], n_alleles: usize) -> Self {
        let n_haps = alleles.len();
        if n_haps == 0 {
            return Self::Dense(DenseColumn::new(0, 1));
        }

        // Count ALT carriers for MAF calculation
        let alt_count = alleles.iter().filter(|&&a| a > 0).count();
        let maf = (alt_count as f64 / n_haps as f64).min(1.0 - alt_count as f64 / n_haps as f64);

        // Use sparse storage for rare variants (MAF < 1%)
        if maf < 0.01 && n_alleles == 2 {
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
            Self::Dense(DenseColumn::from_alleles(alleles.iter().copied(), n_alleles))
        }
    }

    /// Check if this is a biallelic column
    pub fn is_biallelic(&self) -> bool {
        match self {
            Self::Dense(col) => col.bits_per_allele() == 1,
            Self::Sparse(_) => true, // Sparse is always biallelic
            Self::Dictionary(col) => col.is_biallelic(),
        }
    }

    /// Iterate over all alleles
    pub fn iter(&self) -> Box<dyn Iterator<Item = u8> + '_> {
        match self {
            Self::Dense(col) => Box::new(col.iter()),
            Self::Sparse(col) => Box::new(col.iter()),
            Self::Dictionary(col) => Box::new(col.iter_marker(0)),
        }
    }
}

impl Default for GenotypeColumn {
    fn default() -> Self {
        Self::Dense(DenseColumn::new(0, 1))
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