//! # Mutable Genotype Storage
//!
//! A simple mutable storage for genotypes during phasing iterations.
//! This allows efficient in-place updates without rebuilding columns.
//!
//! Supports multiallelic markers by storing full byte values (0-254 for alleles,
//! 255 for missing). This matches Java Beagle's `BitArrayGTRec` which uses
//! variable-width bit storage based on allele count.

use crate::data::HapIdx;

/// Mutable genotype storage for phasing
///
/// Uses byte storage (1 byte per allele) to support multiallelic markers.
/// Outer vector is indexed by marker, inner vector by haplotype.
///
/// Allele values: 0 = REF, 1+ = ALT alleles, 255 = missing
#[derive(Clone, Debug)]
pub struct MutableGenotypes {
    /// Alleles indexed by [marker][haplotype]
    /// Values: 0 = REF, 1-254 = ALT alleles, 255 = missing
    alleles: Vec<Vec<u8>>,
    /// Number of haplotypes
    n_haps: usize,
}

impl MutableGenotypes {
    /// Create from a function that provides alleles
    ///
    /// Allele values: 0 = REF, 1-254 = ALT alleles, 255 = missing
    pub fn from_fn<F>(n_markers: usize, n_haps: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> u8,
    {
        let alleles: Vec<Vec<u8>> = (0..n_markers)
            .map(|m| (0..n_haps).map(|h| f(m, h)).collect())
            .collect();

        Self { alleles, n_haps }
    }

    /// Number of markers
    #[inline]
    pub fn n_markers(&self) -> usize {
        self.alleles.len()
    }

    /// Number of haplotypes
    #[inline]
    pub fn n_haps(&self) -> usize {
        self.n_haps
    }

    /// Get allele at (marker, haplotype)
    ///
    /// Returns 0 (REF), 1-254 (ALT alleles), or 255 (missing)
    #[inline]
    pub fn get(&self, marker: usize, hap: HapIdx) -> u8 {
        self.alleles[marker][hap.as_usize()]
    }

    /// Check if position is missing
    #[inline]
    pub fn is_missing(&self, marker: usize, hap: HapIdx) -> bool {
        self.alleles[marker][hap.as_usize()] == 255
    }

    /// Set allele at (marker, haplotype)
    ///
    /// Allele values: 0 = REF, 1-254 = ALT alleles, 255 = missing
    #[inline]
    pub fn set(&mut self, marker: usize, hap: HapIdx, allele: u8) {
        self.alleles[marker][hap.as_usize()] = allele;
    }

    /// Get all alleles at a marker as a slice
    #[inline]
    pub fn marker_alleles(&self, marker: usize) -> &[u8] {
        &self.alleles[marker]
    }

    /// Get all alleles for a haplotype
    ///
    /// Returns a vector with values: 0 (REF), 1-254 (ALT), or 255 (missing)
    pub fn haplotype(&self, hap: HapIdx) -> Vec<u8> {
        let h = hap.as_usize();
        self.alleles.iter().map(|row| row[h]).collect()
    }

    /// Swap alleles between two haplotypes at a marker
    #[inline]
    pub fn swap(&mut self, marker: usize, hap1: HapIdx, hap2: HapIdx) {
        let h1 = hap1.as_usize();
        let h2 = hap2.as_usize();
        self.alleles[marker].swap(h1, h2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutable_genotypes() {
        let mut geno = MutableGenotypes::from_fn(3, 4, |_, _| 0);

        assert_eq!(geno.n_markers(), 3);
        assert_eq!(geno.n_haps(), 4);

        // Set some values
        geno.set(0, HapIdx::new(0), 1);
        geno.set(1, HapIdx::new(1), 1);
        geno.set(2, HapIdx::new(2), 1);

        assert_eq!(geno.get(0, HapIdx::new(0)), 1);
        assert_eq!(geno.get(0, HapIdx::new(1)), 0);
        assert_eq!(geno.get(1, HapIdx::new(1)), 1);
    }

    #[test]
    fn test_swap() {
        let mut geno = MutableGenotypes::from_fn(3, 4, |_, h| if h == 0 { 1 } else { 0 });

        assert_eq!(geno.get(0, HapIdx::new(0)), 1);
        assert_eq!(geno.get(0, HapIdx::new(1)), 0);

        geno.swap(0, HapIdx::new(0), HapIdx::new(1));

        assert_eq!(geno.get(0, HapIdx::new(0)), 0);
        assert_eq!(geno.get(0, HapIdx::new(1)), 1);
    }

    #[test]
    fn test_haplotype() {
        let geno = MutableGenotypes::from_fn(5, 2, |m, h| if h == 0 { (m % 2) as u8 } else { 0 });

        let hap0 = geno.haplotype(HapIdx::new(0));
        assert_eq!(hap0, vec![0, 1, 0, 1, 0]);

        let hap1 = geno.haplotype(HapIdx::new(1));
        assert_eq!(hap1, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_missing_data() {
        // Test that missing data (255) is preserved correctly
        let geno = MutableGenotypes::from_fn(4, 2, |m, h| {
            match (m, h) {
                (0, 0) => 0,   // REF
                (0, 1) => 1,   // ALT
                (1, 0) => 255, // Missing
                (1, 1) => 0,   // REF
                (2, 0) => 1,   // ALT
                (2, 1) => 255, // Missing
                (3, _) => 255, // Both missing
                _ => 0,
            }
        });

        // Check get returns correct values
        assert_eq!(geno.get(0, HapIdx::new(0)), 0);
        assert_eq!(geno.get(0, HapIdx::new(1)), 1);
        assert_eq!(geno.get(1, HapIdx::new(0)), 255);  // Missing preserved!
        assert_eq!(geno.get(1, HapIdx::new(1)), 0);
        assert_eq!(geno.get(2, HapIdx::new(0)), 1);
        assert_eq!(geno.get(2, HapIdx::new(1)), 255);  // Missing preserved!
        assert_eq!(geno.get(3, HapIdx::new(0)), 255);
        assert_eq!(geno.get(3, HapIdx::new(1)), 255);

        // Check is_missing
        assert!(!geno.is_missing(0, HapIdx::new(0)));
        assert!(geno.is_missing(1, HapIdx::new(0)));
        assert!(geno.is_missing(2, HapIdx::new(1)));
        assert!(geno.is_missing(3, HapIdx::new(0)));
        assert!(geno.is_missing(3, HapIdx::new(1)));

        // Check haplotype returns 255 for missing
        let hap0 = geno.haplotype(HapIdx::new(0));
        assert_eq!(hap0, vec![0, 255, 1, 255]);

        let hap1 = geno.haplotype(HapIdx::new(1));
        assert_eq!(hap1, vec![1, 0, 255, 255]);
    }

    #[test]
    fn test_set_missing() {
        let mut geno = MutableGenotypes::from_fn(3, 2, |_, _| 0);

        // Set some values including missing
        geno.set(0, HapIdx::new(0), 1);
        geno.set(1, HapIdx::new(0), 255);  // Set missing
        geno.set(2, HapIdx::new(0), 0);

        assert_eq!(geno.get(0, HapIdx::new(0)), 1);
        assert_eq!(geno.get(1, HapIdx::new(0)), 255);
        assert_eq!(geno.get(2, HapIdx::new(0)), 0);
        assert!(geno.is_missing(1, HapIdx::new(0)));

        // Now set a missing position to non-missing
        geno.set(1, HapIdx::new(0), 1);
        assert_eq!(geno.get(1, HapIdx::new(0)), 1);
        assert!(!geno.is_missing(1, HapIdx::new(0)));
    }

    #[test]
    fn test_swap_with_missing() {
        let mut geno = MutableGenotypes::from_fn(3, 2, |m, h| {
            match (m, h) {
                (0, 0) => 1,   // ALT
                (0, 1) => 255, // Missing
                (1, 0) => 255, // Missing
                (1, 1) => 0,   // REF
                (2, 0) => 0,   // REF
                (2, 1) => 1,   // ALT
                _ => 0,
            }
        });

        // Before swap
        assert_eq!(geno.get(0, HapIdx::new(0)), 1);
        assert_eq!(geno.get(0, HapIdx::new(1)), 255);
        assert!(geno.is_missing(0, HapIdx::new(1)));

        // Swap at marker 0
        geno.swap(0, HapIdx::new(0), HapIdx::new(1));

        // After swap - missing should move with the haplotype
        assert_eq!(geno.get(0, HapIdx::new(0)), 255);  // Was 1, now missing
        assert_eq!(geno.get(0, HapIdx::new(1)), 1);    // Was missing, now ALT
        assert!(geno.is_missing(0, HapIdx::new(0)));
        assert!(!geno.is_missing(0, HapIdx::new(1)));
    }

    #[test]
    fn test_multiallelic() {
        // Test multiallelic support (alleles 0, 1, 2, 3)
        let geno = MutableGenotypes::from_fn(4, 2, |m, h| {
            match (m, h) {
                (0, 0) => 0, // REF
                (0, 1) => 1, // ALT1
                (1, 0) => 2, // ALT2
                (1, 1) => 3, // ALT3
                (2, 0) => 0,
                (2, 1) => 2, // ALT2
                (3, 0) => 255, // Missing
                (3, 1) => 3,   // ALT3
                _ => 0,
            }
        });

        assert_eq!(geno.get(0, HapIdx::new(0)), 0);
        assert_eq!(geno.get(0, HapIdx::new(1)), 1);
        assert_eq!(geno.get(1, HapIdx::new(0)), 2);
        assert_eq!(geno.get(1, HapIdx::new(1)), 3);
        assert_eq!(geno.get(2, HapIdx::new(1)), 2);
        assert_eq!(geno.get(3, HapIdx::new(0)), 255);
        assert_eq!(geno.get(3, HapIdx::new(1)), 3);

        let hap0 = geno.haplotype(HapIdx::new(0));
        assert_eq!(hap0, vec![0, 2, 0, 255]);

        let hap1 = geno.haplotype(HapIdx::new(1));
        assert_eq!(hap1, vec![1, 3, 2, 3]);
    }
}
