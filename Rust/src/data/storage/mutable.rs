//! # Mutable Genotype Storage
//!
//! A simple mutable storage for genotypes during phasing iterations.
//! This allows efficient in-place updates without rebuilding columns.
//! Now optimized using `bitvec` for 8x memory reduction.

use crate::data::HapIdx;
use bitvec::prelude::*;

/// Mutable genotype storage for phasing
///
/// Uses `Vec<BitBox>` for memory efficiency (1 bit per allele).
/// Outer vector is indexed by marker, inner BitBox by haplotype.
#[derive(Clone, Debug)]
pub struct MutableGenotypes {
    /// Alleles indexed by [marker][haplotype]
    /// Using Lsb0 order: bit 0 is hap 0, bit 1 is hap 1, etc.
    /// Changed to u64 for SIMD/Bit-parallel alignment.
    alleles: Vec<BitBox<u64, Lsb0>>,
    /// Number of haplotypes
    n_haps: usize,
}

impl MutableGenotypes {
    /// Create from existing allele data
    pub fn new(n_markers: usize, n_haps: usize) -> Self {
        // Initialize with all zeros
        let empty_row = bitbox![u64, Lsb0; 0; n_haps];
        Self {
            alleles: vec![empty_row; n_markers],
            n_haps,
        }
    }

    /// Create from a function that provides alleles
    pub fn from_fn<F>(n_markers: usize, n_haps: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> u8,
    {
        let mut alleles = Vec::with_capacity(n_markers);
        
        for m in 0..n_markers {
            let mut row = BitVec::<u64, Lsb0>::with_capacity(n_haps);
            for h in 0..n_haps {
                let val = f(m, h) != 0;
                row.push(val);
            }
            alleles.push(row.into_boxed_bitslice());
        }
        
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

    /// Number of samples (assuming diploid)
    #[inline]
    pub fn n_samples(&self) -> usize {
        self.n_haps / 2
    }

    /// Get allele at (marker, haplotype)
    #[inline]
    pub fn get(&self, marker: usize, hap: HapIdx) -> u8 {
        self.alleles[marker][hap.as_usize()] as u8
    }

    /// Set allele at (marker, haplotype)
    #[inline]
    pub fn set(&mut self, marker: usize, hap: HapIdx, allele: u8) {
        self.alleles[marker].set(hap.as_usize(), allele != 0);
    }

    /// Get all alleles at a marker as a BitSlice
    #[inline]
    pub fn marker_alleles(&self, marker: usize) -> &BitSlice<u64, Lsb0> {
        &self.alleles[marker]
    }
    
    /// Get all alleles at a marker as a slice of packed u64 words
    #[inline]
    pub fn marker_alleles_packed(&self, marker: usize) -> &[u64] {
        self.alleles[marker].as_raw_slice()
    }

    /// Get all alleles for a haplotype
    /// Note: This is now slower than before as it iterates columns
    pub fn haplotype(&self, hap: HapIdx) -> Vec<u8> {
        let h = hap.as_usize();
        self.alleles.iter().map(|m| m[h] as u8).collect()
    }

    /// Swap alleles between two haplotypes at a marker
    #[inline]
    pub fn swap(&mut self, marker: usize, hap1: HapIdx, hap2: HapIdx) {
        let h1 = hap1.as_usize();
        let h2 = hap2.as_usize();
        let row = &mut self.alleles[marker];
        // BitSlice::swap is efficient
        row.swap(h1, h2);
    }

    /// Swap alleles between two haplotypes for a contiguous range of markers
    #[inline]
    pub fn swap_contiguous(&mut self, start: usize, end: usize, hap1: HapIdx, hap2: HapIdx) {
        let h1 = hap1.as_usize();
        let h2 = hap2.as_usize();
        for m in start..end {
            self.alleles[m].swap(h1, h2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutable_genotypes() {
        let mut geno = MutableGenotypes::new(3, 4);

        assert_eq!(geno.n_markers(), 3);
        assert_eq!(geno.n_haps(), 4);
        assert_eq!(geno.n_samples(), 2);

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
        let geno = MutableGenotypes::from_fn(5, 2, |m, h| if h == 0 { m as u8 } else { 0 });

        // Note: bitvec only stores 0/1. The test in original code used `m as u8` which could be > 1.
        // We must ensure the test data is binary.
        // If m % 2 == 1 -> 1, else 0
        let geno_binary = MutableGenotypes::from_fn(5, 2, |m, h| if h == 0 { (m % 2) as u8 } else { 0 });

        let hap0 = geno_binary.haplotype(HapIdx::new(0));
        assert_eq!(hap0, vec![0, 1, 0, 1, 0]);

        let hap1 = geno_binary.haplotype(HapIdx::new(1));
        assert_eq!(hap1, vec![0, 0, 0, 0, 0]);
    }
}
