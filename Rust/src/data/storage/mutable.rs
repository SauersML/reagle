//! # Mutable Genotype Storage
//!
//! A simple mutable storage for genotypes during phasing iterations.
//! This allows efficient in-place updates without rebuilding columns.

use crate::data::HapIdx;

/// Mutable genotype storage for phasing
/// 
/// Uses a simple Vec<Vec<u8>> layout for easy mutation.
/// Less memory-efficient than dense/sparse but allows O(1) updates.
#[derive(Clone, Debug)]
pub struct MutableGenotypes {
    /// Alleles indexed by [marker][haplotype]
    alleles: Vec<Vec<u8>>,
    /// Number of haplotypes
    n_haps: usize,
}

impl MutableGenotypes {
    /// Create from existing allele data
    pub fn new(n_markers: usize, n_haps: usize) -> Self {
        Self {
            alleles: vec![vec![0u8; n_haps]; n_markers],
            n_haps,
        }
    }

    /// Create from a function that provides alleles
    pub fn from_fn<F>(n_markers: usize, n_haps: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> u8,
    {
        let alleles = (0..n_markers)
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
    #[inline]
    pub fn get(&self, marker: usize, hap: HapIdx) -> u8 {
        self.alleles[marker][hap.as_usize()]
    }

    /// Set allele at (marker, haplotype)
    #[inline]
    pub fn set(&mut self, marker: usize, hap: HapIdx, allele: u8) {
        self.alleles[marker][hap.as_usize()] = allele;
    }

    /// Get all alleles at a marker
    #[inline]
    pub fn marker_alleles(&self, marker: usize) -> &[u8] {
        &self.alleles[marker]
    }

    /// Get mutable alleles at a marker
    #[inline]
    pub fn marker_alleles_mut(&mut self, marker: usize) -> &mut [u8] {
        &mut self.alleles[marker]
    }

    /// Get all alleles for a haplotype
    pub fn haplotype(&self, hap: HapIdx) -> Vec<u8> {
        let h = hap.as_usize();
        self.alleles.iter().map(|m| m[h]).collect()
    }

    /// Swap alleles between two haplotypes at a marker
    #[inline]
    pub fn swap(&mut self, marker: usize, hap1: HapIdx, hap2: HapIdx) {
        let h1 = hap1.as_usize();
        let h2 = hap2.as_usize();
        self.alleles[marker].swap(h1, h2);
    }

    /// Swap alleles between two haplotypes at multiple markers
    pub fn swap_range(&mut self, markers: &[usize], hap1: HapIdx, hap2: HapIdx) {
        let h1 = hap1.as_usize();
        let h2 = hap2.as_usize();
        for &m in markers {
            self.alleles[m].swap(h1, h2);
        }
    }

    /// Copy alleles from another MutableGenotypes
    pub fn copy_from(&mut self, other: &MutableGenotypes) {
        debug_assert_eq!(self.n_markers(), other.n_markers());
        debug_assert_eq!(self.n_haps, other.n_haps);
        for (dst, src) in self.alleles.iter_mut().zip(other.alleles.iter()) {
            dst.copy_from_slice(src);
        }
    }

    /// Get raw alleles (for reading into GenotypeColumn)
    pub fn raw_alleles(&self) -> &[Vec<u8>] {
        &self.alleles
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mutable_genotypes() {
        let mut geno = MutableGenotypes::new(3, 4);
        
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
        let mut geno = MutableGenotypes::from_fn(3, 4, |_, h| {
            if h == 0 { 1 } else { 0 }
        });
        
        assert_eq!(geno.get(0, HapIdx::new(0)), 1);
        assert_eq!(geno.get(0, HapIdx::new(1)), 0);
        
        geno.swap(0, HapIdx::new(0), HapIdx::new(1));
        
        assert_eq!(geno.get(0, HapIdx::new(0)), 0);
        assert_eq!(geno.get(0, HapIdx::new(1)), 1);
    }

    #[test]
    fn test_haplotype() {
        let geno = MutableGenotypes::from_fn(5, 2, |m, h| {
            if h == 0 { m as u8 } else { 0 }
        });
        
        let hap0 = geno.haplotype(HapIdx::new(0));
        assert_eq!(hap0, vec![0, 1, 2, 3, 4]);
        
        let hap1 = geno.haplotype(HapIdx::new(1));
        assert_eq!(hap1, vec![0, 0, 0, 0, 0]);
    }
}