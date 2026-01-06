//! # Dense Bit-Packed Storage
//!
//! Efficient storage for common variants using bit vectors.
//! Replaces `vcf/BitArrayGTRec.java`.

use bitvec::prelude::*;

use crate::data::HapIdx;

/// Dense bit-packed storage for genotype data
#[derive(Clone, Debug)]
pub struct DenseColumn {
    /// Bit vector storing allele data
    /// For biallelic: 1 bit per haplotype (0=REF, 1=ALT)
    /// For multi-allelic: ceil(log2(n_alleles)) bits per haplotype
    bits: BitVec<u64, Lsb0>,

    /// Bits per allele (1 for biallelic, 2 for 3-4 alleles, etc.)
    bits_per_allele: u8,

    /// Number of haplotypes stored
    n_haplotypes: u32,
}

impl DenseColumn {
    /// Create a new empty dense column
    pub fn new(n_haplotypes: usize, n_alleles: usize) -> Self {
        let bits_per_allele = Self::calculate_bits_per_allele(n_alleles);
        let total_bits = n_haplotypes * bits_per_allele as usize;
        Self {
            bits: bitvec![u64, Lsb0; 0; total_bits],
            bits_per_allele,
            n_haplotypes: n_haplotypes as u32,
        }
    }

    /// Create from an iterator of alleles
    pub fn from_alleles(alleles: impl Iterator<Item = u8>, n_alleles: usize) -> Self {
        let alleles: Vec<u8> = alleles.collect();
        let n_haplotypes = alleles.len();
        let bits_per_allele = Self::calculate_bits_per_allele(n_alleles);

        let total_bits = n_haplotypes * bits_per_allele as usize;
        let mut bits = bitvec![u64, Lsb0; 0; total_bits];

        for (i, &allele) in alleles.iter().enumerate() {
            let start = i * bits_per_allele as usize;
            for b in 0..bits_per_allele as usize {
                if (allele >> b) & 1 == 1 {
                    bits.set(start + b, true);
                }
            }
        }

        Self {
            bits,
            bits_per_allele,
            n_haplotypes: n_haplotypes as u32,
        }
    }

    /// Calculate bits needed per allele
    fn calculate_bits_per_allele(n_alleles: usize) -> u8 {
        if n_alleles <= 1 {
            1
        } else {
            (usize::BITS - (n_alleles - 1).leading_zeros()) as u8
        }
    }

    /// Get allele for haplotype
    #[inline]
    pub fn get(&self, hap: HapIdx) -> u8 {
        let idx = hap.as_usize();
        if idx >= self.n_haplotypes as usize {
            return 0;
        }

        let start = idx * self.bits_per_allele as usize;
        let mut allele = 0u8;
        for b in 0..self.bits_per_allele as usize {
            if self.bits[start + b] {
                allele |= 1 << b;
            }
        }
        allele
    }

    /// Set allele for haplotype
    pub fn set(&mut self, hap: HapIdx, allele: u8) {
        let idx = hap.as_usize();
        if idx >= self.n_haplotypes as usize {
            return;
        }

        let start = idx * self.bits_per_allele as usize;
        for b in 0..self.bits_per_allele as usize {
            self.bits.set(start + b, (allele >> b) & 1 == 1);
        }
    }

    /// Number of haplotypes
    pub fn n_haplotypes(&self) -> usize {
        self.n_haplotypes as usize
    }

    /// Bits per allele
    pub fn bits_per_allele(&self) -> u8 {
        self.bits_per_allele
    }

    /// Count of ALT alleles (for biallelic)
    pub fn alt_count(&self) -> usize {
        if self.bits_per_allele == 1 {
            self.bits.count_ones()
        } else {
            self.iter().filter(|&a| a > 0).count()
        }
    }

    /// Iterate all alleles
    pub fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        (0..self.n_haplotypes as usize).map(move |i| self.get(HapIdx::new(i as u32)))
    }

    /// Memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        self.bits.as_raw_slice().len() * std::mem::size_of::<u64>() + std::mem::size_of::<Self>()
    }

    /// Access the underlying raw u64 storage slice
    ///
    /// This allows for bit-parallel operations (SIMD/SWAR) on blocks of 64 haplotypes.
    pub fn as_raw_slice(&self) -> &[u64] {
        self.bits.as_raw_slice()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biallelic() {
        let alleles = vec![0, 1, 0, 1, 1, 0, 0, 1];
        let col = DenseColumn::from_alleles(alleles.iter().copied(), 2);

        assert_eq!(col.n_haplotypes(), 8);
        assert_eq!(col.bits_per_allele(), 1);

        for (i, &expected) in alleles.iter().enumerate() {
            assert_eq!(col.get(HapIdx::new(i as u32)), expected);
        }

        assert_eq!(col.alt_count(), 4);
    }

    #[test]
    fn test_multiallelic() {
        let alleles = vec![0, 1, 2, 3, 0, 1, 2, 3];
        let col = DenseColumn::from_alleles(alleles.iter().copied(), 4);

        assert_eq!(col.n_haplotypes(), 8);
        assert_eq!(col.bits_per_allele(), 2);

        for (i, &expected) in alleles.iter().enumerate() {
            assert_eq!(col.get(HapIdx::new(i as u32)), expected);
        }
    }

    #[test]
    fn test_set() {
        let mut col = DenseColumn::new(4, 2);
        col.set(HapIdx::new(0), 1);
        col.set(HapIdx::new(2), 1);

        assert_eq!(col.get(HapIdx::new(0)), 1);
        assert_eq!(col.get(HapIdx::new(1)), 0);
        assert_eq!(col.get(HapIdx::new(2)), 1);
        assert_eq!(col.get(HapIdx::new(3)), 0);
    }
}
