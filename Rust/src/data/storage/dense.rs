//! # Dense Bit-Packed Storage
//!
//! Efficient storage for common variants using bit vectors.
//! Replaces `vcf/BitArrayGTRec.java`.

use bitvec::prelude::*;

use crate::data::marker::bits_per_allele;
use crate::data::HapIdx;

/// Dense bit-packed storage for genotype data
#[derive(Clone, Debug)]
pub struct DenseColumn {
    /// Bit vector storing allele data
    /// For biallelic: 1 bit per haplotype (0=REF, 1=ALT)
    /// For multi-allelic: ceil(log2(n_alleles)) bits per haplotype
    bits: BitVec<u64, Lsb0>,

    /// Bit vector tracking missing data (1 = missing, 0 = present)
    missing: BitVec<u64, Lsb0>,

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
            missing: bitvec![u64, Lsb0; 0; n_haplotypes],
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
        let mut missing = bitvec![u64, Lsb0; 0; n_haplotypes];

        for (i, &allele) in alleles.iter().enumerate() {
            if allele == 255 {
                missing.set(i, true);
                continue;
            }
            let start = i * bits_per_allele as usize;
            for b in 0..bits_per_allele as usize {
                if (allele >> b) & 1 == 1 {
                    bits.set(start + b, true);
                }
            }
        }

        Self {
            bits,
            missing,
            bits_per_allele,
            n_haplotypes: n_haplotypes as u32,
        }
    }

    /// Calculate bits needed per allele (minimum 1 for storage)
    fn calculate_bits_per_allele(n_alleles: usize) -> u8 {
        bits_per_allele(n_alleles).max(1)
    }

    /// Get allele for haplotype
    #[inline]
    pub fn get(&self, hap: HapIdx) -> u8 {
        let idx = hap.as_usize();
        if idx >= self.n_haplotypes as usize {
            return 0;
        }

        if self.missing[idx] {
            return 255;
        }

        // Fast path for biallelic sites (99% of cases): single bit lookup
        if self.bits_per_allele == 1 {
            return self.bits[idx] as u8;
        }

        // General multi-allelic path
        let start = idx * self.bits_per_allele as usize;
        let mut allele = 0u8;
        for b in 0..self.bits_per_allele as usize {
            if self.bits[start + b] {
                allele |= 1 << b;
            }
        }
        allele
    }

    /// Number of haplotypes
    pub fn n_haplotypes(&self) -> usize {
        self.n_haplotypes as usize
    }

    /// Count of ALT alleles (for biallelic)
    pub fn alt_count(&self) -> usize {
        if self.bits_per_allele == 1 {
            // bits.count_ones() includes bits that might have been set for missing data 
            // if we didn't clear them. But our set/from_alleles clears them.
            // Still, it's safer and clearer to use the iter or bit-parallel logic that respects 'missing'.
            self.iter().filter(|&a| a > 0 && a != 255).count()
        } else {
            self.iter().filter(|&a| a > 0 && a != 255).count()
        }
    }

    /// Iterate all alleles
    pub fn iter(&self) -> impl Iterator<Item = u8> + '_ {
        (0..self.n_haplotypes as usize).map(move |i| self.get(HapIdx::new(i as u32)))
    }

    /// Memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        self.bits.as_raw_slice().len() * std::mem::size_of::<u64>() 
            + self.missing.as_raw_slice().len() * std::mem::size_of::<u64>()
            + std::mem::size_of::<Self>()
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

        for (i, &expected) in alleles.iter().enumerate() {
            assert_eq!(col.get(HapIdx::new(i as u32)), expected);
        }
    }

}
