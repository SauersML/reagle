//! # Dense Bit-Packed Storage
//!
//! Efficient storage for common variants using bit vectors.
//! Replaces `vcf/BitArrayGTRec.java`.

use bitvec::prelude::*;

use crate::data::marker::bits_per_allele;
use crate::data::HapIdx;
use crate::model::types::RefHapId;

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

    /// Stable column fingerprint for fast equality prefiltering in hot paths.
    fingerprint: u64,
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
            fingerprint: 0,
        }
        .with_fingerprint()
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
            if allele == crate::data::storage::AlleleCode::MISSING.raw() {
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
            fingerprint: 0,
        }
        .with_fingerprint()
    }

    #[inline]
    fn with_fingerprint(mut self) -> Self {
        self.fingerprint = Self::compute_fingerprint(
            self.bits.as_raw_slice(),
            self.missing.as_raw_slice(),
            self.bits_per_allele,
            self.n_haplotypes,
        );
        self
    }

    #[inline]
    fn compute_fingerprint(
        bits: &[u64],
        missing: &[u64],
        bits_per_allele: u8,
        n_haplotypes: u32,
    ) -> u64 {
        // FNV-1a over immutable column payload.
        let mut h: u64 = 0xcbf29ce484222325;
        h ^= bits_per_allele as u64;
        h = h.wrapping_mul(0x100000001b3);
        h ^= n_haplotypes as u64;
        h = h.wrapping_mul(0x100000001b3);
        for &w in bits {
            h ^= w;
            h = h.wrapping_mul(0x100000001b3);
        }
        for &w in missing {
            h ^= w;
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }

    /// Calculate bits needed per allele (minimum 1 for storage)
    fn calculate_bits_per_allele(n_alleles: usize) -> u8 {
        bits_per_allele(n_alleles).max(1)
    }

    /// Get allele for haplotype
    #[inline]
    pub fn get(&self, hap: HapIdx) -> u8 {
        self.get_idx(hap.as_usize())
    }

    /// Type-safe reference-panel accessor.
    ///
    /// Using `RefHapId` here prevents mixing target/combined hap index spaces
    /// in reference-only hot paths.
    #[inline]
    pub fn get_ref(&self, hap: RefHapId) -> u8 {
        assert!(
            hap.as_usize() < self.n_haplotypes as usize,
            "reference hap index {} out of bounds for {} haplotypes",
            hap.as_usize(),
            self.n_haplotypes
        );
        self.get_idx(hap.as_usize())
    }

    #[inline]
    fn get_idx(&self, idx: usize) -> u8 {
        if idx >= self.n_haplotypes as usize {
            return crate::data::storage::AlleleCode::MISSING.raw();
        }
        if self.missing[idx] {
            return crate::data::storage::AlleleCode::MISSING.raw();
        }
        if self.bits_per_allele == 1 {
            return self.bits[idx] as u8;
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

    /// Number of haplotypes
    pub fn n_haplotypes(&self) -> usize {
        self.n_haplotypes as usize
    }

    #[inline]
    pub fn bits_per_allele(&self) -> u8 {
        self.bits_per_allele
    }

    #[inline]
    pub fn bits_raw(&self) -> &[u64] {
        self.bits.as_raw_slice()
    }

    #[inline]
    pub fn missing_raw(&self) -> &[u64] {
        self.missing.as_raw_slice()
    }

    #[inline]
    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Count of ALT alleles (for biallelic)
    pub fn alt_count(&self) -> usize {
        if self.bits_per_allele == 1 {
            // bits.count_ones() includes bits that might have been set for missing data
            // if we didn't clear them. But our set/from_alleles clears them.
            // Still, it's safer and clearer to use the iter or bit-parallel logic that respects 'missing'.
            self.iter()
                .filter(|&a| a > 0 && a != crate::data::storage::AlleleCode::MISSING.raw())
                .count()
        } else {
            self.iter()
                .filter(|&a| a > 0 && a != crate::data::storage::AlleleCode::MISSING.raw())
                .count()
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

    #[test]
    fn test_out_of_range_get_returns_missing() {
        let col = DenseColumn::from_alleles([0u8, 1, 0].into_iter(), 2);
        assert_eq!(
            col.get(HapIdx::new(9)),
            crate::data::storage::AlleleCode::MISSING.raw()
        );
    }
}
