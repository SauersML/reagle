//! # Sparse Storage for Rare Variants
//!
//! Efficient storage for rare variants (MAF < 1%) by storing only carrier indices.
//! Replaces `vcf/LowMafGTRec.java`.

use crate::data::HapIdx;

/// Sparse storage for rare variants
#[derive(Clone, Debug)]
pub struct SparseColumn {
    /// Sorted indices of carrier haplotypes
    carriers: Vec<HapIdx>,

    /// Total number of haplotypes
    n_haplotypes: u32,

    /// If true, carriers are REF (0) and non-carriers are ALT (1)
    /// If false, carriers are ALT (1) and non-carriers are REF (0)
    inverted: bool,
}

impl SparseColumn {
    /// Create from carrier indices
    pub fn from_carriers(mut carriers: Vec<HapIdx>, n_haplotypes: u32, inverted: bool) -> Self {
        carriers.sort_unstable();
        Self {
            carriers,
            n_haplotypes,
            inverted,
        }
    }

    /// Get allele for haplotype (binary search)
    #[inline]
    pub fn get(&self, hap: HapIdx) -> u8 {
        let is_carrier = self.carriers.binary_search(&hap).is_ok();
        if self.inverted {
            if is_carrier { 0 } else { 1 }
        } else {
            if is_carrier { 1 } else { 0 }
        }
    }

    /// Number of carriers
    pub fn n_carriers(&self) -> usize {
        if self.inverted {
            self.n_haplotypes as usize - self.carriers.len()
        } else {
            self.carriers.len()
        }
    }

    /// Number of haplotypes
    pub fn n_haplotypes(&self) -> usize {
        self.n_haplotypes as usize
    }

    /// Memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        self.carriers.len() * std::mem::size_of::<HapIdx>() + std::mem::size_of::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_column() {
        let carriers = vec![HapIdx::new(2), HapIdx::new(5), HapIdx::new(7)];
        let col = SparseColumn::from_carriers(carriers, 10, false);

        assert_eq!(col.get(HapIdx::new(0)), 0);
        assert_eq!(col.get(HapIdx::new(2)), 1);
        assert_eq!(col.get(HapIdx::new(5)), 1);
        assert_eq!(col.get(HapIdx::new(7)), 1);
        assert_eq!(col.get(HapIdx::new(9)), 0);

        assert_eq!(col.n_carriers(), 3);
    }

}
