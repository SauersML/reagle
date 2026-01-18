//! # Mutable Genotype Storage
//!
//! Bit-packed mutable storage for genotypes during phasing iterations.
//! This allows efficient in-place updates without rebuilding columns.
//!
//! Memory model:
//! - Primary storage: BitVec (1 bit per allele) for biallelic SNPs
//! - Exception store: block-sparse vectors for multiallelic (2+) and missing (255)
//!
//! For typical datasets (99%+ biallelic, <1% missing), this provides
//! ~8x memory reduction vs byte-per-allele storage.

use crate::data::HapIdx;
use bitvec::prelude::*;

/// Bit-packed mutable genotype storage for phasing
///
/// Uses 1 bit per allele for the common biallelic case (0 or 1),
/// with a sparse exception map for multiallelic (2-254) and missing (255).
///
/// Memory efficiency:
/// - 1M markers × 6K haps biallelic: ~750 MB (vs ~6 GB byte-packed)
/// - Exception blocks add ~9 bytes per exception (typically <1%)
///
/// Allele values: 0 = REF, 1 = ALT1, 2+ = ALT2+, 255 = missing
#[derive(Clone, Debug)]
pub struct MutableGenotypes {
    /// Bit-packed alleles: 1 bit per position
    /// Layout: bits[marker * n_haps + hap]
    /// Value: 0 = REF or exception, 1 = ALT1 (biallelic)
    bits: BitVec<u64, Lsb0>,
    /// Sparse exception store for non-biallelic values (per block).
    /// Each block holds sorted (key, allele) pairs, where key = offset * n_haps + hap.
    exceptions: Vec<Vec<(u64, u8)>>,
    /// Total number of exception entries
    exc_count: usize,
    /// Number of markers
    n_markers: usize,
    /// Number of haplotypes (stride for indexing)
    n_haps: usize,
}

impl MutableGenotypes {
    /// Exception block size in markers (keeps per-block scans cache-friendly).
    const EXC_BLOCK_SIZE: usize = 256;

    /// Pack (marker offset, hap) into a u64 key for the exception block
    #[inline(always)]
    fn pack_key(offset: usize, hap: usize, n_haps: usize) -> u64 {
        (offset as u64) * (n_haps as u64) + (hap as u64)
    }

    #[inline(always)]
    fn unpack_key(key: u64, n_haps: usize) -> (usize, usize) {
        let hap = (key % n_haps as u64) as usize;
        let offset = (key / n_haps as u64) as usize;
        (offset, hap)
    }

    #[inline(always)]
    fn block_index(marker: usize) -> (usize, usize) {
        let block_idx = marker / Self::EXC_BLOCK_SIZE;
        let offset = marker % Self::EXC_BLOCK_SIZE;
        (block_idx, offset)
    }

    /// Create from a function that provides alleles
    ///
    /// Allele values: 0 = REF, 1-254 = ALT alleles, 255 = missing
    pub fn from_fn<F>(n_markers: usize, n_haps: usize, mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> u8,
    {
        let total_bits = n_markers * n_haps;
        let mut bits = bitvec![u64, Lsb0; 0; total_bits];
        let n_blocks = (n_markers + Self::EXC_BLOCK_SIZE - 1) / Self::EXC_BLOCK_SIZE;
        let mut exceptions: Vec<Vec<(u64, u8)>> = vec![Vec::new(); n_blocks];
        let mut exc_count = 0usize;

        for m in 0..n_markers {
            let base = m * n_haps;
            for h in 0..n_haps {
                let allele = f(m, h);
                match allele {
                    0 => {} // bit already 0
                    1 => bits.set(base + h, true),
                    _ => {
                        let (block_idx, offset) = Self::block_index(m);
                        let key = Self::pack_key(offset, h, n_haps);
                        let block = &mut exceptions[block_idx];
                        match block.binary_search_by_key(&key, |(k, _)| *k) {
                            Ok(pos) => block[pos].1 = allele,
                            Err(pos) => {
                                block.insert(pos, (key, allele));
                                exc_count += 1;
                            }
                        }
                    }
                }
            }
        }

        Self { bits, exceptions, exc_count, n_markers, n_haps }
    }

    /// Number of markers
    #[inline]
    pub fn n_markers(&self) -> usize {
        self.n_markers
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
        let h = hap.as_usize();
        let idx = marker * self.n_haps + h;

        if self.exc_count == 0 {
            return self.bits[idx] as u8;
        }

        let (block_idx, offset) = Self::block_index(marker);
        if let Some(block) = self.exceptions.get(block_idx) {
            if !block.is_empty() {
                let key = Self::pack_key(offset, h, self.n_haps);
                if let Ok(pos) = block.binary_search_by_key(&key, |(k, _)| *k) {
                    return block[pos].1;
                }
            }
        }

        self.bits[idx] as u8
    }

    /// Check if position is missing
    #[inline]
    pub fn is_missing(&self, marker: usize, hap: HapIdx) -> bool {
        self.get(marker, hap) == 255
    }

    /// Set allele at (marker, haplotype)
    ///
    /// Allele values: 0 = REF, 1-254 = ALT alleles, 255 = missing
    #[inline]
    pub fn set(&mut self, marker: usize, hap: HapIdx, allele: u8) {
        let h = hap.as_usize();
        let idx = marker * self.n_haps + h;
        match allele {
            0 => {
                self.bits.set(idx, false);
                let (block_idx, offset) = Self::block_index(marker);
                if let Some(block) = self.exceptions.get_mut(block_idx) {
                    if !block.is_empty() {
                        let key = Self::pack_key(offset, h, self.n_haps);
                        if let Ok(pos) = block.binary_search_by_key(&key, |(k, _)| *k) {
                            block.remove(pos);
                            self.exc_count = self.exc_count.saturating_sub(1);
                        }
                    }
                }
            }
            1 => {
                self.bits.set(idx, true);
                let (block_idx, offset) = Self::block_index(marker);
                if let Some(block) = self.exceptions.get_mut(block_idx) {
                    if !block.is_empty() {
                        let key = Self::pack_key(offset, h, self.n_haps);
                        if let Ok(pos) = block.binary_search_by_key(&key, |(k, _)| *k) {
                            block.remove(pos);
                            self.exc_count = self.exc_count.saturating_sub(1);
                        }
                    }
                }
            }
            _ => {
                // Non-biallelic: store in exception map
                // Set bit to 0 as a sentinel (exception takes precedence)
                self.bits.set(idx, false);
                let (block_idx, offset) = Self::block_index(marker);
                let key = Self::pack_key(offset, h, self.n_haps);
                let block = &mut self.exceptions[block_idx];
                match block.binary_search_by_key(&key, |(k, _)| *k) {
                    Ok(pos) => block[pos].1 = allele,
                    Err(pos) => {
                        block.insert(pos, (key, allele));
                        self.exc_count += 1;
                    }
                }
            }
        }
    }

    /// Get all alleles at a marker as a Vec
    ///
    /// Note: Returns owned Vec since bit-packed data cannot be returned as slice
    ///
    /// Optimized: avoids per-haplotype HashMap lookups by using a two-pass approach.
    #[inline]
    pub fn marker_alleles(&self, marker: usize) -> Vec<u8> {
        let base = marker * self.n_haps;
        let mut result = Vec::with_capacity(self.n_haps);

        // Fast path: if no exceptions, just extract bits directly
        if self.exc_count == 0 {
            for h in 0..self.n_haps {
                result.push(self.bits[base + h] as u8);
            }
            return result;
        }

        // Two-pass approach for sparse exceptions:
        // Pass 1: Extract all bits (no HashMap lookups)
        for h in 0..self.n_haps {
            result.push(self.bits[base + h] as u8);
        }

        // Pass 2: Fix up exceptions for this marker (only block-local entries)
        let (block_idx, offset) = Self::block_index(marker);
        if let Some(block) = self.exceptions.get(block_idx) {
            for &(key, val) in block {
                let (exc_offset, exc_h) = Self::unpack_key(key, self.n_haps);
                if exc_offset == offset {
                    result[exc_h] = val;
                }
            }
        }

        result
    }

    /// Get all alleles for a haplotype
    ///
    /// Returns a vector with values: 0 (REF), 1-254 (ALT), or 255 (missing)
    ///
    /// Optimized: avoids per-marker HashMap lookups by using a two-pass approach:
    /// 1. Extract all bits directly (fast, cache-friendly)
    /// 2. Fix up exceptions for this haplotype (sparse, typically <1%)
    pub fn haplotype(&self, hap: HapIdx) -> Vec<u8> {
        let h = hap.as_usize();
        let mut result = Vec::with_capacity(self.n_markers);

        // Fast path: if no exceptions, just extract bits directly
        if self.exc_count == 0 {
            for m in 0..self.n_markers {
                result.push(self.bits[m * self.n_haps + h] as u8);
            }
            return result;
        }

        // Two-pass approach for sparse exceptions:
        // Pass 1: Extract all bits (no HashMap lookups)
        for m in 0..self.n_markers {
            result.push(self.bits[m * self.n_haps + h] as u8);
        }

        // Pass 2: Fix up exceptions for this haplotype (sparse, typically <1%)
        for (block_idx, block) in self.exceptions.iter().enumerate() {
            let block_start = block_idx * Self::EXC_BLOCK_SIZE;
            for &(key, val) in block {
                let (offset, exc_h) = Self::unpack_key(key, self.n_haps);
                if exc_h == h {
                    let marker = block_start + offset;
                    if marker < self.n_markers {
                        result[marker] = val;
                    }
                }
            }
        }

        result
    }

    /// Swap alleles between two haplotypes at a marker
    #[inline]
    pub fn swap(&mut self, marker: usize, hap1: HapIdx, hap2: HapIdx) {
        // Get current values
        let v1 = self.get(marker, hap1);
        let v2 = self.get(marker, hap2);

        // Set swapped values
        self.set(marker, hap1, v2);
        self.set(marker, hap2, v1);
    }

    /// Swap alleles between two haplotypes for markers indicated by mask
    ///
    /// Efficiently swaps alleles where the mask bit is set.
    pub fn swap_haplotypes(&mut self, hap1: HapIdx, hap2: HapIdx, mask: &BitSlice<u8, Lsb0>) {
        assert_eq!(mask.len(), self.n_markers, "Swap mask length mismatch");
        
        for m in mask.iter_ones() {
            self.swap(m, hap1, hap2);
        }
    }

}

// Test-only diagnostic methods
#[cfg(test)]
impl MutableGenotypes {
    /// Get number of exceptions (for diagnostics)
    fn n_exceptions(&self) -> usize {
        self.exc_count
    }

    /// Get approximate memory usage in bytes
    fn memory_bytes(&self) -> usize {
        // BitVec: bits / 8 bytes
        let bits_bytes = (self.bits.len() + 7) / 8;
        // Exception blocks: key/value pairs (u64 + u8) plus Vec overhead
        let exceptions_bytes: usize = self.exceptions.iter().map(|b| b.len() * (8 + 1)).sum();
        // Struct overhead
        let struct_bytes = std::mem::size_of::<Self>();

        bits_bytes + exceptions_bytes + struct_bytes
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

        // Verify exceptions are stored
        assert_eq!(geno.n_exceptions(), 4); // 4 missing values
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

        // 5 exceptions: 2, 3, 2, 255, 3
        assert_eq!(geno.n_exceptions(), 5);
    }

    #[test]
    fn test_marker_alleles() {
        let geno = MutableGenotypes::from_fn(3, 4, |m, h| ((m + h) % 2) as u8);

        // Marker 0: h=0->0, h=1->1, h=2->0, h=3->1
        assert_eq!(geno.marker_alleles(0), vec![0, 1, 0, 1]);
        // Marker 1: h=0->1, h=1->0, h=2->1, h=3->0
        assert_eq!(geno.marker_alleles(1), vec![1, 0, 1, 0]);
        // Marker 2: h=0->0, h=1->1, h=2->0, h=3->1
        assert_eq!(geno.marker_alleles(2), vec![0, 1, 0, 1]);
    }

    #[test]
    fn test_memory_layout() {
        // Verify that values are stored and retrieved correctly
        let geno = MutableGenotypes::from_fn(3, 2, |m, h| (m * 10 + h) as u8);

        // Marker 0: [0, 1] - both fit in bits
        // Marker 1: [10, 11] - both go to exceptions
        // Marker 2: [20, 21] - both go to exceptions
        assert_eq!(geno.get(0, HapIdx::new(0)), 0);
        assert_eq!(geno.get(0, HapIdx::new(1)), 1);
        assert_eq!(geno.get(1, HapIdx::new(0)), 10);
        assert_eq!(geno.get(1, HapIdx::new(1)), 11);
        assert_eq!(geno.get(2, HapIdx::new(0)), 20);
        assert_eq!(geno.get(2, HapIdx::new(1)), 21);

        // 4 exceptions (10, 11, 20, 21)
        assert_eq!(geno.n_exceptions(), 4);
    }

    #[test]
    fn test_memory_efficiency() {
        // Verify memory savings for pure biallelic data
        let n_markers = 10000;
        let n_haps = 1000;

        let geno = MutableGenotypes::from_fn(n_markers, n_haps, |m, h| ((m + h) % 2) as u8);

        // Should have zero exceptions (all 0 or 1)
        assert_eq!(geno.n_exceptions(), 0);

        // Memory should be ~1.25 MB (10M bits / 8 = 1.25 MB)
        let mem = geno.memory_bytes();
        let expected_max = 2_000_000; // 2 MB with overhead
        assert!(mem < expected_max, "Memory {} exceeds expected {}", mem, expected_max);

        // Compare to byte-packed: 10M bytes = 10 MB
        // 8x savings confirmed
    }

    #[test]
    fn test_swap_multiallelic() {
        // Test swapping when one value is multiallelic
        let mut geno = MutableGenotypes::from_fn(2, 2, |m, h| {
            match (m, h) {
                (0, 0) => 1,   // biallelic
                (0, 1) => 3,   // multiallelic
                (1, 0) => 0,
                (1, 1) => 1,
                _ => 0,
            }
        });

        assert_eq!(geno.get(0, HapIdx::new(0)), 1);
        assert_eq!(geno.get(0, HapIdx::new(1)), 3);

        geno.swap(0, HapIdx::new(0), HapIdx::new(1));

        assert_eq!(geno.get(0, HapIdx::new(0)), 3);  // multiallelic moved
        assert_eq!(geno.get(0, HapIdx::new(1)), 1);  // biallelic moved
    }
}
