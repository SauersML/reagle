//! # Dictionary-Compressed Storage
//!
//! Run-length / dictionary compression for haplotype patterns.
//! Replaces `bref/SeqCoder3.java` logic.

use bitvec::prelude::*;
use std::collections::HashMap;

use crate::data::HapIdx;

/// Dictionary-compressed storage for haplotype blocks
#[derive(Clone, Debug)]
pub struct DictionaryColumn {
    /// Unique haplotype patterns (each pattern spans multiple markers)
    /// patterns[i] is the allele sequence for dictionary entry i
    patterns: Vec<BitVec<u64, Lsb0>>,

    /// For each haplotype, which pattern index it uses
    hap_to_pattern: Vec<u16>,

    /// Number of markers covered by this block
    n_markers: u32,

    /// Bits per allele (1 for biallelic)
    bits_per_allele: u8,
}

impl DictionaryColumn {
    /// Compress a set of marker columns into a dictionary block
    pub fn compress(
        columns: &[impl Fn(HapIdx) -> u8],
        n_markers: usize,
        n_haplotypes: usize,
        bits_per_allele: u8,
    ) -> Self {
        // Build pattern for each haplotype
        // We use (bits_per_allele + 1) bits per marker. The last bit is the missing flag.
        let bits_per_marker = bits_per_allele as usize + 1;
        let pattern_bits = n_markers * bits_per_marker;

        let mut pattern_map: HashMap<BitVec<u64, Lsb0>, u16> = HashMap::new();
        let mut patterns: Vec<BitVec<u64, Lsb0>> = Vec::new();
        let mut hap_to_pattern: Vec<u16> = Vec::with_capacity(n_haplotypes);

        for h in 0..n_haplotypes {
            let hap = HapIdx::new(h as u32);
            let mut pattern = bitvec![u64, Lsb0; 0; pattern_bits];

            for m in 0..n_markers {
                let allele = columns[m](hap);
                let start = m * bits_per_marker;
                if allele == 255 {
                    // Set the missing bit (the last bit of this marker's segment)
                    pattern.set(start + bits_per_marker - 1, true);
                } else {
                    for b in 0..bits_per_allele as usize {
                        if (allele >> b) & 1 == 1 {
                            pattern.set(start + b, true);
                        }
                    }
                }
            }

            let pattern_idx = if let Some(&idx) = pattern_map.get(&pattern) {
                idx
            } else {
                let idx = patterns.len() as u16;
                pattern_map.insert(pattern.clone(), idx);
                patterns.push(pattern);
                idx
            };

            hap_to_pattern.push(pattern_idx);
        }

        Self {
            patterns,
            hap_to_pattern,
            n_markers: n_markers as u32,
            bits_per_allele,
        }
    }

    /// Get allele at (marker_offset, haplotype)
    #[inline]
    pub fn get(&self, marker_offset: usize, hap: HapIdx) -> u8 {
        let pattern_idx = self.hap_to_pattern[hap.as_usize()] as usize;
        let pattern = &self.patterns[pattern_idx];

        let bits_per_marker = self.bits_per_allele as usize + 1;
        let start = marker_offset * bits_per_marker;

        // Check the missing bit
        if pattern[start + bits_per_marker - 1] {
            return 255;
        }

        let mut allele = 0u8;
        for b in 0..self.bits_per_allele as usize {
            if pattern[start + b] {
                allele |= 1 << b;
            }
        }
        allele
    }

    /// Number of haplotypes
    pub fn n_haplotypes(&self) -> usize {
        self.hap_to_pattern.len()
    }

    /// Number of markers in this block
    pub fn n_markers(&self) -> usize {
        self.n_markers as usize
    }

    /// Number of unique patterns
    pub fn n_patterns(&self) -> usize {
        self.patterns.len()
    }

    /// Compression ratio (patterns / haplotypes)
    pub fn compression_ratio(&self) -> f64 {
        self.n_patterns() as f64 / self.n_haplotypes() as f64
    }

    /// Count ALT alleles at a marker offset
    pub fn alt_count(&self, marker_offset: usize) -> usize {
        let mut count = 0;
        for hap_idx in 0..self.hap_to_pattern.len() {
            let a = self.get(marker_offset, HapIdx::new(hap_idx as u32));
            if a > 0 && a != 255 {
                count += 1;
            }
        }
        count
    }

    /// Memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        let pattern_bytes: usize = self
            .patterns
            .iter()
            .map(|p| p.as_raw_slice().len() * std::mem::size_of::<u64>())
            .sum();
        pattern_bytes
            + self.hap_to_pattern.len() * std::mem::size_of::<u16>()
            + std::mem::size_of::<Self>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dictionary_compression() {
        // Create 4 haplotypes with 3 markers
        // Hap 0: [0, 1, 0]
        // Hap 1: [0, 1, 0]  <- same as hap 0
        // Hap 2: [1, 0, 1]
        // Hap 3: [1, 0, 1]  <- same as hap 2

        let data = vec![
            vec![0u8, 0, 1, 1], // Marker 0
            vec![1u8, 1, 0, 0], // Marker 1
            vec![0u8, 0, 1, 1], // Marker 2
        ];

        let columns: Vec<Box<dyn Fn(HapIdx) -> u8>> = data
            .iter()
            .map(|col| {
                let col = col.clone();
                Box::new(move |h: HapIdx| col[h.as_usize()]) as Box<dyn Fn(HapIdx) -> u8>
            })
            .collect();

        let dict = DictionaryColumn::compress(
            &columns.iter().map(|f| |h| f(h)).collect::<Vec<_>>(),
            3,
            4,
            1,
        );

        assert_eq!(dict.n_haplotypes(), 4);
        assert_eq!(dict.n_markers(), 3);
        assert_eq!(dict.n_patterns(), 2); // Only 2 unique patterns

        // Verify values
        assert_eq!(dict.get(0, HapIdx::new(0)), 0);
        assert_eq!(dict.get(1, HapIdx::new(0)), 1);
        assert_eq!(dict.get(2, HapIdx::new(0)), 0);

        assert_eq!(dict.get(0, HapIdx::new(2)), 1);
        assert_eq!(dict.get(1, HapIdx::new(2)), 0);
        assert_eq!(dict.get(2, HapIdx::new(2)), 1);

    }
}
