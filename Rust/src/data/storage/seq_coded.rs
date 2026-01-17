//! Sequence-coded genotype storage for BREF3 data
//!
//! This format keeps data in BREF3's native compression without expansion.
//! Memory savings: ~6x compared to Dense storage for typical reference panels.

use std::sync::Arc;
use crate::data::HapIdx;

/// A block of markers sharing the same haplotype-to-sequence mapping
///
/// This preserves BREF3's efficient encoding where:
/// - Each haplotype maps to a sequence pattern index (shared across markers in block)
/// - Each marker stores only the allele for each sequence pattern
///
/// Memory per marker = n_seq bytes (typically 16-64)
/// vs Dense = n_haps/8 bytes (12.5KB for 100K haps)
#[derive(Clone, Debug)]
pub struct SeqCodedBlock {
    /// Maps haplotype index -> sequence pattern index (shared across all markers in block)
    hap_to_seq: Arc<Vec<u16>>,
    /// For each marker: maps sequence pattern index -> allele
    /// Outer vec indexed by local marker offset within block
    seq_to_allele: Vec<Vec<u8>>,
    /// Number of haplotypes
    n_haps: usize,
}

impl SeqCodedBlock {
    /// Create a new sequence-coded block
    pub fn new(hap_to_seq: Vec<u16>) -> Self {
        let n_haps = hap_to_seq.len();
        Self {
            hap_to_seq: Arc::new(hap_to_seq),
            seq_to_allele: Vec::new(),
            n_haps,
        }
    }

    /// Add a marker's sequence-to-allele mapping to the block
    pub fn push_marker(&mut self, seq_alleles: Vec<u8>) {
        self.seq_to_allele.push(seq_alleles);
    }

    /// Number of markers in this block
    pub fn n_markers(&self) -> usize {
        self.seq_to_allele.len()
    }

    /// Number of haplotypes
    pub fn n_haplotypes(&self) -> usize {
        self.n_haps
    }

    /// Get allele for a haplotype at a marker (by local offset within block)
    #[inline]
    pub fn get(&self, marker_offset: usize, hap: HapIdx) -> u8 {
        let seq_idx = self.hap_to_seq[hap.as_usize()] as usize;
        self.seq_to_allele[marker_offset][seq_idx]
    }

    /// Memory usage in bytes
    pub fn size_bytes(&self) -> usize {
        let hap_to_seq_size = self.hap_to_seq.len() * 2;
        let seq_to_allele_size: usize = self.seq_to_allele.iter().map(|v| v.len()).sum();
        hap_to_seq_size + seq_to_allele_size + std::mem::size_of::<Self>()
    }

    /// Count ALT allele carriers at a marker
    pub fn alt_count(&self, marker_offset: usize) -> usize {
        let seq_alleles = &self.seq_to_allele[marker_offset];
        let mut count = 0usize;
        for &seq_idx in self.hap_to_seq.iter() {
            if seq_alleles[seq_idx as usize] > 0 {
                count += 1;
            }
        }
        count
    }
}

/// A single marker view into a SeqCodedBlock
#[derive(Clone, Debug)]
pub struct SeqCodedColumn {
    block: Arc<SeqCodedBlock>,
    marker_offset: usize,
}

impl SeqCodedColumn {
    /// Create a view into a specific marker within a block
    pub fn new(block: Arc<SeqCodedBlock>, marker_offset: usize) -> Self {
        Self { block, marker_offset }
    }

    /// Get allele for a haplotype
    #[inline]
    pub fn get(&self, hap: HapIdx) -> u8 {
        self.block.get(self.marker_offset, hap)
    }

    #[inline]
    pub fn hap_to_seq(&self) -> &[u16] {
        &self.block.hap_to_seq
    }

    #[inline]
    pub fn seq_alleles(&self) -> &[u8] {
        &self.block.seq_to_allele[self.marker_offset]
    }

    /// Number of haplotypes
    pub fn n_haplotypes(&self) -> usize {
        self.block.n_haplotypes()
    }

    /// Count ALT allele carriers
    pub fn alt_count(&self) -> usize {
        self.block.alt_count(self.marker_offset)
    }

    /// Memory usage (amortized per marker)
    pub fn size_bytes(&self) -> usize {
        self.block.size_bytes() / self.block.n_markers().max(1)
    }
}
