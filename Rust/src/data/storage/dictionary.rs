//! # Dictionary-Compressed Storage
//!
//! ## Role
//! Run-length / dictionary compression for haplotype patterns.
//! This is the in-memory representation; I/O is in `io/bref3.rs` (future).
//! Replaces `bref/SeqCoder3.java` logic.
//!
//! ## Spec
//!
//! ### DictionaryBlock Struct
//! ```rust,ignore
//! pub struct DictionaryBlock {
//!     /// Unique haplotype patterns (each pattern spans multiple markers).
//!     /// patterns[i] is the allele sequence for dictionary entry i.
//!     patterns: Vec<BitVec<u64, Lsb0>>,
//!
//!     /// For each haplotype, which pattern index it uses.
//!     /// hap_to_pattern[h] gives the pattern index for haplotype h.
//!     hap_to_pattern: Vec<u16>,
//!
//!     /// Number of markers covered by this block.
//!     n_markers: u32,
//! }
//! ```
//!
//! ### Access Pattern
//! ```rust,ignore
//! impl DictionaryBlock {
//!     /// Get allele at (marker_offset, haplotype).
//!     #[inline]
//!     pub fn get(&self, marker_offset: usize, hap: HapIdx) -> u8 {
//!         let pattern_idx = self.hap_to_pattern[hap.0 as usize];
//!         let pattern = &self.patterns[pattern_idx as usize];
//!         pattern[marker_offset] as u8
//!     }
//! }
//! ```
//!
//! ### Compression Rationale
//! Real haplotypes are highly redundant due to linkage disequilibrium.
//! In a block of 100 markers with 10,000 haplotypes:
//! - Naive: 100 * 10,000 = 1,000,000 bits
//! - With 500 unique patterns: 500 * 100 + 10,000 * 9 = 140,000 bits
//! - Savings: 7x
//!
//! ### Building a DictionaryBlock
//! ```rust,ignore
//! impl DictionaryBlock {
//!     /// Compress a slice of markers into a dictionary block.
//!     pub fn compress(
//!         columns: &[GenotypeColumn],
//!         n_haplotypes: usize,
//!     ) -> Self;
//! }
//! ```
//!
//! ### Performance Notes
//! - Pattern lookup is O(1).
//! - Good for HMM iteration where we access all markers for one haplotype.
