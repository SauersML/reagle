//! # Sparse Storage for Rare Variants
//!
//! ## Role
//! Efficient storage for rare variants (MAF < 1%) by storing only carrier indices.
//! Replaces `vcf/LowMafGTRec.java`.
//!
//! ## Spec
//!
//! ### SparseColumn Struct
//! ```rust,ignore
//! pub struct SparseColumn {
//!     /// Sorted indices of haplotypes carrying the ALT allele.
//!     /// For multi-allelic: Vec<(HapIdx, u8)> storing (haplotype, allele).
//!     carriers: Vec<HapIdx>,
//!
//!     /// Total number of haplotypes (needed to know who has REF).
//!     n_haplotypes: u32,
//! }
//! ```
//!
//! ### Methods
//! ```rust,ignore
//! impl SparseColumn {
//!     /// Create from carrier indices.
//!     pub fn from_carriers(carriers: Vec<HapIdx>, n_haplotypes: u32) -> Self;
//!
//!     /// Get allele for haplotype (binary search).
//!     #[inline]
//!     pub fn get(&self, hap: HapIdx) -> u8 {
//!         match self.carriers.binary_search(&hap) {
//!             Ok(_) => 1,  // Found: has ALT
//!             Err(_) => 0, // Not found: has REF
//!         }
//!     }
//!
//!     /// Iterate carrier indices.
//!     pub fn carriers(&self) -> &[HapIdx];
//!
//!     /// Number of carriers (ALT allele count).
//!     pub fn n_carriers(&self) -> usize;
//!
//!     /// Minor allele frequency.
//!     pub fn maf(&self) -> f64;
//! }
//! ```
//!
//! ### When to Use
//! - MAF < 0.01: Sparse is more memory-efficient
//! - For a marker with 10,000 haplotypes and MAF=0.001:
//!   - Dense: 10,000 bits = 1,250 bytes
//!   - Sparse: 10 carriers * 4 bytes = 40 bytes
//!   - Savings: 31x
//!
//! ### Multi-allelic Extension
//! ```rust,ignore
//! pub struct SparseMultiAllelic {
//!     /// (haplotype_index, allele) pairs, sorted by haplotype.
//!     entries: Vec<(HapIdx, u8)>,
//!     n_haplotypes: u32,
//! }
//! ```
