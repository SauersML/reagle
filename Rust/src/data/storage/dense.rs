//! # Dense Bit-Packed Storage
//!
//! ## Role
//! Efficient storage for common variants using bit vectors.
//! Replaces `vcf/BitArrayGTRec.java`.
//!
//! ## Spec
//!
//! ### DenseColumn Struct
//! ```rust,ignore
//! pub struct DenseColumn {
//!     /// Bit vector: 1 bit per haplotype for biallelic markers.
//!     /// For multi-allelic: ceil(log2(n_alleles)) bits per haplotype.
//!     bits: BitVec<u64, Lsb0>,
//!
//!     /// Bits per allele (1 for biallelic, 2 for 3-4 alleles, etc.)
//!     bits_per_allele: u8,
//!
//!     /// Number of haplotypes stored.
//!     n_haplotypes: u32,
//! }
//! ```
//!
//! ### Methods
//! ```rust,ignore
//! impl DenseColumn {
//!     /// Create from allele iterator.
//!     pub fn from_alleles(alleles: impl Iterator<Item = u8>, n_alleles: usize) -> Self;
//!
//!     /// Get allele for haplotype.
//!     #[inline]
//!     pub fn get(&self, hap: HapIdx) -> u8;
//!
//!     /// Set allele for haplotype.
//!     pub fn set(&mut self, hap: HapIdx, allele: u8);
//!
//!     /// Iterate all alleles.
//!     pub fn iter(&self) -> impl Iterator<Item = u8>;
//!
//!     /// Memory usage in bytes.
//!     pub fn size_bytes(&self) -> usize;
//! }
//! ```
//!
//! ### Memory Layout
//! For biallelic markers with 1000 haplotypes:
//! - Dense: 1000 bits = 125 bytes
//! - Naive Vec<u8>: 1000 bytes
//! - Savings: 8x
//!
//! ### Performance Notes
//! - Use `#[inline]` on hot paths.
//! - Consider SIMD for population-level operations (allele counting).
