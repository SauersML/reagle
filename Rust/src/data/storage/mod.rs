//! # Genotype Storage Backends
//!
//! ## Role
//! Polymorphic storage for genotype data. Replaces Java's `GTRec` class hierarchy
//! with a single Rust enum.
//!
//! ## Design: The Core Enum
//! This is the most important type in the entire project:
//!
//! ```rust,ignore
//! pub enum GenotypeColumn {
//!     /// High-frequency variants (MAF > 0.05).
//!     /// Bit-packed: 1 bit per haplotype for biallelic sites.
//!     Dense(DenseColumn),
//!
//!     /// Rare variants (MAF < 0.01).
//!     /// Store only indices of ALT allele carriers.
//!     Sparse(SparseColumn),
//!
//!     /// Dictionary-compressed blocks.
//!     /// For runs of similar haplotype patterns.
//!     Dictionary(DictionaryBlock),
//! }
//! ```
//!
//! ## Why Enum Instead of Trait Objects?
//! - **Cache locality:** Enum variants are stack-allocated, no pointer chasing.
//! - **Match exhaustiveness:** Compiler ensures all variants are handled.
//! - **Inlining:** No dynamic dispatch overhead in hot loops.
//!
//! ## Common Interface
//! ```rust,ignore
//! impl GenotypeColumn {
//!     /// Get allele for a specific haplotype (0 = REF, 1+ = ALT).
//!     pub fn get_allele(&self, hap: HapIdx) -> u8;
//!
//!     /// Number of haplotypes in this column.
//!     pub fn n_haplotypes(&self) -> usize;
//!
//!     /// Count of each allele.
//!     pub fn allele_counts(&self) -> Vec<u32>;
//! }
//! ```

pub mod dense;
pub mod dictionary;
pub mod matrix;
pub mod sparse;
