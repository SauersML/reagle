//! # Data Module
//!
//! ## Role
//! In-memory representations of genomic data. This is the core "Model" layer.
//!
//! ## Design Philosophy: Data-Oriented Design
//! - **Structure of Arrays (SoA):** Store markers and genotype columns separately
//!   for cache-friendly iteration.
//! - **Zero-cost newtypes:** `MarkerIdx`, `HapIdx`, `SampleIdx` prevent index bugs
//!   at compile time with no runtime overhead.
//! - **Enum-based polymorphism:** `GenotypeColumn` variants (Dense/Sparse/Dictionary)
//!   replace Java's class hierarchy with a single stack-allocated enum.
//!
//! ## Sub-modules
//! - `marker`: Genomic position and allele definitions
//! - `haplotype`: Index types for samples and haplotypes
//! - `genetic_map`: Physical-to-genetic distance interpolation
//! - `storage`: Genotype storage backends

pub mod genetic_map;
pub mod haplotype;
pub mod marker;
pub mod storage;
