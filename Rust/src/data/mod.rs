//! # Data Module
//!
//! In-memory representations of genomic data. This is the core "Model" layer.
//!
//! ## Design Philosophy: Data-Oriented Design
//! - **Structure of Arrays (SoA):** Store markers and genotype columns separately
//!   for cache-friendly iteration.
//! - **Zero-cost newtypes:** `MarkerIdx`, `HapIdx`, `SampleIdx` prevent index bugs
//!   at compile time with no runtime overhead.
//! - **Enum-based polymorphism:** `GenotypeColumn` variants (Dense/Sparse/Dictionary)
//!   replace Java's class hierarchy with a single stack-allocated enum.

pub mod genetic_map;
pub mod haplotype;
pub mod marker;
pub mod storage;

// Re-export commonly used types
pub use genetic_map::GeneticMap;
pub use haplotype::{HapIdx, SampleIdx, Samples};
pub use marker::{Allele, Marker, MarkerIdx, Markers};
pub use storage::{
    DenseColumn, DictionaryColumn, GenotypeColumn, GenotypeMatrix, MutableGenotypes, SparseColumn,
};

/// Chromosome identifier (0-based index into chromosome name table)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChromIdx(pub u16);

impl ChromIdx {
    pub fn new(idx: u16) -> Self {
        Self(idx)
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl From<u16> for ChromIdx {
    fn from(idx: u16) -> Self {
        Self(idx)
    }
}

impl From<ChromIdx> for usize {
    fn from(idx: ChromIdx) -> usize {
        idx.0 as usize
    }
}