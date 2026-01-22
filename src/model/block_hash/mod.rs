//! # Block-Hash Clade HMM
//!
//! This module implements the Adaptive Full-Panel Block-Hash Clade HMM to fix
//! the critical HMM state continuity bug in Reagle's imputation and phasing pipelines.
//!
//! ## The Bug
//!
//! The current implementation (`build_pbwt_hap_indices_for_batch`) selects different
//! neighbor sets at each cluster, causing "index scrambling" where the same state index
//! refers to different reference haplotypes across marker clusters. This breaks the
//! Li-Stephens assumption of haplotype continuity.
//!
//! ## The Solution
//!
//! Instead of dynamic neighbor selection, we:
//! 1. Compress the ENTIRE reference panel into unique patterns per window
//! 2. Track probability via Global Haplotype IDs (not local pattern indices)
//! 3. Use TransitionBridge to correctly map probability between windows
//!
//! ## Architecture
//!
//! - MicroWindow: Wraps `DictionaryColumn` with HMM state (reuses existing compression)
//! - TransitionBridge: Maps probability between windows via `zip()` on hap_to_pattern vectors
//! - Compression: Builds `DictionaryColumn` from `GenotypeMatrix` windows
//! - HMM Kernel: Reuses existing `HmmUpdater` for SIMD-optimized forward/backward passes
//!
//! ## Key Advantages
//!
//! - Maximum Accuracy: No donor truncation for 1KG+HGDP-scale panels
//! - Mathematically Correct: Probability follows physical DNA molecules
//! - High Performance: SIMD-friendly execution on compressed states
//! - Code Reuse: Leverages ~2,000 lines of existing, tested infrastructure

mod types;
mod compressed_block;
mod workspace;
mod reference_map;
mod transition;
mod compression;
mod hmm;

// Legacy compatibility (deprecated - use ReferenceMap instead)

mod micro_window;

// Public API exports
pub use types::{GlobalId, PatternId};
pub use compressed_block::CompressedBlock;
pub use workspace::BlockHmmWorkspace;
pub use reference_map::ReferenceMap;
pub use compression::{
    build_compressed_block, CompressionStats,
    DEFAULT_WINDOW_SIZE, DEFAULT_MAX_STATES
};

// Legacy exports (deprecated)

pub use micro_window::MicroWindow;
