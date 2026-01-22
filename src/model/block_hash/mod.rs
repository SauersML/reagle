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
mod micro_window;
mod transition;
mod compression;
mod hmm;

// Public API exports
pub use micro_window::MicroWindow;
pub use types::{GlobalId, PatternId};
pub use compression::{
    build_micro_window, build_all_windows, CompressionStats,
    DEFAULT_WINDOW_SIZE, DEFAULT_MAX_STATES
};
pub use hmm::{forward_pass_within_window, forward_pass_all_windows, backward_pass_all_windows};
