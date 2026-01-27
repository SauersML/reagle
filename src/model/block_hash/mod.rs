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
//! 2. Track probability per pattern (plus reservoir), while retaining global IDs for mapping/bridging
//! 3. Use TransitionBridge to correctly map probability between windows
//!
//! ## Architecture
//!
//! ### Immutable/Mutable Separation (Critical for Parallelization)
//!
//! - CompressedBlock (immutable): Built once, Arc-wrapped, shared across all samples
//!   - Contains reference panel compression, pattern counts, global ID mappings
//!   - Zero-cost sharing across threads
//!
//! - BlockHmmWorkspace (mutable): Per-sample workspace, thread-local or pooled
//!   - Contains forward/backward buffers, emissions, checkpoints
//!   - Prevents re-compression overhead
//!
//! - ReferenceMap (immutable): Pre-computed container of blocks and transitions
//!   - Builds all CompressedBlocks and TransitionBridges once
//!   - Enables efficient parallel imputation across samples
//!
//! ### Core Components
//!
//! - types.rs: Type-safe GlobalId and PatternId newtypes to prevent index confusion
//! - compressed_block.rs: Immutable reference data with pattern mappings
//! - workspace.rs: Mutable per-sample HMM state buffers
//! - transition.rs: CSR sparse format for deterministic, cache-friendly transitions
//! - compression.rs: Builds CompressedBlock from GenotypeMatrix using DictionaryColumn
//! - hmm.rs: Forward/backward passes reusing existing HmmUpdater SIMD kernels
//!
//! ### Multiallelic Safety
//!
//! Uses exact allele matching instead of biallelic 0/1 assumption:
//! - Match: 1.0 - error_rate (regardless of allele value)
//! - Mismatch: error_rate (works for alleles {0, 1, 2, 3, ...})
//!
//! ### Performance Characteristics
//!
//! - Memory: <15GB for 1KG+HGDP (~8k haplotypes)
//! - Compression: ~5-10x (8k haps → 800-1600 unique patterns per window)
//! - Parallelization: Zero-cost reference sharing via Arc
//! - SIMD: Leverages AVX-512 kernels from existing HmmUpdater

pub mod compressed_block;
pub mod compression;
pub mod hmm;
pub mod reference_map;
pub mod transition;
pub mod types;
pub mod weighted_kernel;
pub mod workspace;

// Crate-internal API exports (will become public when integrated)
pub use compressed_block::CompressedBlock;
pub use reference_map::ReferenceMap;
pub use workspace::BlockHmmWorkspace;
