//! # Sliding Window Management
//!
//! ## Role
//! Chunk chromosome-scale data into overlapping windows for parallel processing.
//! Replaces `vcf/SlidingWindow.java`.
//!
//! ## Why Windows?
//! - Whole-genome data is too large to fit in memory at once.
//! - Phasing/imputation can be done independently per window.
//! - Overlap ensures continuity at window boundaries.
//!
//! ## Spec
//!
//! ### WindowConfig
//! ```rust,ignore
//! pub struct WindowConfig {
//!     /// Target markers per window.
//!     pub window_size: usize,    // default: 40_000
//!
//!     /// Overlap with previous window.
//!     pub overlap: usize,        // default: 2_000
//! }
//! ```
//!
//! ### WindowIterator
//! ```rust,ignore
//! pub struct WindowIterator<R: VcfReader> {
//!     reader: R,
//!     config: WindowConfig,
//!     buffer: VecDeque<VcfRecord>,
//!     window_idx: usize,
//! }
//!
//! impl<R: VcfReader> Iterator for WindowIterator<R> {
//!     type Item = Result<Window>;
//!
//!     fn next(&mut self) -> Option<Self::Item>;
//! }
//! ```
//!
//! ### Window Struct
//! ```rust,ignore
//! pub struct Window {
//!     /// The genotype data for this window.
//!     pub matrix: GenotypeMatrix,
//!
//!     /// Index of this window (0-based).
//!     pub window_idx: usize,
//!
//!     /// Genomic range: (chrom, start_bp, end_bp).
//!     pub region: (String, u32, u32),
//!
//!     /// Marker indices that overlap with the previous window.
//!     /// These need special handling during result merging.
//!     pub overlap_start: usize,
//!
//!     /// Marker indices that overlap with the next window.
//!     pub overlap_end: usize,
//! }
//! ```
//!
//! ### Result Merging
//! ```rust,ignore
//! /// Merge phased results from overlapping windows.
//! /// Uses Li & Stephens switch error minimization in overlap regions.
//! pub fn merge_windows(
//!     windows: &[PhasedWindow],
//! ) -> GenotypeMatrix;
//! ```
//!
//! ### Memory Management
//! - Only keep 2 windows in memory at a time (current + overlap buffer).
//! - Stream results to output VCF as windows complete.
