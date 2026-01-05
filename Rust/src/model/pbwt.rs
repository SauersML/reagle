//! # Positional Burrows-Wheeler Transform (PBWT)
//!
//! ## Role
//! Find long haplotype matches efficiently in O(N*M) time.
//! Replaces `phase/PbwtPhaser.java`, `phase/PbwtPhaseIbs.java`.
//!
//! ## Background
//! The PBWT (Durbin 2014) maintains a sorted order of haplotypes such that
//! haplotypes with identical prefixes are adjacent. This enables O(1)
//! match extension per marker.
//!
//! ## Spec
//!
//! ### Core Data Structures
//! ```rust,ignore
//! pub struct PbwtState {
//!     /// Prefix array: permutation of haplotype indices sorted by
//!     /// reverse prefix (alleles seen so far, reading backwards).
//!     pub a: Vec<HapIdx>,
//!
//!     /// Divergence array: d[i] = position where a[i] and a[i-1]
//!     /// first differ (reading backwards from current position).
//!     pub d: Vec<u32>,
//!
//!     /// Current marker position.
//!     pub pos: MarkerIdx,
//! }
//! ```
//!
//! ### Forward Sweep
//! ```rust,ignore
//! impl PbwtState {
//!     /// Initialize at first marker.
//!     pub fn new(n_haplotypes: usize) -> Self;
//!
//!     /// Advance to next marker, updating a and d arrays.
//!     pub fn advance(&mut self, column: &GenotypeColumn);
//!
//!     /// Process entire matrix, returning final state.
//!     pub fn sweep(matrix: &GenotypeMatrix) -> Self;
//! }
//! ```
//!
//! ### Finding Long Matches
//! ```rust,ignore
//! /// A match between two haplotypes.
//! pub struct HaplotypeMatch {
//!     pub hap1: HapIdx,
//!     pub hap2: HapIdx,
//!     pub start: MarkerIdx,  // First marker of match
//!     pub end: MarkerIdx,    // Last marker of match (exclusive)
//! }
//!
//! /// Find all matches longer than min_length markers.
//! pub fn find_long_matches(
//!     matrix: &GenotypeMatrix,
//!     min_length: usize,
//! ) -> Vec<HaplotypeMatch>;
//!
//! /// Find the K best matches for a specific target haplotype.
//! /// Used to select "surrogate parents" for phasing.
//! pub fn find_best_matches(
//!     matrix: &GenotypeMatrix,
//!     target: HapIdx,
//!     k: usize,
//! ) -> Vec<HaplotypeMatch>;
//! ```
//!
//! ### Performance Notes
//! - This is the **hottest loop** in the program.
//! - Use `unsafe` unchecked indexing after bounds validation.
//! - Consider SIMD for the partition step.
//! - Memory access pattern is cache-friendly (sequential a[] access).
//!
//! ### Bidirectional PBWT
//! For phasing, we need matches in both directions:
//! ```rust,ignore
//! pub struct BidirectionalPbwt {
//!     forward: Vec<PbwtState>,   // States at each position, forward
//!     backward: Vec<PbwtState>,  // States at each position, backward
//! }
//! ```
