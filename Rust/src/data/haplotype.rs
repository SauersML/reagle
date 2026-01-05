//! # Haplotype and Sample Index Types
//!
//! ## Role
//! Type-safe indices for samples and haplotypes.
//!
//! ## Spec
//!
//! ### Newtype Indices
//! ```rust,ignore
//! #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
//! pub struct SampleIdx(pub u32);
//!
//! #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
//! pub struct HapIdx(pub u32);
//! ```
//!
//! ### Conversion Functions
//! For diploid organisms, each sample has two haplotypes:
//! ```rust,ignore
//! impl SampleIdx {
//!     /// Returns the two haplotype indices for this sample.
//!     pub fn to_haplotypes(self) -> (HapIdx, HapIdx) {
//!         let h = self.0 * 2;
//!         (HapIdx(h), HapIdx(h + 1))
//!     }
//! }
//!
//! impl HapIdx {
//!     /// Returns the sample this haplotype belongs to.
//!     pub fn to_sample(self) -> SampleIdx {
//!         SampleIdx(self.0 / 2)
//!     }
//!
//!     /// Returns true if this is the first (maternal) haplotype.
//!     pub fn is_first(self) -> bool {
//!         self.0 % 2 == 0
//!     }
//! }
//! ```
//!
//! ### Sample Registry
//! ```rust,ignore
//! pub struct Samples {
//!     ids: Vec<String>,           // Sample ID strings
//!     index: HashMap<String, SampleIdx>, // Fast lookup
//! }
//! ```
