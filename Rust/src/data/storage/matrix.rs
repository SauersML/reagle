//! # Genotype Matrix
//!
//! ## Role
//! The main data structure: a matrix of genotypes (markers x haplotypes).
//! Replaces `vcf/RefGT.java`, `vcf/BasicGT.java`, and related classes.
//!
//! ## Spec
//!
//! ### Structure of Arrays Layout
//! ```rust,ignore
//! pub struct GenotypeMatrix {
//!     /// Marker metadata (one per row).
//!     markers: Vec<Marker>,
//!
//!     /// Genotype data (one column per marker, stored by row for iteration).
//!     columns: Vec<GenotypeColumn>,
//!
//!     /// Sample metadata.
//!     samples: Samples,
//!
//!     /// Number of haplotypes (2 * n_samples for diploid).
//!     n_haplotypes: usize,
//! }
//! ```
//!
//! ### Core Methods
//! ```rust,ignore
//! impl GenotypeMatrix {
//!     /// Get allele at (marker, haplotype).
//!     pub fn get_allele(&self, m: MarkerIdx, h: HapIdx) -> u8;
//!
//!     /// Get entire column for a marker.
//!     pub fn column(&self, m: MarkerIdx) -> &GenotypeColumn;
//!
//!     /// Number of markers.
//!     pub fn n_markers(&self) -> usize;
//!
//!     /// Number of haplotypes.
//!     pub fn n_haplotypes(&self) -> usize;
//!
//!     /// Slice a range of markers (for windowing).
//!     /// Returns a view, not a copy.
//!     pub fn slice(&self, range: Range<MarkerIdx>) -> GenotypeMatrixView;
//!
//!     /// Iterate over markers.
//!     pub fn iter_markers(&self) -> impl Iterator<Item = (MarkerIdx, &Marker, &GenotypeColumn)>;
//! }
//! ```
//!
//! ### Phased vs Unphased (Type State Pattern)
//! ```rust,ignore
//! pub struct Phased;
//! pub struct Unphased;
//!
//! pub struct GenotypeMatrix<Phase = Phased> {
//!     // ... fields ...
//!     _phase: PhantomData<Phase>,
//! }
//! ```
//! This ensures you can't accidentally pass unphased data to algorithms expecting phased.
