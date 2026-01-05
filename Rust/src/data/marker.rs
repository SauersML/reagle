//! # Marker Definitions
//!
//! ## Role
//! Genomic marker (variant site) representation. Replaces `vcf/Marker.java`.
//!
//! ## Spec
//!
//! ### Newtype Index
//! ```rust,ignore
//! #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
//! pub struct MarkerIdx(pub u32);
//! ```
//! Prevents accidental confusion with `HapIdx` or `SampleIdx`.
//!
//! ### Marker Struct
//! ```rust,ignore
//! pub struct Marker {
//!     pub chrom: u8,           // Chromosome (1-22, 23=X, 24=Y, 25=MT)
//!     pub pos: u32,            // 1-based genomic position
//!     pub id: Option<String>,  // rsID if available
//!     pub ref_allele: Allele,  // Reference allele
//!     pub alt_alleles: Vec<Allele>, // Alternate allele(s)
//! }
//! ```
//!
//! ### Allele Representation
//! ```rust,ignore
//! pub enum Allele {
//!     Base(u8),      // A=0, C=1, G=2, T=3 for SNVs
//!     Seq(String),   // Indels/complex variants
//! }
//! ```
//!
//! ### Methods
//! - `is_snv() -> bool`: True if ref and all alts are single nucleotides.
//! - `is_biallelic() -> bool`: True if exactly one alt allele.
//! - `n_alleles() -> usize`: Total number of alleles (ref + alts).
