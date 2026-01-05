//! # Genetic Map Interpolation
//!
//! ## Role
//! Convert physical positions (base pairs) to genetic distances (centiMorgans).
//! Replaces `vcf/GeneticMap.java`.
//!
//! ## Spec
//!
//! ### GeneticMap Struct
//! ```rust,ignore
//! pub struct GeneticMap {
//!     /// Sorted (base_pair, centimorgan) pairs from a PLINK .map file
//!     points: Vec<(u32, f64)>,
//! }
//! ```
//!
//! ### Methods
//! ```rust,ignore
//! impl GeneticMap {
//!     /// Load from PLINK-format genetic map file.
//!     pub fn from_file(path: &Path) -> Result<Self>;
//!
//!     /// Interpolate genetic position at a physical position.
//!     /// Uses linear interpolation between surrounding map points.
//!     pub fn get_cm(&self, bp: u32) -> f64;
//!
//!     /// Calculate recombination probability between two positions.
//!     /// Uses Haldane's mapping function: r = 0.5 * (1 - e^(-2d))
//!     /// where d is genetic distance in Morgans.
//!     pub fn recomb_prob(&self, bp1: u32, bp2: u32) -> f64;
//! }
//! ```
//!
//! ### Default Behavior
//! If no genetic map is provided, use a constant rate:
//! - Default: 1 cM per 1 Mb (1e-8 per bp)
//!
//! ### File Format (PLINK .map)
//! ```text
//! chr1  rs123  0.05  10000
//! chr1  rs456  0.10  20000
//! ```
//! Columns: chrom, id, cM, bp
