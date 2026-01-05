//! # Model Parameters
//!
//! ## Role
//! Pure data structures for algorithm hyperparameters.
//! Replaces `phase/ParamEstimates.java`.
//!
//! ## Spec
//!
//! ### ModelParams
//! ```rust,ignore
//! pub struct ModelParams {
//!     /// Effective population size (Ne).
//!     /// Controls recombination probability: higher Ne = more recombination.
//!     /// Default: 1_000_000
//!     pub ne: f64,
//!
//!     /// Per-site allelic error probability.
//!     /// Probability of observing wrong allele due to genotyping error.
//!     /// Default: calculated from data, typically 0.0001 - 0.001
//!     pub err: f64,
//!
//!     /// Reference to genetic map for recombination rates.
//!     pub genetic_map: Option<GeneticMap>,
//! }
//! ```
//!
//! ### Li-Stephens Transition Probability
//! The probability of **not** switching to a different reference haplotype:
//!
//! ```text
//! P(no_switch) = exp(-4 * Ne * r / n)
//!
//! where:
//!   Ne = effective population size
//!   r  = recombination rate between markers (genetic distance in Morgans)
//!   n  = number of reference haplotypes
//! ```
//!
//! ### Implementation
//! ```rust,ignore
//! impl ModelParams {
//!     /// Calculate transition probability between two markers.
//!     pub fn transition_prob(
//!         &self,
//!         bp1: u32,
//!         bp2: u32,
//!         n_ref_haps: usize,
//!     ) -> f64 {
//!         let r = self.genetic_map
//!             .as_ref()
//!             .map(|gm| gm.recomb_prob(bp1, bp2))
//!             .unwrap_or_else(|| (bp2 - bp1) as f64 * 1e-8);
//!
//!         (-4.0 * self.ne * r / n_ref_haps as f64).exp()
//!     }
//!
//!     /// Emission probability: P(observed | true_allele).
//!     pub fn emission_prob(&self, observed: u8, true_allele: u8) -> f64 {
//!         if observed == true_allele {
//!             1.0 - self.err
//!         } else {
//!             self.err
//!         }
//!     }
//! }
//! ```
//!
//! ### Auto-calibration
//! Error rate can be estimated from data by examining homozygous genotypes:
//! ```rust,ignore
//! pub fn estimate_error_rate(matrix: &GenotypeMatrix) -> f64;
//! ```
