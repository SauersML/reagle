//! # Li-Stephens Hidden Markov Model
//!
//! ## Role
//! The core statistical model for phasing and imputation.
//! Replaces `phase/PhaseLS.java`, `imp/ImpLS.java`.
//!
//! ## Background
//! The Li-Stephens model (Li & Stephens 2003) treats a target haplotype as
//! an imperfect mosaic of reference haplotypes. Hidden states are which
//! reference haplotype we're currently copying from.
//!
//! ## Spec
//!
//! ### Forward-Backward Algorithm
//! ```rust,ignore
//! /// Run forward-backward algorithm, returning posterior state probabilities.
//! pub fn forward_backward(
//!     target: &[u8],              // Target haplotype alleles
//!     reference: &GenotypeMatrix, // Reference panel
//!     params: &ModelParams,       // Ne, error rate
//!     workspace: &mut HmmWorkspace, // Pre-allocated buffers
//! ) -> HmmResult;
//!
//! pub struct HmmResult {
//!     /// Posterior probability of each state at each marker.
//!     /// Shape: [n_markers, n_ref_haps]
//!     pub state_probs: Vec<Vec<f64>>,
//!
//!     /// Log-likelihood of the observation sequence.
//!     pub log_likelihood: f64,
//! }
//! ```
//!
//! ### Forward Pass
//! ```rust,ignore
//! /// Compute forward probabilities (scaled to prevent underflow).
//! fn forward_pass(
//!     target: &[u8],
//!     reference: &GenotypeMatrix,
//!     params: &ModelParams,
//!     workspace: &mut HmmWorkspace,
//! ) -> f64;  // Returns log-likelihood
//! ```
//!
//! ### Backward Pass
//! ```rust,ignore
//! /// Compute backward probabilities.
//! fn backward_pass(
//!     target: &[u8],
//!     reference: &GenotypeMatrix,
//!     params: &ModelParams,
//!     workspace: &mut HmmWorkspace,
//! );
//! ```
//!
//! ### Genotype Probability (for imputation)
//! ```rust,ignore
//! /// Compute P(genotype | observations) at a specific marker.
//! pub fn genotype_probability(
//!     result: &HmmResult,
//!     reference: &GenotypeMatrix,
//!     marker: MarkerIdx,
//! ) -> [f64; 3];  // P(0/0), P(0/1), P(1/1) for diploid
//! ```
//!
//! ### Viterbi (Most Likely Path)
//! ```rust,ignore
//! /// Find most likely sequence of hidden states.
//! pub fn viterbi(
//!     target: &[u8],
//!     reference: &GenotypeMatrix,
//!     params: &ModelParams,
//!     workspace: &mut HmmWorkspace,
//! ) -> Vec<HapIdx>;
//! ```
//!
//! ### Sampling (for phasing iterations)
//! ```rust,ignore
//! /// Sample a path from the posterior distribution.
//! pub fn sample_path(
//!     result: &HmmResult,
//!     rng: &mut impl Rng,
//! ) -> Vec<HapIdx>;
//! ```
//!
//! ### Numerical Stability
//! - Use log-space or scaling factors to prevent underflow.
//! - Scaling: Normalize forward probabilities at each position.
//! - Track sum of log(scale_factors) for total log-likelihood.
//!
//! ### Performance Notes
//! - This is O(n_markers * n_ref_haps) per target haplotype.
//! - For 1M markers and 10K reference haps: 10 billion operations.
//! - Use sparse reference panels (PBWT-selected subset).
//! - Consider vectorization for emission probability computation.
