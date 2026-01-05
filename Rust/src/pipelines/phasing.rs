//! # Phasing Pipeline
//!
//! ## Role
//! Orchestrate statistical phasing: infer haplotypes from unphased genotypes.
//! Replaces logic scattered across `main/Main.java` and `phase/` classes.
//!
//! ## Algorithm Overview
//! Uses iterative conditional sampling (Browning & Browning 2007):
//!
//! 1. **Initialize:** Random phase assignment
//! 2. **For each iteration:**
//!    a. Build PBWT index of current haplotypes
//!    b. For each sample (parallel):
//!       - Find surrogate parents (long PBWT matches)
//!       - Run Li-Stephens HMM against surrogates
//!       - Sample new phase from posterior
//!    c. Update haplotypes
//! 3. **Output:** Final phased haplotypes
//!
//! ## Spec
//!
//! ### Entry Point
//! ```rust,ignore
//! pub fn run(config: &Config) -> Result<()>;
//! ```
//!
//! ### Per-Window Processing
//! ```rust,ignore
//! fn phase_window(
//!     window: &mut GenotypeMatrix<Unphased>,
//!     params: &ModelParams,
//!     config: &Config,
//! ) -> Result<GenotypeMatrix<Phased>>;
//! ```
//!
//! ### Single Iteration
//! ```rust,ignore
//! fn phasing_iteration(
//!     current_phase: &mut GenotypeMatrix<Phased>,
//!     original_genotypes: &GenotypeMatrix<Unphased>,
//!     params: &ModelParams,
//!     iteration: usize,
//!     rng: &mut impl Rng,
//! ) -> Result<()>;
//! ```
//!
//! ### Per-Sample Phasing (parallelized via rayon)
//! ```rust,ignore
//! fn phase_sample(
//!     sample: SampleIdx,
//!     current_haplotypes: &GenotypeMatrix<Phased>,
//!     genotypes: &GenotypeMatrix<Unphased>,
//!     pbwt: &BidirectionalPbwt,
//!     params: &ModelParams,
//!     workspace: &mut HmmWorkspace,
//!     rng: &mut impl Rng,
//! ) -> (Vec<u8>, Vec<u8>);  // (hap1, hap2)
//! ```
//!
//! ### Burn-in vs Main Iterations
//! - **Burn-in:** More aggressive exploration, higher "temperature"
//! - **Main:** Converge to high-probability phase
//!
//! ### Progress Reporting
//! ```rust,ignore
//! // Log progress at each iteration
//! log::info!(
//!     "Iteration {}/{}: window {} of {}, log-likelihood: {:.2}",
//!     iter, total_iters, window_idx, n_windows, ll
//! );
//! ```
