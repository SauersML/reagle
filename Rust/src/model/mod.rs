//! # Model Module (The Math)
//!
//! ## Role
//! Pure algorithmic implementations. No I/O, no side effects.
//! These functions take data structures and return results.
//!
//! ## Design Philosophy
//! - **Pure functions:** Given the same input, always produce the same output.
//! - **No allocations in hot loops:** Use `Workspace` buffers passed by caller.
//! - **Generic over storage:** Algorithms work with `GenotypeMatrix` regardless
//!   of whether underlying storage is Dense, Sparse, or Dictionary.
//!
//! ## Sub-modules
//! - `parameters`: Model hyperparameters (Ne, error rates)
//! - `pbwt`: Positional Burrows-Wheeler Transform for finding matches
//! - `hmm`: Li-Stephens Hidden Markov Model for phasing/imputation
//!
//! ## Algorithm Overview
//!
//! ### Phasing Pipeline
//! 1. **PBWT:** Find long haplotype matches (surrogate parents)
//! 2. **HMM:** Run forward-backward to compute state probabilities
//! 3. **Sample:** Draw new phase from posterior distribution
//! 4. **Iterate:** Repeat for burn-in + main iterations
//!
//! ### Imputation Pipeline
//! 1. **Build reference:** Load phased reference panel
//! 2. **HMM:** For each target haplotype, run LS model against reference
//! 3. **Posterior:** Compute genotype probabilities at missing sites
//! 4. **Output:** Write dosages (DS) and probabilities (GP)

pub mod hmm;
pub mod parameters;
pub mod pbwt;
