//! # Imputation Pipeline
//!
//! ## Role
//! Orchestrate genotype imputation: infer missing genotypes using a reference panel.
//! Replaces logic in `imp/ImpLS.java` and related classes.
//!
//! ## Algorithm Overview
//! 1. **Load reference panel:** Phased haplotypes (from VCF or bref3)
//! 2. **Load target genotypes:** May be sparse (genotyping array data)
//! 3. **For each target sample (parallel):**
//!    a. Phase the target if needed
//!    b. Run Li-Stephens HMM for each haplotype against reference
//!    c. Compute posterior genotype probabilities at all reference sites
//! 4. **Output:** VCF with DS (dosage) and GP (genotype probability) fields
//!
//! ## Spec
//!
//! ### Entry Point
//! ```rust,ignore
//! pub fn run(config: &Config) -> Result<()>;
//! ```
//!
//! ### Reference Panel Loading
//! ```rust,ignore
//! fn load_reference(path: &Path) -> Result<GenotypeMatrix<Phased>>;
//! ```
//!
//! ### Per-Sample Imputation (parallelized)
//! ```rust,ignore
//! fn impute_sample(
//!     sample: SampleIdx,
//!     target: &GenotypeMatrix,          // Target genotypes (sparse)
//!     reference: &GenotypeMatrix<Phased>, // Reference panel
//!     params: &ModelParams,
//!     workspace: &mut HmmWorkspace,
//! ) -> ImputationResult;
//!
//! pub struct ImputationResult {
//!     /// Dosage at each imputed site (expected ALT allele count).
//!     pub dosages: Vec<f32>,
//!
//!     /// Genotype probabilities [P(0/0), P(0/1), P(1/1)] at each site.
//!     pub probs: Vec<[f32; 3]>,
//!
//!     /// Imputation quality score (INFO score) per site.
//!     pub info_scores: Vec<f32>,
//! }
//! ```
//!
//! ### Dosage Calculation
//! ```rust,ignore
//! /// Compute expected dosage from diploid HMM posteriors.
//! fn compute_dosage(
//!     hap1_probs: &HmmResult,
//!     hap2_probs: &HmmResult,
//!     reference: &GenotypeMatrix,
//!     marker: MarkerIdx,
//! ) -> f32 {
//!     // Sum over all state combinations
//!     // DS = E[allele_count] = sum_i sum_j P(state_i) * P(state_j) * (ref[i] + ref[j])
//! }
//! ```
//!
//! ### INFO Score (Imputation Quality)
//! ```rust,ignore
//! /// Compute INFO score (ratio of observed to expected variance).
//! /// INFO = 1 - (observed_variance / expected_variance_under_HWE)
//! fn compute_info_score(dosages: &[f32], probs: &[[f32; 3]]) -> f32;
//! ```
//!
//! ### Output Format
//! VCF with FORMAT fields:
//! - `GT`: Best-guess genotype
//! - `DS`: Dosage (0.0 - 2.0)
//! - `GP`: Genotype probabilities (3 comma-separated floats)
