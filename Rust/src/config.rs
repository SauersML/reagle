//! # Configuration Logic
//!
//! ## Role
//! CLI argument parsing and validation. Replaces `main/Par.java`.
//!
//! ## Spec
//! - Define `struct Config` deriving `clap::Parser`.
//! - Fields:
//!   - `gt: PathBuf` - Input VCF file (target genotypes)
//!   - `ref_panel: Option<PathBuf>` - Reference panel (for imputation)
//!   - `out: PathBuf` - Output prefix
//!   - `ne: f64` - Effective population size (default: 1_000_000)
//!   - `err: f64` - Allelic error rate (default: auto-calculated)
//!   - `window: usize` - Window size in markers (default: 40_000)
//!   - `overlap: usize` - Overlap between windows (default: 2_000)
//!   - `burnin: usize` - Burn-in iterations (default: 6)
//!   - `iterations: usize` - Main phasing iterations (default: 12)
//!   - `nthreads: usize` - Number of threads (default: all cores)
//!   - `seed: Option<u64>` - Random seed for reproducibility
//!
//! ## Validation
//! - Ensure `window > overlap`
//! - Ensure `ne > 0`
//! - Ensure input files exist
//! - Calculate default error rate based on sample size if not provided
//!
//! ## Example CLI
//! ```bash
//! reagle --gt input.vcf.gz --out phased --ne 1000000 --nthreads 8
//! ```
