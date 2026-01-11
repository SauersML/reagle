//! # Configuration Logic
//!
//! CLI argument parsing and validation using clap derive.
//! Replaces `main/Par.java`.

use clap::Parser;
use std::path::PathBuf;

use crate::error::{ReagleError, Result};

/// Reagle: High-performance genotype phasing and imputation
#[derive(Parser, Debug, Clone)]
#[command(name = "reagle")]
#[command(author = "Reagle Authors")]
#[command(version = "0.1.0")]
#[command(about = "High-performance genotype phasing and imputation", long_about = None)]
pub struct Config {
    // ============ Data Parameters ============
    /// Input VCF file with GT FORMAT field (required)
    #[arg(long, value_name = "FILE")]
    pub gt: PathBuf,

    /// Reference panel (bref3 or VCF file with phased genotypes)
    #[arg(long, value_name = "FILE")]
    pub r#ref: Option<PathBuf>,

    /// Output file prefix (required)
    #[arg(long, short, value_name = "PREFIX")]
    pub out: PathBuf,

    /// PLINK map file with cM units
    #[arg(long, value_name = "FILE")]
    pub map: Option<PathBuf>,

    /// Chromosome or region [chrom] or [chrom]:[start]-[end]
    #[arg(long, value_name = "REGION")]
    pub chrom: Option<String>,

    /// File with sample IDs to exclude (one per line)
    #[arg(long, value_name = "FILE")]
    pub excludesamples: Option<PathBuf>,

    /// File with marker IDs to exclude (one per line)
    #[arg(long, value_name = "FILE")]
    pub excludemarkers: Option<PathBuf>,

    // ============ Phasing Parameters ============
    /// Maximum burn-in iterations
    #[arg(long, default_value = "3")]
    pub burnin: usize,

    /// Phasing iterations
    #[arg(long, default_value = "12")]
    pub iterations: usize,

    /// MCMC burn-in sweeps per sample (phase sampling)
    #[arg(long = "mcmc-burnin", default_value = "2")]
    pub mcmc_burnin: usize,

    /// MCMC collected sweeps per sample (phase sampling)
    #[arg(long = "mcmc-samples", default_value = "4")]
    pub mcmc_samples: usize,

    /// Number of independent MCMC chains per sample
    #[arg(long = "mcmc-chains", default_value = "2")]
    pub mcmc_chains: usize,

    /// Model states for phasing
    #[arg(long = "phase-states", default_value = "280")]
    pub phase_states: usize,

    /// Rare variant frequency threshold
    #[arg(long, default_value = "0.002")]
    pub rare: f32,

    // ============ Imputation Parameters ============
    /// Impute ungenotyped markers
    #[arg(long, default_value = "true")]
    pub impute: bool,

    /// Model states for imputation
    #[arg(long = "imp-states", default_value = "1600")]
    pub imp_states: usize,

    /// Imputation segment length in cM
    #[arg(long = "imp-segment", default_value = "6.0")]
    pub imp_segment: f32,

    /// Imputation step size in cM
    #[arg(long = "imp-step", default_value = "0.1")]
    pub imp_step: f32,

    /// Number of imputation steps
    #[arg(long = "imp-nsteps", default_value = "7")]
    pub imp_nsteps: usize,

    /// Maximum cM in a marker cluster
    #[arg(long, default_value = "0.005")]
    pub cluster: f32,

    /// Print posterior allele probabilities
    #[arg(long, default_value = "false")]
    pub ap: bool,

    /// Print posterior genotype probabilities
    #[arg(long, default_value = "false")]
    pub gp: bool,

    // ============ General Parameters ============
    /// Effective population size
    #[arg(long, default_value = "100000")]
    pub ne: f32,

    /// Allele mismatch probability (auto-calculated if not specified)
    #[arg(long)]
    pub err: Option<f32>,

    /// Estimate ne and err parameters
    #[arg(long, default_value = "true")]
    pub em: bool,

    /// Window length in cM
    #[arg(long, default_value = "40.0")]
    pub window: f32,

    /// Maximum markers per window
    #[arg(long = "window-markers", default_value = "4000000")]
    pub window_markers: usize,

    /// Window overlap in cM
    #[arg(long, default_value = "2.0")]
    pub overlap: f32,

    /// Enable streaming mode for large datasets
    #[arg(long)]
    pub streaming: Option<bool>,

    /// Random seed for reproducibility
    #[arg(long, default_value = "-99999")]
    pub seed: i64,

    /// Number of threads (default: all available cores)
    #[arg(long)]
    pub nthreads: Option<usize>,

    /// Enable profiling output (hierarchical timing tree)
    #[arg(long, default_value = "false")]
    pub profile: bool,
}

impl Config {
    /// Parse command line arguments and validate
    pub fn parse_and_validate() -> Result<Self> {
        let config = Self::parse();
        config.validate()?;
        Ok(config)
    }

    /// Load sample IDs to exclude from the exclusion file
    ///
    /// Returns an empty set if no exclusion file is specified.
    pub fn load_exclude_samples(&self) -> Result<std::collections::HashSet<String>> {
        use std::io::{BufRead, BufReader};
        use std::fs::File;

        let mut exclude_set = std::collections::HashSet::new();

        if let Some(ref path) = self.excludesamples {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                let id = line.trim();
                if !id.is_empty() && !id.starts_with('#') {
                    exclude_set.insert(id.to_string());
                }
            }
        }

        Ok(exclude_set)
    }

    /// Load marker IDs to exclude from the exclusion file
    ///
    /// Returns an empty set if no exclusion file is specified.
    pub fn load_exclude_markers(&self) -> Result<std::collections::HashSet<String>> {
        use std::io::{BufRead, BufReader};
        use std::fs::File;

        let mut exclude_set = std::collections::HashSet::new();

        if let Some(ref path) = self.excludemarkers {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                let line = line?;
                let id = line.trim();
                if !id.is_empty() && !id.starts_with('#') {
                    exclude_set.insert(id.to_string());
                }
            }
        }

        Ok(exclude_set)
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        // Check input file exists
        if !self.gt.exists() {
            return Err(ReagleError::FileNotFound {
                path: self.gt.clone(),
            });
        }

        // Check reference file exists if specified
        if let Some(ref ref_path) = self.r#ref {
            if !ref_path.exists() {
                return Err(ReagleError::FileNotFound {
                    path: ref_path.clone(),
                });
            }
        }

        // Check map file exists if specified
        if let Some(ref map_path) = self.map {
            if !map_path.exists() {
                return Err(ReagleError::FileNotFound {
                    path: map_path.clone(),
                });
            }
        }

        // Validate window > overlap
        if self.window < 1.1 * self.overlap {
            return Err(ReagleError::config(
                "The 'window' parameter must be at least 1.1 times the 'overlap' parameter",
            ));
        }

        // Validate ne > 0
        if self.ne <= 0.0 {
            return Err(ReagleError::config(
                "Effective population size (ne) must be positive",
            ));
        }

        // Check output prefix is not a directory
        if self.out.is_dir() {
            return Err(ReagleError::config(format!(
                "'out' parameter cannot be a directory: {:?}",
                self.out
            )));
        }

        Ok(())
    }

    /// Get the number of threads to use
    pub fn nthreads(&self) -> usize {
        self.nthreads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
    }

    /// Check if imputation mode (reference panel provided)
    pub fn is_imputation_mode(&self) -> bool {
        self.r#ref.is_some()
    }
}
