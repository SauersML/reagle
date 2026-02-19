//! # Configuration Logic
//!
//! CLI argument parsing and validation using clap derive.
//! Replaces `main/Par.java`.

use clap::Parser;
use serde::Deserialize;
use std::path::PathBuf;
use tracing::info_span;

use crate::error::{ReagleError, Result};

const DEFAULT_CONFIG_FILE: &str = "reagle.toml";

/// Minimal CLI arguments (everything else lives in `reagle.toml`).
#[derive(Parser, Debug, Clone)]
#[command(name = "reagle")]
#[command(author = "Reagle Authors")]
#[command(version = "0.1.0")]
#[command(about = "High-performance genotype phasing and imputation", long_about = None)]
pub struct CliArgs {
    /// Input VCF file with GT FORMAT field (required)
    #[arg(long, value_name = "FILE")]
    pub target: PathBuf,

    /// Reference panel (bref3 or VCF file with phased genotypes)
    #[arg(long, value_name = "FILE")]
    pub r#ref: Option<PathBuf>,

    /// Output file prefix (required)
    #[arg(long, short, value_name = "PREFIX")]
    pub out: PathBuf,

    /// PLINK map file with cM units
    #[arg(long, value_name = "FILE")]
    pub map: Option<PathBuf>,

    /// Enable profiling output (hierarchical timing tree)
    #[arg(long, default_value = "false")]
    pub profile: bool,

    /// Force phasing-only mode, even when a reference panel is provided
    #[arg(long = "phase", default_value = "false")]
    pub phase_only: bool,

    /// Random seed for reproducibility
    #[arg(long, value_name = "INT")]
    pub seed: Option<i64>,
}

/// Reagle: High-performance genotype phasing and imputation
#[derive(Debug, Clone)]
pub struct Config {
    // ============ Data Parameters ============
    /// Input VCF file with GT FORMAT field (required)
    pub target: PathBuf,

    /// Reference panel (bref3 or VCF file with phased genotypes)
    pub r#ref: Option<PathBuf>,

    /// Output file prefix (required)
    pub out: PathBuf,

    /// PLINK map file with cM units
    pub map: Option<PathBuf>,

    /// Chromosome or region [chrom] or [chrom]:[start]-[end]
    pub chrom: Option<String>,

    /// File with sample IDs to exclude (one per line)
    pub excludesamples: Option<PathBuf>,

    /// File with marker IDs to exclude (one per line)
    pub excludemarkers: Option<PathBuf>,

    // ============ Phasing Parameters ============
    /// Maximum burn-in iterations
    pub burnin: usize,

    /// Phasing iterations
    pub iterations: usize,

    /// Dynamic MCMC neighbor cap (K).
    pub dynamic_k: usize,

    /// Number of MCMC steps per outer iteration (for dynamic MCMC)
    pub mcmc_steps: usize,

    /// Number of MCMC samples used to estimate phase LR (higher = more stable)
    pub mcmc_lr_samples: usize,

    /// Model states for phasing (0 = auto by memory budget)
    pub phase_states: usize,

    /// Rare variant frequency threshold
    pub rare: f32,

    // ============ Imputation Parameters ============
    /// Impute ungenotyped markers
    pub impute: bool,

    /// Imputation segment length in cM
    pub imp_segment: f32,

    /// Imputation step size in cM
    pub imp_step: f32,

    /// Number of imputation steps
    pub imp_nsteps: usize,

    /// Maximum cM in a marker cluster
    pub cluster: f32,

    /// PBWT batch memory budget (MB) for imputation state selection
    pub pbwt_batch_mb: usize,

    /// Print posterior allele probabilities
    pub ap: bool,

    /// Print posterior genotype probabilities
    pub gp: bool,

    /// Per-window prescan top-k for abyss ranking.
    ///
    /// Methodology note:
    /// chr21 sweeps (10 target / 1000 ref) showed a quality sweet spot around
    /// 25-30 for phase/IQS/Hellinger, while dosage R² peaked closer to ~50.
    /// Defaulting to 50 prioritizes dosage-first workflows.
    pub window_top_k: usize,

    /// Base fraction of state budget from carried priors.
    pub state_mix_prior_frac: f32,

    /// Base fraction of state budget from window-local LMS states.
    pub state_mix_window_frac: f32,

    /// Base fraction of state budget from donor shortlist.
    pub state_mix_donor_frac: f32,

    /// Base fraction of state budget from global core states.
    pub state_mix_core_frac: f32,

    /// Minimum prior fraction when non-empty priors are available.
    pub state_mix_prior_boost_min_frac: f32,

    /// Minimum donor fraction when non-empty priors are available.
    pub state_mix_prior_boost_donor_min_frac: f32,

    /// Maximum core fraction when non-empty priors are available.
    pub state_mix_prior_boost_core_max_frac: f32,

    /// Guaranteed prior floor fraction when non-empty priors are available.
    pub state_mix_prior_floor_frac: f32,

    /// Weak-signal threshold for informative ratio.
    pub state_mix_weak_signal_threshold: f32,

    /// Weak-signal prior fraction when priors are empty.
    pub state_mix_weak_prior_frac: f32,

    /// Weak-signal window fraction when priors are empty.
    pub state_mix_weak_window_frac: f32,

    /// Weak-signal donor fraction when priors are empty.
    pub state_mix_weak_donor_frac: f32,

    /// Weak-signal core fraction when priors are empty.
    pub state_mix_weak_core_frac: f32,

    // ============ General Parameters ============
    /// Effective population size
    pub ne: f32,

    /// Allele mismatch probability (auto-calculated if not specified)
    pub err: Option<f32>,

    /// Estimate ne and err parameters
    pub em: bool,

    /// Window length in cM
    pub window: f32,

    /// Maximum markers per window
    pub window_markers: usize,

    /// Window overlap in cM
    pub overlap: f32,

    /// Random seed for reproducibility
    pub seed: i64,

    /// Number of threads (default: all available cores)
    pub nthreads: Option<usize>,

    /// Enable profiling output (hierarchical timing tree)
    pub profile: bool,

    /// Force phasing-only mode, even when a reference panel is provided
    pub phase_only: bool,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct TomlConfig {
    // Data parameters
    pub map: Option<PathBuf>,
    pub chrom: Option<String>,
    pub excludesamples: Option<PathBuf>,
    pub excludemarkers: Option<PathBuf>,

    // Phasing parameters
    pub burnin: Option<usize>,
    pub iterations: Option<usize>,
    pub dynamic_k: Option<usize>,
    pub mcmc_steps: Option<usize>,
    pub mcmc_lr_samples: Option<usize>,
    pub phase_states: Option<usize>,
    pub rare: Option<f32>,

    // Imputation parameters
    pub impute: Option<bool>,
    pub imp_segment: Option<f32>,
    pub imp_step: Option<f32>,
    pub imp_nsteps: Option<usize>,
    pub cluster: Option<f32>,
    pub pbwt_batch_mb: Option<usize>,
    pub ap: Option<bool>,
    pub gp: Option<bool>,
    pub window_top_k: Option<usize>,
    pub state_mix_prior_frac: Option<f32>,
    pub state_mix_window_frac: Option<f32>,
    pub state_mix_donor_frac: Option<f32>,
    pub state_mix_core_frac: Option<f32>,
    pub state_mix_prior_boost_min_frac: Option<f32>,
    pub state_mix_prior_boost_donor_min_frac: Option<f32>,
    pub state_mix_prior_boost_core_max_frac: Option<f32>,
    pub state_mix_prior_floor_frac: Option<f32>,
    pub state_mix_weak_signal_threshold: Option<f32>,
    pub state_mix_weak_prior_frac: Option<f32>,
    pub state_mix_weak_window_frac: Option<f32>,
    pub state_mix_weak_donor_frac: Option<f32>,
    pub state_mix_weak_core_frac: Option<f32>,

    // General parameters
    pub ne: Option<f32>,
    pub err: Option<f32>,
    pub em: Option<bool>,
    pub window: Option<f32>,
    pub window_markers: Option<usize>,
    pub overlap: Option<f32>,
    pub seed: Option<i64>,
    pub nthreads: Option<usize>,
    pub profile: Option<bool>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            target: std::path::PathBuf::from(""),
            r#ref: None,
            out: std::path::PathBuf::from(""),
            map: None,
            chrom: None,
            excludesamples: None,
            excludemarkers: None,
            burnin: 6,
            iterations: 12,
            dynamic_k: 48,
            mcmc_steps: 8,
            mcmc_lr_samples: 32,
            phase_states: 0,
            rare: 0.002,
            impute: true,
            imp_segment: 6.0,
            imp_step: 0.1,
            imp_nsteps: 7,
            cluster: 0.005,
            pbwt_batch_mb: 256,
            ap: true,
            gp: true,
            window_top_k: 50,
            state_mix_prior_frac: 0.20,
            state_mix_window_frac: 0.35,
            state_mix_donor_frac: 0.25,
            state_mix_core_frac: 0.20,
            state_mix_prior_boost_min_frac: 0.50,
            state_mix_prior_boost_donor_min_frac: 0.10,
            state_mix_prior_boost_core_max_frac: 0.10,
            state_mix_prior_floor_frac: 0.50,
            state_mix_weak_signal_threshold: 0.25,
            state_mix_weak_prior_frac: 0.10,
            state_mix_weak_window_frac: 0.10,
            state_mix_weak_donor_frac: 0.30,
            state_mix_weak_core_frac: 0.50,
            ne: 50000.0,
            err: None,
            em: true,
            window: 40.0,
            window_markers: 100000,
            overlap: 2.0,
            seed: -99999,
            nthreads: None,
            profile: false,
            phase_only: false,
        }
    }
}

impl Config {
    fn apply_toml(&mut self, cfg: TomlConfig) {
        if let Some(value) = cfg.map {
            self.map = Some(value);
        }
        if let Some(value) = cfg.chrom {
            self.chrom = Some(value);
        }
        if let Some(value) = cfg.excludesamples {
            self.excludesamples = Some(value);
        }
        if let Some(value) = cfg.excludemarkers {
            self.excludemarkers = Some(value);
        }

        if let Some(value) = cfg.burnin {
            self.burnin = value;
        }
        if let Some(value) = cfg.iterations {
            self.iterations = value;
        }
        if let Some(value) = cfg.dynamic_k {
            self.dynamic_k = value;
        }
        if let Some(value) = cfg.mcmc_steps {
            self.mcmc_steps = value;
        }
        if let Some(value) = cfg.mcmc_lr_samples {
            self.mcmc_lr_samples = value;
        }
        if let Some(value) = cfg.phase_states {
            self.phase_states = value;
        }
        if let Some(value) = cfg.rare {
            self.rare = value;
        }

        if let Some(value) = cfg.impute {
            self.impute = value;
        }
        if let Some(value) = cfg.imp_segment {
            self.imp_segment = value;
        }
        if let Some(value) = cfg.imp_step {
            self.imp_step = value;
        }
        if let Some(value) = cfg.imp_nsteps {
            self.imp_nsteps = value;
        }
        if let Some(value) = cfg.cluster {
            self.cluster = value;
        }
        if let Some(value) = cfg.pbwt_batch_mb {
            self.pbwt_batch_mb = value;
        }
        if let Some(value) = cfg.ap {
            self.ap = value;
        }
        if let Some(value) = cfg.gp {
            self.gp = value;
        }
        if let Some(value) = cfg.window_top_k {
            self.window_top_k = value;
        }
        if let Some(value) = cfg.state_mix_prior_frac {
            self.state_mix_prior_frac = value;
        }
        if let Some(value) = cfg.state_mix_window_frac {
            self.state_mix_window_frac = value;
        }
        if let Some(value) = cfg.state_mix_donor_frac {
            self.state_mix_donor_frac = value;
        }
        if let Some(value) = cfg.state_mix_core_frac {
            self.state_mix_core_frac = value;
        }
        if let Some(value) = cfg.state_mix_prior_boost_min_frac {
            self.state_mix_prior_boost_min_frac = value;
        }
        if let Some(value) = cfg.state_mix_prior_boost_donor_min_frac {
            self.state_mix_prior_boost_donor_min_frac = value;
        }
        if let Some(value) = cfg.state_mix_prior_boost_core_max_frac {
            self.state_mix_prior_boost_core_max_frac = value;
        }
        if let Some(value) = cfg.state_mix_prior_floor_frac {
            self.state_mix_prior_floor_frac = value;
        }
        if let Some(value) = cfg.state_mix_weak_signal_threshold {
            self.state_mix_weak_signal_threshold = value;
        }
        if let Some(value) = cfg.state_mix_weak_prior_frac {
            self.state_mix_weak_prior_frac = value;
        }
        if let Some(value) = cfg.state_mix_weak_window_frac {
            self.state_mix_weak_window_frac = value;
        }
        if let Some(value) = cfg.state_mix_weak_donor_frac {
            self.state_mix_weak_donor_frac = value;
        }
        if let Some(value) = cfg.state_mix_weak_core_frac {
            self.state_mix_weak_core_frac = value;
        }

        if let Some(value) = cfg.ne {
            self.ne = value;
        }
        if let Some(value) = cfg.err {
            self.err = Some(value);
        }
        if let Some(value) = cfg.em {
            self.em = value;
        }
        if let Some(value) = cfg.window {
            self.window = value;
        }
        if let Some(value) = cfg.window_markers {
            self.window_markers = value;
        }
        if let Some(value) = cfg.overlap {
            self.overlap = value;
        }
        if let Some(value) = cfg.seed {
            self.seed = value;
        }
        if let Some(value) = cfg.nthreads {
            self.nthreads = Some(value);
        }
        if let Some(value) = cfg.profile {
            self.profile = value;
        }
    }

    /// Parse command line arguments and validate
    pub fn parse_and_validate() -> Result<Self> {
        info_span!("config_parse_and_validate").in_scope(|| Self::parse_from(std::env::args_os()))
    }

    /// Parse provided CLI arguments and validate.
    ///
    /// Intended for tests and programmatic callers.
    pub fn parse_from<I, T>(args: I) -> Result<Self>
    where
        I: IntoIterator<Item = T>,
        T: Into<std::ffi::OsString> + Clone,
    {
        let cli = CliArgs::parse_from(args);
        let mut config = Self::default();

        let mut applied_toml = false;
        if let Some(toml_cfg) = load_toml_config(PathBuf::from(DEFAULT_CONFIG_FILE))? {
            config.apply_toml(toml_cfg);
            applied_toml = true;
        }
        if !applied_toml {
            if let Some(parent) = cli.target.parent() {
                let path = parent.join(DEFAULT_CONFIG_FILE);
                if let Some(toml_cfg) = load_toml_config(path)? {
                    config.apply_toml(toml_cfg);
                }
            }
        }

        config.target = cli.target;
        config.r#ref = cli.r#ref;
        config.out = cli.out;
        if cli.map.is_some() {
            config.map = cli.map;
        }
        if cli.profile {
            config.profile = true;
        }
        if cli.phase_only {
            config.phase_only = true;
        }
        if let Some(seed) = cli.seed {
            config.seed = seed;
        }

        config.validate()?;
        Ok(config)
    }

    /// Load sample IDs to exclude from the exclusion file
    ///
    /// Returns an empty set if no exclusion file is specified.
    pub fn load_exclude_samples(&self) -> Result<std::collections::HashSet<String>> {
        info_span!("load_exclude_samples").in_scope(|| {
            use std::fs::File;
            use std::io::{BufRead, BufReader};

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
        })
    }

    /// Load marker IDs to exclude from the exclusion file
    ///
    /// Returns an empty set if no exclusion file is specified.
    pub fn load_exclude_markers(&self) -> Result<std::collections::HashSet<String>> {
        info_span!("load_exclude_markers").in_scope(|| {
            use std::fs::File;
            use std::io::{BufRead, BufReader};

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
        })
    }

    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        // Check input file exists
        if !self.target.exists() {
            return Err(ReagleError::FileNotFound {
                path: self.target.clone(),
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

        if self.mcmc_lr_samples == 0 {
            return Err(ReagleError::config("mcmc-lr-samples must be positive"));
        }
        if self.window_top_k == 0 {
            return Err(ReagleError::config("window_top_k must be positive"));
        }

        let frac_fields = [
            ("state_mix_prior_frac", self.state_mix_prior_frac),
            ("state_mix_window_frac", self.state_mix_window_frac),
            ("state_mix_donor_frac", self.state_mix_donor_frac),
            ("state_mix_core_frac", self.state_mix_core_frac),
            (
                "state_mix_prior_boost_min_frac",
                self.state_mix_prior_boost_min_frac,
            ),
            (
                "state_mix_prior_boost_donor_min_frac",
                self.state_mix_prior_boost_donor_min_frac,
            ),
            (
                "state_mix_prior_boost_core_max_frac",
                self.state_mix_prior_boost_core_max_frac,
            ),
            (
                "state_mix_prior_floor_frac",
                self.state_mix_prior_floor_frac,
            ),
            (
                "state_mix_weak_signal_threshold",
                self.state_mix_weak_signal_threshold,
            ),
            ("state_mix_weak_prior_frac", self.state_mix_weak_prior_frac),
            (
                "state_mix_weak_window_frac",
                self.state_mix_weak_window_frac,
            ),
            ("state_mix_weak_donor_frac", self.state_mix_weak_donor_frac),
            ("state_mix_weak_core_frac", self.state_mix_weak_core_frac),
        ];
        for (name, value) in frac_fields {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(ReagleError::config(format!(
                    "{} must be a finite value in [0, 1], got {}",
                    name, value
                )));
            }
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
        self.r#ref.is_some() && !self.phase_only
    }
}

fn load_toml_config(path: PathBuf) -> Result<Option<TomlConfig>> {
    if !path.exists() {
        return Ok(None);
    }

    let contents = std::fs::read_to_string(&path)?;
    let config: TomlConfig = toml::from_str(&contents).map_err(|err| {
        ReagleError::config(format!(
            "Failed to parse TOML config {}: {}",
            path.display(),
            err
        ))
    })?;

    Ok(Some(config))
}
