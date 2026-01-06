//! # Model Parameters
//!
//! Hyperparameters for the Li-Stephens HMM model.
//! This matches the Java Beagle implementation in `phase/PhaseData.java`
//! and `main/Par.java`.

/// Model parameters for Li-Stephens HMM
///
/// Parameters are tuned based on the Java Beagle defaults and formulas.
#[derive(Clone, Debug)]
pub struct ModelParams {
    /// Per-site allele mismatch probability
    pub p_mismatch: f32,

    /// Recombination intensity (controls state transition rate)
    /// Formula from Java: recombIntensity = 0.04 * ne / nHaps
    pub recomb_intensity: f32,

    /// Number of HMM states (reference haplotypes to consider)
    /// Default for phasing: 280
    /// Default for imputation: 1600
    pub n_states: usize,

    /// Number of total haplotypes in panel
    pub n_haps: usize,

    /// Number of burnin iterations
    pub burnin: usize,

    /// Number of phasing iterations (after burnin)
    pub iterations: usize,

    /// Likelihood ratio threshold for marking heterozygotes as phased
    pub lr_threshold: f32,

    /// Initial likelihood ratio threshold
    pub initial_lr: f32,
}

impl ModelParams {
    /// Default phase states
    pub const DEFAULT_PHASE_STATES: usize = 280;

    /// Default imputation states
    pub const DEFAULT_IMP_STATES: usize = 1600;

    /// Default burnin iterations
    pub const DEFAULT_BURNIN: usize = 3;

    /// Default phasing iterations
    pub const DEFAULT_ITERATIONS: usize = 12;

    /// Default initial LR threshold
    pub const DEFAULT_INITIAL_LR: f32 = 10000.0;

    /// Create default parameters
    pub fn new() -> Self {
        Self {
            p_mismatch: 0.0001,
            recomb_intensity: 1.0,
            n_states: Self::DEFAULT_PHASE_STATES,
            n_haps: 0,
            burnin: Self::DEFAULT_BURNIN,
            iterations: Self::DEFAULT_ITERATIONS,
            lr_threshold: f32::INFINITY,
            initial_lr: Self::DEFAULT_INITIAL_LR,
        }
    }

    /// Create parameters for phasing
    ///
    /// # Arguments
    /// * `n_haps` - Total number of haplotypes (target + reference)
    /// * `ne` - Effective population size (from CLI or default)
    /// * `err` - Optional allele mismatch probability (None = use Li-Stephens formula)
    pub fn for_phasing(n_haps: usize, ne: f32, err: Option<f32>) -> Self {
        // Formula from Java PhaseData constructor
        let recomb_intensity = 0.04 * ne / n_haps as f32;

        let p_mismatch = err.unwrap_or_else(|| Self::li_stephens_p_mismatch(n_haps));

        Self {
            p_mismatch,
            recomb_intensity,
            n_states: Self::DEFAULT_PHASE_STATES.min(n_haps.saturating_sub(2)),
            n_haps,
            burnin: Self::DEFAULT_BURNIN,
            iterations: Self::DEFAULT_ITERATIONS,
            lr_threshold: f32::INFINITY, // Set per iteration
            initial_lr: Self::DEFAULT_INITIAL_LR,
        }
    }

    /// Create parameters for imputation
    ///
    /// # Arguments
    /// * `n_ref_haps` - Number of reference haplotypes
    /// * `ne` - Effective population size (from CLI or default)
    /// * `err` - Optional allele mismatch probability (None = use Li-Stephens formula)
    pub fn for_imputation(n_ref_haps: usize, ne: f32, err: Option<f32>) -> Self {
        let p_mismatch = err.unwrap_or_else(|| Self::li_stephens_p_mismatch(n_ref_haps));
        let recomb_intensity = 0.04 * ne / n_ref_haps as f32;

        Self {
            p_mismatch,
            recomb_intensity,
            n_states: Self::DEFAULT_IMP_STATES.min(n_ref_haps),
            n_haps: n_ref_haps,
            burnin: 0,
            iterations: 1,
            lr_threshold: 1.0,
            initial_lr: Self::DEFAULT_INITIAL_LR,
        }
    }

    /// Li-Stephens approximation for allele mismatch probability
    ///
    /// From Java `Par.liStephensPMismatch`:
    /// ```java
    /// double theta = 1.0 / (Math.log(nHaps) + 0.5);
    /// return (float) (theta / (2*(theta + nHaps)));
    /// ```
    ///
    /// Based on Li N, Stephens M. Genetics 2003 Dec;165(4):2213-33
    pub fn li_stephens_p_mismatch(n_haps: usize) -> f32 {
        if n_haps <= 1 {
            return 0.0001;
        }
        let n = n_haps as f64;
        let theta = 1.0 / (n.ln() + 0.5);
        (theta / (2.0 * (theta + n))) as f32
    }

    /// Calculate LR threshold for a given iteration
    ///
    /// From Java `PhaseData.lrThreshold`:
    /// - During burnin: infinity (don't mark anything as phased)
    /// - Final iteration: 1.0 (mark everything)
    /// - Otherwise: exponential decay from initial_lr to 4.0
    pub fn lr_threshold_for_iteration(&self, it: usize) -> f32 {
        if it < self.burnin {
            f32::INFINITY
        } else if it == self.burnin + self.iterations - 1 {
            1.0
        } else {
            let n_its_m1 = (self.iterations - 1) as f64;
            let last_val = 4.0;
            let exp = (n_its_m1 - (it - self.burnin) as f64) / n_its_m1;
            let base = self.initial_lr as f64 / last_val;
            (last_val * base.powf(exp)) as f32
        }
    }

    /// Calculate recombination probability from genetic distance
    ///
    /// From Java `MarkerMap.pRecomb`:
    /// ```java
    /// double c = -recombIntensity;
    /// pRecomb[m] = -Math.expm1(c * genDist.get(m))
    /// ```
    ///
    /// Note: -expm1(x) = 1 - exp(x), which is more numerically stable
    pub fn p_recomb(&self, gen_dist_cm: f64) -> f32 {
        let c = -(self.recomb_intensity as f64);
        (-f64::exp_m1(c * gen_dist_cm)) as f32
    }

    /// Calculate emission probability for matching allele
    pub fn emit_match(&self) -> f32 {
        1.0 - self.p_mismatch
    }

    /// Calculate emission probability for mismatching allele
    pub fn emit_mismatch(&self) -> f32 {
        self.p_mismatch
    }

    /// Update mismatch probability (for EM estimation)
    ///
    /// From Java `PhaseData.updatePMismatch`:
    /// Only update if new value is valid and greater than current
    pub fn update_p_mismatch(&mut self, new_p: f32) {
        if new_p.is_finite() && new_p > self.p_mismatch && new_p < 0.5 {
            self.p_mismatch = new_p;
        }
    }

    /// Update recombination intensity (for EM estimation)
    ///
    /// From Java `PhaseData.updateRecombIntensity`:
    /// Only update if new value is valid and positive
    pub fn update_recomb_intensity(&mut self, new_intensity: f32) {
        if new_intensity.is_finite() && new_intensity > 0.0 {
            self.recomb_intensity = new_intensity;
        }
    }

    /// Calculate Ne from recombIntensity
    ///
    /// From Java `PhaseData.ne`:
    /// ```java
    /// return (long) Math.ceil(25 * recombIntensity * nHaps);
    /// ```
    pub fn ne_from_recomb_intensity(&self) -> u64 {
        (25.0 * self.recomb_intensity as f64 * self.n_haps as f64).ceil() as u64
    }

    /// Set number of states
    pub fn set_n_states(&mut self, n_states: usize) {
        self.n_states = n_states;
    }
}

impl Default for ModelParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Parameter estimates from EM iteration
///
/// Thread-safe accumulator for EM parameter estimation.
/// Matches Java `phase/ParamEstimates.java`.
#[derive(Clone, Debug, Default)]
pub struct ParamEstimates {
    /// Sum of genetic distances (for recombIntensity estimation)
    /// Note: Java Beagle uses genDist, NOT switch probabilities
    sum_gen_dist: f64,
    /// Sum of expected switches
    sum_expected_switches: f64,
    /// Sum of match probabilities
    sum_match_probs: f64,
    /// Sum of mismatch probabilities
    sum_mismatch_probs: f64,
    /// Total observations for switch estimation
    n_switch_obs: usize,
    /// Total observations for emission estimation
    n_emit_obs: usize,
}

impl ParamEstimates {
    /// Create new empty estimates
    pub fn new() -> Self {
        Self::default()
    }

    /// Add switch observation
    ///
    /// # Arguments
    /// * `gen_dist` - Genetic distance in cM (NOT recombination probability)
    /// * `expected_switches` - Expected number of switches from Baum-Welch
    pub fn add_switch(&mut self, gen_dist: f64, expected_switches: f64) {
        self.sum_gen_dist += gen_dist;
        self.sum_expected_switches += expected_switches;
        self.n_switch_obs += 1;
    }

    /// Add emission observation
    pub fn add_emission(&mut self, match_prob: f64, mismatch_prob: f64) {
        self.sum_match_probs += match_prob;
        self.sum_mismatch_probs += mismatch_prob;
        self.n_emit_obs += 1;
    }

    /// Merge with another estimate (thread-safe reduction)
    pub fn merge(&mut self, other: &ParamEstimates) {
        self.sum_gen_dist += other.sum_gen_dist;
        self.sum_expected_switches += other.sum_expected_switches;
        self.sum_match_probs += other.sum_match_probs;
        self.sum_mismatch_probs += other.sum_mismatch_probs;
        self.n_switch_obs += other.n_switch_obs;
        self.n_emit_obs += other.n_emit_obs;
    }

    /// Sum of genetic distances (for checking convergence)
    pub fn sum_gen_dist(&self) -> f64 {
        self.sum_gen_dist
    }

    /// Estimate recombination intensity
    ///
    /// From Java `ParamEstimates.recombIntensity()`:
    /// Returns ratio of expected switches to total genetic distance
    /// λ = Σ(expected_switches) / Σ(genetic_distances)
    pub fn recomb_intensity(&self) -> f32 {
        if self.sum_gen_dist <= 0.0 {
            return 1.0;
        }
        (self.sum_expected_switches / self.sum_gen_dist) as f32
    }

    /// Estimate mismatch probability
    ///
    /// From Java `ParamEstimates.pMismatch()`:
    /// Returns proportion of mismatches
    pub fn p_mismatch(&self) -> f32 {
        let total = self.sum_match_probs + self.sum_mismatch_probs;
        if total <= 0.0 {
            return 0.0001;
        }
        (self.sum_mismatch_probs / total) as f32
    }

    /// Number of switch observations
    pub fn n_switch_obs(&self) -> usize {
        self.n_switch_obs
    }

    /// Number of emission observations
    pub fn n_emit_obs(&self) -> usize {
        self.n_emit_obs
    }
}

/// Thread-safe wrapper for ParamEstimates using Mutex
///
/// Uses interior mutability for correct float accumulation across threads.
#[derive(Debug)]
pub struct AtomicParamEstimates {
    inner: std::sync::Mutex<ParamEstimates>,
}

impl AtomicParamEstimates {
    /// Create new empty estimates
    pub fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(ParamEstimates::new()),
        }
    }

    /// Add estimation data from a local ParamEstimates
    pub fn add_estimation_data(&self, estimates: &ParamEstimates) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.merge(estimates);
        }
    }

    /// Convert to regular ParamEstimates
    pub fn to_estimates(&self) -> ParamEstimates {
        self.inner.lock().map(|g| g.clone()).unwrap_or_default()
    }
}

impl Default for AtomicParamEstimates {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_li_stephens_p_mismatch() {
        let p = ModelParams::li_stephens_p_mismatch(1000);
        assert!(p > 0.0 && p < 0.01);

        // More haplotypes -> lower mismatch probability
        let p2 = ModelParams::li_stephens_p_mismatch(10000);
        assert!(p2 < p);

        // Edge case
        let p0 = ModelParams::li_stephens_p_mismatch(0);
        assert_eq!(p0, 0.0001);
    }

    #[test]
    fn test_recomb_intensity_formula() {
        let params = ModelParams::for_phasing(1000, 1_000_000.0, None);

        // Should be 0.04 * 1_000_000 / 1000 = 40.0
        let expected = 0.04 * 1_000_000.0 / 1000.0;
        assert!((params.recomb_intensity - expected as f32).abs() < 0.01);
    }

    #[test]
    fn test_p_recomb() {
        let params = ModelParams::for_phasing(1000, 1_000_000.0, None);

        // No distance -> no recomb
        let p0 = params.p_recomb(0.0);
        assert!(p0.abs() < 0.0001);

        // Small distance -> small prob
        let p1 = params.p_recomb(0.001);
        assert!(p1 > 0.0 && p1 < 0.5);

        // Larger distance -> higher prob
        let p2 = params.p_recomb(0.01);
        assert!(p2 > p1);
    }

    #[test]
    fn test_lr_threshold_for_iteration() {
        let params = ModelParams::for_phasing(1000, 1_000_000.0, None);

        // Burnin: infinity
        for it in 0..params.burnin {
            assert!(params.lr_threshold_for_iteration(it).is_infinite());
        }

        // Final iteration: 1.0
        let final_it = params.burnin + params.iterations - 1;
        assert!((params.lr_threshold_for_iteration(final_it) - 1.0).abs() < 0.0001);

        // Middle iterations: between initial_lr and 4.0
        let mid_it = params.burnin + params.iterations / 2;
        let mid_lr = params.lr_threshold_for_iteration(mid_it);
        assert!(mid_lr > 4.0);
        assert!(mid_lr < params.initial_lr);
    }

    #[test]
    fn test_emit_probs() {
        let params = ModelParams::new();
        let match_p = params.emit_match();
        let mismatch_p = params.emit_mismatch();

        assert!(match_p > 0.99);
        assert!(mismatch_p < 0.01);
        assert!((match_p + mismatch_p - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_param_estimates_merge() {
        let mut e1 = ParamEstimates::new();
        e1.add_switch(0.5, 2.0); // 0.5 cM genetic distance
        e1.add_emission(0.9, 0.1);

        let mut e2 = ParamEstimates::new();
        e2.add_switch(0.3, 1.5); // 0.3 cM genetic distance
        e2.add_emission(0.8, 0.2);

        e1.merge(&e2);

        assert!((e1.sum_gen_dist() - 0.8).abs() < 0.0001);
        assert_eq!(e1.n_switch_obs(), 2);
    }

    #[test]
    fn test_ne_from_recomb_intensity() {
        let mut params = ModelParams::new();
        params.n_haps = 1000;
        params.recomb_intensity = 40.0;

        let ne = params.ne_from_recomb_intensity();
        // Should be ceil(25 * 40 * 1000) = 1_000_000
        assert_eq!(ne, 1_000_000);
    }
}
