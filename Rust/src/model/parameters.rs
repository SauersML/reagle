//! # Model Parameters
//!
//! Hyperparameters for the Li-Stephens HMM model.
//! Replaces `phase/PhaseData.java` parameter handling.

/// Model parameters for Li-Stephens HMM
#[derive(Clone, Debug)]
pub struct ModelParams {
    /// Effective population size (Ne)
    pub ne: f32,

    /// Per-site allele mismatch probability
    pub p_mismatch: f32,

    /// Recombination intensity multiplier
    pub recomb_intensity: f32,

    /// Number of HMM states (reference haplotypes to consider)
    pub n_states: usize,

    /// Minimum match length for PBWT state selection (in markers)
    pub min_match_length: usize,
}

impl ModelParams {
    /// Create default parameters
    pub fn new() -> Self {
        Self {
            ne: 100_000.0,
            p_mismatch: 0.0001,
            recomb_intensity: 1.0,
            n_states: 280,
            min_match_length: 10,
        }
    }

    /// Create parameters for phasing
    pub fn for_phasing(n_haps: usize) -> Self {
        Self {
            ne: 100_000.0,
            p_mismatch: Self::li_stephens_p_mismatch(n_haps),
            recomb_intensity: 1.0,
            n_states: 280.min(n_haps.saturating_sub(2)),
            min_match_length: 10,
        }
    }

    /// Create parameters for imputation
    pub fn for_imputation(n_ref_haps: usize) -> Self {
        Self {
            ne: 100_000.0,
            p_mismatch: Self::li_stephens_p_mismatch(n_ref_haps),
            recomb_intensity: 1.0,
            n_states: 1600.min(n_ref_haps),
            min_match_length: 10,
        }
    }

    /// Li-Stephens approximation for allele mismatch probability
    /// Based on Li N, Stephens M. Genetics 2003 Dec;165(4):2213-33
    pub fn li_stephens_p_mismatch(n_haps: usize) -> f32 {
        if n_haps <= 1 {
            return 0.0001;
        }
        let n = n_haps as f64;
        let theta = 1.0 / (n.ln() + 0.5);
        (theta / (2.0 * (theta + n))) as f32
    }

    /// Calculate switch probability between markers
    ///
    /// # Arguments
    /// * `gen_dist` - Genetic distance in cM
    ///
    /// # Returns
    /// Probability of switching to a different haplotype
    pub fn switch_prob(&self, gen_dist: f64) -> f32 {
        // Li-Stephens recombination probability
        // P(switch) = 1 - exp(-4 * Ne * r / n_states)
        // where r is recombination rate (cM/100)
        let r = gen_dist / 100.0; // Convert cM to Morgan
        let rate = 4.0 * self.ne as f64 * r * self.recomb_intensity as f64 / self.n_states as f64;
        (1.0 - (-rate).exp()) as f32
    }

    /// Calculate no-switch probability
    pub fn no_switch_prob(&self, gen_dist: f64) -> f32 {
        1.0 - self.switch_prob(gen_dist)
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
    pub fn update_p_mismatch(&mut self, new_p: f32) {
        if new_p.is_finite() && new_p > 0.0 && new_p < 0.5 {
            self.p_mismatch = new_p;
        }
    }

    /// Update recombination intensity (for EM estimation)
    pub fn update_recomb_intensity(&mut self, new_intensity: f32) {
        if new_intensity.is_finite() && new_intensity > 0.0 {
            self.recomb_intensity = new_intensity;
        }
    }

    /// Set effective population size
    pub fn set_ne(&mut self, ne: f32) {
        self.ne = ne;
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
#[derive(Clone, Debug, Default)]
pub struct ParamEstimates {
    /// Sum of switch probabilities
    sum_switch_probs: f64,
    /// Sum of expected switches
    sum_expected_switches: f64,
    /// Sum of match probabilities
    sum_match_probs: f64,
    /// Sum of mismatch probabilities
    sum_mismatch_probs: f64,
    /// Number of observations
    n_obs: usize,
}

impl ParamEstimates {
    /// Create new empty estimates
    pub fn new() -> Self {
        Self::default()
    }

    /// Add switch observation
    pub fn add_switch(&mut self, switch_prob: f64, expected_switches: f64) {
        self.sum_switch_probs += switch_prob;
        self.sum_expected_switches += expected_switches;
        self.n_obs += 1;
    }

    /// Add emission observation
    pub fn add_emission(&mut self, match_prob: f64, mismatch_prob: f64) {
        self.sum_match_probs += match_prob;
        self.sum_mismatch_probs += mismatch_prob;
    }

    /// Merge with another estimate
    pub fn merge(&mut self, other: &ParamEstimates) {
        self.sum_switch_probs += other.sum_switch_probs;
        self.sum_expected_switches += other.sum_expected_switches;
        self.sum_match_probs += other.sum_match_probs;
        self.sum_mismatch_probs += other.sum_mismatch_probs;
        self.n_obs += other.n_obs;
    }

    /// Estimate recombination intensity
    pub fn recomb_intensity(&self) -> f32 {
        if self.sum_switch_probs <= 0.0 {
            return 1.0;
        }
        (self.sum_expected_switches / self.sum_switch_probs) as f32
    }

    /// Estimate mismatch probability
    pub fn p_mismatch(&self) -> f32 {
        let total = self.sum_match_probs + self.sum_mismatch_probs;
        if total <= 0.0 {
            return 0.0001;
        }
        (self.sum_mismatch_probs / total) as f32
    }

    /// Number of observations
    pub fn n_obs(&self) -> usize {
        self.n_obs
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
    }

    #[test]
    fn test_switch_prob() {
        let params = ModelParams::new();

        // No distance -> no switch
        let p0 = params.switch_prob(0.0);
        assert!(p0 < 0.001);

        // Small distance -> small switch prob
        let p1 = params.switch_prob(0.01);
        assert!(p1 > 0.0 && p1 < 0.5);

        // Large distance -> high switch prob
        let p2 = params.switch_prob(1.0);
        assert!(p2 > p1);
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
}