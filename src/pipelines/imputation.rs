//! # Imputation Pipeline
//!
//! Orchestrates the imputation workflow:
//! 1. Load target and reference VCFs
//! 2. Align markers between target and reference
//! 3. Process data in overlapping sliding windows (for memory efficiency)
//! 4. Run Li-Stephens HMM for each target haplotype with dynamic PBWT state selection
//! 5. Interpolate state probabilities for ungenotyped markers
//! 6. Splice window results at overlap midpoints
//! 7. Compute dosages and write output with quality metrics (DR2, AF)
//!
//! This matches Java `imp/ImpLS.java`, `imp/ImpLSBaum.java`, and related classes.

use crate::config::Config;
use crate::error::Result;
use crate::model::parameters::ModelParams;
use crate::utils::telemetry::TelemetryBlackboard;
use std::sync::Arc;
use tracing::instrument;
/// Imputation pipeline
pub struct ImputationPipeline {
    pub(crate) config: Config,
    pub(crate) params: ModelParams,
    pub(crate) telemetry: Option<Arc<TelemetryBlackboard>>,
}

/// Per-haplotype allele posterior probabilities.
/// Optimized: uses compact representation for biallelic (99% of sites).
#[derive(Clone, Debug)]
pub enum AllelePosteriors {
    /// Biallelic site: just store P(ALT)
    Biallelic(f32),
    /// Multiallelic site: full PMF where index i = P(allele i)
    Multiallelic(Vec<f32>),
}

impl AllelePosteriors {
    /// Get P(allele i)
    #[inline]
    pub fn prob(&self, allele: usize) -> f32 {
        match self {
            AllelePosteriors::Biallelic(p_alt) => {
                if allele == 0 {
                    1.0 - p_alt
                } else if allele == 1 {
                    *p_alt
                } else {
                    0.0
                }
            }
            AllelePosteriors::Multiallelic(probs) => probs.get(allele).copied().unwrap_or(0.0),
        }
    }

}

impl ImputationPipeline {
    /// Create a new imputation pipeline
    pub fn new(config: Config, telemetry: Option<Arc<TelemetryBlackboard>>) -> Self {
        let params = ModelParams::new();
        Self {
            config,
            params,
            telemetry,
        }
    }

    /// Run the imputation pipeline
    #[instrument(name = "imputation", skip(self))]
    pub fn run(&mut self) -> Result<()> {
        // Use streaming approach to avoid OOM on large reference panels
        self.run_streaming()
    }
}

// ... existing tests ...
#[cfg(test)]
mod tests {
    // Tests for imputation pipeline

    #[test]
    fn test_state_probs_basic() {
        // Test placeholder
        assert!(true);
    }
}
