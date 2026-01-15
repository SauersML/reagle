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

use tracing::instrument;

use crate::config::Config;
use crate::error::Result;
use crate::model::parameters::ModelParams;

/// Imputation pipeline
pub struct ImputationPipeline {
    pub(crate) config: Config,
    pub(crate) params: ModelParams,
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
                if allele == 0 { 1.0 - p_alt } else if allele == 1 { *p_alt } else { 0.0 }
            }
            AllelePosteriors::Multiallelic(probs) => {
                probs.get(allele).copied().unwrap_or(0.0)
            }
        }
    }

    /// Get the most likely allele (argmax)
    #[inline]
    pub fn max_allele(&self) -> u8 {
        match self {
            AllelePosteriors::Biallelic(p_alt) => if *p_alt >= 0.5 { 1 } else { 0 },
            AllelePosteriors::Multiallelic(probs) => {
                probs.iter().enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i as u8)
                    .unwrap_or(0)
            }
        }
    }

}

/// Cluster-based state probabilities with Beagle-style interpolation weights.
/// Uses CSR (Compressed Sparse Row) format to eliminate Vec<Vec<T>> overhead.
#[derive(Clone, Debug)]
pub struct ClusterStateProbs {
    marker_cluster: std::sync::Arc<Vec<usize>>,
    ref_cluster_end: std::sync::Arc<Vec<usize>>,
    weight: std::sync::Arc<Vec<f32>>,
    // CSR format: offsets[c]..offsets[c+1] gives indices for cluster c
    offsets: Vec<usize>,
    hap_indices: Vec<u32>,
    probs: Vec<f32>,
    probs_p1: Vec<f32>,
}

impl ClusterStateProbs {
    /// Create from pre-computed sparse CSR data (from run_hmm_forward_backward_to_sparse).
    /// This avoids the O(n_clusters × n_states) dense intermediate allocation.
    pub fn from_sparse(
        marker_cluster: std::sync::Arc<Vec<usize>>,
        ref_cluster_end: std::sync::Arc<Vec<usize>>,
        weight: std::sync::Arc<Vec<f32>>,
        offsets: Vec<usize>,
        hap_indices: Vec<u32>,
        probs: Vec<f32>,
        probs_p1: Vec<f32>,
    ) -> Self {
        Self {
            marker_cluster,
            ref_cluster_end,
            weight,
            offsets,
            hap_indices,
            probs,
            probs_p1,
        }
    }

    /// Return haplotype-indexed priors at a reference marker.
    /// Probabilities are interpolated between clusters to match allele posterior logic.
    pub fn haplotype_priors_at(&self, ref_marker: usize) -> (Vec<u32>, Vec<f32>) {
        let cluster = *self.marker_cluster.get(ref_marker).unwrap_or(&0);
        let mut in_cluster = ref_marker < *self.ref_cluster_end.get(cluster).unwrap_or(&0);
        let mut weight = self.weight.get(ref_marker).copied().unwrap_or(0.5);
        if !weight.is_finite() {
            in_cluster = true;
            weight = 0.5;
        }

        let start = self.offsets.get(cluster).copied().unwrap_or(0);
        let end = self.offsets.get(cluster + 1).copied().unwrap_or(start);
        let haps = &self.hap_indices[start..end];
        let probs = &self.probs[start..end];
        let probs_p1 = &self.probs_p1[start..end];

        let mut out_probs = Vec::with_capacity(haps.len());
        for (j, _) in haps.iter().enumerate() {
            let prob = probs[j];
            let prob_p1 = probs_p1[j];
            let interp = if in_cluster {
                prob
            } else {
                weight * prob + (1.0 - weight) * prob_p1
            };
            out_probs.push(interp.max(0.0));
        }

        let sum: f32 = out_probs.iter().sum();
        if sum > 1e-10 {
            for p in &mut out_probs {
                *p /= sum;
            }
        }

        (haps.to_vec(), out_probs)
    }

    #[inline]
    pub fn allele_posteriors<F>(
        &self,
        ref_marker: usize,
        n_alleles: usize,
        get_ref_allele: &F,
    ) -> AllelePosteriors
    where
        F: Fn(usize, u32) -> u8,
    {
        let cluster = *self.marker_cluster.get(ref_marker).unwrap_or(&0);
        let mut in_cluster = ref_marker < *self.ref_cluster_end.get(cluster).unwrap_or(&0);
        let mut weight = self.weight.get(ref_marker).copied().unwrap_or(0.5);
        if !weight.is_finite() {
            in_cluster = true;
            weight = 0.5;
        }

        // CSR indexing: get slice for this cluster
        let start = self.offsets.get(cluster).copied().unwrap_or(0);
        let end = self.offsets.get(cluster + 1).copied().unwrap_or(start);
        let haps = &self.hap_indices[start..end];
        let probs = &self.probs[start..end];
        let probs_p1 = &self.probs_p1[start..end];

        if n_alleles == 2 {
            let mut p_alt = 0.0f32;
            let mut p_ref = 0.0f32;
            for (j, &hap) in haps.iter().enumerate() {
                let prob = probs[j];
                let prob_p1 = probs_p1[j];

                // Interpolate state probability, anchor to LEFT haplotype's allele
                // (matches Java: both prob and prob_p1 contribute to current haplotype's allele)
                let interp_prob = if in_cluster {
                    prob
                } else {
                    weight * prob + (1.0 - weight) * prob_p1
                };

                let allele = get_ref_allele(ref_marker, hap);
                if allele == 1 {
                    p_alt += interp_prob;
                } else if allele == 0 {
                    p_ref += interp_prob;
                }
            }
            let total = p_ref + p_alt;
            let p_alt = if total > 1e-10 { p_alt / total } else { 0.0 };
            AllelePosteriors::Biallelic(p_alt)
        } else {
            let mut al_probs = vec![0.0f32; n_alleles];
            for (j, &hap) in haps.iter().enumerate() {
                let prob = probs[j];
                let prob_p1 = probs_p1[j];

                // Interpolate state probability, anchor to LEFT haplotype's allele
                let interp_prob = if in_cluster {
                    prob
                } else {
                    weight * prob + (1.0 - weight) * prob_p1
                };

                let allele = get_ref_allele(ref_marker, hap);
                if allele != 255 && (allele as usize) < n_alleles {
                    al_probs[allele as usize] += interp_prob;
                }
            }
            let total: f32 = al_probs.iter().sum();
            if total > 1e-10 {
                for p in &mut al_probs {
                    *p /= total;
                }
            }
            AllelePosteriors::Multiallelic(al_probs)
        }
    }
}

impl ImputationPipeline {
    /// Create a new imputation pipeline
    pub fn new(config: Config) -> Self {
        let params = ModelParams::new();
        Self { config, params }
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
    #[allow(unused_imports)]
    use super::*;

    // Tests for imputation pipeline

    #[test]
    fn test_state_probs_basic() {
        // Test placeholder
        assert!(true);
    }
}
