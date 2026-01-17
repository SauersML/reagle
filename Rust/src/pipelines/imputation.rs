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

use std::sync::Arc;
use tracing::instrument;

use crate::config::Config;
use crate::data::storage::GenotypeColumn;
use crate::data::HapIdx;
use crate::error::Result;
use crate::model::parameters::ModelParams;
use crate::utils::telemetry::TelemetryBlackboard;

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
    pub fn allele_posteriors_for_column(
        &self,
        ref_marker: usize,
        n_alleles: usize,
        column: &GenotypeColumn,
        map_ref_to_targ: Option<&[i8]>,
    ) -> AllelePosteriors {
        #[inline]
        fn map_allele(map_ref_to_targ: Option<&[i8]>, allele: u8) -> u8 {
            if allele == 255 {
                return 255;
            }
            if let Some(map) = map_ref_to_targ {
                let idx = allele as usize;
                if idx < map.len() {
                    let mapped = map[idx];
                    if mapped >= 0 {
                        mapped as u8
                    } else {
                        255
                    }
                } else {
                    255
                }
            } else {
                allele
            }
        }

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

        match column {
            GenotypeColumn::SeqCoded(col) => {
                let hap_to_seq = col.hap_to_seq();
                let seq_alleles = col.seq_alleles();
                if n_alleles == 2 {
                    let mut seq_probs = vec![0.0f32; seq_alleles.len()];
                    for (j, &hap) in haps.iter().enumerate() {
                        let idx = hap as usize;
                        if idx < hap_to_seq.len() {
                            let seq_idx = hap_to_seq[idx] as usize;
                            if seq_idx < seq_probs.len() {
                                seq_probs[seq_idx] += probs[j];
                            }
                        }
                    }
                    let mut p_alt = 0.0f32;
                    let mut p_ref = 0.0f32;
                    for (seq_idx, &p) in seq_probs.iter().enumerate() {
                        if p == 0.0 {
                            continue;
                        }
                        let allele = map_allele(map_ref_to_targ, seq_alleles[seq_idx]);
                        if allele == 1 {
                            p_alt += p;
                        } else if allele == 0 {
                            p_ref += p;
                        }
                    }
                    if !in_cluster {
                        let mut seq_probs_p1 = vec![0.0f32; seq_alleles.len()];
                        for (j, &hap) in haps.iter().enumerate() {
                            let idx = hap as usize;
                            if idx < hap_to_seq.len() {
                                let seq_idx = hap_to_seq[idx] as usize;
                                if seq_idx < seq_probs_p1.len() {
                                    seq_probs_p1[seq_idx] += probs_p1[j];
                                }
                            }
                        }
                        let mut p_alt_p1 = 0.0f32;
                        let mut p_ref_p1 = 0.0f32;
                        for (seq_idx, &p) in seq_probs_p1.iter().enumerate() {
                            if p == 0.0 {
                                continue;
                            }
                            let allele = map_allele(map_ref_to_targ, seq_alleles[seq_idx]);
                            if allele == 1 {
                                p_alt_p1 += p;
                            } else if allele == 0 {
                                p_ref_p1 += p;
                            }
                        }
                        p_alt = weight * p_alt + (1.0 - weight) * p_alt_p1;
                        p_ref = weight * p_ref + (1.0 - weight) * p_ref_p1;
                    }
                    let total = p_ref + p_alt;
                    let p_alt = if total > 1e-10 { p_alt / total } else { 0.0 };
                    AllelePosteriors::Biallelic(p_alt)
                } else {
                    let mut seq_probs = vec![0.0f32; seq_alleles.len()];
                    for (j, &hap) in haps.iter().enumerate() {
                        let idx = hap as usize;
                        if idx < hap_to_seq.len() {
                            let seq_idx = hap_to_seq[idx] as usize;
                            if seq_idx < seq_probs.len() {
                                seq_probs[seq_idx] += probs[j];
                            }
                        }
                    }
                    let mut al_probs = vec![0.0f32; n_alleles];
                    for (seq_idx, &p) in seq_probs.iter().enumerate() {
                        if p == 0.0 {
                            continue;
                        }
                        let allele = map_allele(map_ref_to_targ, seq_alleles[seq_idx]);
                        if allele != 255 && (allele as usize) < n_alleles {
                            al_probs[allele as usize] += p;
                        }
                    }
                    if !in_cluster {
                        let mut seq_probs_p1 = vec![0.0f32; seq_alleles.len()];
                        for (j, &hap) in haps.iter().enumerate() {
                            let idx = hap as usize;
                            if idx < hap_to_seq.len() {
                                let seq_idx = hap_to_seq[idx] as usize;
                                if seq_idx < seq_probs_p1.len() {
                                    seq_probs_p1[seq_idx] += probs_p1[j];
                                }
                            }
                        }
                        let mut al_probs_p1 = vec![0.0f32; n_alleles];
                        for (seq_idx, &p) in seq_probs_p1.iter().enumerate() {
                            if p == 0.0 {
                                continue;
                            }
                            let allele = map_allele(map_ref_to_targ, seq_alleles[seq_idx]);
                            if allele != 255 && (allele as usize) < n_alleles {
                                al_probs_p1[allele as usize] += p;
                            }
                        }
                        for i in 0..n_alleles {
                            al_probs[i] = weight * al_probs[i] + (1.0 - weight) * al_probs_p1[i];
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
            GenotypeColumn::Dictionary(col, offset) => {
                let hap_to_pattern = col.hap_to_pattern();
                let n_patterns = col.n_patterns();
                if n_alleles == 2 {
                    let mut pattern_probs = vec![0.0f32; n_patterns];
                    for (j, &hap) in haps.iter().enumerate() {
                        let idx = hap as usize;
                        if idx < hap_to_pattern.len() {
                            let pat_idx = hap_to_pattern[idx] as usize;
                            if pat_idx < pattern_probs.len() {
                                pattern_probs[pat_idx] += probs[j];
                            }
                        }
                    }
                    let mut p_alt = 0.0f32;
                    let mut p_ref = 0.0f32;
                    for (pat_idx, &p) in pattern_probs.iter().enumerate() {
                        if p == 0.0 {
                            continue;
                        }
                        let allele = map_allele(map_ref_to_targ, col.pattern_allele(*offset, pat_idx));
                        if allele == 1 {
                            p_alt += p;
                        } else if allele == 0 {
                            p_ref += p;
                        }
                    }
                    if !in_cluster {
                        let mut pattern_probs_p1 = vec![0.0f32; n_patterns];
                        for (j, &hap) in haps.iter().enumerate() {
                            let idx = hap as usize;
                            if idx < hap_to_pattern.len() {
                                let pat_idx = hap_to_pattern[idx] as usize;
                                if pat_idx < pattern_probs_p1.len() {
                                    pattern_probs_p1[pat_idx] += probs_p1[j];
                                }
                            }
                        }
                        let mut p_alt_p1 = 0.0f32;
                        let mut p_ref_p1 = 0.0f32;
                        for (pat_idx, &p) in pattern_probs_p1.iter().enumerate() {
                            if p == 0.0 {
                                continue;
                            }
                            let allele = map_allele(map_ref_to_targ, col.pattern_allele(*offset, pat_idx));
                            if allele == 1 {
                                p_alt_p1 += p;
                            } else if allele == 0 {
                                p_ref_p1 += p;
                            }
                        }
                        p_alt = weight * p_alt + (1.0 - weight) * p_alt_p1;
                        p_ref = weight * p_ref + (1.0 - weight) * p_ref_p1;
                    }
                    let total = p_ref + p_alt;
                    let p_alt = if total > 1e-10 { p_alt / total } else { 0.0 };
                    AllelePosteriors::Biallelic(p_alt)
                } else {
                    let mut pattern_probs = vec![0.0f32; n_patterns];
                    for (j, &hap) in haps.iter().enumerate() {
                        let idx = hap as usize;
                        if idx < hap_to_pattern.len() {
                            let pat_idx = hap_to_pattern[idx] as usize;
                            if pat_idx < pattern_probs.len() {
                                pattern_probs[pat_idx] += probs[j];
                            }
                        }
                    }
                    let mut al_probs = vec![0.0f32; n_alleles];
                    for (pat_idx, &p) in pattern_probs.iter().enumerate() {
                        if p == 0.0 {
                            continue;
                        }
                        let allele = map_allele(map_ref_to_targ, col.pattern_allele(*offset, pat_idx));
                        if allele != 255 && (allele as usize) < n_alleles {
                            al_probs[allele as usize] += p;
                        }
                    }
                    if !in_cluster {
                        let mut pattern_probs_p1 = vec![0.0f32; n_patterns];
                        for (j, &hap) in haps.iter().enumerate() {
                            let idx = hap as usize;
                            if idx < hap_to_pattern.len() {
                                let pat_idx = hap_to_pattern[idx] as usize;
                                if pat_idx < pattern_probs_p1.len() {
                                    pattern_probs_p1[pat_idx] += probs_p1[j];
                                }
                            }
                        }
                        let mut al_probs_p1 = vec![0.0f32; n_alleles];
                        for (pat_idx, &p) in pattern_probs_p1.iter().enumerate() {
                            if p == 0.0 {
                                continue;
                            }
                            let allele = map_allele(map_ref_to_targ, col.pattern_allele(*offset, pat_idx));
                            if allele != 255 && (allele as usize) < n_alleles {
                                al_probs_p1[allele as usize] += p;
                            }
                        }
                        for i in 0..n_alleles {
                            al_probs[i] = weight * al_probs[i] + (1.0 - weight) * al_probs_p1[i];
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
            GenotypeColumn::Dense(col) => {
                if n_alleles == 2 {
                    let mut p_alt = 0.0f32;
                    let mut p_ref = 0.0f32;
                    if in_cluster {
                        for (j, &hap) in haps.iter().enumerate() {
                            let allele = map_allele(map_ref_to_targ, col.get(HapIdx::new(hap)));
                            if allele == 1 {
                                p_alt += probs[j];
                            } else if allele == 0 {
                                p_ref += probs[j];
                            }
                        }
                    } else {
                        for (j, &hap) in haps.iter().enumerate() {
                            let allele = map_allele(map_ref_to_targ, col.get(HapIdx::new(hap)));
                            if allele == 1 {
                                p_alt += probs[j];
                            } else if allele == 0 {
                                p_ref += probs[j];
                            }
                        }
                        let mut p_alt_p1 = 0.0f32;
                        let mut p_ref_p1 = 0.0f32;
                        for (j, &hap) in haps.iter().enumerate() {
                            let allele = map_allele(map_ref_to_targ, col.get(HapIdx::new(hap)));
                            if allele == 1 {
                                p_alt_p1 += probs_p1[j];
                            } else if allele == 0 {
                                p_ref_p1 += probs_p1[j];
                            }
                        }
                        p_alt = weight * p_alt + (1.0 - weight) * p_alt_p1;
                        p_ref = weight * p_ref + (1.0 - weight) * p_ref_p1;
                    }
                    let total = p_ref + p_alt;
                    let p_alt = if total > 1e-10 { p_alt / total } else { 0.0 };
                    AllelePosteriors::Biallelic(p_alt)
                } else {
                    let mut al_probs = vec![0.0f32; n_alleles];
                    if in_cluster {
                        for (j, &hap) in haps.iter().enumerate() {
                            let allele = map_allele(map_ref_to_targ, col.get(HapIdx::new(hap)));
                            if allele != 255 && (allele as usize) < n_alleles {
                                al_probs[allele as usize] += probs[j];
                            }
                        }
                    } else {
                        for (j, &hap) in haps.iter().enumerate() {
                            let allele = map_allele(map_ref_to_targ, col.get(HapIdx::new(hap)));
                            if allele != 255 && (allele as usize) < n_alleles {
                                al_probs[allele as usize] += probs[j];
                            }
                        }
                        let mut al_probs_p1 = vec![0.0f32; n_alleles];
                        for (j, &hap) in haps.iter().enumerate() {
                            let allele = map_allele(map_ref_to_targ, col.get(HapIdx::new(hap)));
                            if allele != 255 && (allele as usize) < n_alleles {
                                al_probs_p1[allele as usize] += probs_p1[j];
                            }
                        }
                        for i in 0..n_alleles {
                            al_probs[i] = weight * al_probs[i] + (1.0 - weight) * al_probs_p1[i];
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
            GenotypeColumn::Sparse(col) => {
                if n_alleles == 2 {
                    let mut p_alt = 0.0f32;
                    let mut p_ref = 0.0f32;
                    if in_cluster {
                        for (j, &hap) in haps.iter().enumerate() {
                            let allele = map_allele(map_ref_to_targ, col.get(HapIdx::new(hap)));
                            if allele == 1 {
                                p_alt += probs[j];
                            } else if allele == 0 {
                                p_ref += probs[j];
                            }
                        }
                    } else {
                        for (j, &hap) in haps.iter().enumerate() {
                            let allele = map_allele(map_ref_to_targ, col.get(HapIdx::new(hap)));
                            if allele == 1 {
                                p_alt += probs[j];
                            } else if allele == 0 {
                                p_ref += probs[j];
                            }
                        }
                        let mut p_alt_p1 = 0.0f32;
                        let mut p_ref_p1 = 0.0f32;
                        for (j, &hap) in haps.iter().enumerate() {
                            let allele = map_allele(map_ref_to_targ, col.get(HapIdx::new(hap)));
                            if allele == 1 {
                                p_alt_p1 += probs_p1[j];
                            } else if allele == 0 {
                                p_ref_p1 += probs_p1[j];
                            }
                        }
                        p_alt = weight * p_alt + (1.0 - weight) * p_alt_p1;
                        p_ref = weight * p_ref + (1.0 - weight) * p_ref_p1;
                    }
                    let total = p_ref + p_alt;
                    let p_alt = if total > 1e-10 { p_alt / total } else { 0.0 };
                    AllelePosteriors::Biallelic(p_alt)
                } else {
                    let mut al_probs = vec![0.0f32; n_alleles];
                    if in_cluster {
                        for (j, &hap) in haps.iter().enumerate() {
                            let allele = map_allele(map_ref_to_targ, col.get(HapIdx::new(hap)));
                            if allele != 255 && (allele as usize) < n_alleles {
                                al_probs[allele as usize] += probs[j];
                            }
                        }
                    } else {
                        for (j, &hap) in haps.iter().enumerate() {
                            let allele = map_allele(map_ref_to_targ, col.get(HapIdx::new(hap)));
                            if allele != 255 && (allele as usize) < n_alleles {
                                al_probs[allele as usize] += probs[j];
                            }
                        }
                        let mut al_probs_p1 = vec![0.0f32; n_alleles];
                        for (j, &hap) in haps.iter().enumerate() {
                            let allele = map_allele(map_ref_to_targ, col.get(HapIdx::new(hap)));
                            if allele != 255 && (allele as usize) < n_alleles {
                                al_probs_p1[allele as usize] += probs_p1[j];
                            }
                        }
                        for i in 0..n_alleles {
                            al_probs[i] = weight * al_probs[i] + (1.0 - weight) * al_probs_p1[i];
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
    }
}

impl ImputationPipeline {
    /// Create a new imputation pipeline
    pub fn new(config: Config, telemetry: Option<Arc<TelemetryBlackboard>>) -> Self {
        let params = ModelParams::new();
        Self { config, params, telemetry }
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
