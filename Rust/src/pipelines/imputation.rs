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

/// Cluster-based state probabilities with exact decay bridging between anchors.
/// Uses CSR (Compressed Sparse Row) format to eliminate Vec<Vec<T>> overhead.
#[derive(Clone, Debug)]
pub struct ClusterStateProbs {
    marker_cluster: std::sync::Arc<Vec<usize>>,
    ref_cluster_end: std::sync::Arc<Vec<usize>>,
    gen_positions: std::sync::Arc<Vec<f64>>,
    cluster_midpoints_pos: std::sync::Arc<Vec<f64>>,
    recomb_intensity: f32,
    n_states: usize,
    // CSR format: offsets[c]..offsets[c+1] gives indices for cluster c
    offsets: Vec<usize>,
    hap_indices: Vec<u32>,
    probs: Vec<f32>,
    probs_p1: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct AllelePosteriorCache {
    seq_block_id: usize,
    seq_cluster: usize,
    seq_probs: Vec<f32>,
    seq_probs_p1: Vec<f32>,
    seq_probs_ab: Vec<f32>,
    seq_counts: Vec<u32>,
    dict_block_id: usize,
    dict_cluster: usize,
    dict_probs: Vec<f32>,
    dict_probs_p1: Vec<f32>,
    dict_probs_ab: Vec<f32>,
    dict_counts: Vec<u32>,
    pattern_block_id: usize,
    pattern_cluster: usize,
    pattern_probs: Vec<f32>,
    pattern_probs_p1: Vec<f32>,
    pattern_probs_ab: Vec<f32>,
    pattern_counts: Vec<u32>,
}

impl ClusterStateProbs {
    /// Create from pre-computed sparse CSR data (from run_hmm_forward_backward_to_sparse).
    /// This avoids the O(n_clusters × n_states) dense intermediate allocation.
    pub fn from_sparse(
        marker_cluster: std::sync::Arc<Vec<usize>>,
        ref_cluster_end: std::sync::Arc<Vec<usize>>,
        gen_positions: std::sync::Arc<Vec<f64>>,
        cluster_midpoints_pos: std::sync::Arc<Vec<f64>>,
        recomb_intensity: f32,
        n_states: usize,
        offsets: Vec<usize>,
        hap_indices: Vec<u32>,
        probs: Vec<f32>,
        probs_p1: Vec<f32>,
    ) -> Self {
        Self {
            marker_cluster,
            ref_cluster_end,
            gen_positions,
            cluster_midpoints_pos,
            recomb_intensity,
            n_states,
            offsets,
            hap_indices,
            probs,
            probs_p1,
        }
    }

    /// Return haplotype-indexed priors at a reference marker.
    /// Probabilities are interpolated between clusters to match allele posterior logic.
    pub fn haplotype_priors_at(&self, ref_marker: usize) -> (Vec<u32>, Vec<f32>) {
        let (cluster, in_cluster, decay_a, decay_b, noise_a, noise_b) =
            self.bridge_terms(ref_marker);

        let start = self.offsets.get(cluster).copied().unwrap_or(0);
        let end = self.offsets.get(cluster + 1).copied().unwrap_or(start);
        let haps = &self.hap_indices[start..end];
        let probs = &self.probs[start..end];
        let probs_p1 = &self.probs_p1[start..end];

        let mut out_probs = Vec::with_capacity(haps.len());
        if in_cluster {
            for &prob in probs {
                out_probs.push(prob.max(0.0));
            }
        } else {
            for (j, _) in haps.iter().enumerate() {
                let prob = probs[j];
                let prob_p1 = probs_p1[j];
                let fa = prob * decay_a + noise_a;
                let fb = prob_p1 * decay_b + noise_b;
                let p = fa * fb;
                out_probs.push(p.max(0.0));
            }
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
    pub fn allele_posteriors_for_column_cached(
        &self,
        ref_marker: usize,
        n_alleles: usize,
        column: &GenotypeColumn,
        map_ref_to_targ: Option<&[i8]>,
        cache: &mut AllelePosteriorCache,
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

        let (cluster, in_cluster, decay_a, decay_b, noise_a, noise_b) =
            self.bridge_terms(ref_marker);

        let start = self.offsets.get(cluster).copied().unwrap_or(0);
        let end = self.offsets.get(cluster + 1).copied().unwrap_or(start);
        let haps = &self.hap_indices[start..end];
        let probs = &self.probs[start..end];
        let probs_p1 = &self.probs_p1[start..end];

        match column {
            GenotypeColumn::SeqCoded(col) => {
                let hap_to_seq = col.hap_to_seq();
                let seq_alleles = col.seq_alleles();
                let block_id = col.block_id();
                if cache.seq_block_id != block_id || cache.seq_cluster != cluster {
                    cache.seq_block_id = block_id;
                    cache.seq_cluster = cluster;
                    cache.seq_probs.clear();
                    cache.seq_probs.resize(seq_alleles.len(), 0.0);
                    cache.seq_probs_p1.clear();
                    cache.seq_probs_p1.resize(seq_alleles.len(), 0.0);
                    cache.seq_probs_ab.clear();
                    cache.seq_probs_ab.resize(seq_alleles.len(), 0.0);
                    cache.seq_counts.clear();
                    cache.seq_counts.resize(seq_alleles.len(), 0);
                    for (j, &hap) in haps.iter().enumerate() {
                        let idx = hap as usize;
                        if idx < hap_to_seq.len() {
                            let seq_idx = hap_to_seq[idx] as usize;
                            if seq_idx < cache.seq_probs.len() {
                                cache.seq_probs[seq_idx] += probs[j];
                                cache.seq_probs_p1[seq_idx] += probs_p1[j];
                                cache.seq_probs_ab[seq_idx] += probs[j] * probs_p1[j];
                                cache.seq_counts[seq_idx] += 1;
                            }
                        }
                    }
                }
                if n_alleles == 2 {
                    let mut p_alt = 0.0f32;
                    let mut p_ref = 0.0f32;
                    if in_cluster {
                        for (seq_idx, &p) in cache.seq_probs.iter().enumerate() {
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
                    } else {
                        let missing_mass = self.missing_mass(haps.len(), noise_a, noise_b);
                        let base_freq = self.base_allele_freqs(column, 2, map_ref_to_targ);
                        for seq_idx in 0..cache.seq_probs.len() {
                            let count = cache.seq_counts[seq_idx] as f32;
                            if count == 0.0 {
                                continue;
                            }
                            let sum_a = cache.seq_probs[seq_idx];
                            let sum_b = cache.seq_probs_p1[seq_idx];
                            let sum_ab = cache.seq_probs_ab[seq_idx];
                            let group_mass = decay_a * decay_b * sum_ab
                                + decay_a * noise_b * sum_a
                                + decay_b * noise_a * sum_b
                                + noise_a * noise_b * count;
                            if group_mass == 0.0 {
                                continue;
                            }
                            let allele = map_allele(map_ref_to_targ, seq_alleles[seq_idx]);
                            if allele == 1 {
                                p_alt += group_mass;
                            } else if allele == 0 {
                                p_ref += group_mass;
                            }
                        }
                        p_alt += missing_mass * base_freq[1];
                        p_ref += missing_mass * base_freq[0];
                    }
                    let total = p_ref + p_alt;
                    let p_alt = if total > 1e-10 { p_alt / total } else { 0.0 };
                    AllelePosteriors::Biallelic(p_alt)
                } else {
                    let mut al_probs = vec![0.0f32; n_alleles];
                    if in_cluster {
                        for (seq_idx, &p) in cache.seq_probs.iter().enumerate() {
                            if p == 0.0 {
                                continue;
                            }
                            let allele = map_allele(map_ref_to_targ, seq_alleles[seq_idx]);
                            if allele != 255 && (allele as usize) < n_alleles {
                                al_probs[allele as usize] += p;
                            }
                        }
                    } else {
                        let missing_mass = self.missing_mass(haps.len(), noise_a, noise_b);
                        let base_freq = self.base_allele_freqs(column, n_alleles, map_ref_to_targ);
                        for seq_idx in 0..cache.seq_probs.len() {
                            let count = cache.seq_counts[seq_idx] as f32;
                            if count == 0.0 {
                                continue;
                            }
                            let sum_a = cache.seq_probs[seq_idx];
                            let sum_b = cache.seq_probs_p1[seq_idx];
                            let sum_ab = cache.seq_probs_ab[seq_idx];
                            let group_mass = decay_a * decay_b * sum_ab
                                + decay_a * noise_b * sum_a
                                + decay_b * noise_a * sum_b
                                + noise_a * noise_b * count;
                            if group_mass == 0.0 {
                                continue;
                            }
                            let allele = map_allele(map_ref_to_targ, seq_alleles[seq_idx]);
                            if allele != 255 && (allele as usize) < n_alleles {
                                al_probs[allele as usize] += group_mass;
                            }
                        }
                        for i in 0..n_alleles {
                            al_probs[i] += missing_mass * base_freq[i];
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
                let block_id = Arc::as_ptr(col) as usize;
                if cache.dict_block_id != block_id || cache.dict_cluster != cluster {
                    cache.dict_block_id = block_id;
                    cache.dict_cluster = cluster;
                    cache.dict_probs.clear();
                    cache.dict_probs.resize(n_patterns, 0.0);
                    cache.dict_probs_p1.clear();
                    cache.dict_probs_p1.resize(n_patterns, 0.0);
                    cache.dict_probs_ab.clear();
                    cache.dict_probs_ab.resize(n_patterns, 0.0);
                    cache.dict_counts.clear();
                    cache.dict_counts.resize(n_patterns, 0);
                    for (j, &hap) in haps.iter().enumerate() {
                        let idx = hap as usize;
                        if idx < hap_to_pattern.len() {
                            let pat_idx = hap_to_pattern[idx] as usize;
                            if pat_idx < cache.dict_probs.len() {
                                cache.dict_probs[pat_idx] += probs[j];
                                cache.dict_probs_p1[pat_idx] += probs_p1[j];
                                cache.dict_probs_ab[pat_idx] += probs[j] * probs_p1[j];
                                cache.dict_counts[pat_idx] += 1;
                            }
                        }
                    }
                }
                if n_alleles == 2 {
                    let mut p_alt = 0.0f32;
                    let mut p_ref = 0.0f32;
                    if in_cluster {
                        for (pat_idx, &p) in cache.dict_probs.iter().enumerate() {
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
                    } else {
                        let missing_mass = self.missing_mass(haps.len(), noise_a, noise_b);
                        let base_freq = self.base_allele_freqs(column, 2, map_ref_to_targ);
                        for pat_idx in 0..n_patterns {
                            let count = cache.dict_counts[pat_idx] as f32;
                            if count == 0.0 {
                                continue;
                            }
                            let sum_a = cache.dict_probs[pat_idx];
                            let sum_b = cache.dict_probs_p1[pat_idx];
                            let sum_ab = cache.dict_probs_ab[pat_idx];
                            let group_mass = decay_a * decay_b * sum_ab
                                + decay_a * noise_b * sum_a
                                + decay_b * noise_a * sum_b
                                + noise_a * noise_b * count;
                            if group_mass == 0.0 {
                                continue;
                            }
                            let allele = map_allele(map_ref_to_targ, col.pattern_allele(*offset, pat_idx));
                            if allele == 1 {
                                p_alt += group_mass;
                            } else if allele == 0 {
                                p_ref += group_mass;
                            }
                        }
                        p_alt += missing_mass * base_freq[1];
                        p_ref += missing_mass * base_freq[0];
                    }
                    let total = p_ref + p_alt;
                    let p_alt = if total > 1e-10 { p_alt / total } else { 0.0 };
                    AllelePosteriors::Biallelic(p_alt)
                } else {
                    let mut al_probs = vec![0.0f32; n_alleles];
                    if in_cluster {
                        for (pat_idx, &p) in cache.dict_probs.iter().enumerate() {
                            if p == 0.0 {
                                continue;
                            }
                            let allele = map_allele(map_ref_to_targ, col.pattern_allele(*offset, pat_idx));
                            if allele != 255 && (allele as usize) < n_alleles {
                                al_probs[allele as usize] += p;
                            }
                        }
                    } else {
                        let missing_mass = self.missing_mass(haps.len(), noise_a, noise_b);
                        let base_freq = self.base_allele_freqs(column, n_alleles, map_ref_to_targ);
                        for pat_idx in 0..n_patterns {
                            let count = cache.dict_counts[pat_idx] as f32;
                            if count == 0.0 {
                                continue;
                            }
                            let sum_a = cache.dict_probs[pat_idx];
                            let sum_b = cache.dict_probs_p1[pat_idx];
                            let sum_ab = cache.dict_probs_ab[pat_idx];
                            let group_mass = decay_a * decay_b * sum_ab
                                + decay_a * noise_b * sum_a
                                + decay_b * noise_a * sum_b
                                + noise_a * noise_b * count;
                            if group_mass == 0.0 {
                                continue;
                            }
                            let allele = map_allele(map_ref_to_targ, col.pattern_allele(*offset, pat_idx));
                            if allele != 255 && (allele as usize) < n_alleles {
                                al_probs[allele as usize] += group_mass;
                            }
                        }
                        for i in 0..n_alleles {
                            al_probs[i] += missing_mass * base_freq[i];
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
                        let missing_mass = self.missing_mass(haps.len(), noise_a, noise_b);
                        let base_freq = self.base_allele_freqs(column, 2, map_ref_to_targ);
                        for (j, &hap) in haps.iter().enumerate() {
                            let allele = map_allele(map_ref_to_targ, col.get(HapIdx::new(hap)));
                            let fa = probs[j] * decay_a + noise_a;
                            let fb = probs_p1[j] * decay_b + noise_b;
                            let p = fa * fb;
                            if allele == 1 {
                                p_alt += p;
                            } else if allele == 0 {
                                p_ref += p;
                            }
                        }
                        p_alt += missing_mass * base_freq[1];
                        p_ref += missing_mass * base_freq[0];
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
                        let missing_mass = self.missing_mass(haps.len(), noise_a, noise_b);
                        let base_freq = self.base_allele_freqs(column, n_alleles, map_ref_to_targ);
                        for (j, &hap) in haps.iter().enumerate() {
                            let allele = map_allele(map_ref_to_targ, col.get(HapIdx::new(hap)));
                            let fa = probs[j] * decay_a + noise_a;
                            let fb = probs_p1[j] * decay_b + noise_b;
                            let p = fa * fb;
                            if allele != 255 && (allele as usize) < n_alleles {
                                al_probs[allele as usize] += p;
                            }
                        }
                        for i in 0..n_alleles {
                            al_probs[i] += missing_mass * base_freq[i];
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
                        let missing_mass = self.missing_mass(haps.len(), noise_a, noise_b);
                        let base_freq = self.base_allele_freqs(column, 2, map_ref_to_targ);
                        for (j, &hap) in haps.iter().enumerate() {
                            let allele = map_allele(map_ref_to_targ, col.get(HapIdx::new(hap)));
                            let fa = probs[j] * decay_a + noise_a;
                            let fb = probs_p1[j] * decay_b + noise_b;
                            let p = fa * fb;
                            if allele == 1 {
                                p_alt += p;
                            } else if allele == 0 {
                                p_ref += p;
                            }
                        }
                        p_alt += missing_mass * base_freq[1];
                        p_ref += missing_mass * base_freq[0];
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
                        let missing_mass = self.missing_mass(haps.len(), noise_a, noise_b);
                        let base_freq = self.base_allele_freqs(column, n_alleles, map_ref_to_targ);
                        for (j, &hap) in haps.iter().enumerate() {
                            let allele = map_allele(map_ref_to_targ, col.get(HapIdx::new(hap)));
                            let fa = probs[j] * decay_a + noise_a;
                            let fb = probs_p1[j] * decay_b + noise_b;
                            let p = fa * fb;
                            if allele != 255 && (allele as usize) < n_alleles {
                                al_probs[allele as usize] += p;
                            }
                        }
                        for i in 0..n_alleles {
                            al_probs[i] += missing_mass * base_freq[i];
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

    #[inline]
    pub fn allele_posteriors_for_patterns_cached(
        &self,
        ref_marker: usize,
        n_alleles: usize,
        hap_to_pattern: &[u32],
        pattern_alleles: &[u8],
        map_ref_to_targ: Option<&[i8]>,
        cache: &mut AllelePosteriorCache,
        block_id: usize,
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

        let (cluster, in_cluster, decay_a, decay_b, noise_a, noise_b) =
            self.bridge_terms(ref_marker);
        let start = self.offsets.get(cluster).copied().unwrap_or(0);
        let end = self.offsets.get(cluster + 1).copied().unwrap_or(start);
        let haps = &self.hap_indices[start..end];
        let probs = &self.probs[start..end];
        let probs_p1 = &self.probs_p1[start..end];

        if cache.pattern_block_id != block_id || cache.pattern_cluster != cluster {
            cache.pattern_block_id = block_id;
            cache.pattern_cluster = cluster;
            cache.pattern_probs.clear();
            cache.pattern_probs.resize(pattern_alleles.len(), 0.0);
            cache.pattern_probs_p1.clear();
            cache.pattern_probs_p1.resize(pattern_alleles.len(), 0.0);
            cache.pattern_probs_ab.clear();
            cache.pattern_probs_ab.resize(pattern_alleles.len(), 0.0);
            cache.pattern_counts.clear();
            cache.pattern_counts.resize(pattern_alleles.len(), 0);
            for (j, &hap) in haps.iter().enumerate() {
                let idx = hap as usize;
                if idx < hap_to_pattern.len() {
                    let pat_idx = hap_to_pattern[idx] as usize;
                    if pat_idx < cache.pattern_probs.len() {
                        cache.pattern_probs[pat_idx] += probs[j];
                        cache.pattern_probs_p1[pat_idx] += probs_p1[j];
                        cache.pattern_probs_ab[pat_idx] += probs[j] * probs_p1[j];
                        cache.pattern_counts[pat_idx] += 1;
                    }
                }
            }
        }

        if n_alleles == 2 {
            let mut p_alt = 0.0f32;
            let mut p_ref = 0.0f32;
            if in_cluster {
                for (pat_idx, &p) in cache.pattern_probs.iter().enumerate() {
                    if p == 0.0 {
                        continue;
                    }
                    let allele = map_allele(map_ref_to_targ, pattern_alleles[pat_idx]);
                    if allele == 1 {
                        p_alt += p;
                    } else if allele == 0 {
                        p_ref += p;
                    }
                }
            } else {
                let missing_mass = self.missing_mass(haps.len(), noise_a, noise_b);
                let base_freq = {
                    let mut counts = [0.0f32; 2];
                    let mut total = 0.0f32;
                    for (pat_idx, &count) in cache.pattern_counts.iter().enumerate() {
                        let count = count as f32;
                        if count == 0.0 {
                            continue;
                        }
                        let allele = map_allele(map_ref_to_targ, pattern_alleles[pat_idx]);
                        if allele == 1 {
                            counts[1] += count;
                            total += count;
                        } else if allele == 0 {
                            counts[0] += count;
                            total += count;
                        }
                    }
                    if total > 0.0 {
                        vec![counts[0] / total, counts[1] / total]
                    } else {
                        vec![0.5, 0.5]
                    }
                };
                for pat_idx in 0..cache.pattern_probs.len() {
                    let count = cache.pattern_counts[pat_idx] as f32;
                    if count == 0.0 {
                        continue;
                    }
                    let sum_a = cache.pattern_probs[pat_idx];
                    let sum_b = cache.pattern_probs_p1[pat_idx];
                    let sum_ab = cache.pattern_probs_ab[pat_idx];
                    let group_mass = decay_a * decay_b * sum_ab
                        + decay_a * noise_b * sum_a
                        + decay_b * noise_a * sum_b
                        + noise_a * noise_b * count;
                    if group_mass == 0.0 {
                        continue;
                    }
                    let allele = map_allele(map_ref_to_targ, pattern_alleles[pat_idx]);
                    if allele == 1 {
                        p_alt += group_mass;
                    } else if allele == 0 {
                        p_ref += group_mass;
                    }
                }
                p_alt += missing_mass * base_freq[1];
                p_ref += missing_mass * base_freq[0];
            }
            let total = p_ref + p_alt;
            let p_alt = if total > 1e-10 { p_alt / total } else { 0.0 };
            AllelePosteriors::Biallelic(p_alt)
        } else {
            let mut al_probs = vec![0.0f32; n_alleles];
            if in_cluster {
                for (pat_idx, &p) in cache.pattern_probs.iter().enumerate() {
                    if p == 0.0 {
                        continue;
                    }
                    let allele = map_allele(map_ref_to_targ, pattern_alleles[pat_idx]);
                    if allele != 255 && (allele as usize) < n_alleles {
                        al_probs[allele as usize] += p;
                    }
                }
            } else {
                let missing_mass = self.missing_mass(haps.len(), noise_a, noise_b);
                let base_freq = {
                    let mut counts = vec![0.0f32; n_alleles];
                    let mut total = 0.0f32;
                    for (pat_idx, &count) in cache.pattern_counts.iter().enumerate() {
                        let count = count as f32;
                        if count == 0.0 {
                            continue;
                        }
                        let allele = map_allele(map_ref_to_targ, pattern_alleles[pat_idx]);
                        if allele != 255 {
                            let idx = allele as usize;
                            if idx < n_alleles {
                                counts[idx] += count;
                                total += count;
                            }
                        }
                    }
                    if total > 0.0 {
                        counts.into_iter().map(|c| c / total).collect()
                    } else {
                        vec![1.0 / n_alleles as f32; n_alleles]
                    }
                };
                for pat_idx in 0..cache.pattern_probs.len() {
                    let count = cache.pattern_counts[pat_idx] as f32;
                    if count == 0.0 {
                        continue;
                    }
                    let sum_a = cache.pattern_probs[pat_idx];
                    let sum_b = cache.pattern_probs_p1[pat_idx];
                    let sum_ab = cache.pattern_probs_ab[pat_idx];
                    let group_mass = decay_a * decay_b * sum_ab
                        + decay_a * noise_b * sum_a
                        + decay_b * noise_a * sum_b
                        + noise_a * noise_b * count;
                    if group_mass == 0.0 {
                        continue;
                    }
                    let allele = map_allele(map_ref_to_targ, pattern_alleles[pat_idx]);
                    if allele != 255 && (allele as usize) < n_alleles {
                        al_probs[allele as usize] += group_mass;
                    }
                }
                for i in 0..n_alleles {
                    al_probs[i] += missing_mass * base_freq[i];
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

    #[inline]
    fn bridge_terms(&self, ref_marker: usize) -> (usize, bool, f32, f32, f32, f32) {
        let cluster = *self.marker_cluster.get(ref_marker).unwrap_or(&0);
        let in_cluster = ref_marker < *self.ref_cluster_end.get(cluster).unwrap_or(&0);
        if in_cluster || cluster + 1 >= self.cluster_midpoints_pos.len() {
            return (cluster, true, 1.0, 1.0, 0.0, 0.0);
        }
        let pos = *self.gen_positions.get(ref_marker).unwrap_or(&0.0);
        let pos_a = self.cluster_midpoints_pos[cluster];
        let pos_b = self.cluster_midpoints_pos[cluster + 1];
        let d_a = (pos - pos_a).abs();
        let d_b = (pos_b - pos).abs();
        let decay_a = (-self.recomb_intensity as f64 * d_a).exp() as f32;
        let decay_b = (-self.recomb_intensity as f64 * d_b).exp() as f32;
        let noise_a = (1.0 - decay_a) / self.n_states as f32;
        let noise_b = (1.0 - decay_b) / self.n_states as f32;
        (cluster, false, decay_a, decay_b, noise_a, noise_b)
    }

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

    #[inline]
    fn missing_mass(&self, listed: usize, noise_a: f32, noise_b: f32) -> f32 {
        if listed >= self.n_states {
            0.0
        } else {
            let missing = (self.n_states - listed) as f32;
            missing * noise_a * noise_b
        }
    }

    fn base_allele_freqs(
        &self,
        column: &GenotypeColumn,
        n_alleles: usize,
        map_ref_to_targ: Option<&[i8]>,
    ) -> Vec<f32> {
        if n_alleles == 2 && map_ref_to_targ.is_none() {
            let n_haps = column.n_haplotypes().max(1) as f32;
            let alt = column.alt_count() as f32;
            let alt_freq = (alt / n_haps).clamp(0.0, 1.0);
            return vec![1.0 - alt_freq, alt_freq];
        }
        let n_haps = column.n_haplotypes();
        let mut counts = vec![0usize; n_alleles];
        for h in 0..n_haps {
            let allele = Self::map_allele(map_ref_to_targ, column.get(HapIdx::new(h as u32)));
            if allele != 255 {
                let idx = allele as usize;
                if idx < n_alleles {
                    counts[idx] += 1;
                }
            }
        }
        let denom: usize = counts.iter().sum();
        if denom == 0 {
            return vec![1.0 / n_alleles as f32; n_alleles];
        }
        counts
            .into_iter()
            .map(|c| c as f32 / denom as f32)
            .collect()
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
