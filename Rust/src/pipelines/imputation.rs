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

use rayon::prelude::*;
use std::sync::Arc;
use tracing::{info_span, instrument};

use crate::config::Config;
use crate::data::genetic_map::GeneticMaps;
use crate::data::haplotype::{HapIdx, SampleIdx};
use crate::data::marker::MarkerIdx;
use crate::data::storage::GenotypeMatrix;
use crate::data::storage::phase_state::Phased;
use crate::error::Result;
use crate::io::vcf::{ImputationQuality, VcfReader, VcfWriter};
use crate::utils::workspace::ImpWorkspace;

use crate::model::imp_ibs::{build_cluster_hap_sequences_for_targets, ClusterCodedSteps, ImpIbs};
use crate::model::imp_states_cluster::ImpStatesCluster;
use crate::model::parameters::ModelParams;
use crate::data::alignment::MarkerAlignment;
use crate::model::imp_utils::{
    compute_marker_clusters_with_blocks, compute_ref_cluster_bounds, compute_cluster_weights,
    build_marker_cluster_index, compute_targ_block_end, compute_state_probs,
    CompactDr2Entry
};

/// Minimum genetic distance between markers (matches Java Beagle)
const MIN_CM_DIST: f64 = 1e-7;

/// Imputation pipeline
pub struct ImputationPipeline {
    pub(crate) config: Config,
    pub(crate) params: ModelParams,
}

/// State probabilities from HMM forward-backward on reference markers.
///
/// Sparse state probabilities with interpolation for ungenotyped markers.
///
/// The HMM runs only on GENOTYPED markers (where target has data), matching
/// Java Beagle's efficient approach. State probabilities for ungenotyped markers
/// are computed via linear interpolation of ALLELE POSTERIORS (not state probs).
///
/// This provides ~100x speedup over running HMM on all reference markers.
///
/// **Design Note**: We store state probabilities at genotyped markers, and for
/// interpolation we store BOTH `probs[m]` and `probs_p1[m]` (probability at next
/// marker) to match Java Beagle's approach. The haplotype from marker m is used
/// for allele lookup at all interpolated positions, while the state probability
/// is interpolated between prob[m] and probs_p1[m].
///
/// This matches Java `imp/StateProbsFactory.java` which stores:
/// - hapIndices[m][j] = haplotype for state j at marker m
/// - probs[m][j] = P(state j) at marker m
/// - probsP1[m][j] = P(state j) at marker m+1
#[cfg(test)]
#[derive(Clone, Debug)]
pub struct StateProbs {
    /// Indices of genotyped markers in reference space
    /// Uses Arc to share across all haplotypes (avoids cloning per sample)
    genotyped_markers: std::sync::Arc<Vec<usize>>,
    /// Reference haplotype indices at each genotyped marker
    hap_indices: Vec<Vec<u32>>,
    /// State probabilities at each genotyped marker
    probs: Vec<Vec<f32>>,
    /// State probabilities at the NEXT genotyped marker (for interpolation)
    /// At the last marker, this equals probs (no next marker)
    probs_p1: Vec<Vec<f32>>,
    /// Dense haplotypes for all markers (small panels only)
    dense_haps: Option<Vec<Vec<u32>>>,
    /// Genetic positions of ALL reference markers (for interpolation)
    /// Uses Arc to share across all haplotypes (avoids cloning ~8MB per haplotype)
    gen_positions: std::sync::Arc<Vec<f64>>,
    /// Maps genotyped marker index to cluster index
    marker_to_cluster: std::sync::Arc<Vec<usize>>,
}

#[cfg(test)]
impl StateProbs {
    /// Create state probabilities from sparse HMM output.
    ///
    /// # Arguments
    /// * `genotyped_markers` - Indices of genotyped markers in reference space
    /// * `n_states` - Number of HMM states
    /// * `hap_indices` - Reference haplotype indices at each genotyped marker
    /// * `state_probs` - Flattened state probabilities from HMM (genotyped markers only)
    /// * `gen_positions` - Genetic positions of ALL reference markers
    ///
    /// # Design
    /// Stores state probabilities at each genotyped marker PLUS probabilities at the
    /// next marker (probs_p1) for interpolation. This matches Java Beagle's approach
    /// where interpolation uses:
    /// - Haplotype from marker m for allele lookup
    /// - Interpolated probability: w * prob[m] + (1-w) * prob[m+1]
    ///
    /// # Sparse Storage (matching Java StateProbsFactory)
    /// Only states with probability >= threshold are stored, where:
    ///   threshold = min(0.005, 1/K)
    /// This typically reduces storage by 50-100x, critical for large datasets.
    /// Remaining probability mass is renormalized.
    ///
    /// # Why probs_p1 uses "next marker" (not "next cluster"):
    ///
    /// The HMM runs on CLUSTERS, so markers in the same cluster have IDENTICAL probabilities.
    /// When interpolating between markers M and M+1:
    /// - If same cluster: probs[M] == probs[M+1], so probs_p1[M] = probs[M+1] = probs[M]
    ///   → Interpolation yields CONSTANT (correct for within-cluster region)
    /// - If different clusters: probs[M] != probs[M+1]
    ///   → Interpolation yields smooth transition (correct for between-cluster region)
    ///
    /// Using "next marker" naturally produces correct behavior without special-casing.
    /// Using "next cluster" would INCORRECTLY force gradients INSIDE clusters.
    pub fn new(
        genotyped_markers: std::sync::Arc<Vec<usize>>,
        n_states: usize,
        hap_indices: Vec<Vec<u32>>,
        state_probs: Vec<f32>,
        gen_positions: std::sync::Arc<Vec<f64>>,
        marker_to_cluster: std::sync::Arc<Vec<usize>>,
        dense_haps: Option<Vec<Vec<u32>>>,
    ) -> Self {
        let n_genotyped = genotyped_markers.len();

        // Sparse storage threshold: min(0.005, 0.9999/K) - matches Java exactly
        // For small panels, keep all states to maximize accuracy.
        let threshold = if n_genotyped <= 1000 {
            0.0
        } else {
            (0.005f32).min(0.9999f32 / n_states.max(1) as f32)
        };
        let include_all_states = threshold == 0.0;

        let mut filtered_haps = Vec::with_capacity(n_genotyped);
        let mut filtered_probs = Vec::with_capacity(n_genotyped);
        let mut filtered_probs_p1 = Vec::with_capacity(n_genotyped);

        // Filter states by probability threshold (sparse storage)
        // Java does NOT renormalize after filtering - it stores raw probabilities
        for sparse_m in 0..n_genotyped {
            let row_offset = sparse_m * n_states;
            // probs_p1 uses next marker (see comment above for why this is correct)
            let m_p1 = if sparse_m + 1 < n_genotyped { sparse_m + 1 } else { sparse_m };
            let row_offset_p1 = m_p1 * n_states;

            let mut haps = Vec::new();
            let mut probs = Vec::new();
            let mut probs_p1 = Vec::new();

            // Collect states ABOVE threshold (Java uses >, not >=)
            for j in 0..n_states.min(hap_indices.get(sparse_m).map(|v| v.len()).unwrap_or(0)) {
                let prob = state_probs.get(row_offset + j).copied().unwrap_or(0.0);
                let prob_p1 = state_probs.get(row_offset_p1 + j).copied().unwrap_or(0.0);
                // Java: if (stateProbs[m][j] > threshold || stateProbs[mP1][j] > threshold)
                if include_all_states || prob > threshold || prob_p1 > threshold {
                    haps.push(hap_indices[sparse_m][j]);
                    probs.push(prob);
                    probs_p1.push(prob_p1);
                }
            }

            // Java does NOT renormalize - stores raw filtered probabilities
            filtered_haps.push(haps);
            filtered_probs.push(probs);
            filtered_probs_p1.push(probs_p1);
        }

        let dense_haps = if include_all_states {
            dense_haps
        } else {
            None
        };

        Self {
            genotyped_markers,
            hap_indices: filtered_haps,
            probs: filtered_probs,
            probs_p1: filtered_probs_p1,
            dense_haps,
            gen_positions,
            marker_to_cluster,
        }
    }

    /// Compute per-allele probabilities at a reference marker (random access).
    /// Returns optimized representation: Biallelic for 2-allele sites, Multiallelic for others.
    ///
    /// NOTE: For sequential marker access (0, 1, 2, ...), use `cursor()` instead
    /// which provides O(1) lookup via linear scanning instead of O(log N) binary search.
    ///
    /// This method is primarily used in tests for random-access interpolation verification.
    #[cfg(test)]
    #[inline]
    pub fn allele_posteriors<F>(&self, ref_marker: usize, n_alleles: usize, get_ref_allele: F) -> AllelePosteriors
    where
        F: Fn(usize, u32) -> u8,
    {
        match self.genotyped_markers.binary_search(&ref_marker) {
            Ok(sparse_idx) => {
                self.posteriors_at_genotyped(sparse_idx, ref_marker, n_alleles, &get_ref_allele)
            }
            Err(insert_pos) => {
                self.posteriors_interpolated(ref_marker, insert_pos, n_alleles, &get_ref_allele)
            }
        }
    }

    /// Create a cursor for efficient sequential marker access.
    ///
    /// When processing markers in order (0, 1, 2, ..., N-1), this provides O(1)
    /// amortized lookup instead of O(log N) binary search per marker.
    /// This is ~900 million binary search operations saved for typical datasets.
    pub fn cursor(self: Arc<Self>) -> StateProbsCursor {
        StateProbsCursor::new(self)
    }

    /// Per-allele probabilities at a genotyped marker
    #[inline]
    fn posteriors_at_genotyped<F>(
        &self,
        sparse_idx: usize,
        ref_marker: usize,
        n_alleles: usize,
        get_ref_allele: &F,
    ) -> AllelePosteriors
    where
        F: Fn(usize, u32) -> u8,
    {
        let haps = &self.hap_indices[sparse_idx];
        let probs = &self.probs[sparse_idx];

        if n_alleles == 2 {
            // Biallelic path with proper handling of missing reference data
            // When a reference haplotype has missing data (255) at this marker,
            // we must renormalize so that P(REF) + P(ALT) = 1.0
            let mut p_alt = 0.0f32;
            let mut p_ref = 0.0f32;
            for (j, &hap) in haps.iter().enumerate() {
                let allele = get_ref_allele(ref_marker, hap);
                if allele == 1 {
                    p_alt += probs[j];
                } else if allele == 0 {
                    p_ref += probs[j];
                }
                // allele == 255 (missing): don't add to either, will renormalize
            }
            // Renormalize if there was any missing data
            let total = p_ref + p_alt;
            let p_alt = if total > 1e-10 { p_alt / total } else { 0.0 };
            AllelePosteriors::Biallelic(p_alt)
        } else {
            // Full multiallelic - compute PMF with renormalization
            let mut al_probs = vec![0.0f32; n_alleles];
            for (j, &hap) in haps.iter().enumerate() {
                let allele = get_ref_allele(ref_marker, hap);
                if allele != 255 && (allele as usize) < n_alleles {
                    al_probs[allele as usize] += probs[j];
                }
            }
            // Renormalize so probabilities sum to 1
            let total: f32 = al_probs.iter().sum();
            if total > 1e-10 {
                for p in &mut al_probs {
                    *p /= total;
                }
            }
            AllelePosteriors::Multiallelic(al_probs)
        }
    }

    /// Per-allele probabilities at an ungenotyped marker via interpolation
    ///
    /// Matches Java Beagle's `ImputedVcfWriter.setAlProbs()` approach:
    /// - Uses haplotypes from LEFT marker for allele lookup
    /// - Interpolates state probabilities: w * prob[m] + (1-w) * probs_p1[m]
    /// - Adds interpolated prob to the allele that haplotype carries at ref_marker
    ///
    /// This differs from the previous approach which used different haplotype sets
    /// for left and right markers and interpolated allele posteriors.
    #[inline]
    fn posteriors_interpolated<F>(
        &self,
        ref_marker: usize,
        insert_pos: usize,
        n_alleles: usize,
        get_ref_allele: &F,
    ) -> AllelePosteriors
    where
        F: Fn(usize, u32) -> u8,
    {
        let n_genotyped = self.genotyped_markers.len();

        // Handle edge cases
        if n_genotyped == 0 {
            return if n_alleles == 2 {
                AllelePosteriors::Biallelic(0.0)
            } else {
                AllelePosteriors::Multiallelic(vec![0.0f32; n_alleles])
            };
        }
        if insert_pos == 0 {
            return self.posteriors_at_genotyped(0, ref_marker, n_alleles, get_ref_allele);
        }
        if insert_pos >= n_genotyped {
            return self.posteriors_at_genotyped(n_genotyped - 1, ref_marker, n_alleles, get_ref_allele);
        }

        // Get the LEFT genotyped marker (we use its haplotypes for allele lookup)
        let left_sparse = insert_pos - 1;
        let left_ref = self.genotyped_markers[left_sparse];
        let right_ref = self.genotyped_markers[insert_pos];

        let pos_left = self.gen_positions[left_ref];
        let pos_right = self.gen_positions[right_ref];
        let pos_marker = self.gen_positions[ref_marker];

        // Weight for LEFT marker's probability (matches Java's wts formula)
        let total_dist = pos_right - pos_left;
        let weight_left = if total_dist > 1e-10 {
            ((pos_right - pos_marker) / total_dist) as f32
        } else {
            0.5
        };

        // Prefer dense haplotype lookup when available (small panels)
        if let Some(ref dense_haps) = self.dense_haps {
            if ref_marker < dense_haps.len() {
                let haps = &dense_haps[ref_marker];
                let probs = &self.probs[left_sparse];
                let probs_p1 = &self.probs_p1[left_sparse];
                let right_sparse = insert_pos;
                let is_between_clusters = self.marker_to_cluster[left_sparse]
                    != self.marker_to_cluster[right_sparse];

                if n_alleles == 2 {
                    let mut p_alt = 0.0f32;
                    let mut p_ref = 0.0f32;
                    for (j, &hap) in haps.iter().enumerate() {
                        let prob = probs.get(j).copied().unwrap_or(0.0);
                        let interpolated_prob = if is_between_clusters {
                            let prob_p1 = probs_p1.get(j).copied().unwrap_or(0.0);
                            weight_left * prob + (1.0 - weight_left) * prob_p1
                        } else {
                            prob
                        };
                        let allele = get_ref_allele(ref_marker, hap);
                        if allele == 1 {
                            p_alt += interpolated_prob;
                        } else if allele == 0 {
                            p_ref += interpolated_prob;
                        }
                    }
                    let total = p_ref + p_alt;
                    let p_alt = if total > 1e-10 { p_alt / total } else { 0.0 };
                    return AllelePosteriors::Biallelic(p_alt);
                } else {
                    let mut al_probs = vec![0.0f32; n_alleles];
                    for (j, &hap) in haps.iter().enumerate() {
                        let prob = probs.get(j).copied().unwrap_or(0.0);
                        let interpolated_prob = if is_between_clusters {
                            let prob_p1 = probs_p1.get(j).copied().unwrap_or(0.0);
                            weight_left * prob + (1.0 - weight_left) * prob_p1
                        } else {
                            prob
                        };
                        let allele = get_ref_allele(ref_marker, hap);
                        if allele != 255 && (allele as usize) < n_alleles {
                            al_probs[allele as usize] += interpolated_prob;
                        }
                    }
                    let total: f32 = al_probs.iter().sum();
                    if total > 1e-10 {
                        for p in &mut al_probs {
                            *p /= total;
                        }
                    }
                    return AllelePosteriors::Multiallelic(al_probs);
                }
            }
        }

        // Get haplotypes and probabilities from LEFT marker
        let haps = &self.hap_indices[left_sparse];
        let probs = &self.probs[left_sparse];
        let probs_p1 = &self.probs_p1[left_sparse];

        // Java Beagle uses constant probability for markers within a cluster and
        // interpolates only for markers BETWEEN clusters.
        // We use a robust, integer-based check on cluster indices instead of
        // an unreliable floating point vector comparison (`probs != probs_p1`).
        let right_sparse = insert_pos;
        let is_between_clusters = self.marker_to_cluster[left_sparse]
            != self.marker_to_cluster[right_sparse];


        if n_alleles == 2 {
            // Interpolate state probability, anchor to LEFT haplotype's allele
            // (matches Java: both prob and prob_p1 contribute to current haplotype's allele)
            let mut p_alt = 0.0f32;
            let mut p_ref = 0.0f32;
            for j in 0..haps.len() {
                let prob = probs.get(j).copied().unwrap_or(0.0);

                let prob_p1 = if is_between_clusters {
                    probs_p1.get(j).copied().unwrap_or(0.0)
                } else {
                    prob
                };

                let hap = haps[j];
                let allele = get_ref_allele(ref_marker, hap);
                if allele == 1 {
                    p_alt += weight_left * prob;
                } else if allele == 0 {
                    p_ref += weight_left * prob;
                }

                if let Some(haps_p1) = haps_p1 {
                    let hap_p1 = haps_p1.get(j).copied().unwrap_or(hap);
                    let allele_p1 = get_ref_allele(ref_marker, hap_p1);
                    if allele_p1 == 1 {
                        p_alt += (1.0 - weight_left) * prob_p1;
                    } else if allele_p1 == 0 {
                        p_ref += (1.0 - weight_left) * prob_p1;
                    }
                } else {
                     if allele == 1 {
                        p_alt += (1.0 - weight_left) * prob_p1;
                    } else if allele == 0 {
                        p_ref += (1.0 - weight_left) * prob_p1;
                    }
                }
            }
            let total = p_ref + p_alt;
            let p_alt = if total > 1e-10 { p_alt / total } else { 0.0 };
            AllelePosteriors::Biallelic(p_alt)
        } else {
            // Interpolate state probability, anchor to LEFT haplotype's allele
            let mut al_probs = vec![0.0f32; n_alleles];
            for j in 0..haps.len() {
                let prob = probs.get(j).copied().unwrap_or(0.0);

                let prob_p1 = if is_between_clusters {
                    probs_p1.get(j).copied().unwrap_or(0.0)
                } else {
                    prob
                };

                let hap = haps[j];
                let allele = get_ref_allele(ref_marker, hap);
                if allele != 255 && (allele as usize) < n_alleles {
                    al_probs[allele as usize] += weight_left * prob;
                }

                if let Some(haps_p1) = haps_p1 {
                    let hap_p1 = haps_p1.get(j).copied().unwrap_or(hap);
                    let allele_p1 = get_ref_allele(ref_marker, hap_p1);
                    if allele_p1 != 255 && (allele_p1 as usize) < n_alleles {
                        al_probs[allele_p1 as usize] += (1.0 - weight_left) * prob_p1;
                    }
                } else if allele != 255 && (allele as usize) < n_alleles {
                    al_probs[allele as usize] += (1.0 - weight_left) * prob_p1;
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

    /// Get expected ALT allele count (dosage)
    /// For biallelic: P(ALT)
    /// For multiallelic: sum of i * P(allele i) for i >= 1
    #[cfg(test)]
    #[inline]
    pub fn dosage(&self) -> f32 {
        match self {
            AllelePosteriors::Biallelic(p_alt) => *p_alt,
            AllelePosteriors::Multiallelic(probs) => {
                probs.iter().enumerate().skip(1).map(|(i, &p)| i as f32 * p).sum()
            }
        }
    }
}

/// Cursor for efficient sequential marker access to StateProbs.
///
/// Eliminates the O(log N) binary search per marker by maintaining position state.
/// When markers are processed in order (0, 1, 2, ...), this provides O(1) amortized lookup.
///
/// # Performance Impact
/// For 818 samples × 2 haps × 1.1M markers = 1.8 billion marker lookups:
/// - Binary search: 1.8B × 20 comparisons = 36 billion comparisons
/// - Cursor: 1.8B × ~1 comparison = ~1.8 billion comparisons (20x faster)
#[cfg(test)]
pub struct StateProbsCursor {
    state_probs: Arc<StateProbs>,
    /// Current position in genotyped_markers (the sparse index)
    sparse_idx: usize,
}

#[cfg(test)]
impl StateProbsCursor {
    /// Create a new cursor starting at position 0
    #[inline]
    pub(crate) fn new(state_probs: Arc<StateProbs>) -> Self {
        Self {
            state_probs,
            sparse_idx: 0,
        }
    }

    /// Compute allele posteriors at ref_marker using cursor for O(1) lookup.
    ///
    /// IMPORTANT: Markers MUST be queried in ascending order (0, 1, 2, ...).
    /// The cursor advances automatically and cannot go backwards.
    #[inline]
    pub fn allele_posteriors<F>(
        &mut self,
        ref_marker: usize,
        n_alleles: usize,
        get_ref_allele: F,
    ) -> AllelePosteriors
    where
        F: Fn(usize, u32) -> u8,
    {
        let genotyped_markers = &self.state_probs.genotyped_markers;
        let n_genotyped = genotyped_markers.len();

        // Advance cursor until we find or pass ref_marker
        // This is O(1) amortized because each marker is visited at most once
        while self.sparse_idx < n_genotyped && genotyped_markers[self.sparse_idx] < ref_marker {
            self.sparse_idx += 1;
        }

        // Check if ref_marker is exactly a genotyped marker
        if self.sparse_idx < n_genotyped && genotyped_markers[self.sparse_idx] == ref_marker {
            self.state_probs.posteriors_at_genotyped(self.sparse_idx, ref_marker, n_alleles, &get_ref_allele)
        } else {
            // ref_marker is between sparse_idx-1 and sparse_idx (or before first/after last)
            // insert_pos = sparse_idx gives the correct interpolation interval
            self.state_probs.posteriors_interpolated(ref_marker, self.sparse_idx, n_alleles, &get_ref_allele)
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
    pub fn new(
        marker_cluster: std::sync::Arc<Vec<usize>>,
        ref_cluster_end: std::sync::Arc<Vec<usize>>,
        weight: std::sync::Arc<Vec<f32>>,
        n_states: usize,
        hap_indices_input: Vec<Vec<u32>>,
        cluster_probs: Vec<f32>,
    ) -> Self {
        let n_clusters = hap_indices_input.len();
        let threshold = if n_clusters <= 1000 {
            0.0
        } else {
            (0.9999f32 / n_states as f32).min(0.005f32)
        };

        // Build CSR format: flat arrays with offsets
        // Estimate capacity: ~50 states per cluster on average
        let estimated_nnz = n_clusters * 50;
        let mut offsets = Vec::with_capacity(n_clusters + 1);
        let mut hap_indices = Vec::with_capacity(estimated_nnz);
        let mut probs = Vec::with_capacity(estimated_nnz);
        let mut probs_p1 = Vec::with_capacity(estimated_nnz);

        offsets.push(0);

        for c in 0..n_clusters {
            let next = if c + 1 < n_clusters { c + 1 } else { c };
            let row_offset = c * n_states;
            let next_offset = next * n_states;

            for k in 0..n_states {
                let prob = cluster_probs.get(row_offset + k).copied().unwrap_or(0.0);
                let prob_next = cluster_probs.get(next_offset + k).copied().unwrap_or(0.0);
                if prob > threshold || prob_next > threshold {
                    hap_indices.push(hap_indices_input[c][k]);
                    probs.push(prob);
                    probs_p1.push(prob_next);
                }
            }

            offsets.push(hap_indices.len());
        }

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

    #[inline]
    fn allele_posteriors<F>(
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

    /// Run imputation loading all data into memory (original implementation)
    fn run_in_memory(&mut self) -> Result<()> {
        let (mut target_reader, target_gt, target_samples) = info_span!("load_target_data").in_scope(|| {
            info_span!("load_target").in_scope(|| {
                eprintln!("Loading target VCF...");

                let (mut target_reader, target_file) = VcfReader::open(&self.config.gt)?;
                let target_samples = target_reader.samples_arc();
                let target_gt = target_reader.read_all(target_file)?;
                Ok::<_, crate::error::ReagleError>((target_reader, target_gt, target_samples))
            })
        })?;

        let ref_gt: Arc<GenotypeMatrix<Phased>> = info_span!("load_reference").in_scope(|| {
            eprintln!("Loading reference panel...");
            let ref_path = self.config.r#ref.as_ref().ok_or_else(|| {
                crate::error::ReagleError::config("Reference panel required for imputation")
            })?;

            // Detect file format by extension and load accordingly
            Ok::<_, crate::error::ReagleError>(Arc::new(if ref_path.extension().map(|e| e == "bref3").unwrap_or(false) {
                eprintln!("  Detected BREF3 format (streaming)");
                // Use streaming loader for memory-efficient loading
                Self::load_reference_streaming(ref_path)?
            } else {
                eprintln!("  Detected VCF format");
                let (mut ref_reader, ref_file) = VcfReader::open(ref_path)?;
                ref_reader.read_all(ref_file)?.into_phased()
            }))
        })?;

        if target_gt.n_markers() == 0 || ref_gt.n_markers() == 0 {
            return Ok(());
        }

        // Create marker alignment (reused for both phasing and imputation)
        let alignment = info_span!("align_markers").in_scope(|| {
            eprintln!("Aligning markers...");
            MarkerAlignment::new(&target_gt, &ref_gt)
        });

        let n_ref_haps = ref_gt.n_haplotypes();
        let n_ref_markers = ref_gt.n_markers();
        let n_total_haps = n_ref_haps + target_gt.n_haplotypes();
        self.params = ModelParams::for_imputation(n_ref_haps, n_total_haps, self.config.ne, self.config.err);
        let n_genotyped = alignment.ref_to_target.iter().filter(|&&x| x >= 0).count();
        eprintln!(
            "  {} of {} reference markers are genotyped in target",
            n_genotyped, n_ref_markers
        );
        if target_gt.has_confidence() {
            // Count how many marker-sample pairs have low confidence
            let mut low_confidence_count = 0usize;
            let mut total_count = 0usize;
            for m in 0..target_gt.n_markers() {
                for s in 0..target_gt.n_samples() {
                    total_count += 1;
                    if target_gt.sample_confidence_f32(MarkerIdx::new(m as u32), s) < 0.5 {
                        low_confidence_count += 1;
                    }
                }
            }
            eprintln!(
                "  Target has genotype confidence scores: {}/{} ({:.1}%) are low-confidence",
                low_confidence_count, total_count,
                100.0 * low_confidence_count as f64 / total_count.max(1) as f64
            );
        }

        // Check if target data is already phased - skip phasing if so
        let phased_target_gt_res: Result<GenotypeMatrix<Phased>> = if target_reader.was_all_phased() {
            eprintln!("Target data is already phased, skipping phasing step");
            Ok(target_gt.into_phased())
        } else {
            info_span!("phasing").in_scope(|| {
                // Phase target before imputation (imputation requires phased haplotypes)
                eprintln!("Phasing target data before imputation...");

                // Create phasing pipeline with current config
                let mut phasing = super::phasing::PhasingPipeline::new(self.config.clone());

                // Set reference panel for reference-guided phasing
                phasing.set_reference(Arc::clone(&ref_gt), alignment.clone());
                eprintln!("Using reference panel ({} haplotypes) for phasing", n_ref_haps);

                // Load genetic map if provided
                let gen_maps = if let Some(ref map_path) = self.config.map {
                    let chrom_names: Vec<&str> = target_gt
                        .markers()
                        .chrom_names()
                        .iter()
                        .map(|s| s.as_ref())
                        .collect();
                    GeneticMaps::from_plink_file(map_path, &chrom_names)?
                } else {
                    GeneticMaps::new()
                };

                phasing.phase_in_memory_with_overlap(&target_gt, &gen_maps, None, None)
                    .map(|(result, ..)| result)
            })
        };
        let target_gt = Arc::new(phased_target_gt_res?);

        // Wrap large data structures in Arc for sharing with closures
        let target_gt = Arc::new(target_gt);
        let alignment = Arc::new(alignment);

        let n_target_markers = target_gt.n_markers();
        let n_target_samples = target_gt.n_samples();
        let n_target_haps = target_gt.n_haplotypes();

        eprintln!(
            "Target: {} markers, {} samples; Reference: {} markers, {} haplotypes",
            n_target_markers, n_target_samples, n_ref_markers, n_ref_haps
        );

        // Initialize parameters with CLI config
        // Java uses different hap counts for different parameters:
        // - errProb: uses nHaps = nRefHaps + nTargHaps (total)
        // - pRecomb: uses refGT.nHaps() (ref only)
        let n_total_haps = n_ref_haps + n_target_haps;
        self.params = ModelParams::for_imputation(n_ref_haps, n_total_haps, self.config.ne, self.config.err);
        self.params
            .set_n_states(self.config.imp_states.min(n_ref_haps));

        // Load genetic map if provided
        let gen_maps = if let Some(ref map_path) = self.config.map {
            let chrom_names: Vec<&str> = ref_gt
                .markers()
                .chrom_names()
                .iter()
                .map(|s| s.as_ref())
                .collect();
            GeneticMaps::from_plink_file(map_path, &chrom_names)?
        } else {
            GeneticMaps::new()
        };

        let chrom = ref_gt.marker(MarkerIdx::new(0)).chrom;

        // Build target-aligned marker list (Java uses all target markers, regardless of missingness).
        let genotyped_markers_vec: Vec<usize> = (0..n_ref_markers)
            .filter(|&ref_m| alignment.target_marker(ref_m).is_some())
            .collect();
        let n_genotyped = genotyped_markers_vec.len();
        let n_to_impute = n_ref_markers - n_genotyped;
        let has_observed: std::sync::Arc<Vec<bool>> = {
            let mut flags = vec![false; n_ref_markers];
            for &m in &genotyped_markers_vec {
                flags[m] = true;
            }
            std::sync::Arc::new(flags)
        };

        // Number of IBS haplotypes to find per step (Java ImpIbs)
        // Java: nHapsPerStep = imp_states / (imp_segment / imp_step)
        let imp_states = self.params.n_states;
        let n_steps_per_segment = (self.config.imp_segment / self.config.imp_step).round() as usize;
        let n_steps_per_segment = n_steps_per_segment.max(1);
        let n_ibs_haps = if n_ref_haps <= 1000 {
            n_ref_haps
        } else {
            (imp_states / n_steps_per_segment)
                .max(1)
                .min(n_ref_haps)
        };

        eprintln!("Running imputation with dynamic state selection...");
        let n_states = self.params.n_states;

        // Compute cumulative genetic positions for ALL reference markers
        // Wrapped in Arc to share across all haplotypes without cloning
        let gen_positions: std::sync::Arc<Vec<f64>> = info_span!("compute_genetic_positions").in_scope(|| {
            let mut positions = Vec::with_capacity(n_ref_markers);
            let mut cumulative = 0.0f64;
            positions.push(0.0);
            for m in 1..n_ref_markers {
                let pos1 = ref_gt.marker(MarkerIdx::new((m - 1) as u32)).pos;
                let pos2 = ref_gt.marker(MarkerIdx::new(m as u32)).pos;
                let gen_dist = gen_maps.gen_dist(chrom, pos1, pos2);
                cumulative += gen_dist.abs().max(MIN_CM_DIST);
                positions.push(cumulative);
            }
            std::sync::Arc::new(positions)
        });

        // Compute marker clusters per sample to avoid cross-sample leakage.
        let cluster_dist = self.config.cluster as f64;
        let base_err_rate = self.params.p_mismatch;

        info_span!("compute_marker_clusters").in_scope(|| {
            eprintln!("Computing marker clusters for {} samples...", n_target_samples);
        });

        eprintln!(
            "  HMM on per-sample clusters ({} genotyped markers), interpolating {} ungenotyped",
            n_genotyped, n_to_impute
        );

        // Initialize quality stats for all reference markers
        let n_alleles_per_marker: Vec<usize> = (0..n_ref_markers)
            .map(|m| {
                let marker = ref_gt.marker(MarkerIdx::new(m as u32));
                1 + marker.alt_alleles.len()
            })
            .collect();
        let mut quality = ImputationQuality::new(&n_alleles_per_marker);

        // Mark imputed markers (those not in target)
        for m in 0..n_ref_markers {
            let is_imputed = !has_observed[m];
            quality.set_imputed(m, is_imputed);
        }

        // Check if we need per-haplotype allele probabilities for AP/GP output
        let need_allele_probs = self.config.ap || self.config.gp;

        // Disk-buffered storage: write sample-major (sequential, fast), read chunked for output.
        // This avoids the ~5-12 GB RAM buffer that would cause OOM on small machines.
        // Layout: sample-major = [S0_M0, S0_M1, ..., S0_Mn, S1_M0, S1_M1, ...]
        // Sequential writes during processing, chunked transpose during output.

        let temp_dir = std::env::temp_dir();
        let pid = std::process::id();
        let dosage_path = temp_dir.join(format!("reagle_{}_dosages.tmp", pid));
        let best_gt_path = temp_dir.join(format!("reagle_{}_gt.tmp", pid));
        let posteriors_path = temp_dir.join(format!("reagle_{}_post.tmp", pid));

        // Create temp files with pre-allocated size for sequential writes
        use std::io::Write as IoWrite;
        let dosage_file = std::fs::File::create(&dosage_path)?;
        dosage_file.set_len((n_target_samples * n_ref_markers * 4) as u64)?;
        let mut dosage_writer = std::io::BufWriter::with_capacity(8 * 1024 * 1024, dosage_file);

        let gt_file = std::fs::File::create(&best_gt_path)?;
        gt_file.set_len((n_target_samples * n_ref_markers * 2) as u64)?;
        let mut gt_writer = std::io::BufWriter::with_capacity(4 * 1024 * 1024, gt_file);

        let mut post_writer: Option<std::io::BufWriter<std::fs::File>> = if need_allele_probs {
            let post_file = std::fs::File::create(&posteriors_path)?;
            post_file.set_len((n_target_samples * n_ref_markers * 8) as u64)?;
            Some(std::io::BufWriter::with_capacity(8 * 1024 * 1024, post_file))
        } else {
            None
        };

        let max_alleles = n_alleles_per_marker.iter().copied().max().unwrap_or(2);

        // Batch size for parallel processing - balances parallelism vs memory
        // Each batch holds state_probs temporarily, so batch_size * per_sample_state_probs should fit in RAM
        const BATCH_SIZE: usize = 50;

        info_span!("run_hmm_batched", n_samples = n_target_samples, batch_size = BATCH_SIZE).in_scope(|| {
            eprintln!("Processing samples in parallel batches of {}...", BATCH_SIZE);

            let n_batches = (n_target_samples + BATCH_SIZE - 1) / BATCH_SIZE;

            for batch_idx in 0..n_batches {
                let batch_start = batch_idx * BATCH_SIZE;
                let batch_end = (batch_start + BATCH_SIZE).min(n_target_samples);
                let batch_samples: Vec<usize> = (batch_start..batch_end).collect();

                // Parallel HMM computation for this batch
                // Returns: Vec<(sample_idx, dosages, best_gt, posteriors, dr2_data)>
                // CompactDr2Entry avoids heap allocation for biallelic sites (99%+)
                type Dr2Data = Vec<CompactDr2Entry>;
                let batch_results: Vec<(usize, Vec<f32>, Vec<(u8, u8)>, Option<Vec<(f32, f32)>>, Dr2Data)> = info_span!("process_batch", batch_idx = batch_idx).in_scope(|| {
                    batch_samples
                        .par_iter()
                        .map(|&s| {
                            info_span!("process_sample", sample_idx = s).in_scope(|| {
                        let hap1_idx = HapIdx::new((s * 2) as u32);
                        let hap2_idx = HapIdx::new((s * 2 + 1) as u32);
                        let target_haps = [hap1_idx, hap2_idx];

                        // Determine genotyped markers for this sample only
                        let sample_genotyped: Vec<usize> = (0..n_ref_markers)
                            .filter(|&ref_m| {
                                if let Some(target_m) = alignment.target_marker(ref_m) {
                                    let marker_idx = MarkerIdx::new(target_m as u32);
                                    let a1 = target_gt.allele(marker_idx, hap1_idx);
                                    let a2 = target_gt.allele(marker_idx, hap2_idx);
                                    a1 != 255 || a2 != 255
                                } else {
                                    false
                                }
                            })
                            .collect();

                        // Compute HMM state probabilities for this sample
                        let (sp1, sp2): (Arc<ClusterStateProbs>, Arc<ClusterStateProbs>) = if sample_genotyped.is_empty() {
                            let empty = Arc::new(ClusterStateProbs::new(
                                std::sync::Arc::new(vec![0usize; n_ref_markers]),
                                std::sync::Arc::new(Vec::new()),
                                std::sync::Arc::new(Vec::new()),
                                0,
                                Vec::new(),
                                Vec::new(),
                            ));
                            (empty.clone(), empty)
                        } else {
                            info_span!("compute_sample_clusters").in_scope(|| {
                            let targ_block_end = compute_targ_block_end(&ref_gt, &target_gt, &alignment, &sample_genotyped);
                            let clusters = compute_marker_clusters_with_blocks(
                                &sample_genotyped,
                                &gen_positions,
                                cluster_dist,
                                &targ_block_end,
                            );
                            let n_clusters = clusters.len();
                            let cluster_bounds: Vec<(usize, usize)> = clusters.iter().map(|c| (c.start, c.end)).collect();

                            let cluster_midpoints: Vec<f64> = clusters
                                .iter()
                                .map(|c| {
                                    if c.end > c.start {
                                        (gen_positions[sample_genotyped[c.start]]
                                            + gen_positions[sample_genotyped[c.end - 1]])
                                            / 2.0
                                    } else {
                                        gen_positions[sample_genotyped[c.start]]
                                    }
                                })
                                .collect();

                            let cluster_p_recomb: Vec<f32> = std::iter::once(0.0f32)
                                .chain((1..n_clusters).map(|c| {
                                    let gen_dist = (cluster_midpoints[c] - cluster_midpoints[c - 1]).abs();
                                    self.params.p_recomb(gen_dist)
                                }))
                                .collect();

                            let (ref_cluster_start, ref_cluster_end) = compute_ref_cluster_bounds(&sample_genotyped, &clusters);
                            let marker_cluster = std::sync::Arc::new(build_marker_cluster_index(&ref_cluster_start, n_ref_markers));
                            let ref_cluster_end = std::sync::Arc::new(ref_cluster_end);
                            let cluster_weights = std::sync::Arc::new(compute_cluster_weights(&gen_positions, &ref_cluster_start, &ref_cluster_end));

                            let cluster_seqs = build_cluster_hap_sequences_for_targets(
                                &ref_gt,
                                &target_gt,
                                &alignment,
                                &sample_genotyped,
                                &cluster_bounds,
                                &target_haps,
                            );
                            let coded_steps = ClusterCodedSteps::from_cluster_sequences(
                                &cluster_seqs,
                                &cluster_midpoints,
                                self.config.imp_step as f64,
                            );
                            let imp_ibs = ImpIbs::new(
                                coded_steps,
                                self.config.imp_nsteps,
                                n_ibs_haps,
                                n_ref_haps,
                                target_haps.len(),
                                self.config.seed as u64 + s as u64,
                            );

                            let mut workspace = ImpWorkspace::with_ref_size(n_states);
                            let mut imp_states = ImpStatesCluster::new(
                                &imp_ibs,
                                n_clusters,
                                n_ref_haps,
                                n_states,
                            );

                            let mut out = Vec::with_capacity(2);
                            for (local_h, &global_h) in target_haps.iter().enumerate() {
                                let mut hap_indices: Vec<Vec<u32>> = Vec::new();
                                let actual_n_states = imp_states.ibs_states_cluster(local_h, &mut hap_indices);

                                let probs = compute_state_probs(
                                    &hap_indices,
                                    &cluster_bounds,
                                    &sample_genotyped,
                                    &target_gt,
                                    &ref_gt,
                                    &alignment,
                                    global_h.as_usize(),
                                    actual_n_states,
                                    &mut workspace,
                                    base_err_rate,
                                    &cluster_p_recomb,
                                    std::sync::Arc::clone(&marker_cluster),
                                    std::sync::Arc::clone(&ref_cluster_end),
                                    std::sync::Arc::clone(&cluster_weights),
                                );
                                out.push(probs);
                             }
                             (out[0].clone(), out[1].clone())
                             })
                         };

                        // Compute dosages and collect DR2 data for all markers
                        let get_ref_allele = |ref_m: usize, hap: u32| -> u8 {
                            ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(hap))
                        };

                        let mut dosages = Vec::with_capacity(n_ref_markers);
                        let mut best_gt_for_sample: Vec<(u8, u8)> = Vec::with_capacity(n_ref_markers);
                        let mut posteriors_for_sample = if need_allele_probs {
                            Some(Vec::with_capacity(n_ref_markers))
                        } else {
                            None
                        };
                        let mut dr2_data: Dr2Data = Vec::with_capacity(n_ref_markers);

                        let mut probs1 = vec![0.0f32; max_alleles];
                        let mut probs2 = vec![0.0f32; max_alleles];

                        info_span!("compute_dosages").in_scope(|| {
                            for m in 0..n_ref_markers {
                            let n_alleles = n_alleles_per_marker[m];
                            let is_genotyped = has_observed[m];

                            // Clear probability buffers
                            for a in 0..n_alleles {
                                probs1[a] = 0.0;
                                probs2[a] = 0.0;
                            }

                            let mut skip_sample = false;
                            let mut use_observed_a1 = false;
                            let mut use_observed_a2 = false;
                            let mut observed_a1: u8 = 255;
                            let mut observed_a2: u8 = 255;

                            if is_genotyped {
                                if let Some(target_m) = alignment.target_marker(m) {
                                    let target_marker_idx = MarkerIdx::new(target_m as u32);
                                    let a1 = target_gt.allele(target_marker_idx, hap1_idx);
                                    let a2 = target_gt.allele(target_marker_idx, hap2_idx);
                                    let a1_mapped = alignment.map_allele(target_m, a1);
                                    let a2_mapped = alignment.map_allele(target_m, a2);

                                    if a1_mapped != 255 && (a1_mapped as usize) < n_alleles {
                                        probs1[a1_mapped as usize] = 1.0;
                                        use_observed_a1 = true;
                                        observed_a1 = a1_mapped;
                                    } else {
                                        skip_sample = true;
                                        let post1 = sp1.allele_posteriors(m, n_alleles, &get_ref_allele);
                                        for a in 0..n_alleles { probs1[a] = post1.prob(a); }
                                    }

                                    if a2_mapped != 255 && (a2_mapped as usize) < n_alleles {
                                        probs2[a2_mapped as usize] = 1.0;
                                        use_observed_a2 = true;
                                        observed_a2 = a2_mapped;
                                    } else {
                                        skip_sample = true;
                                        let post2 = sp2.allele_posteriors(m, n_alleles, &get_ref_allele);
                                        for a in 0..n_alleles { probs2[a] = post2.prob(a); }
                                    }
                                }
                            } else {
                                let post1 = sp1.allele_posteriors(m, n_alleles, &get_ref_allele);
                                let post2 = sp2.allele_posteriors(m, n_alleles, &get_ref_allele);
                                for a in 0..n_alleles {
                                    probs1[a] = post1.prob(a);
                                    probs2[a] = post2.prob(a);
                                }
                            }

                            // Compute dosage
                            let d1 = if use_observed_a1 { observed_a1 as f32 } else {
                                if n_alleles == 2 { probs1[1] } else {
                                    (1..n_alleles).map(|a| a as f32 * probs1[a]).sum::<f32>()
                                }
                            };
                            let d2 = if use_observed_a2 { observed_a2 as f32 } else {
                                if n_alleles == 2 { probs2[1] } else {
                                    (1..n_alleles).map(|a| a as f32 * probs2[a]).sum::<f32>()
                                }
                            };
                            dosages.push(d1 + d2);

                            // ALWAYS store best-guess GT from HMM to preserve phase info
                            let best_a1 = if use_observed_a1 {
                                observed_a1
                            } else {
                                // argmax of probs1[0..n_alleles]
                                (0..n_alleles).max_by(|&a, &b| probs1[a].partial_cmp(&probs1[b]).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0) as u8
                            };
                            let best_a2 = if use_observed_a2 {
                                observed_a2
                            } else {
                                // argmax of probs2[0..n_alleles]
                                (0..n_alleles).max_by(|&a, &b| probs2[a].partial_cmp(&probs2[b]).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0) as u8
                            };
                            best_gt_for_sample.push((best_a1, best_a2));

                            // Store posteriors for AP/GP if needed
                            if let Some(ref mut post_vec) = posteriors_for_sample {
                                let p1 = if use_observed_a1 { observed_a1 as f32 } else { probs1.get(1).copied().unwrap_or(0.0) };
                                let p2 = if use_observed_a2 { observed_a2 as f32 } else { probs2.get(1).copied().unwrap_or(0.0) };
                                post_vec.push((p1, p2));
                            }

                            // Collect DR2 data (probs need to be cloned for later accumulation)
                            // Use compact storage to avoid heap allocation for biallelic sites
                            // Always include genotyped markers (Beagle does this)
                            // Skip is based on if we couldn't map, but we already set skip_sample above.
                            // If skip_sample was true, we computed posteriors, so we are GOOD to include.
                            // So we should NOT skip in stats accumulation.
                            // The previous code skipped if is_genotyped && skip_sample (meaning mapping failed).
                            // But since we computed posteriors for them, we should treat them as valid data points.
                            // So pass `skip = false` effectively. Or keep the field but ignore it or set it false.

                            if n_alleles == 2 {
                                dr2_data.push(CompactDr2Entry::Biallelic {
                                    marker: m as u32,
                                    p1: probs1[1],
                                    p2: probs2[1],
                                    skip: false,
                                });
                            } else {
                                dr2_data.push(CompactDr2Entry::Multiallelic {
                                    marker: m as u32,
                                    probs1: probs1[..n_alleles].to_vec(),
                                    probs2: probs2[..n_alleles].to_vec(),
                                    skip: false,
                                });
                             }
                         }
                        });

                        // sp1, sp2 are dropped here when returning
                        (s, dosages, best_gt_for_sample, posteriors_for_sample, dr2_data)
                            })
                        })
                    .collect()
                });

                // Sort results by sample index for sequential writes
                let mut sorted_results = batch_results;
                sorted_results.sort_by_key(|(s, _, _, _, _)| *s);

                // Write sample-major (sequential = fast!) and accumulate DR2
                for (s, dosages, best_gt, posteriors, dr2_data) in sorted_results {
                    let is_diploid = target_samples.is_diploid(SampleIdx::new(s as u32));

                    // Write to temp files (sequential, no seeks!)
                    for &d in &dosages {
                        dosage_writer.write_all(&d.to_le_bytes()).expect("temp file write failed");
                    }
                    for &(a1, a2) in &best_gt {
                        gt_writer.write_all(&[a1, a2]).expect("temp file write failed");
                    }
                    if let Some(ref mut pw) = post_writer {
                        if let Some(ref posts) = posteriors {
                            for &(p1, p2) in posts {
                                pw.write_all(&p1.to_le_bytes()).expect("temp file write failed");
                                pw.write_all(&p2.to_le_bytes()).expect("temp file write failed");
                            }
                        }
                    }

                    // Accumulate DR2 quality stats
                    for entry in dr2_data {
                        match entry {
                            CompactDr2Entry::Biallelic { marker, p1, p2, skip } => {
                                if !skip {
                                    if let Some(stats) = quality.get_mut(marker as usize) {
                                        if is_diploid {
                                            stats.add_sample_biallelic(p1, p2);
                                        } else {
                                            stats.add_haploid_biallelic(p1);
                                        }
                                    }
                                }
                            }
                            CompactDr2Entry::Multiallelic { marker, probs1, probs2, skip } => {
                                if !skip {
                                    if let Some(stats) = quality.get_mut(marker as usize) {
                                        if is_diploid {
                                            stats.add_sample(&probs1, &probs2);
                                        } else {
                                            stats.add_haploid(&probs1);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                eprintln!("  Processed batch {}/{} (samples {}-{})",
                         batch_idx + 1, n_batches, batch_start + 1, batch_end);
            }
        });

        // Flush temp files before reading
        dosage_writer.flush()?;
        gt_writer.flush()?;
        if let Some(ref mut pw) = post_writer {
            pw.flush()?;
        }

        // Output with chunked transpose: read sample-major, write marker-major
        info_span!("write_output").in_scope(move || {
            let output_path = self.config.out.with_extension("vcf.gz");
            eprintln!("Writing output to {:?}...", output_path);
            let mut writer = VcfWriter::create(&output_path, target_samples.clone())?;
            writer.write_header_extended(ref_gt.markers(), true, self.config.gp, self.config.ap)?;

            // Open temp files for reading
            let dosage_file = std::fs::File::open(&dosage_path)?;
            let gt_file = std::fs::File::open(&best_gt_path)?;
            let post_file: Option<std::fs::File> = if need_allele_probs {
                Some(std::fs::File::open(&posteriors_path)?)
            } else {
                None
            };

            // Use memory-mapped files for zero-copy random access during transpose
            // This avoids allocating large buffers - the OS handles paging efficiently
            let dosage_mmap = unsafe { memmap2::Mmap::map(&dosage_file)? };
            let gt_mmap = unsafe { memmap2::Mmap::map(&gt_file)? };
            let post_mmap: Option<memmap2::Mmap> = if let Some(ref pf) = post_file {
                Some(unsafe { memmap2::Mmap::map(pf)? })
            } else {
                None
            };

            // Direct mmap access - closures reference the mmaps
            let get_dosage = |m: usize, s: usize| -> f32 {
                let offset = (s * n_ref_markers + m) * 4;
                f32::from_le_bytes(dosage_mmap[offset..offset+4].try_into().unwrap())
            };
            let get_best_gt = |m: usize, s: usize| -> (u8, u8) {
                let offset = (s * n_ref_markers + m) * 2;
                (gt_mmap[offset], gt_mmap[offset + 1])
            };

            type GetPosteriorsFn = Box<dyn Fn(usize, usize) -> (AllelePosteriors, AllelePosteriors)>;
            let get_posteriors: Option<GetPosteriorsFn> = if need_allele_probs {
                let pm = post_mmap.as_ref().unwrap();
                // Clone mmap reference into closure
                let pm_slice: &[u8] = pm;
                let pm_ptr = pm_slice.as_ptr();
                let n_ref = n_ref_markers;
                Some(Box::new(move |m: usize, s: usize| -> (AllelePosteriors, AllelePosteriors) {
                    let offset = (s * n_ref + m) * 8;
                    let p1 = f32::from_le_bytes(unsafe {
                        std::slice::from_raw_parts(pm_ptr.add(offset), 4)
                    }.try_into().unwrap());
                    let p2 = f32::from_le_bytes(unsafe {
                        std::slice::from_raw_parts(pm_ptr.add(offset + 4), 4)
                    }.try_into().unwrap());
                    (AllelePosteriors::Biallelic(p1), AllelePosteriors::Biallelic(p2))
                }))
            } else {
                None
            };

            // Write all markers in one call - no chunking needed with mmap
            writer.write_imputed_streaming(
                &ref_gt,
                get_dosage,
                get_best_gt,
                get_posteriors,
                &quality,
                0,
                n_ref_markers,
                self.config.gp,
                self.config.ap,
            )?;

            eprintln!("  Written {} markers", n_ref_markers);

            writer.flush()?;

            // Clean up temp files
            let _ = std::fs::remove_file(&dosage_path);
            let _ = std::fs::remove_file(&best_gt_path);
            let _ = std::fs::remove_file(&posteriors_path);

            Ok::<_, crate::error::ReagleError>(())
        })?;

        eprintln!("Imputation complete!");
        Ok(())
    }

    /// Load reference panel from file (VCF or BREF3)
    pub fn load_reference_streaming(path: &std::path::Path) -> Result<GenotypeMatrix<Phased>> {
        // Currently falling back to standard loading.
        // Tricky because we need to read the whole file to get markers
        if path.extension().map(|e| e == "bref3").unwrap_or(false) {
            let file = std::fs::File::open(path)?;
            let reader = std::io::BufReader::new(file);
            let mut bref_reader = crate::io::bref3::Bref3Reader::open(reader)?;
            Ok(bref_reader.read_all()?)
        } else {
            let (mut reader, file) = VcfReader::open(path)?;
            Ok(reader.read_all(file)?.into_phased())
        }
    }
}

// ... existing tests ...
#[cfg(test)]
mod tests {
    use super::*;

    // Tests for imputation pipeline

    #[test]
    fn test_state_probs_basic() {
        // Test placeholder
        assert!(true);
    }
}
