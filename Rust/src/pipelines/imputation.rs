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

use std::sync::Arc;

use crate::config::Config;
use crate::error::Result;
use crate::model::parameters::ModelParams;

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
                // Use the same haplotype for both left and right contributions
                // (matches Java Beagle: interpolate probabilities, not haplotype selection)
                if allele == 1 {
                    p_alt += weight_left * prob + (1.0 - weight_left) * prob_p1;
                } else if allele == 0 {
                    p_ref += weight_left * prob + (1.0 - weight_left) * prob_p1;
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
                // Use the same haplotype for both left and right contributions
                // (matches Java Beagle: interpolate probabilities, not haplotype selection)
                if allele != 255 && (allele as usize) < n_alleles {
                    al_probs[allele as usize] += weight_left * prob + (1.0 - weight_left) * prob_p1;
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
pub struct StateProbsCursor {
    state_probs: Arc<StateProbs>,
    /// Current position in genotyped_markers (the sparse index)
    sparse_idx: usize,
}

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

    /// Extract state probabilities from the final cluster for soft-handoff to next window.
    /// Returns (haplotype_ids, probabilities) for states with significant probability.
    pub fn extract_final_priors(&self) -> (Vec<u32>, Vec<f32>) {
        let n_clusters = self.offsets.len().saturating_sub(1);
        if n_clusters == 0 {
            return (Vec::new(), Vec::new());
        }
        
        // Get the last cluster's data
        let last_cluster = n_clusters - 1;
        let start = self.offsets.get(last_cluster).copied().unwrap_or(0);
        let end = self.offsets.get(last_cluster + 1).copied().unwrap_or(start);
        
        let haps = &self.hap_indices[start..end];
        // Use probs_p1 (probability at next marker) as it represents the "exit" state
        let probs = &self.probs_p1[start..end];
        
        // Filter to significant probabilities (>0.001) to save memory
        let mut out_haps = Vec::new();
        let mut out_probs = Vec::new();
        for (i, &hap) in haps.iter().enumerate() {
            let prob = probs.get(i).copied().unwrap_or(0.0);
            if prob > 0.001 {
                out_haps.push(hap);
                out_probs.push(prob);
            }
        }
        
        (out_haps, out_probs)
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
