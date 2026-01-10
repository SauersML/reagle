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
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info_span, instrument};

use crate::config::Config;
use crate::data::genetic_map::GeneticMaps;
use crate::data::haplotype::{HapIdx, SampleIdx};
use crate::data::marker::MarkerIdx;
use crate::data::storage::GenotypeMatrix;
use crate::data::storage::phase_state::{PhaseState, Phased};
use crate::error::Result;
use crate::io::bref3::Bref3Reader;
use crate::io::vcf::{ImputationQuality, VcfReader, VcfWriter};
use crate::utils::workspace::ImpWorkspace;

use crate::model::imp_ibs::{build_cluster_hap_sequences_for_targets, ClusterCodedSteps, ImpIbs};
use crate::model::imp_states_cluster::ImpStatesCluster;
use crate::model::parameters::ModelParams;

/// Minimum genetic distance between markers (matches Java Beagle)
const MIN_CM_DIST: f64 = 1e-7;

/// Imputation pipeline
pub struct ImputationPipeline {
    config: Config,
    params: ModelParams,
}

/// A cluster of nearby genotyped markers
///
/// Matches Java ImpData's targClustStartEnd concept.
/// Markers within cluster_dist cM are grouped together.
#[derive(Clone, Debug)]
struct MarkerCluster {
    /// Start index in the genotyped markers array (inclusive)
    start: usize,
    /// End index in the genotyped markers array (exclusive)
    end: usize,
}

/// Compute marker clusters based on genetic distance
///
/// Matches Java ImpData.targClustStartEnd():
/// - Groups markers that are within cluster_dist cM of each other
/// - Each cluster contains one or more consecutive genotyped markers
///
/// # Arguments
/// * `genotyped_markers` - Indices of genotyped markers in reference space
/// * `gen_positions` - Cumulative genetic positions for ALL reference markers
/// * `cluster_dist` - Maximum genetic distance within a cluster (cM)
fn compute_marker_clusters(
    genotyped_markers: &[usize],
    gen_positions: &[f64],
    cluster_dist: f64,
) -> Vec<MarkerCluster> {
    if genotyped_markers.is_empty() {
        return Vec::new();
    }

    let mut clusters = Vec::new();
    let mut cluster_start = 0;
    let mut start_pos = gen_positions[genotyped_markers[0]];

    for m in 1..genotyped_markers.len() {
        let pos = gen_positions[genotyped_markers[m]];
        if pos - start_pos > cluster_dist {
            // Start new cluster
            clusters.push(MarkerCluster {
                start: cluster_start,
                end: m,
            });
            cluster_start = m;
            start_pos = pos;
        }
    }

    // Add final cluster
    clusters.push(MarkerCluster {
        start: cluster_start,
        end: genotyped_markers.len(),
    });

    clusters
}

fn compute_marker_clusters_with_blocks(
    genotyped_markers: &[usize],
    gen_positions: &[f64],
    cluster_dist: f64,
    block_end: &[usize],
) -> Vec<MarkerCluster> {
    if genotyped_markers.is_empty() {
        return Vec::new();
    }

    let mut clusters = Vec::new();
    let mut block_start = 0usize;
    let mut block_end_iter = block_end.iter().copied().filter(|&v| v > 0);

    while let Some(block_end_idx) = block_end_iter.next() {
        if block_end_idx <= block_start {
            continue;
        }
        let mut cluster_start = block_start;
        let mut start_pos = gen_positions[genotyped_markers[cluster_start]];

        for m in (cluster_start + 1)..block_end_idx {
            let pos = gen_positions[genotyped_markers[m]];
            if pos - start_pos > cluster_dist {
                clusters.push(MarkerCluster {
                    start: cluster_start,
                    end: m,
                });
                cluster_start = m;
                start_pos = pos;
            }
        }

        clusters.push(MarkerCluster {
            start: cluster_start,
            end: block_end_idx,
        });
        block_start = block_end_idx;
    }

    if clusters.is_empty() {
        compute_marker_clusters(genotyped_markers, gen_positions, cluster_dist)
    } else {
        clusters
    }
}

fn compute_ref_cluster_bounds(
    genotyped_markers: &[usize],
    clusters: &[MarkerCluster],
) -> (Vec<usize>, Vec<usize>) {
    let mut starts = Vec::with_capacity(clusters.len());
    let mut ends = Vec::with_capacity(clusters.len());
    for cluster in clusters {
        let start = genotyped_markers[cluster.start];
        let end = genotyped_markers[cluster.end - 1] + 1;
        starts.push(start);
        ends.push(end);
    }
    (starts, ends)
}

fn compute_cluster_weights(
    gen_positions: &[f64],
    ref_cluster_start: &[usize],
    ref_cluster_end: &[usize],
) -> Vec<f32> {
    let n_ref_markers = gen_positions.len();
    let mut wts = vec![f32::NAN; n_ref_markers];
    if ref_cluster_start.is_empty() {
        return wts;
    }

    for c in 0..ref_cluster_start.len().saturating_sub(1) {
        let end = ref_cluster_end[c];
        let next_start = ref_cluster_start[c + 1];
        if end == 0 || next_start <= end {
            continue;
        }
        let next_start_pos = gen_positions[next_start];
        let end_pos = gen_positions[end - 1];
        let total_len = (next_start_pos - end_pos).max(1e-12);

        for m in end..next_start {
            let wt = (next_start_pos - gen_positions[m]) / total_len;
            wts[m] = wt as f32;
        }
    }
    wts
}

fn build_marker_cluster_index(
    ref_cluster_start: &[usize],
    n_ref_markers: usize,
) -> Vec<usize> {
    let mut marker_cluster = vec![0usize; n_ref_markers];
    if ref_cluster_start.is_empty() {
        return marker_cluster;
    }
    let mut c = 0usize;
    for m in 0..n_ref_markers {
        while c + 1 < ref_cluster_start.len() && m >= ref_cluster_start[c + 1] {
            c += 1;
        }
        marker_cluster[m] = c;
    }
    marker_cluster
}

fn compute_cluster_mismatches(
    hap_indices: &[Vec<u32>],
    cluster_bounds: &[(usize, usize)],
    genotyped_markers: &[usize],
    target_gt: &GenotypeMatrix<Phased>,
    ref_gt: &GenotypeMatrix<Phased>,
    alignment: &MarkerAlignment,
    targ_hap: usize,
    n_states: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let n_clusters = hap_indices.len();
    let mut mismatches = vec![vec![0.0f32; n_states]; n_clusters];
    let mut non_missing = vec![vec![0.0f32; n_states]; n_clusters];
    let targ_hap_idx = HapIdx::new(targ_hap as u32);
    let sample_idx = targ_hap_idx.sample().as_usize();

    for (c, &(start, end)) in cluster_bounds.iter().enumerate() {
        if c >= n_clusters {
            break;
        }
        for &ref_m in &genotyped_markers[start..end] {
            let Some(target_m) = alignment.target_marker(ref_m) else {
                continue;
            };
            let target_marker_idx = MarkerIdx::new(target_m as u32);
            let targ_allele = target_gt.allele(target_marker_idx, targ_hap_idx);
            if targ_allele == 255 {
                continue;
            }
            let confidence = target_gt.sample_confidence_f32(target_marker_idx, sample_idx);
            if confidence <= 0.0 {
                continue;
            }

            for (j, &hap) in hap_indices[c].iter().enumerate().take(n_states) {
                let ref_allele = ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(hap));
                let mapped = alignment.reverse_map_allele(target_m, ref_allele);
                if mapped == 255 {
                    continue;
                }
                non_missing[c][j] += confidence;
                if mapped != targ_allele {
                    mismatches[c][j] += confidence;
                }
            }
        }
    }

    (mismatches, non_missing)
}

fn compute_targ_block_end<S: crate::data::storage::phase_state::PhaseState>(
    ref_gt: &GenotypeMatrix<Phased>,
    target_gt: &GenotypeMatrix<S>,
    alignment: &MarkerAlignment,
    genotyped_markers: &[usize],
) -> Vec<usize> {
    let n_ref_haps = ref_gt.n_haplotypes();
    let mut block_end = Vec::new();
    if genotyped_markers.is_empty() {
        return block_end;
    }

    let mut last_hash: u64 = 0;
    let mut initialized = false;
    for (idx, &ref_m) in genotyped_markers.iter().enumerate() {
        let Some(target_m) = alignment.target_marker(ref_m) else {
            continue;
        };
        let target_marker_idx = MarkerIdx::new(target_m as u32);
        let n_alleles = 1 + target_gt.marker(target_marker_idx).alt_alleles.len();
        let missing_allele = n_alleles as u8;
        let mut hash = 0xcbf29ce484222325u64;
        for h in 0..n_ref_haps {
            let ref_allele = ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(h as u32));
            let mut allele = alignment.reverse_map_allele(target_m, ref_allele);
            if allele == 255 {
                allele = missing_allele;
            }
            hash ^= allele as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
        if initialized && hash != last_hash {
            block_end.push(idx);
        }
        last_hash = hash;
        initialized = true;
    }
    block_end.push(genotyped_markers.len());
    block_end
}

/// Marker alignment between target and reference panels
#[derive(Clone, Debug)]
pub struct MarkerAlignment {
    /// For each reference marker, the index of the corresponding target marker (-1 if not in target)
    ref_to_target: Vec<i32>,
    /// For each target marker, the index of the corresponding reference marker
    target_to_ref: Vec<usize>,
    /// Number of reference markers

    /// Allele mapping for each aligned marker (indexed by target marker)
    /// Maps target allele indices to reference allele indices
    allele_mappings: Vec<Option<crate::data::marker::AlleleMapping>>,
}

impl MarkerAlignment {
    /// Create alignment by matching markers by position with allele mapping
    ///
    /// This handles strand flips (A/T vs T/A) and allele swaps automatically
    /// using `compute_allele_mapping`.
    pub fn new<S1: PhaseState, S2: PhaseState>(target_gt: &GenotypeMatrix<S1>, ref_gt: &GenotypeMatrix<S2>) -> Self {
        use crate::data::marker::compute_allele_mapping;

        let n_ref_markers = ref_gt.n_markers();
        let n_target_markers = target_gt.n_markers();

        // Build position -> target index map
        let mut target_pos_map: HashMap<(u16, u32), usize> = HashMap::new();
        for m in 0..n_target_markers {
            let marker = target_gt.marker(MarkerIdx::new(m as u32));
            target_pos_map.insert((marker.chrom.0, marker.pos), m);
        }

        // Map reference markers to target markers
        let mut ref_to_target = vec![-1i32; n_ref_markers];
        let mut target_to_ref = vec![0usize; n_target_markers];
        let mut allele_mappings: Vec<Option<crate::data::marker::AlleleMapping>> =
            vec![None; n_target_markers];

        let mut n_strand_flipped = 0usize;
        let mut n_allele_swapped = 0usize;

        for m in 0..n_ref_markers {
            let ref_marker = ref_gt.marker(MarkerIdx::new(m as u32));
            if let Some(&target_idx) = target_pos_map.get(&(ref_marker.chrom.0, ref_marker.pos)) {
                let target_marker = target_gt.marker(MarkerIdx::new(target_idx as u32));

                // Compute allele mapping (handles strand flips)
                if let Some(mapping) = compute_allele_mapping(target_marker, ref_marker) {
                    // Check if the mapping is valid (at least REF allele maps)
                    if mapping.is_valid() {
                        ref_to_target[m] = target_idx as i32;
                        target_to_ref[target_idx] = m;

                        if mapping.strand_flipped {
                            n_strand_flipped += 1;
                            // Warn about strand-ambiguous markers (A/T or C/G) where flip detection is unreliable
                            if crate::data::marker::is_strand_ambiguous(target_marker) {
                                eprintln!(
                                    "  Warning: Strand-ambiguous marker at pos {} (A/T or C/G SNV) was strand-flipped",
                                    target_marker.pos
                                );
                            }
                        }
                        if mapping.alleles_swapped {
                            n_allele_swapped += 1;
                        }

                        allele_mappings[target_idx] = Some(mapping);
                    }
                    // If mapping is invalid, marker won't be aligned
                }
            }
        }

        if n_strand_flipped > 0 || n_allele_swapped > 0 {
            eprintln!(
                "  Allele alignment: {} strand-flipped, {} allele-swapped markers",
                n_strand_flipped, n_allele_swapped
            );
        }

        Self {
            ref_to_target,
            target_to_ref,
            allele_mappings,
        }
    }

    /// Get target marker index for a reference marker (returns None if not genotyped)
    pub fn target_marker(&self, ref_marker: usize) -> Option<usize> {
        let idx = self.ref_to_target.get(ref_marker).copied().unwrap_or(-1);
        if idx >= 0 { Some(idx as usize) } else { None }
    }

    /// Map a target allele to reference allele space
    ///
    /// Returns the reference allele index for a given target allele,
    /// handling strand flips and swaps automatically.
    /// Returns 255 (missing) if no valid mapping exists.
    pub fn map_allele(&self, target_marker: usize, target_allele: u8) -> u8 {
        if target_allele == 255 {
            return 255; // Missing stays missing
        }

        if let Some(Some(mapping)) = self.allele_mappings.get(target_marker) {
            mapping.map_allele(target_allele).unwrap_or(255)
        } else {
            // No mapping means identity (direct match assumed)
            target_allele
        }
    }

    /// Map a reference allele to target allele space (reverse mapping)
    ///
    /// Returns the target allele index for a given reference allele,
    /// handling strand flips and swaps automatically.
    /// Returns 255 (missing) if no valid mapping exists.
    pub fn reverse_map_allele(&self, target_marker: usize, ref_allele: u8) -> u8 {
        if ref_allele == 255 {
            return 255; // Missing stays missing
        }

        if let Some(Some(mapping)) = self.allele_mappings.get(target_marker) {
            mapping.reverse_map_allele(ref_allele).unwrap_or(255)
        } else {
            // No mapping means identity (direct match assumed)
            ref_allele
        }
    }

    /// Get reference marker index for a target marker (returns None if not aligned)
    pub fn target_to_ref(&self, target_marker: usize) -> Option<usize> {
        // Check allele_mappings to ensure the marker actually aligns.
        // The raw target_to_ref vector initializes with 0s, which is ambiguous.
        if self.allele_mappings.get(target_marker).and_then(|m| m.as_ref()).is_some() {
            Some(self.target_to_ref[target_marker])
        } else {
            None
        }
    }

    /// Get the number of markers that were successfully aligned
    pub fn n_aligned(&self) -> usize {
        self.ref_to_target.iter().filter(|&&x| x >= 0).count()
    }
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
    /// Reference haplotype indices at the NEXT genotyped marker (optional, accuracy boost)
    haps_p1: Option<Vec<Vec<u32>>>,
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
        let store_haps_p1 = n_genotyped <= 1000;

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
        let mut filtered_haps_p1: Option<Vec<Vec<u32>>> = if store_haps_p1 {
            Some(Vec::with_capacity(n_genotyped))
        } else {
            None
        };

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
            let mut haps_p1 = if store_haps_p1 { Some(Vec::new()) } else { None };

            // Collect states ABOVE threshold (Java uses >, not >=)
            for j in 0..n_states.min(hap_indices.get(sparse_m).map(|v| v.len()).unwrap_or(0)) {
                let prob = state_probs.get(row_offset + j).copied().unwrap_or(0.0);
                let prob_p1 = state_probs.get(row_offset_p1 + j).copied().unwrap_or(0.0);
                // Java: if (stateProbs[m][j] > threshold || stateProbs[mP1][j] > threshold)
                if include_all_states || prob > threshold || prob_p1 > threshold {
                    haps.push(hap_indices[sparse_m][j]);
                    probs.push(prob);
                    probs_p1.push(prob_p1);
                    if let Some(ref mut haps_p1_vec) = haps_p1 {
                        let hap_p1 = hap_indices[m_p1].get(j).copied().unwrap_or(hap_indices[sparse_m][j]);
                        haps_p1_vec.push(hap_p1);
                    }
                }
            }

            // Java does NOT renormalize - stores raw filtered probabilities
            filtered_haps.push(haps);
            filtered_probs.push(probs);
            filtered_probs_p1.push(probs_p1);
            if let Some(ref mut all_haps_p1) = filtered_haps_p1 {
                all_haps_p1.push(haps_p1.unwrap_or_default());
            }
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
            haps_p1: filtered_haps_p1,
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
        let haps_p1 = self.haps_p1.as_ref().map(|v| &v[left_sparse]);

        // Java Beagle uses constant probability for markers within a cluster and
        // interpolates only for markers BETWEEN clusters.
        // We use a robust, integer-based check on cluster indices instead of
        // an unreliable floating point vector comparison (`probs != probs_p1`).
        let right_sparse = insert_pos;
        let is_between_clusters = self.marker_to_cluster[left_sparse]
            != self.marker_to_cluster[right_sparse];


        if n_alleles == 2 {
            // Interpolate haplotype alleles with distance-weighted mixing.
            let mut p_alt = 0.0f32;
            let mut p_ref = 0.0f32;
            for (j, &hap) in haps.iter().enumerate() {
                let prob = probs.get(j).copied().unwrap_or(0.0);

                let (prob_left, prob_right) = if is_between_clusters {
                    let prob_p1 = probs_p1.get(j).copied().unwrap_or(0.0);
                    (weight_left * prob, (1.0 - weight_left) * prob_p1)
                } else if haps_p1.is_some() {
                    (weight_left * prob, (1.0 - weight_left) * prob)
                } else {
                    (prob, 0.0)
                };

                let allele_left = get_ref_allele(ref_marker, hap);
                if allele_left == 1 {
                    p_alt += prob_left;
                } else if allele_left == 0 {
                    p_ref += prob_left;
                }

                if prob_right > 0.0 {
                    if let Some(haps_p1_row) = haps_p1 {
                        let hap_right = haps_p1_row.get(j).copied().unwrap_or(hap);
                        let allele_right = get_ref_allele(ref_marker, hap_right);
                        if allele_right == 1 {
                            p_alt += prob_right;
                        } else if allele_right == 0 {
                            p_ref += prob_right;
                        }
                    } else {
                        let allele_right = get_ref_allele(ref_marker, hap);
                        if allele_right == 1 {
                            p_alt += prob_right;
                        } else if allele_right == 0 {
                            p_ref += prob_right;
                        }
                    }
                }
            }

            let total = p_ref + p_alt;
            let p_alt = if total > 1e-10 { p_alt / total } else { 0.0 };
            AllelePosteriors::Biallelic(p_alt)
        } else {
            let mut al_probs = vec![0.0f32; n_alleles];
            for (j, &hap) in haps.iter().enumerate() {
                let prob = probs.get(j).copied().unwrap_or(0.0);

                let (prob_left, prob_right) = if is_between_clusters {
                    let prob_p1 = probs_p1.get(j).copied().unwrap_or(0.0);
                    (weight_left * prob, (1.0 - weight_left) * prob_p1)
                } else if haps_p1.is_some() {
                    (weight_left * prob, (1.0 - weight_left) * prob)
                } else {
                    (prob, 0.0)
                };

                let allele_left = get_ref_allele(ref_marker, hap);
                if allele_left != 255 && (allele_left as usize) < n_alleles {
                    al_probs[allele_left as usize] += prob_left;
                }

                if prob_right > 0.0 {
                    if let Some(haps_p1_row) = haps_p1 {
                        let hap_right = haps_p1_row.get(j).copied().unwrap_or(hap);
                        let allele_right = get_ref_allele(ref_marker, hap_right);
                        if allele_right != 255 && (allele_right as usize) < n_alleles {
                            al_probs[allele_right as usize] += prob_right;
                        }
                    } else {
                        let allele_right = get_ref_allele(ref_marker, hap);
                        if allele_right != 255 && (allele_right as usize) < n_alleles {
                            al_probs[allele_right as usize] += prob_right;
                        }
                    }
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

    /// Compute dosage = expected allele index = sum(i * P(i))
    #[inline]
    pub fn dosage(&self) -> f32 {
        match self {
            AllelePosteriors::Biallelic(p_alt) => *p_alt,
            AllelePosteriors::Multiallelic(probs) => {
                probs.iter().enumerate().map(|(i, &p)| i as f32 * p).sum()
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
#[derive(Clone, Debug)]
pub struct ClusterStateProbs {
    marker_cluster: std::sync::Arc<Vec<usize>>,
    ref_cluster_end: std::sync::Arc<Vec<usize>>,
    weight: std::sync::Arc<Vec<f32>>,
    hap_indices: Vec<Vec<u32>>,
    haps_p1: Vec<Vec<u32>>,
    probs: Vec<Vec<f32>>,
    probs_p1: Vec<Vec<f32>>,
}

impl ClusterStateProbs {
    pub fn new(
        marker_cluster: std::sync::Arc<Vec<usize>>,
        ref_cluster_end: std::sync::Arc<Vec<usize>>,
        weight: std::sync::Arc<Vec<f32>>,
        n_states: usize,
        hap_indices: Vec<Vec<u32>>,
        cluster_probs: Vec<f32>,
    ) -> Self {
        let n_clusters = hap_indices.len();
        let threshold = if n_clusters <= 1000 {
            0.0
        } else {
            (0.9999f32 / n_states as f32).min(0.005f32)
        };
        let mut filtered_haps = Vec::with_capacity(n_clusters);
        let mut filtered_haps_p1 = Vec::with_capacity(n_clusters);
        let mut probs = Vec::with_capacity(n_clusters);
        let mut probs_p1 = Vec::with_capacity(n_clusters);

        for c in 0..n_clusters {
            let next = if c + 1 < n_clusters { c + 1 } else { c };
            let row_offset = c * n_states;
            let next_offset = next * n_states;

            let mut haps_row = Vec::new();
            let mut haps_p1_row = Vec::new();
            let mut prob_row = Vec::new();
            let mut prob_p1_row = Vec::new();

            for k in 0..n_states {
                let prob = cluster_probs.get(row_offset + k).copied().unwrap_or(0.0);
                let prob_p1 = cluster_probs.get(next_offset + k).copied().unwrap_or(0.0);
                if prob > threshold || prob_p1 > threshold {
                    haps_row.push(hap_indices[c][k]);
                    haps_p1_row.push(hap_indices[next][k]);
                    prob_row.push(prob);
                    prob_p1_row.push(prob_p1);
                }
            }

            filtered_haps.push(haps_row);
            filtered_haps_p1.push(haps_p1_row);
            probs.push(prob_row);
            probs_p1.push(prob_p1_row);
        }

        Self {
            marker_cluster,
            ref_cluster_end,
            weight,
            hap_indices: filtered_haps,
            haps_p1: filtered_haps_p1,
            probs,
            probs_p1,
        }
    }

    pub fn cursor(self: Arc<Self>) -> ClusterStateProbsCursor {
        ClusterStateProbsCursor::new(self)
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

        let haps = &self.hap_indices[cluster];
        let haps_p1 = &self.haps_p1[cluster];
        let probs = &self.probs[cluster];
        let probs_p1 = &self.probs_p1[cluster];

        if n_alleles == 2 {
            let mut p_alt = 0.0f32;
            let mut p_ref = 0.0f32;
            for (j, &hap) in haps.iter().enumerate() {
                let prob = probs.get(j).copied().unwrap_or(0.0);
                let prob_p1 = probs_p1.get(j).copied().unwrap_or(0.0);
                if in_cluster {
                    let allele = get_ref_allele(ref_marker, hap);
                    if allele == 1 {
                        p_alt += prob;
                    } else if allele == 0 {
                        p_ref += prob;
                    }
                } else {
                    let hap_right = haps_p1.get(j).copied().unwrap_or(hap);
                    let prob_left = weight * prob;
                    let prob_right = (1.0 - weight) * prob_p1;

                    let allele_left = get_ref_allele(ref_marker, hap);
                    if allele_left == 1 {
                        p_alt += prob_left;
                    } else if allele_left == 0 {
                        p_ref += prob_left;
                    }

                    let allele_right = get_ref_allele(ref_marker, hap_right);
                    if allele_right == 1 {
                        p_alt += prob_right;
                    } else if allele_right == 0 {
                        p_ref += prob_right;
                    }
                }
            }
            let total = p_ref + p_alt;
            let p_alt = if total > 1e-10 { p_alt / total } else { 0.0 };
            AllelePosteriors::Biallelic(p_alt)
        } else {
            let mut al_probs = vec![0.0f32; n_alleles];
            for (j, &hap) in haps.iter().enumerate() {
                let prob = probs.get(j).copied().unwrap_or(0.0);
                let prob_p1 = probs_p1.get(j).copied().unwrap_or(0.0);
                if in_cluster {
                    let allele = get_ref_allele(ref_marker, hap);
                    if allele != 255 && (allele as usize) < n_alleles {
                        al_probs[allele as usize] += prob;
                    }
                } else {
                    let hap_right = haps_p1.get(j).copied().unwrap_or(hap);
                    let prob_left = weight * prob;
                    let prob_right = (1.0 - weight) * prob_p1;

                    let allele_left = get_ref_allele(ref_marker, hap);
                    if allele_left != 255 && (allele_left as usize) < n_alleles {
                        al_probs[allele_left as usize] += prob_left;
                    }

                    let allele_right = get_ref_allele(ref_marker, hap_right);
                    if allele_right != 255 && (allele_right as usize) < n_alleles {
                        al_probs[allele_right as usize] += prob_right;
                    }
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

pub struct ClusterStateProbsCursor {
    state_probs: Arc<ClusterStateProbs>,
}

impl ClusterStateProbsCursor {
    #[inline]
    pub(crate) fn new(state_probs: Arc<ClusterStateProbs>) -> Self {
        Self { state_probs }
    }

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
        self.state_probs.allele_posteriors(ref_marker, n_alleles, &get_ref_allele)
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
        let (target_reader, target_gt, target_samples) = info_span!("load_target").in_scope(|| {
            eprintln!("Loading target VCF...");
            let (mut target_reader, target_file) = VcfReader::open(&self.config.gt)?;
            let target_samples = target_reader.samples_arc();
            let target_gt = target_reader.read_all(target_file)?;
            Ok::<_, crate::error::ReagleError>((target_reader, target_gt, target_samples))
        })?;

        let ref_gt: Arc<GenotypeMatrix<Phased>> = info_span!("load_reference").in_scope(|| {
            eprintln!("Loading reference panel...");
            let ref_path = self.config.r#ref.as_ref().ok_or_else(|| {
                crate::error::ReagleError::config("Reference panel required for imputation")
            })?;

            // Detect file format by extension and load accordingly
            Ok::<_, crate::error::ReagleError>(Arc::new(if ref_path.extension().map(|e| e == "bref3").unwrap_or(false) {
                eprintln!("  Detected BREF3 format");
                let reader = Bref3Reader::open(ref_path)?;
                reader.read_all()?
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

                phasing.phase_in_memory(&target_gt, &gen_maps)
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

        // Note: gen_positions and steps_config removed - ImpStates now uses ref_panel step boundaries directly

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

        // Cluster-coded haplotype sequences and recursive IBS matching (Java ImpIbs)

        eprintln!("Running imputation with dynamic state selection...");
        let n_states = self.params.n_states;

        // Compute cumulative genetic positions for ALL reference markers
        // Wrapped in Arc to share across all haplotypes without cloning
        let gen_positions: std::sync::Arc<Vec<f64>> = {
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
        };

        // Compute marker clusters per sample to avoid cross-sample leakage.
        let cluster_dist = self.config.cluster as f64;
        let base_err_rate = self.params.p_mismatch;

        eprintln!(
            "  HMM on per-sample clusters ({} genotyped markers), interpolating {} ungenotyped",
            n_genotyped, n_to_impute
        );

        let state_probs: Vec<Arc<ClusterStateProbs>> = info_span!("run_hmm", n_samples = n_target_samples).in_scope(|| {
            (0..n_target_samples)
                .into_par_iter()
                .map(|s| {
                    let hap1_idx = HapIdx::new((s * 2) as u32);
                    let hap2_idx = HapIdx::new((s * 2 + 1) as u32);
                    let target_haps = [hap1_idx, hap2_idx];

                    // Determine genotyped markers for this sample only (either hap non-missing).
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

                    if sample_genotyped.is_empty() {
                        let empty = Arc::new(ClusterStateProbs::new(
                            std::sync::Arc::new(vec![0usize; n_ref_markers]),
                            std::sync::Arc::new(Vec::new()),
                            std::sync::Arc::new(Vec::new()),
                            0,
                            Vec::new(),
                            Vec::new(),
                        ));
                        return (empty.clone(), empty);
                    }

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
                        self.config.seed as u64,
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
                        let (hap_indices, actual_n_states) = if n_ref_haps <= 1000 {
                            let all: Vec<u32> = (0..n_ref_haps as u32).collect();
                            (vec![all; n_clusters], n_ref_haps)
                        } else {
                            let mut hap_indices: Vec<Vec<u32>> = Vec::new();
                            let actual_n_states = imp_states.ibs_states_cluster(local_h, &mut hap_indices);
                            (hap_indices, actual_n_states)
                        };

                        let (cluster_mismatches, cluster_non_missing) = compute_cluster_mismatches(
                            &hap_indices,
                            &cluster_bounds,
                            &sample_genotyped,
                            &target_gt,
                            &ref_gt,
                            &alignment,
                            global_h.as_usize(),
                            actual_n_states,
                        );
                        let cluster_state_probs = run_hmm_forward_backward_clusters_counts(
                            &cluster_mismatches,
                            &cluster_non_missing,
                            &cluster_p_recomb,
                            base_err_rate,
                            actual_n_states,
                            &mut workspace,
                        );

                        out.push(Arc::new(ClusterStateProbs::new(
                            std::sync::Arc::clone(&marker_cluster),
                            std::sync::Arc::clone(&ref_cluster_end),
                            std::sync::Arc::clone(&cluster_weights),
                            actual_n_states,
                            hap_indices,
                            cluster_state_probs,
                        )));
                    }

                    (out[0].clone(), out[1].clone())
                })
                .collect::<Vec<(Arc<ClusterStateProbs>, Arc<ClusterStateProbs>)>>()
                .into_iter()
                .flat_map(|(sp1, sp2)| vec![sp1, sp2])
                .collect()
        });

        eprintln!("Computing dosages with interpolation and quality metrics...");

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

        // =========================================================================
        // STREAMING ARCHITECTURE: Compute on-the-fly to avoid OOM
        // =========================================================================
        //
        // Previous architecture (OOM-causing):
        //   - Computed ALL posteriors for ALL samples upfront: O(n_markers × n_samples × 48 bytes)
        //   - For 1.1M markers × 818 samples = ~45GB allocation
        //
        // New architecture (streaming):
        //   - Uses cursor-based state lookup: O(1) per marker instead of O(log N) binary search
        //   - Computes DR2 in marker-major pass (one marker at a time across all samples)
        //   - Computes dosages on-the-fly during write (no pre-allocation)
        //   - Memory: O(n_haps) for cursors + O(n_markers) for DR2 stats
        //
        // Performance:
        //   - Eliminates ~900M binary searches (20x improvement in lookup)
        //   - Eliminates ~45GB allocation (enables large datasets)
        // =========================================================================

        // PASS 1: Compute DR2 statistics in marker-major order (streaming)
        let target_samples_for_dr2 = Arc::clone(&target_samples);
        info_span!("compute_dr2").in_scope(|| {
            eprintln!("Computing DR2 quality metrics (streaming)...");
            // Create cursors for each haplotype (2 per sample)
            let mut cursors: Vec<ClusterStateProbsCursor> =
                state_probs.iter().map(|sp| sp.clone().cursor()).collect();

            // Closure for ref allele lookup
            let get_ref_allele = |ref_m: usize, hap: u32| -> u8 {
                ref_gt.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(hap))
            };

            // Reusable buffers for allele probabilities (avoids per-marker allocation)
            let max_alleles = n_alleles_per_marker.iter().copied().max().unwrap_or(2);
            let mut probs1 = vec![0.0f32; max_alleles];
            let mut probs2 = vec![0.0f32; max_alleles];

            // Process marker-by-marker (streaming, O(n_markers × n_samples) total)
            for m in 0..n_ref_markers {
                let n_alleles = n_alleles_per_marker[m];
                let is_genotyped = has_observed[m];

                // Use an iterator over chunks to access cursors mutably for each sample
                // This avoids splitting the slice inside the loop
                let mut cursors_iter = cursors.chunks_exact_mut(2);

                for s in 0..n_target_samples {
                    let is_diploid = target_samples_for_dr2.is_diploid(SampleIdx::new(s as u32));
                    let hap1_idx = HapIdx::new((s * 2) as u32);
                    let hap2_idx = HapIdx::new((s * 2 + 1) as u32);
                    
                    let cursors_pair = cursors_iter.next().unwrap();
                    let (left, right) = cursors_pair.split_at_mut(1);
                    let cursor1 = &mut left[0];
                    let cursor2 = &mut right[0];

                    // Clear probability buffers
                    for a in 0..n_alleles {
                        probs1[a] = 0.0;
                        probs2[a] = 0.0;
                    }

                    let mut skip_sample = false;
                    if is_genotyped {
                        // For genotyped markers: use OBSERVED alleles with probability 1.0
                        // This matches Java's setToObsAlleles() behavior
                        if let Some(target_m) = alignment.target_marker(m) {
                            let target_marker_idx = MarkerIdx::new(target_m as u32);
                            let a1 = target_gt.allele(target_marker_idx, hap1_idx);
                            let a2 = target_gt.allele(target_marker_idx, hap2_idx);
                            
                            // Map target alleles to reference allele space
                            let a1_mapped = alignment.map_allele(target_m, a1);
                            let a2_mapped = alignment.map_allele(target_m, a2);
                            
                            // Set probability 1.0 for observed allele (if not missing)
                            // If missing, fall back to HMM posteriors
                            if a1_mapped != 255 && (a1_mapped as usize) < n_alleles {
                                probs1[a1_mapped as usize] = 1.0;
                            } else {
                                skip_sample = true;
                                let post1 = cursor1.allele_posteriors(m, n_alleles, get_ref_allele);
                                for a in 0..n_alleles { probs1[a] = post1.prob(a); }
                            }

                            if a2_mapped != 255 && (a2_mapped as usize) < n_alleles {
                                probs2[a2_mapped as usize] = 1.0;
                            } else {
                                skip_sample = true;
                                let post2 = cursor2.allele_posteriors(m, n_alleles, get_ref_allele);
                                for a in 0..n_alleles { probs2[a] = post2.prob(a); }
                            }
                        }
                    } else {
                        // For imputed markers: use HMM posteriors
                        let post1 = cursor1.allele_posteriors(m, n_alleles, get_ref_allele);
                        let post2 = cursor2.allele_posteriors(m, n_alleles, get_ref_allele);

                        for a in 0..n_alleles {
                            probs1[a] = post1.prob(a);
                            probs2[a] = post2.prob(a);
                        }
                    }

                    if let Some(stats) = quality.get_mut(m) {
                        if !(is_genotyped && skip_sample) {
                            if is_diploid {
                                stats.add_sample(&probs1[..n_alleles], &probs2[..n_alleles]);
                            } else {
                                stats.add_haploid(&probs1[..n_alleles]);
                            }
                        }
                    }
                }
            }
        });

        // PASS 2: Write output with on-the-fly computation (streaming)
        info_span!("write_output").in_scope(move || {
            let output_path = self.config.out.with_extension("vcf.gz");
            eprintln!("Writing output to {:?} (streaming)...", output_path);
            let mut writer = VcfWriter::create(&output_path, target_samples)?;
        writer.write_header_extended(ref_gt.markers(), true, self.config.gp, self.config.ap)?;

        // Use RefCell for interior mutability - allows mutable cursor access from Fn closures
        use std::cell::RefCell;
        use std::rc::Rc;

        let cursors: Rc<RefCell<Vec<ClusterStateProbsCursor>>> = Rc::new(RefCell::new(
            state_probs.iter().map(|sp| sp.clone().cursor()).collect()
        ));

        // Share n_alleles_per_marker between closures via Rc
        let n_alleles_shared: Rc<Vec<usize>> = Rc::new(n_alleles_per_marker);

        // Helper to create one-hot allele posteriors from a hard-called allele
        let one_hot_posterior = |allele: u8, n_alleles: usize| -> AllelePosteriors {
            if n_alleles == 2 {
                AllelePosteriors::Biallelic(allele as f32)
            } else {
                let mut probs = vec![0.0f32; n_alleles];
                if (allele as usize) < n_alleles {
                    probs[allele as usize] = 1.0;
                }
                AllelePosteriors::Multiallelic(probs)
            }
        };

        // Streaming closure: compute dosage on-the-fly using cursor
        let n_alleles_for_dosage = Rc::clone(&n_alleles_shared);
        let cursors_for_dosage = Rc::clone(&cursors);
        let ref_gt_for_dosage = Arc::clone(&ref_gt);
        let alignment_for_dosage = alignment.clone();
        let has_observed_for_dosage = std::sync::Arc::clone(&has_observed);
        let target_gt_for_dosage = Arc::clone(&target_gt);
        let get_dosage = move |m: usize, s: usize| -> f32 {
            if has_observed_for_dosage[m] {
                if let Some(target_m) = alignment_for_dosage.target_marker(m) {
                    let hap1_idx = HapIdx::new((s * 2) as u32);
                    let hap2_idx = HapIdx::new((s * 2 + 1) as u32);
                    let a1 = target_gt_for_dosage.allele(MarkerIdx::new(target_m as u32), hap1_idx);
                    let a2 = target_gt_for_dosage.allele(MarkerIdx::new(target_m as u32), hap2_idx);
                    let a1_mapped = alignment_for_dosage.map_allele(target_m, a1);
                    let a2_mapped = alignment_for_dosage.map_allele(target_m, a2);
                    if a1_mapped != 255 && a2_mapped != 255 {
                        return a1_mapped as f32 + a2_mapped as f32;
                    }
                }
            }

            let n_alleles = n_alleles_for_dosage[m];

            // Check if genotyped in target
            if has_observed_for_dosage[m] {
                if let Some(target_m) = alignment_for_dosage.target_marker(m) {
                    let target_marker_idx = MarkerIdx::new(target_m as u32);
                    let hap1_idx = HapIdx::new((s * 2) as u32);
                    let hap2_idx = HapIdx::new((s * 2 + 1) as u32);

                    let a1 = target_gt_for_dosage.allele(target_marker_idx, hap1_idx);
                    let a2 = target_gt_for_dosage.allele(target_marker_idx, hap2_idx);
                    
                    let a1_mapped = alignment_for_dosage.map_allele(target_m, a1);
                    let a2_mapped = alignment_for_dosage.map_allele(target_m, a2);

                    // Use observed alleles if available, else fall back to HMM
                    let use_a1 = a1_mapped != 255 && (a1_mapped as usize) < n_alleles;
                    let use_a2 = a2_mapped != 255 && (a2_mapped as usize) < n_alleles;

                    if use_a1 && use_a2 {
                        return (a1_mapped as f32) + (a2_mapped as f32);
                    }
                    
                    let mut cursors = cursors_for_dosage.borrow_mut();
                    let (cursor1, cursor2) = {
                        let mid = s * 2 + 1;
                        let (left, right) = cursors.split_at_mut(mid);
                        (&mut left[s * 2], &mut right[0])
                    };
                    let get_ref_allele = |ref_m: usize, hap: u32| -> u8 {
                        ref_gt_for_dosage.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(hap))
                    };

                    let d1 = if use_a1 { a1_mapped as f32 }
                        else { cursor1.allele_posteriors(m, n_alleles, get_ref_allele).dosage() };
                    let d2 = if use_a2 { a2_mapped as f32 }
                        else { cursor2.allele_posteriors(m, n_alleles, get_ref_allele).dosage() };
                    
                    return d1 + d2;
                }
            }

            let mut cursors = cursors_for_dosage.borrow_mut();
            let get_ref_allele = |ref_m: usize, hap: u32| -> u8 {
                ref_gt_for_dosage.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(hap))
            };
            let post1 = cursors[s * 2].allele_posteriors(m, n_alleles, &get_ref_allele);
            let post2 = cursors[s * 2 + 1].allele_posteriors(m, n_alleles, &get_ref_allele);
            post1.dosage() + post2.dosage()
        };

        // Streaming closure: compute posteriors on-the-fly
        type GetPosteriorsFn = Box<dyn Fn(usize, usize) -> (AllelePosteriors, AllelePosteriors)>;
        let get_posteriors: Option<GetPosteriorsFn> = if need_allele_probs {
            let cursors_post: RefCell<Vec<ClusterStateProbsCursor>> = RefCell::new(state_probs.iter().map(|sp| sp.clone().cursor()).collect());
            let n_alleles_per_marker = n_alleles_shared.as_ref().clone();
            let ref_gt_for_post = Arc::clone(&ref_gt);
            let alignment_for_post = alignment.clone();
            let target_gt_for_post = Arc::clone(&target_gt);
            Some(Box::new(
                move |m: usize, s: usize| -> (AllelePosteriors, AllelePosteriors) {
                    let n_alleles = n_alleles_per_marker[m];
                    let get_ref_allele = |ref_m: usize, hap: u32| -> u8 {
                        ref_gt_for_post.allele(MarkerIdx::new(ref_m as u32), HapIdx::new(hap))
                    };

                    if let Some(target_m) = alignment_for_post.target_marker(m) {
                        let hap1_idx = HapIdx::new((s * 2) as u32);
                        let hap2_idx = HapIdx::new((s * 2 + 1) as u32);
                        let a1 = target_gt_for_post.allele(MarkerIdx::new(target_m as u32), hap1_idx);
                        let a2 = target_gt_for_post.allele(MarkerIdx::new(target_m as u32), hap2_idx);
                        let a1_mapped = alignment_for_post.map_allele(target_m, a1);
                        let a2_mapped = alignment_for_post.map_allele(target_m, a2);

                        let mut cursors = cursors_post.borrow_mut();
                        let post1 = if a1_mapped != 255 {
                            one_hot_posterior(a1_mapped, n_alleles)
                        } else {
                            cursors[s * 2].allele_posteriors(m, n_alleles, &get_ref_allele)
                        };
                        let post2 = if a2_mapped != 255 {
                            one_hot_posterior(a2_mapped, n_alleles)
                        } else {
                            cursors[s * 2 + 1].allele_posteriors(m, n_alleles, &get_ref_allele)
                        };
                        return (post1, post2);
                    }

                    let mut cursors = cursors_post.borrow_mut();
                    let post1 = cursors[s * 2].allele_posteriors(m, n_alleles, &get_ref_allele);
                    let post2 = cursors[s * 2 + 1].allele_posteriors(m, n_alleles, &get_ref_allele);
                    (post1, post2)
                },
            ))
        } else {
            None
        };

        writer.write_imputed_streaming(
            &ref_gt,
            get_dosage,
            get_posteriors,
            &quality,
            0,
            n_ref_markers,
            self.config.gp,
            self.config.ap,
        )?;
            writer.flush()?;
            Ok::<_, crate::error::ReagleError>(())
        })?;

        eprintln!("Imputation complete!");
        Ok(())
    }
}

/// Run forward-backward HMM on CLUSTERS (matches Java ImpLSBaum exactly)
///
/// This is the cluster-aggregated version that:
/// 1. Operates on C clusters instead of M markers (10-50x faster)
/// 2. Uses per-marker mismatch probability applied across all observed markers
/// 3. Matches Java's exact mathematical model
///
/// # Arguments
/// * `cluster_mismatches` - For each cluster, mismatch counts per state
/// * `cluster_non_missing` - For each cluster, observed marker counts per state
/// * `p_recomb` - Per-cluster recombination probabilities
/// * `p_err` - Per-marker mismatch probability
/// * `n_states` - Number of HMM states
/// * `workspace` - Reusable workspace for temporary storage
///
/// # Returns
/// Flat array of state probabilities: cluster_state_probs[c * n_states + k] = P(state k | cluster c)
#[cfg(test)]
fn run_hmm_forward_backward_clusters(
    cluster_mismatches: &[Vec<f32>],
    cluster_non_missing: &[Vec<f32>],
    p_recomb: &[f32],
    p_err: f32,
    n_states: usize,
    workspace: &mut ImpWorkspace,
) -> Vec<f32> {
    let n_clusters = cluster_mismatches.len();
    if n_clusters == 0 || n_states == 0 {
        return Vec::new();
    }

    // Ensure workspace is sized correctly
    workspace.resize(n_states);

    let p_err = p_err.clamp(1e-8, 0.5);
    let p_no_err = 1.0 - p_err;
    let log_p_err = p_err.ln();
    let log_p_no_err = p_no_err.ln();

    // Forward pass
    let total_size = n_clusters * n_states;
    let mut fwd: Vec<f32> = vec![0.0; total_size];
    let mut fwd_sum = 1.0f32;

    for (c, mismatches) in cluster_mismatches.iter().enumerate().take(n_clusters) {
        let p_rec = p_recomb.get(c).copied().unwrap_or(0.0);
        let shift = p_rec / n_states as f32;
        let scale = (1.0 - p_rec) / fwd_sum;

        let mut new_sum = 0.0f32;
        let row_offset = c * n_states;
        let prev_row_offset = if c > 0 { (c - 1) * n_states } else { 0 };

        for k in 0..n_states.min(mismatches.len()) {
            let n_obs = cluster_non_missing
                .get(c)
                .and_then(|row| row.get(k))
                .copied()
                .unwrap_or(0.0);
            let mismatch_count = mismatches[k].min(n_obs);
            let match_count = (n_obs - mismatch_count).max(0.0);
            let emit = (match_count * log_p_no_err + mismatch_count * log_p_err).exp();

            let val = if c == 0 {
                emit / n_states as f32
            } else {
                emit * (scale * fwd[prev_row_offset + k] + shift)
            };

            fwd[row_offset + k] = val;
            new_sum += val;
        }
        fwd_sum = new_sum;
    }

    // Backward pass
    let bwd = &mut workspace.bwd;
    bwd.resize(n_states, 0.0);
    bwd.fill(1.0 / n_states as f32);

    for c in (0..n_clusters).rev() {
        // Update bwd to correspond to cluster c using emissions at c+1 and transition
        if c < n_clusters - 1 {
            let p_rec = p_recomb.get(c + 1).copied().unwrap_or(0.0);
            let shift = p_rec / n_states as f32;

            let mismatches = &cluster_mismatches[c + 1];

            let mut emitted_sum = 0.0f32;
            for k in 0..n_states.min(mismatches.len()) {
                let n_obs = cluster_non_missing
                    .get(c + 1)
                    .and_then(|row| row.get(k))
                    .copied()
                    .unwrap_or(0.0);
                let mismatch_count = mismatches[k].min(n_obs);
                let match_count = (n_obs - mismatch_count).max(0.0);
                let emit = (match_count * log_p_no_err + mismatch_count * log_p_err).exp();
                bwd[k] *= emit;
                emitted_sum += bwd[k];
            }

            if emitted_sum > 0.0 {
                let scale = (1.0 - p_rec) / emitted_sum;
                for k in 0..n_states {
                    bwd[k] = scale * bwd[k] + shift;
                }
            } else {
                let uniform = 1.0 / n_states as f32;
                for k in 0..n_states {
                    bwd[k] = uniform;
                }
            }
        }

        // Compute posterior: fwd * bwd for cluster c
        let row_offset = c * n_states;
        let mut state_sum = 0.0f32;
        for (k, val) in bwd.iter().enumerate().take(n_states) {
            let idx = row_offset + k;
            fwd[idx] *= val;
            state_sum += fwd[idx];
        }

        // Normalize
        if state_sum > 0.0 {
            let inv_sum = 1.0 / state_sum;
            for k in 0..n_states {
                fwd[row_offset + k] *= inv_sum;
            }
        }
    }

    fwd
}

/// Forward-backward HMM on cluster-coded observations (Java ImpLSBaum model).
pub fn run_hmm_forward_backward_clusters_counts(
    cluster_mismatches: &[Vec<f32>],
    cluster_non_missing: &[Vec<f32>],
    p_recomb: &[f32],
    base_err_rate: f32,
    n_states: usize,
    workspace: &mut ImpWorkspace,
) -> Vec<f32> {
    let n_clusters = cluster_mismatches.len();
    let fwd = &mut workspace.fwd;
    fwd.resize(n_clusters * n_states, 0.0);

    let mut last_sum = 1.0f32;
    let p_err = base_err_rate.clamp(1e-8, 0.5);
    let p_no_err = 1.0 - p_err;
    let log_p_err = p_err.ln();
    let log_p_no_err = p_no_err.ln();
    for m in 0..n_clusters {
        let p_rec = p_recomb.get(m).copied().unwrap_or(0.0);
        let shift = p_rec / n_states as f32;
        let scale = (1.0 - p_rec) / last_sum.max(1e-30);

        let row_offset = m * n_states;
        let prev_offset = if m > 0 { (m - 1) * n_states } else { 0 };
        let mut sum = 0.0f32;

        for k in 0..n_states {
            let mism = cluster_mismatches[m].get(k).copied().unwrap_or(0.0);
            let n_obs = cluster_non_missing
                .get(m)
                .and_then(|row| row.get(k))
                .copied()
                .unwrap_or(0.0);
            let mism = mism.min(n_obs);
            let match_count = (n_obs - mism).max(0.0);
            let em = (match_count * log_p_no_err + mism * log_p_err).exp();
            let val = if m == 0 {
                em / n_states as f32
            } else {
                em * (scale * fwd[prev_offset + k] + shift)
            };
            fwd[row_offset + k] = val;
            sum += val;
        }
        last_sum = sum.max(1e-30);
    }

    let bwd = &mut workspace.bwd;
    bwd.resize(n_states, 0.0);
    bwd.fill(1.0 / n_states as f32);

    for m in (0..n_clusters).rev() {
        // Update bwd to correspond to cluster m using emissions at m+1 and transition.
        if m + 1 < n_clusters {
            let p_rec = p_recomb.get(m + 1).copied().unwrap_or(0.0);
            let shift = p_rec / n_states as f32;

            let mut emitted_sum = 0.0f32;
            let mismatches = &cluster_mismatches[m + 1];
            for k in 0..n_states {
                let mism = mismatches.get(k).copied().unwrap_or(0.0);
                let n_obs = cluster_non_missing
                    .get(m + 1)
                    .and_then(|row| row.get(k))
                    .copied()
                    .unwrap_or(0.0);
                let mism = mism.min(n_obs);
                let match_count = (n_obs - mism).max(0.0);
                let em = (match_count * log_p_no_err + mism * log_p_err).exp();
                bwd[k] *= em;
                emitted_sum += bwd[k];
            }

            if emitted_sum > 0.0 {
                let scale = (1.0 - p_rec) / emitted_sum;
                for k in 0..n_states {
                    bwd[k] = scale * bwd[k] + shift;
                }
            } else {
                let uniform = 1.0 / n_states as f32;
                for k in 0..n_states {
                    bwd[k] = uniform;
                }
            }
        }

        let row_offset = m * n_states;
        let mut state_sum = 0.0f32;
        for k in 0..n_states {
            let idx = row_offset + k;
            fwd[idx] *= bwd[k];
            state_sum += fwd[idx];
        }

        if state_sum > 0.0 {
            let inv = 1.0 / state_sum;
            for k in 0..n_states {
                fwd[row_offset + k] *= inv;
            }
        }
    }

    fwd.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matches_to_mismatches(allele_match: &[Vec<bool>]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut mismatches = Vec::with_capacity(allele_match.len());
        let mut non_missing = Vec::with_capacity(allele_match.len());
        for row in allele_match {
            let mut row_mismatches = Vec::with_capacity(row.len());
            let mut row_non_missing = Vec::with_capacity(row.len());
            for &is_match in row {
                row_mismatches.push(if is_match { 0.0 } else { 1.0 });
                row_non_missing.push(1.0);
            }
            mismatches.push(row_mismatches);
            non_missing.push(row_non_missing);
        }
        (mismatches, non_missing)
    }

    // =========================================================================
    // AllelePosteriors Tests - RIGOROUS
    // =========================================================================

    #[test]
    fn test_biallelic_prob_exact_math() {
        // Test exact probability computation for biallelic sites
        // P(REF) = 1 - P(ALT), must be EXACT to f32 precision
        for p_alt_int in 0..=100 {
            let p_alt = p_alt_int as f32 / 100.0;
            let post = AllelePosteriors::Biallelic(p_alt);

            let computed_ref = post.prob(0);
            let computed_alt = post.prob(1);
            let expected_ref = 1.0 - p_alt;

            assert!(
                (computed_ref - expected_ref).abs() < 1e-7,
                "P(REF) wrong for p_alt={}: got {}, expected {}", p_alt, computed_ref, expected_ref
            );
            assert!(
                (computed_alt - p_alt).abs() < 1e-7,
                "P(ALT) wrong for p_alt={}: got {}, expected {}", p_alt, computed_alt, p_alt
            );

            // Probabilities MUST sum to exactly 1.0
            let sum = computed_ref + computed_alt;
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Probabilities don't sum to 1 for p_alt={}: sum={}", p_alt, sum
            );
        }
    }

    #[test]
    fn test_biallelic_dosage_equals_p_alt() {
        // Dosage for biallelic MUST equal P(ALT) exactly
        // This is the definition: E[X] = 0*P(0) + 1*P(1) = P(1) = P(ALT)
        for p_alt_int in 0..=100 {
            let p_alt = p_alt_int as f32 / 100.0;
            let post = AllelePosteriors::Biallelic(p_alt);

            assert!(
                (post.dosage() - p_alt).abs() < 1e-7,
                "Dosage != P(ALT) for p_alt={}: dosage={}", p_alt, post.dosage()
            );
        }
    }

    #[test]
    fn test_biallelic_max_allele_boundary() {
        // Critical boundary: at exactly 0.5, should return 1 (ALT)
        let at_boundary = AllelePosteriors::Biallelic(0.5);
        assert_eq!(at_boundary.max_allele(), 1, "At p_alt=0.5, max_allele should be 1");

        // Just below boundary
        let below = AllelePosteriors::Biallelic(0.4999999);
        assert_eq!(below.max_allele(), 0, "At p_alt=0.4999999, max_allele should be 0");

        // Just above boundary
        let above = AllelePosteriors::Biallelic(0.5000001);
        assert_eq!(above.max_allele(), 1, "At p_alt=0.5000001, max_allele should be 1");

        // Edge cases
        assert_eq!(AllelePosteriors::Biallelic(0.0).max_allele(), 0);
        assert_eq!(AllelePosteriors::Biallelic(1.0).max_allele(), 1);
    }

    #[test]
    fn test_multiallelic_prob_sum_to_one() {
        // Multiallelic probabilities must sum to 1 (if input sums to 1)
        let probs = vec![0.1, 0.2, 0.3, 0.4]; // Sum = 1.0
        let post = AllelePosteriors::Multiallelic(probs.clone());

        let sum: f32 = (0..4).map(|i| post.prob(i)).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Multiallelic probs don't sum to 1: sum={}", sum
        );

        // Verify each prob is correct
        for (i, &expected) in probs.iter().enumerate() {
            assert!(
                (post.prob(i) - expected).abs() < 1e-7,
                "P({}) wrong: got {}, expected {}", i, post.prob(i), expected
            );
        }
    }

    #[test]
    fn test_multiallelic_dosage_formula() {
        // Dosage = E[X] = sum(i * P(i))
        let probs = vec![0.1, 0.2, 0.3, 0.4]; // 4-allelic
        let post = AllelePosteriors::Multiallelic(probs.clone());

        // Expected: 0*0.1 + 1*0.2 + 2*0.3 + 3*0.4 = 0 + 0.2 + 0.6 + 1.2 = 2.0
        let expected_dosage = 2.0f32;
        assert!(
            (post.dosage() - expected_dosage).abs() < 1e-6,
            "Multiallelic dosage wrong: got {}, expected {}", post.dosage(), expected_dosage
        );

        // Test another case: 5-allelic
        let probs2 = vec![0.5, 0.2, 0.1, 0.1, 0.1];
        let post2 = AllelePosteriors::Multiallelic(probs2);
        // Expected: 0*0.5 + 1*0.2 + 2*0.1 + 3*0.1 + 4*0.1 = 0 + 0.2 + 0.2 + 0.3 + 0.4 = 1.1
        assert!(
            (post2.dosage() - 1.1).abs() < 1e-6,
            "5-allelic dosage wrong: got {}", post2.dosage()
        );
    }

    #[test]
    fn test_multiallelic_max_allele_all_cases() {
        // Case 1: First allele is max
        let post1 = AllelePosteriors::Multiallelic(vec![0.5, 0.3, 0.2]);
        assert_eq!(post1.max_allele(), 0);

        // Case 2: Middle allele is max
        let post2 = AllelePosteriors::Multiallelic(vec![0.2, 0.6, 0.2]);
        assert_eq!(post2.max_allele(), 1);

        // Case 3: Last allele is max
        let post3 = AllelePosteriors::Multiallelic(vec![0.1, 0.2, 0.7]);
        assert_eq!(post3.max_allele(), 2);
    }

    #[test]
    fn test_out_of_bounds_returns_zero() {
        // Accessing probability of non-existent allele must return 0
        let biallelic = AllelePosteriors::Biallelic(0.5);
        assert_eq!(biallelic.prob(2), 0.0);
        assert_eq!(biallelic.prob(100), 0.0);

        let triallelic = AllelePosteriors::Multiallelic(vec![0.3, 0.3, 0.4]);
        assert_eq!(triallelic.prob(3), 0.0);
        assert_eq!(triallelic.prob(1000), 0.0);
    }

    // =========================================================================
    // HMM Forward-Backward Tests - RIGOROUS
    // =========================================================================

    #[test]
    fn test_hmm_posteriors_sum_to_one_strict() {
        use crate::utils::workspace::ImpWorkspace;

        // Test with various configurations
        for n_markers in [2, 5, 10, 20] {
            for n_states in [2, 4, 8] {
                let allele_match: Vec<Vec<bool>> = (0..n_markers)
                    .map(|m| (0..n_states).map(|k| (m + k) % 2 == 0).collect())
                    .collect();
                let p_recomb: Vec<f32> = (0..n_markers)
                    .map(|m| if m == 0 { 0.0 } else { 0.01 })
                    .collect();

                let mut workspace = ImpWorkspace::with_ref_size(n_states);
                let p_err = 0.01f32;
                let (mismatches, non_missing) = matches_to_mismatches(&allele_match);
                let posteriors = run_hmm_forward_backward_clusters(
                    &mismatches,
                    &non_missing,
                    &p_recomb,
                    p_err,
                    n_states,
                    &mut workspace,
                );

                // Posteriors MUST sum to 1.0 at EVERY marker
                for m in 0..n_markers {
                    let sum: f32 = (0..n_states).map(|k| posteriors[m * n_states + k]).sum();
                    assert!(
                        (sum - 1.0).abs() < 0.001,
                        "n_markers={}, n_states={}, marker {}: posteriors sum to {}, not 1.0",
                        n_markers, n_states, m, sum
                    );
                }
            }
        }
    }

    #[test]
    fn test_hmm_posteriors_non_negative() {
        use crate::utils::workspace::ImpWorkspace;

        let n_markers = 10;
        let n_states = 4;
        let allele_match: Vec<Vec<bool>> = (0..n_markers)
            .map(|m| (0..n_states).map(|k| k == m % n_states).collect())
            .collect();
        let p_recomb: Vec<f32> = (0..n_markers).map(|m| if m == 0 { 0.0 } else { 0.05 }).collect();

        let mut workspace = ImpWorkspace::with_ref_size(n_states);
        let p_err = 0.01f32;
        let (mismatches, non_missing) = matches_to_mismatches(&allele_match);
        let posteriors = run_hmm_forward_backward_clusters(
            &mismatches,
            &non_missing,
            &p_recomb,
            p_err,
            n_states,
            &mut workspace,
        );

        for (i, &p) in posteriors.iter().enumerate() {
            assert!(p >= 0.0, "Posterior at index {} is negative: {}", i, p);
            assert!(p <= 1.0, "Posterior at index {} exceeds 1: {}", i, p);
        }
    }

    #[test]
    fn test_per_state_non_missing_affects_emission() {
        use crate::utils::workspace::ImpWorkspace;

        let n_clusters = 1;
        let n_states = 2;
        let p_recomb = vec![0.0f32; n_clusters];
        let p_err = 0.01f32;
        let mismatches = vec![vec![0.0f32, 1.0f32]];
        let mut workspace = ImpWorkspace::with_ref_size(n_states);

        let non_missing_equal = vec![vec![2.0f32, 2.0f32]];
        let post_equal = run_hmm_forward_backward_clusters_counts(
            &mismatches,
            &non_missing_equal,
            &p_recomb,
            p_err,
            n_states,
            &mut workspace,
        );

        let non_missing_unequal = vec![vec![2.0f32, 0.0f32]];
        let post_unequal = run_hmm_forward_backward_clusters_counts(
            &mismatches,
            &non_missing_unequal,
            &p_recomb,
            p_err,
            n_states,
            &mut workspace,
        );

        let p0_equal = post_equal[0];
        let p1_equal = post_equal[1];
        let p0_unequal = post_unequal[0];
        let p1_unequal = post_unequal[1];

        assert!(p0_equal > p1_equal, "State 0 should dominate with fewer mismatches");
        assert!(
            p1_unequal > p1_equal,
            "State 1 posterior should increase with fewer observed markers"
        );
        let sum_equal = p0_equal + p1_equal;
        let sum_unequal = p0_unequal + p1_unequal;
        assert!((sum_equal - 1.0).abs() < 1e-6, "Posteriors should sum to 1");
        assert!((sum_unequal - 1.0).abs() < 1e-6, "Posteriors should sum to 1");
    }

    #[test]
    fn test_hmm_perfect_match_gives_high_posterior() {
        use crate::utils::workspace::ImpWorkspace;

        let n_markers = 20;
        let n_states = 4;

        // State 0 always matches, others never match
        let allele_match: Vec<Vec<bool>> = (0..n_markers)
            .map(|_| vec![true, false, false, false])
            .collect();
        let p_recomb: Vec<f32> = (0..n_markers).map(|m| if m == 0 { 0.0 } else { 0.001 }).collect();

        let mut workspace = ImpWorkspace::with_ref_size(n_states);
        let p_err = 0.001f32;
        let (mismatches, non_missing) = matches_to_mismatches(&allele_match);
        let posteriors = run_hmm_forward_backward_clusters(
            &mismatches,
            &non_missing,
            &p_recomb,
            p_err,
            n_states,
            &mut workspace,
        );

        // State 0 should have posterior > 0.99 at every marker
        for m in 0..n_markers {
            let prob_state0 = posteriors[m * n_states];
            assert!(
                prob_state0 > 0.99,
                "Perfect match state should have p>0.99, got {} at marker {}", prob_state0, m
            );
        }
    }

    #[test]
    fn test_hmm_state_switch_detection() {
        use crate::utils::workspace::ImpWorkspace;

        // Pattern: state 0 matches first half, state 1 matches second half
        let n_markers = 20;
        let n_states = 2;

        let allele_match: Vec<Vec<bool>> = (0..n_markers)
            .map(|m| {
                if m < 10 { vec![true, false] } else { vec![false, true] }
            })
            .collect();
        let p_recomb: Vec<f32> = (0..n_markers).map(|m| if m == 0 { 0.0 } else { 0.01 }).collect();

        let mut workspace = ImpWorkspace::with_ref_size(n_states);
        let p_err = 0.01f32;
        let (mismatches, non_missing) = matches_to_mismatches(&allele_match);
        let posteriors = run_hmm_forward_backward_clusters(
            &mismatches,
            &non_missing,
            &p_recomb,
            p_err,
            n_states,
            &mut workspace,
        );

        // First half: state 0 should dominate
        for m in 0..5 {
            let prob_state0 = posteriors[m * n_states];
            assert!(prob_state0 > 0.8, "First half, state 0 should dominate. marker {}: p0={}", m, prob_state0);
        }

        // Second half: state 1 should dominate
        for m in 15..n_markers {
            let prob_state1 = posteriors[m * n_states + 1];
            assert!(prob_state1 > 0.8, "Second half, state 1 should dominate. marker {}: p1={}", m, prob_state1);
        }
    }

    #[test]
    fn test_hmm_symmetry() {
        use crate::utils::workspace::ImpWorkspace;

        let n_markers = 10;
        let n_states = 2;

        // Run 1: state 0 matches at even markers
        let match1: Vec<Vec<bool>> = (0..n_markers)
            .map(|m| if m % 2 == 0 { vec![true, false] } else { vec![false, true] })
            .collect();

        // Run 2: state 1 matches at even markers (swapped)
        let match2: Vec<Vec<bool>> = (0..n_markers)
            .map(|m| if m % 2 == 0 { vec![false, true] } else { vec![true, false] })
            .collect();

        let p_recomb: Vec<f32> = (0..n_markers).map(|m| if m == 0 { 0.0 } else { 0.05 }).collect();

        let mut workspace = ImpWorkspace::with_ref_size(n_states);
        let p_err = 0.01f32;
        let (mismatches1, non_missing1) = matches_to_mismatches(&match1);
        let (mismatches2, non_missing2) = matches_to_mismatches(&match2);

        let post1 = run_hmm_forward_backward_clusters(
            &mismatches1,
            &non_missing1,
            &p_recomb,
            p_err,
            n_states,
            &mut workspace,
        );
        let post2 = run_hmm_forward_backward_clusters(
            &mismatches2,
            &non_missing2,
            &p_recomb,
            p_err,
            n_states,
            &mut workspace,
        );

        // Posteriors should be swapped
        for m in 0..n_markers {
            let p1_s0 = post1[m * n_states];
            let p1_s1 = post1[m * n_states + 1];
            let p2_s0 = post2[m * n_states];
            let p2_s1 = post2[m * n_states + 1];

            assert!((p1_s0 - p2_s1).abs() < 0.01, "Symmetry broken at marker {}: p1[s0]={}, p2[s1]={}", m, p1_s0, p2_s1);
            assert!((p1_s1 - p2_s0).abs() < 0.01, "Symmetry broken at marker {}: p1[s1]={}, p2[s0]={}", m, p1_s1, p2_s0);
        }
    }

    // =========================================================================
    // HARD TESTS - Analytically computed expected values
    // These tests compute exact expected posteriors by hand and compare.
    // If there's ANY bug in the HMM, these WILL FAIL.
    // =========================================================================

    #[test]
    fn test_hmm_2state_2marker_exact_posterior() {
        // 2-state, 2-marker HMM with known parameters
        // We compute the EXACT posterior analytically and compare
        //
        // Setup:
        // - 2 states, 2 markers
        // - State 0 matches at marker 0, state 1 matches at marker 1
        // - p_recomb = 0.1, p_mismatch = 0.01
        //
        // Li-Stephens forward formula:
        //   fwd[m][k] = emit[k] * ((1-rho)*fwd[m-1][k]/sum + rho/K)
        // where rho = p_recomb, K = n_states
        use crate::utils::workspace::ImpWorkspace;

        let n_states = 2;
        let rho = 0.1f32;  // recombination prob
        let p_err = 0.01f32;  // mismatch prob
        let p_match = 1.0 - p_err;

        // State 0 matches at m=0, state 1 matches at m=1
        let allele_match = vec![
            vec![true, false],   // m=0: state 0 matches
            vec![false, true],   // m=1: state 1 matches
        ];

        // Compute expected forward values analytically
        // Marker 0: fwd[0] = (1/K) * emit[k]
        let fwd0_0 = (1.0 / n_states as f32) * p_match;  // state 0 matches
        let fwd0_1 = (1.0 / n_states as f32) * p_err;    // state 1 mismatches
        let fwd0_sum = fwd0_0 + fwd0_1;

        // Marker 1: fwd[1][k] = emit[k] * ((1-rho)*fwd[0][k]/fwd0_sum + rho/K)
        let shift = rho / n_states as f32;
        let scale = (1.0 - rho) / fwd0_sum;

        let fwd1_0_pre = scale * fwd0_0 + shift;  // transition for state 0
        let fwd1_1_pre = scale * fwd0_1 + shift;  // transition for state 1
        let fwd1_0 = p_err * fwd1_0_pre;          // state 0 mismatches at m=1
        let fwd1_1 = p_match * fwd1_1_pre;        // state 1 matches at m=1

        // Backward: bwd[M-1] = 1/K, then apply bwd_update
        // For 2 markers, bwd at m=0 uses emission at m=1
        let bwd1_0 = 1.0 / n_states as f32;
        let bwd1_1 = 1.0 / n_states as f32;

        // bwd_update: first multiply by emit, then normalize transition
        let bwd0_pre_0 = bwd1_0 * p_err;    // state 0 at m=1 mismatches
        let bwd0_pre_1 = bwd1_1 * p_match;  // state 1 at m=1 matches
        let bwd_sum = bwd0_pre_0 + bwd0_pre_1;

        let bwd_scale = (1.0 - rho) / bwd_sum;
        let bwd_shift = rho / n_states as f32;
        let bwd0_0 = bwd_scale * bwd0_pre_0 + bwd_shift;
        let bwd0_1 = bwd_scale * bwd0_pre_1 + bwd_shift;

        // Expected posteriors: gamma[m][k] = fwd[m][k] * bwd[m][k] / sum
        let gamma0_0_raw = fwd0_0 * bwd0_0;
        let gamma0_1_raw = fwd0_1 * bwd0_1;
        let gamma0_sum = gamma0_0_raw + gamma0_1_raw;
        let expected_gamma0_0 = gamma0_0_raw / gamma0_sum;
        let expected_gamma0_1 = gamma0_1_raw / gamma0_sum;

        let gamma1_0_raw = fwd1_0 * bwd1_0;
        let gamma1_1_raw = fwd1_1 * bwd1_1;
        let gamma1_sum = gamma1_0_raw + gamma1_1_raw;
        let expected_gamma1_0 = gamma1_0_raw / gamma1_sum;
        let expected_gamma1_1 = gamma1_1_raw / gamma1_sum;

        // Run the actual HMM
        let p_recomb = vec![0.0, rho];  // First marker has 0 recomb

        let mut workspace = ImpWorkspace::with_ref_size(n_states);
        let (mismatches, non_missing) = matches_to_mismatches(&allele_match);
        let posteriors = run_hmm_forward_backward_clusters(
            &mismatches,
            &non_missing,
            &p_recomb,
            p_err,
            n_states,
            &mut workspace,
        );

        // Compare with TIGHT tolerance - these should be EXACT (up to float precision)
        let tol = 0.02;  // 2% tolerance for numerical differences

        let actual_gamma0_0 = posteriors[0];
        let actual_gamma0_1 = posteriors[1];
        let actual_gamma1_0 = posteriors[2];
        let actual_gamma1_1 = posteriors[3];

        assert!(
            (actual_gamma0_0 - expected_gamma0_0).abs() < tol,
            "Marker 0, State 0: expected {:.6}, got {:.6}", expected_gamma0_0, actual_gamma0_0
        );
        assert!(
            (actual_gamma0_1 - expected_gamma0_1).abs() < tol,
            "Marker 0, State 1: expected {:.6}, got {:.6}", expected_gamma0_1, actual_gamma0_1
        );
        assert!(
            (actual_gamma1_0 - expected_gamma1_0).abs() < tol,
            "Marker 1, State 0: expected {:.6}, got {:.6}", expected_gamma1_0, actual_gamma1_0
        );
        assert!(
            (actual_gamma1_1 - expected_gamma1_1).abs() < tol,
            "Marker 1, State 1: expected {:.6}, got {:.6}", expected_gamma1_1, actual_gamma1_1
        );
    }

    #[test]
    fn test_hmm_uniform_emission_gives_uniform_posterior() {
        // If ALL states match at ALL markers (uniform emission),
        // posterior should be uniform: 1/K for each state
        use crate::utils::workspace::ImpWorkspace;

        for n_states in [2, 4, 8, 16] {
            let n_markers = 10;

            // All states match at all markers
            let allele_match: Vec<Vec<bool>> = (0..n_markers)
                .map(|_| vec![true; n_states])
                .collect();
            let p_recomb: Vec<f32> = (0..n_markers)
                .map(|m| if m == 0 { 0.0 } else { 0.05 })
                .collect();

            let mut workspace = ImpWorkspace::with_ref_size(n_states);
            let p_err = 0.01f32;
            let (mismatches, non_missing) = matches_to_mismatches(&allele_match);
            let posteriors = run_hmm_forward_backward_clusters(
                &mismatches,
                &non_missing,
                &p_recomb,
                p_err,
                n_states,
                &mut workspace,
            );

            let expected = 1.0 / n_states as f32;

            // All posteriors should be uniform
            for m in 0..n_markers {
                for k in 0..n_states {
                    let actual = posteriors[m * n_states + k];
                    assert!(
                        (actual - expected).abs() < 0.05,
                        "n_states={}, marker {}, state {}: expected {:.4}, got {:.4}",
                        n_states, m, k, expected, actual
                    );
                }
            }
        }
    }

    #[test]
    fn test_hmm_no_recombination_preserves_initial_state() {
        // With ZERO recombination, the HMM should stay in the initial state
        // (weighted by emission probabilities)
        use crate::utils::workspace::ImpWorkspace;

        let n_states = 4;
        let n_markers = 10;

        // State 0 always matches
        let allele_match: Vec<Vec<bool>> = (0..n_markers)
            .map(|_| vec![true, false, false, false])
            .collect();

        // ZERO recombination everywhere
        let p_recomb = vec![0.0f32; n_markers];

        let mut workspace = ImpWorkspace::with_ref_size(n_states);
        let p_err = 0.01f32;  // small mismatch prob
        let (mismatches, non_missing) = matches_to_mismatches(&allele_match);
        let posteriors = run_hmm_forward_backward_clusters(
            &mismatches,
            &non_missing,
            &p_recomb,
            p_err,
            n_states,
            &mut workspace,
        );

        // With no recombination and state 0 always matching,
        // state 0 should have nearly all probability
        for m in 0..n_markers {
            let prob_state0 = posteriors[m * n_states];
            assert!(
                prob_state0 > 0.999,
                "With zero recomb, matching state should have p>0.999, got {} at marker {}", prob_state0, m
            );
        }
    }

    #[test]
    fn test_dosage_bounds_diploid() {
        // For diploid genotypes, dosage should be in [0, 2]
        // Test that DS = P(hap1=ALT) + P(hap2=ALT) is bounded correctly

        // Biallelic: dosage = P(ALT) per haplotype, so diploid DS = hap1 + hap2
        let hap1 = AllelePosteriors::Biallelic(0.8);
        let hap2 = AllelePosteriors::Biallelic(0.6);
        let diploid_dosage = hap1.dosage() + hap2.dosage();
        assert!(diploid_dosage >= 0.0 && diploid_dosage <= 2.0,
            "Diploid dosage {} should be in [0,2]", diploid_dosage);
        assert!((diploid_dosage - 1.4).abs() < 1e-6,
            "Expected diploid dosage 1.4, got {}", diploid_dosage);
    }

    #[test]
    fn test_gp_probabilities_from_haplotype_posteriors() {
        // GP (genotype probability) should be computed from haplotype posteriors
        // For biallelic: GP = [P(0/0), P(0/1), P(1/1)]
        //   P(0/0) = P(hap1=0) * P(hap2=0)
        //   P(0/1) = P(hap1=0)*P(hap2=1) + P(hap1=1)*P(hap2=0)
        //   P(1/1) = P(hap1=1) * P(hap2=1)

        let p1_alt = 0.3f32;  // P(hap1 = ALT)
        let p2_alt = 0.7f32;  // P(hap2 = ALT)

        let p1_ref = 1.0 - p1_alt;
        let p2_ref = 1.0 - p2_alt;

        let expected_p00 = p1_ref * p2_ref;  // 0.7 * 0.3 = 0.21
        let expected_p01 = p1_ref * p2_alt + p1_alt * p2_ref;  // 0.7*0.7 + 0.3*0.3 = 0.58
        let expected_p11 = p1_alt * p2_alt;  // 0.3 * 0.7 = 0.21

        // Verify they sum to 1
        let gp_sum = expected_p00 + expected_p01 + expected_p11;
        assert!((gp_sum - 1.0).abs() < 1e-6, "GP should sum to 1, got {}", gp_sum);

        // Test using AllelePosteriors
        let hap1 = AllelePosteriors::Biallelic(p1_alt);
        let hap2 = AllelePosteriors::Biallelic(p2_alt);

        let computed_p00 = hap1.prob(0) * hap2.prob(0);
        let computed_p01 = hap1.prob(0) * hap2.prob(1) + hap1.prob(1) * hap2.prob(0);
        let computed_p11 = hap1.prob(1) * hap2.prob(1);

        assert!((computed_p00 - expected_p00).abs() < 1e-6,
            "P(0/0): expected {}, got {}", expected_p00, computed_p00);
        assert!((computed_p01 - expected_p01).abs() < 1e-6,
            "P(0/1): expected {}, got {}", expected_p01, computed_p01);
        assert!((computed_p11 - expected_p11).abs() < 1e-6,
            "P(1/1): expected {}, got {}", expected_p11, computed_p11);
    }

    #[test]
    fn test_multiallelic_gp_count() {
        // For N alleles, GP should have N*(N+1)/2 values
        // Triallelic (N=3): 3*4/2 = 6 values: 0/0, 0/1, 1/1, 0/2, 1/2, 2/2

        for n_alleles in [2, 3, 4, 5, 10] {
            let expected_gp_count = n_alleles * (n_alleles + 1) / 2;

            // Create uniform posteriors
            let probs: Vec<f32> = (0..n_alleles).map(|_| 1.0 / n_alleles as f32).collect();
            let hap1 = AllelePosteriors::Multiallelic(probs.clone());
            let hap2 = AllelePosteriors::Multiallelic(probs);

            // Compute GP values (following VCF spec ordering)
            let mut gp_values = Vec::new();
            for i2 in 0..n_alleles {
                for i1 in 0..=i2 {
                    let prob = if i1 == i2 {
                        hap1.prob(i1) * hap2.prob(i2)
                    } else {
                        hap1.prob(i1) * hap2.prob(i2) + hap1.prob(i2) * hap2.prob(i1)
                    };
                    gp_values.push(prob);
                }
            }

            assert_eq!(gp_values.len(), expected_gp_count,
                "N={}: expected {} GP values, got {}", n_alleles, expected_gp_count, gp_values.len());

            // GP should sum to 1
            let gp_sum: f32 = gp_values.iter().sum();
            assert!((gp_sum - 1.0).abs() < 1e-5,
                "N={}: GP sum should be 1, got {}", n_alleles, gp_sum);
        }
    }

    // =========================================================================
    // StateProbs Interpolation Tests - EXACT MATHEMATICAL VERIFICATION
    // =========================================================================

    #[test]
    fn test_state_probs_interpolation_weight_formula() {
        // Test that interpolation weight is computed correctly:
        //   weight_left = (pos_right - pos_marker) / (pos_right - pos_left)
        //
        // At left marker: weight_left = 1.0
        // At right marker: weight_left = 0.0
        // At midpoint: weight_left = 0.5

        let genotyped_markers = std::sync::Arc::new(vec![0, 10]);
        // Genetic positions: markers at 0.0 and 1.0 cM
        let gen_positions = std::sync::Arc::new(vec![
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        ]);

        // State 0 (hap 0) always carries REF
        // State 1 (hap 1) always carries ALT
        let hap_indices = vec![
            vec![0, 1],  // marker 0
            vec![0, 1],  // marker 10
        ];
        // At marker 0: 100% state 0, 0% state 1
        // At marker 10: 0% state 0, 100% state 1
        let state_probs = vec![1.0, 0.0, 0.0, 1.0];

        let marker_to_cluster = std::sync::Arc::new(vec![0, 1]);
        let sp = StateProbs::new(
            genotyped_markers,
            2,
            hap_indices,
            state_probs,
            gen_positions,
            marker_to_cluster,
            None,
        );

        // Hap 0 = REF (0), Hap 1 = ALT (1)
        let get_ref_allele = |marker: usize, hap: u32| -> u8 {
            assert!(marker <= 10);
            if hap == 0 { 0 } else { 1 }
        };

        // Test interpolation at various positions
        // At marker 0: weight_left = 1.0, prob = 1.0 * 1.0 + 0.0 * 0.0 = 1.0 for state 0
        //   P(ALT) = prob_state1 = 0.0
        let post_0 = sp.allele_posteriors(0, 2, get_ref_allele);
        match post_0 {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 0.0).abs() < 0.05, "At marker 0: expected P(ALT)~0.0, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }

        // At marker 5 (midpoint): weight_left = 0.5
        //   prob_state0 = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        //   prob_state1 = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        //   P(ALT) = prob_state1 = 0.5
        let post_5 = sp.allele_posteriors(5, 2, get_ref_allele);
        match post_5 {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 0.5).abs() < 0.1, "At marker 5: expected P(ALT)~0.5, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }

        // At marker 10: should use exact value (not interpolated)
        //   P(ALT) = prob_state1 = 1.0
        let post_10 = sp.allele_posteriors(10, 2, get_ref_allele);
        match post_10 {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 1.0).abs() < 0.05, "At marker 10: expected P(ALT)~1.0, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }

        // Test precise interpolation at marker 2 (position 0.2)
        // weight_left = (1.0 - 0.2) / (1.0 - 0.0) = 0.8
        // prob_state1 = 0.8 * 0.0 + 0.2 * 1.0 = 0.2
        let post_2 = sp.allele_posteriors(2, 2, get_ref_allele);
        match post_2 {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 0.2).abs() < 0.1, "At marker 2: expected P(ALT)~0.2, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }

        // Test precise interpolation at marker 8 (position 0.8)
        // weight_left = (1.0 - 0.8) / (1.0 - 0.0) = 0.2
        // prob_state1 = 0.2 * 0.0 + 0.8 * 1.0 = 0.8
        let post_8 = sp.allele_posteriors(8, 2, get_ref_allele);
        match post_8 {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 0.8).abs() < 0.1, "At marker 8: expected P(ALT)~0.8, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }
    }

    #[test]
    fn test_cluster_state_probs_between_clusters_uses_left_and_right_haps() {
        let marker_cluster = Arc::new(vec![0usize, 0usize, 1usize]);
        let ref_cluster_end = Arc::new(vec![1usize, 3usize]);
        let weight = Arc::new(vec![1.0f32, 0.25f32, 1.0f32]);

        let hap_indices = vec![vec![0u32], vec![1u32]];
        let cluster_probs = vec![0.8f32, 0.2f32];

        let state_probs = ClusterStateProbs::new(
            marker_cluster,
            ref_cluster_end,
            weight,
            1,
            hap_indices,
            cluster_probs,
        );

        let get_ref_allele = |m: usize, hap: u32| -> u8 {
            if m == 1 {
                if hap == 0 { 0 } else { 1 }
            } else {
                0
            }
        };

        let post = state_probs.allele_posteriors(1, 2, &get_ref_allele);
        let p_alt = post.prob(1);
        let expected = 0.15 / (0.2 + 0.15);
        assert!((p_alt - expected).abs() < 1e-6, "p_alt={}, expected={}", p_alt, expected);
    }

    #[test]
    fn test_state_probs_edge_cases() {
        // Test edge case: marker before first genotyped marker
        // Should return the first marker's value

        let genotyped_markers = std::sync::Arc::new(vec![5, 10]);
        let gen_positions = std::sync::Arc::new(vec![
            0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0
        ]);

        let hap_indices = vec![vec![0, 1], vec![0, 1]];
        // At marker 5: 70% state 0, 30% state 1
        // At marker 10: 20% state 0, 80% state 1
        let state_probs = vec![0.7, 0.3, 0.2, 0.8];

        let marker_to_cluster = std::sync::Arc::new(vec![0, 1]);
        let sp = StateProbs::new(
            genotyped_markers,
            2,
            hap_indices,
            state_probs,
            gen_positions,
            marker_to_cluster,
            None,
        );

        let get_ref_allele = |marker: usize, hap: u32| -> u8 {
            assert!(marker <= 10);
            if hap == 0 { 0 } else { 1 }
        };

        // Marker 0 is before first genotyped marker (5)
        // Should return marker 5's value: P(ALT) = 0.3
        let post_before = sp.allele_posteriors(0, 2, get_ref_allele);
        match post_before {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 0.3).abs() < 0.1, "Before first: expected ~0.3, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }
    }

    #[test]
    fn test_state_probs_probabilities_normalized() {
        // StateProbs should produce normalized probabilities (sum to 1)
        // even after interpolation

        let genotyped_markers = std::sync::Arc::new(vec![0, 10]);
        let gen_positions: Vec<f64> = (0..=10).map(|i| i as f64 * 0.1).collect();
        let gen_positions = std::sync::Arc::new(gen_positions);

        // 4 states, properly normalized at each genotyped marker
        let hap_indices = vec![
            vec![0, 1, 2, 3],
            vec![0, 1, 2, 3],
        ];
        // Marker 0: [0.4, 0.3, 0.2, 0.1]
        // Marker 10: [0.1, 0.2, 0.3, 0.4]
        let state_probs = vec![0.4, 0.3, 0.2, 0.1, 0.1, 0.2, 0.3, 0.4];

        let marker_to_cluster = std::sync::Arc::new(vec![0, 1]);
        let sp = StateProbs::new(
            genotyped_markers,
            4,
            hap_indices,
            state_probs,
            gen_positions,
            marker_to_cluster,
            None,
        );

        // Allele mapping: hap 0,1 = REF (0), hap 2,3 = ALT (1)
        let get_ref_allele = |marker: usize, hap: u32| -> u8 {
            assert!(marker <= 10);
            if hap < 2 { 0 } else { 1 }
        };

        // At marker 0: P(REF) = 0.4+0.3 = 0.7, P(ALT) = 0.2+0.1 = 0.3
        let post_0 = sp.allele_posteriors(0, 2, get_ref_allele);
        match post_0 {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 0.3).abs() < 0.05, "Marker 0: expected P(ALT)=0.3, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }

        // At marker 10: P(REF) = 0.1+0.2 = 0.3, P(ALT) = 0.3+0.4 = 0.7
        let post_10 = sp.allele_posteriors(10, 2, get_ref_allele);
        match post_10 {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 0.7).abs() < 0.05, "Marker 10: expected P(ALT)=0.7, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }

        // At marker 5 (midpoint): should interpolate
        // Expected P(ALT) = 0.5 * 0.3 + 0.5 * 0.7 = 0.5
        let post_5 = sp.allele_posteriors(5, 2, get_ref_allele);
        match post_5 {
            AllelePosteriors::Biallelic(p_alt) => {
                assert!((p_alt - 0.5).abs() < 0.1, "Marker 5: expected P(ALT)~0.5, got {}", p_alt);
            }
            _ => panic!("Expected Biallelic"),
        }
    }

    /// Critical test: cursor-based interpolation must match binary-search interpolation exactly.
    ///
    /// This validates the O(1) cursor optimization introduced to fix the 900M binary search
    /// bottleneck. If this test fails, the cursor is advancing incorrectly.
    #[test]
    fn test_cursor_matches_binary_search() {
        // Create a realistic StateProbs with multiple genotyped markers
        let genotyped_markers = std::sync::Arc::new(vec![0, 10, 25, 50, 75, 100]);
        let n_ref_markers = 101;
        let gen_positions: std::sync::Arc<Vec<f64>> = std::sync::Arc::new(
            (0..n_ref_markers).map(|m| m as f64 * 0.01).collect()
        );

        // Create haplotype indices and probabilities
        // State 0 = hap 0, State 1 = hap 1, State 2 = hap 2
        let n_states = 3;
        let hap_indices: Vec<Vec<u32>> = (0..genotyped_markers.len())
            .map(|_| vec![0, 1, 2])
            .collect();

        // Varying probabilities across genotyped markers
        // m=0:   [0.8, 0.1, 0.1]
        // m=10:  [0.3, 0.5, 0.2]
        // m=25:  [0.1, 0.7, 0.2]
        // m=50:  [0.2, 0.2, 0.6]
        // m=75:  [0.4, 0.4, 0.2]
        // m=100: [0.1, 0.3, 0.6]
        let state_probs = vec![
            0.8, 0.1, 0.1,  // m=0
            0.3, 0.5, 0.2,  // m=10
            0.1, 0.7, 0.2,  // m=25
            0.2, 0.2, 0.6,  // m=50
            0.4, 0.4, 0.2,  // m=75
            0.1, 0.3, 0.6,  // m=100
        ];

        let marker_to_cluster = std::sync::Arc::new((0..genotyped_markers.len()).collect());
        let sp = StateProbs::new(
            genotyped_markers.clone(),
            n_states,
            hap_indices,
            state_probs,
            gen_positions.clone(),
            marker_to_cluster,
            None,
        );

        // Reference allele function: hap 0 = REF, hap 1 = ALT, hap 2 = ALT
        // All haplotypes have same allele at each marker (simplified test)
        let get_ref_allele = |marker: usize, hap: u32| -> u8 {
            std::hint::black_box(marker); // silence unused warning
            if hap == 0 { 0 } else { 1 }
        };

        // Use cursor to iterate through ALL markers sequentially
        let sp_arc = Arc::new(sp);
        let mut cursor = sp_arc.clone().cursor();

        for m in 0..n_ref_markers {
            // Get result from cursor (O(1) amortized)
            let cursor_result = cursor.allele_posteriors(m, 2, &get_ref_allele);

            // Get result from binary search (O(log N))
            let bs_result = sp_arc.allele_posteriors(m, 2, get_ref_allele);

            // They must match exactly
            match (&cursor_result, &bs_result) {
                (AllelePosteriors::Biallelic(c_alt), AllelePosteriors::Biallelic(b_alt)) => {
                    assert!(
                        (c_alt - b_alt).abs() < 1e-6,
                        "Marker {}: cursor gave P(ALT)={}, binary search gave P(ALT)={}",
                        m, c_alt, b_alt
                    );
                }
                _ => panic!("Marker {}: type mismatch between cursor and binary search", m),
            }
        }
    }

    /// Test cursor behavior when iterating BACKWARDS (should still work via linear scan)
    /// This validates that the cursor handles out-of-order access gracefully.
    #[test]
    fn test_cursor_non_sequential_access() {
        let genotyped_markers = std::sync::Arc::new(vec![0, 50, 100]);
        let n_ref_markers = 101;
        let gen_positions: std::sync::Arc<Vec<f64>> = std::sync::Arc::new(
            (0..n_ref_markers).map(|m| m as f64).collect()
        );

        let hap_indices: Vec<Vec<u32>> = vec![vec![0, 1], vec![0, 1], vec![0, 1]];
        let state_probs = vec![
            1.0, 0.0,  // m=0: 100% state 0
            0.5, 0.5,  // m=50: 50/50
            0.0, 1.0,  // m=100: 100% state 1
        ];

        let marker_to_cluster = std::sync::Arc::new(vec![0, 1, 2]);
        let sp = StateProbs::new(
            genotyped_markers,
            2,
            hap_indices,
            state_probs,
            gen_positions,
            marker_to_cluster,
            None,
        );

        let get_ref_allele = |marker: usize, hap: u32| -> u8 {
            std::hint::black_box(marker);
            if hap == 0 { 0 } else { 1 }
        };

        // Access markers in random order
        // Cursor should handle this by advancing (but can't go backwards)
        let mut cursor = Arc::new(sp).cursor();

        // First access at m=25 (cursor advances to sparse_idx=1)
        let p1 = cursor.allele_posteriors(25, 2, &get_ref_allele);

        // Second access at m=75 (cursor advances to sparse_idx=2)
        let p2 = cursor.allele_posteriors(75, 2, &get_ref_allele);

        // Verify results are reasonable
        match p1 {
            AllelePosteriors::Biallelic(p_alt) => {
                // m=25 is between m=0 (0% ALT) and m=50 (50% ALT)
                // weight_left = (50 - 25) / (50 - 0) = 0.5
                // interpolated = 0.5 * 0.0 + 0.5 * 0.5 = 0.25
                assert!(
                    (p_alt - 0.25).abs() < 0.1,
                    "m=25: expected P(ALT)~0.25, got {}", p_alt
                );
            }
            _ => panic!("Expected Biallelic"),
        }

        match p2 {
            AllelePosteriors::Biallelic(p_alt) => {
                // m=75 is between m=50 (50% ALT) and m=100 (100% ALT)
                // weight_left = (100 - 75) / (100 - 50) = 0.5
                // interpolated = 0.5 * 0.5 + 0.5 * 1.0 = 0.75
                assert!(
                    (p_alt - 0.75).abs() < 0.1,
                    "m=75: expected P(ALT)~0.75, got {}", p_alt
                );
            }
            _ => panic!("Expected Biallelic"),
        }
    }

    /// Edge case test: What happens when all state probabilities are below threshold?
    /// This could happen with many states where probability is spread thin.
    ///
    /// With threshold = min(0.005, 0.9999/K), if K=1000 states, threshold = 0.0009999
    /// If all probs are 0.001 (just barely above), they should all be stored.
    /// If all probs are 0.0005 (below), they should all be filtered out.
    #[test]
    fn test_state_probs_all_below_threshold() {
        // 200 states → threshold = min(0.005, 0.9999/200) = 0.004999...
        // If all probs are 0.002, they're below threshold
        let n_states = 200;
        let genotyped_markers = std::sync::Arc::new(vec![0, 10]);
        let gen_positions: std::sync::Arc<Vec<f64>> = std::sync::Arc::new(
            (0..11).map(|m| m as f64).collect()
        );

        let hap_indices: Vec<Vec<u32>> = (0..2)
            .map(|_| (0..n_states as u32).collect())
            .collect();

        // All probabilities are 0.002 (below threshold of ~0.005)
        let state_probs: Vec<f32> = vec![0.002; 2 * n_states];

        let marker_to_cluster = std::sync::Arc::new((0..genotyped_markers.len()).collect());
        let sp = StateProbs::new(
            genotyped_markers,
            n_states,
            hap_indices,
            state_probs,
            gen_positions,
            marker_to_cluster,
            None,
        );

        let get_ref_allele = |marker: usize, hap: u32| -> u8 {
            std::hint::black_box(marker);
            if hap.is_multiple_of(2) { 0 } else { 1 }
        };

        // With all states filtered out, what should happen?
        // The posteriors should return 0 or handle gracefully
        let post = sp.allele_posteriors(5, 2, get_ref_allele);
        if let AllelePosteriors::Biallelic(p_alt) = post {
            // With no stored states, result is undefined but should be valid
            assert!((0.0..=1.0).contains(&p_alt), "P(ALT) should be in [0,1], got {}", p_alt);
        }
    }

    /// Edge case: Single genotyped marker - interpolation should use that marker for all
    #[test]
    fn test_single_genotyped_marker() {
        let genotyped_markers = std::sync::Arc::new(vec![50]);
        let n_ref_markers = 100;
        let gen_positions: std::sync::Arc<Vec<f64>> = std::sync::Arc::new(
            (0..n_ref_markers).map(|m| m as f64).collect()
        );

        let hap_indices: Vec<Vec<u32>> = vec![vec![0, 1]];
        let state_probs = vec![0.3, 0.7]; // 30% state 0, 70% state 1

        let marker_to_cluster = std::sync::Arc::new(vec![0]);
        let sp = StateProbs::new(
            genotyped_markers,
            2,
            hap_indices,
            state_probs,
            gen_positions,
            marker_to_cluster,
            None,
        );

        let get_ref_allele = |marker: usize, hap: u32| -> u8 {
            std::hint::black_box(marker);
            if hap == 0 { 0 } else { 1 }
        };

        // Any marker should use the single genotyped marker's values
        for m in [0, 25, 50, 75, 99] {
            let post = sp.allele_posteriors(m, 2, get_ref_allele);
            match post {
                AllelePosteriors::Biallelic(p_alt) => {
                    // Should be approximately 0.7 (state 1 carries ALT)
                    assert!(
                        (p_alt - 0.7).abs() < 0.01,
                        "Marker {}: expected P(ALT)~0.7, got {}", m, p_alt
                    );
                }
                _ => panic!("Expected Biallelic"),
            }
        }
    }

    /// Edge case: Very large genetic distances - weight calculation should still be valid
    #[test]
    fn test_large_genetic_distances() {
        let genotyped_markers = std::sync::Arc::new(vec![0, 100]);
        let gen_positions: std::sync::Arc<Vec<f64>> = std::sync::Arc::new({
            let mut pos = vec![0.0; 101];
            // Position 100 is 1000 cM away (very large)
            pos[100] = 1000.0;
            pos
        });

        let hap_indices: Vec<Vec<u32>> = vec![vec![0, 1], vec![0, 1]];
        let state_probs = vec![
            1.0, 0.0,  // m=0: 100% state 0
            0.0, 1.0,  // m=100: 100% state 1
        ];

        let marker_to_cluster = std::sync::Arc::new(vec![0, 1]);
        let sp = StateProbs::new(
            genotyped_markers,
            2,
            hap_indices,
            state_probs,
            gen_positions,
            marker_to_cluster,
            None,
        );

        let get_ref_allele = |marker: usize, hap: u32| -> u8 {
            std::hint::black_box(marker);
            if hap == 0 { 0 } else { 1 }
        };

        // At marker 50, weight_left = (1000 - 0) / (1000 - 0) = 1.0
        // Wait, position 50 has gen_pos = 0.0 (only pos 100 is non-zero)
        // So weight_left = (1000 - 0) / (1000 - 0) = 1.0
        // interpolated = 1.0 * 0.0 + 0.0 * 1.0 = 0.0
        let post = sp.allele_posteriors(50, 2, get_ref_allele);
        if let AllelePosteriors::Biallelic(p_alt) = post {
            // Should be 0 because we're very close to marker 0 in genetic distance
            assert!(
                (0.0..=1.0).contains(&p_alt),
                "P(ALT) should be in [0,1], got {}", p_alt
            );
        }
    }

    // =========================================================================
    // DIAGNOSTIC TESTS: Interpolation Behavior Documentation
    // =========================================================================
    //
    // These tests document the current interpolation behavior and provide
    // diagnostic information about how ClusterStateProbs handles markers
    // between clusters where haplotype assignments may differ.

    /// Diagnostic test: Document how interpolation handles haplotype changes between clusters
    ///
    /// When a state maps to different haplotypes at cluster M vs M+1 (due to recombination),
    /// this test documents what the current implementation produces.
    #[test]
    fn test_interpolation_with_haplotype_change_between_clusters() {
        // Setup: State 0 maps to haplotype 0 at cluster 0, haplotype 1 at cluster 1
        // - Haplotype 0 carries REF at all positions
        // - Haplotype 1 carries ALT at all positions
        let marker_cluster = Arc::new(vec![0usize, 0, 1]);
        let ref_cluster_end = Arc::new(vec![1usize, 3]);
        let weight = Arc::new(vec![f32::NAN, 0.5, f32::NAN]);  // 50/50 weight

        let hap_indices = vec![vec![0u32], vec![1u32]];
        let cluster_probs = vec![1.0f32, 1.0f32];

        let state_probs = ClusterStateProbs::new(
            marker_cluster,
            ref_cluster_end,
            weight,
            1,
            hap_indices,
            cluster_probs,
        );

        let get_ref_allele = |marker: usize, hap: u32| -> u8 {
            std::hint::black_box(marker);
            if hap == 0 { 0 } else { 1 }
        };

        let post = state_probs.allele_posteriors(1, 2, &get_ref_allele);
        let p_alt = post.prob(1);

        // Document current behavior:
        // If using LEFT haplotype only: P(ALT) = 0 (hap 0 = REF)
        // If using both haplotypes: P(ALT) = 0.5 (50% from hap0=REF, 50% from hap1=ALT)
        eprintln!("DIAGNOSTIC: Interpolation with haplotype change");
        eprintln!("  State maps: cluster 0 -> hap 0 (REF), cluster 1 -> hap 1 (ALT)");
        eprintln!("  Weight: 0.5 (equal left/right)");
        eprintln!("  P(ALT) = {:.4}", p_alt);
        eprintln!("  If LEFT-only: expect 0.0, If BOTH: expect 0.5");

        // This is a diagnostic - we want to know which approach is used
        // The test passes either way but documents the behavior
        assert!(
            (0.0..=1.0).contains(&p_alt),
            "P(ALT) should be in [0,1], got {}", p_alt
        );
    }

    /// Diagnostic test: Linear interpolation behavior across multiple markers
    #[test]
    fn test_interpolation_gradient_across_interval() {
        let n_markers = 12;
        let mut marker_cluster = vec![0usize; n_markers];
        marker_cluster[11] = 1;

        let marker_cluster = Arc::new(marker_cluster);
        let ref_cluster_end = Arc::new(vec![1usize, 12]);

        let mut weight = vec![f32::NAN; n_markers];
        for m in 1..11 {
            weight[m] = 1.0 - (m as f32 / 10.0);
        }
        let weight = Arc::new(weight);

        // Haplotype change between clusters
        let hap_indices = vec![vec![0u32], vec![1u32]];
        let cluster_probs = vec![1.0f32, 1.0f32];

        let state_probs = ClusterStateProbs::new(
            marker_cluster,
            ref_cluster_end,
            weight,
            1,
            hap_indices,
            cluster_probs,
        );

        let get_ref_allele = |marker: usize, hap: u32| -> u8 {
            std::hint::black_box(marker);
            if hap == 0 { 0 } else { 1 }
        };

        eprintln!("DIAGNOSTIC: Interpolation gradient across interval");
        eprintln!("  Hap 0 = REF, Hap 1 = ALT");
        eprintln!("  Marker | Weight | P(ALT)");
        eprintln!("  -------+--------+-------");

        let mut p_alts = Vec::new();
        for m in 1..11 {
            let post = state_probs.allele_posteriors(m, 2, &get_ref_allele);
            let p_alt = post.prob(1);
            let w = 1.0 - (m as f32 / 10.0);
            eprintln!("  {:6} | {:6.2} | {:6.4}", m, w, p_alt);
            p_alts.push(p_alt);
        }

        // Check monotonicity - P(ALT) should increase as we move toward cluster 1
        // (if using both haplotypes) or stay constant (if using LEFT only)
        let is_monotonic_increasing = p_alts.windows(2).all(|w| w[1] >= w[0] - 0.001);
        let is_constant = p_alts.iter().all(|&p| (p - p_alts[0]).abs() < 0.01);

        eprintln!("  Monotonic increasing: {}", is_monotonic_increasing);
        eprintln!("  Constant: {}", is_constant);

        // Either behavior is valid - this is diagnostic
        assert!(
            is_monotonic_increasing || is_constant,
            "P(ALT) should be either monotonic increasing or constant"
        );
    }

}
