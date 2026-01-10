//! # Cluster-Coded Haplotype Sequences and IBS Matching for Imputation
//!
//! This module implements the sequence-based haplotype coding and IBS (Identity By State)
//! set computation used in Beagle's imputation pipeline. It mirrors Java's
//! `HaplotypeCoder`, `CodedSteps`, and `ImpIbs` classes.
//!
//! ## Pipeline Overview
//!
//! The imputation process uses a three-stage sequence coding scheme:
//!
//! 1. Cluster-level coding (`ClusterHapSequences`): Each marker cluster gets a
//!    unique sequence ID per distinct allele pattern. Target haplotypes define the
//!    sequence space; reference haplotypes are mapped into it (with 0 = unmapped).
//!
//! 2. Step-level coding (`ClusterCodedSteps`): Multiple adjacent clusters are
//!    aggregated into "steps" (genetic distance windows). This produces a coarser
//!    sequence ID that spans multiple clusters.
//!
//! 3. IBS set computation (`ImpIbs`): For each target haplotype at each step,
//!    find reference haplotypes with matching sequence IDs. This creates the
//!    "IBS sets" - reference haplotypes that are identical-by-state to the target
//!    across the step's marker clusters.
//!
//! ## Sequence ID Encoding
//!
//! Sequence IDs are built incrementally by processing markers within a cluster:
//! - Start with all haplotypes having sequence ID = 1 (base sequence)
//! - At each marker, split sequences by allele: seq_new = hash(seq_old, allele)
//! - Target haplotypes define the mapping; reference haps map to existing or 0

/// Java Linear Congruential Generator (LCG) multiplier for reproducible random sampling.
/// Using Java's constants ensures identical IBS set selection as Java Beagle.
const JAVA_RNG_MULT: u64 = 0x5DEECE66D;
const JAVA_RNG_ADD: u64 = 0xB;
const JAVA_RNG_MASK: u64 = (1u64 << 48) - 1;

use std::sync::Arc;

use crate::data::haplotype::HapIdx;
use crate::data::marker::MarkerIdx;
use crate::data::storage::phase_state::Phased;
use crate::data::storage::GenotypeMatrix;
use crate::pipelines::imputation::MarkerAlignment;

/// Per-cluster haplotype sequence IDs.
///
/// Each cluster contains one or more adjacent genotyped markers. Within a cluster,
/// haplotypes are assigned sequence IDs based on their allele patterns:
/// - `hap_to_seq[c][h]` = sequence ID for haplotype h at cluster c
/// - Sequence ID 0 means the reference haplotype's pattern wasn't seen in targets
/// - Sequence ID 1 is the initial base sequence before any differentiation
///
/// The combined array stores reference haplotypes first, then target haplotypes:
/// `[ref_hap_0, ref_hap_1, ..., ref_hap_n, targ_hap_0, targ_hap_1, ...]`
#[derive(Clone, Debug)]
pub struct ClusterHapSequences {
    /// Sequence IDs: `hap_to_seq[cluster][haplotype]` where haplotype indices
    /// are: 0..n_ref_haps for reference, n_ref_haps..n_haps for target
    pub hap_to_seq: Vec<Vec<u32>>,
    /// Number of distinct sequence IDs at each cluster (for allocation sizing)
    pub value_sizes: Vec<u32>,
    /// Number of reference panel haplotypes
    pub n_ref_haps: usize,
    /// Total haplotypes (reference + target)
    pub n_haps: usize,
}

/// Coded steps: sequence IDs aggregated across multiple adjacent clusters.
///
/// Steps are defined by genetic distance (step_cm) and provide a coarser-grained
/// view of haplotype identity than individual clusters. This aggregation:
/// 1. Reduces the number of partitioning operations in IBS set computation
/// 2. Provides more stable IBS sets that span multiple markers
///
/// The step_starts array maps step indices to their starting cluster indices.
#[derive(Clone, Debug)]
pub struct ClusterCodedSteps {
    /// Starting cluster index for each step: step i covers clusters step_starts[i]..step_starts[i+1]
    pub step_starts: Vec<usize>,
    /// Sequence IDs per step: `hap_to_seq[step][haplotype]`
    pub hap_to_seq: Vec<Vec<u32>>,
    /// Number of distinct sequences at each step
    pub value_sizes: Vec<u32>,
}

/// IBS (Identity By State) sets for imputation HMM state selection.
///
/// For each target haplotype at each step, stores the set of reference haplotypes
/// that share the same coded sequence. These IBS sets are used to:
/// 1. Initialize HMM state probabilities (states matching the target get higher weight)
/// 2. Prune the HMM state space (only consider IBS-matching reference haplotypes)
///
/// The IBS sets are bounded in size (`n_haps_per_step`) and use deterministic
/// random sampling (Java LCG) for reproducibility with Java Beagle.
#[derive(Clone, Debug)]
pub struct ImpIbs {
    /// The underlying coded steps used to derive IBS sets
    pub coded_steps: Arc<ClusterCodedSteps>,
    /// IBS haplotype indices: `ibs_haps[step][target_hap_idx] = Vec<reference_hap_idx>`
    /// where target_hap_idx is 0-based within target panel (not offset by n_ref_haps)
    pub ibs_haps: Vec<Vec<Vec<u32>>>,
}

/// Java-compatible Linear Congruential Generator for deterministic random sampling.
///
/// Uses the same constants as `java.util.Random` to ensure identical results
/// when sampling IBS sets with the same seed. This is critical for reproducing
/// Java Beagle's exact imputation results.
#[derive(Clone, Debug)]
struct JavaRng {
    seed: u64,
}

impl JavaRng {
    fn new(seed: u64) -> Self {
        let seed = (seed ^ JAVA_RNG_MULT) & JAVA_RNG_MASK;
        Self { seed }
    }

    fn next_bits(&mut self, bits: u32) -> u32 {
        self.seed = (self.seed.wrapping_mul(JAVA_RNG_MULT).wrapping_add(JAVA_RNG_ADD)) & JAVA_RNG_MASK;
        (self.seed >> (48 - bits)) as u32
    }

    fn next_int(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        if bound.is_power_of_two() {
            return (((bound as u64) * (self.next_bits(31) as u64)) >> 31) as usize;
        }
        loop {
            let bits = self.next_bits(31) as i64;
            let val = (bits % bound as i64) as i64;
            if bits - val + (bound as i64 - 1) >= 0 {
                return val as usize;
            }
        }
    }
}

/// Build per-cluster haplotype sequences (equivalent to Java's HaplotypeCoder).
///
/// This function assigns unique sequence IDs to each distinct allele pattern
/// within each marker cluster. The key insight is that target haplotypes define
/// the sequence space:
///
/// 1. Process target haplotypes first to build the sequence ID mapping
/// 2. Map reference haplotypes into the target-defined space
/// 3. Reference patterns not seen in targets get sequence ID 0 (filtered out later)
///
/// This target-driven approach ensures IBS sets contain only reference haplotypes
/// that match observed target patterns, improving imputation accuracy.
///
/// # Arguments
/// * `ref_gt` - Reference panel genotypes (phased)
/// * `target_gt` - Target sample genotypes (phased)
/// * `alignment` - Marker alignment between target and reference panels
/// * `genotyped_markers` - Indices of genotyped markers in reference panel
/// * `cluster_bounds` - (start, end) indices into genotyped_markers for each cluster
pub fn build_cluster_hap_sequences(
    ref_gt: &GenotypeMatrix<Phased>,
    target_gt: &GenotypeMatrix<Phased>,
    alignment: &MarkerAlignment,
    genotyped_markers: &[usize],
    cluster_bounds: &[(usize, usize)],
) -> ClusterHapSequences {
    let n_ref_haps = ref_gt.n_haplotypes();
    let n_targ_haps = target_gt.n_haplotypes();
    let n_haps = n_ref_haps + n_targ_haps;

    let mut hap_to_seq = Vec::with_capacity(cluster_bounds.len());
    let mut value_sizes = Vec::with_capacity(cluster_bounds.len());

    for &(start, end) in cluster_bounds {
        let mut targ_seq = vec![1u32; n_targ_haps];
        let mut ref_seq = vec![1u32; n_ref_haps];
        let mut seq_cnt = 2u32; // 0 reserved, 1 is base

        for &ref_m in &genotyped_markers[start..end] {
            let Some(target_m) = alignment.target_marker(ref_m) else {
                continue;
            };
            let target_marker_idx = MarkerIdx::new(target_m as u32);
            let n_alleles = 1 + target_gt.marker(target_marker_idx).alt_alleles.len();
            if n_alleles == 0 {
                continue;
            }

            let n_alleles_with_missing = n_alleles + 1;
            let mut seq_map = vec![0u32; (seq_cnt as usize) * n_alleles_with_missing];
            seq_cnt = 1;
            let missing_allele = n_alleles;

            // Update target hap sequences first.
            for h in 0..n_targ_haps {
                let hap_idx = HapIdx::new(h as u32);
                let allele_raw = target_gt.allele(target_marker_idx, hap_idx);
                let allele = if allele_raw == 255 || (allele_raw as usize) >= n_alleles {
                    missing_allele
                } else {
                    allele_raw as usize
                };
                let index = (targ_seq[h] as usize) * n_alleles_with_missing + allele;
                if seq_map[index] == 0 {
                    seq_map[index] = seq_cnt;
                    seq_cnt += 1;
                }
                targ_seq[h] = seq_map[index];
            }

            // Map reference hap sequences using target-defined seq_map.
            for h in 0..n_ref_haps {
                if ref_seq[h] == 0 {
                    continue;
                }
                let hap_idx = HapIdx::new(h as u32);
                let ref_allele = ref_gt.allele(MarkerIdx::new(ref_m as u32), hap_idx);
                let mapped = alignment.reverse_map_allele(target_m, ref_allele);
                let allele = if mapped == 255 || (mapped as usize) >= n_alleles {
                    missing_allele
                } else {
                    mapped as usize
                };
                let index = (ref_seq[h] as usize) * n_alleles_with_missing + allele;
                ref_seq[h] = seq_map.get(index).copied().unwrap_or(0);
            }
        }

        let mut combined = Vec::with_capacity(n_haps);
        combined.extend_from_slice(&ref_seq);
        combined.extend_from_slice(&targ_seq);
        hap_to_seq.push(combined);
        value_sizes.push(seq_cnt.max(2));
    }

    ClusterHapSequences {
        hap_to_seq,
        value_sizes,
        n_ref_haps,
        n_haps,
    }
}

impl ClusterCodedSteps {
    pub fn from_cluster_sequences(
        cluster_seqs: &ClusterHapSequences,
        cluster_pos: &[f64],
        step_cm: f64,
    ) -> Self {
        let step_starts = compute_step_starts(cluster_pos, step_cm);
        let n_steps = step_starts.len();
        let n_haps = cluster_seqs.n_haps;
        let n_ref_haps = cluster_seqs.n_ref_haps;

        let mut steps = Vec::with_capacity(n_steps);
        let mut value_sizes = Vec::with_capacity(n_steps);

        for (idx, &start) in step_starts.iter().enumerate() {
            let end = if idx + 1 < step_starts.len() {
                step_starts[idx + 1]
            } else {
                cluster_seqs.hap_to_seq.len()
            };

            let mut hap_to_seq = vec![1u32; n_haps];
            let mut seq_cnt = 2u32;

            for cluster_idx in start..end {
                let alleles = &cluster_seqs.hap_to_seq[cluster_idx];
                let n_alleles = cluster_seqs.value_sizes[cluster_idx] as usize;
                if n_alleles == 0 {
                    continue;
                }

                let mut seq_map = vec![0u32; (seq_cnt as usize) * n_alleles];
                seq_cnt = 1;

                for h in n_ref_haps..n_haps {
                    let mut allele = alleles[h] as usize;
                    if allele >= n_alleles {
                        allele = 0;
                    }
                    let index = (hap_to_seq[h] as usize) * n_alleles + allele;
                    if seq_map[index] == 0 {
                        seq_map[index] = seq_cnt;
                        seq_cnt += 1;
                    }
                    hap_to_seq[h] = seq_map[index];
                }

                for h in 0..n_ref_haps {
                    if hap_to_seq[h] == 0 {
                        continue;
                    }
                    let mut allele = alleles[h] as usize;
                    if allele >= n_alleles {
                        allele = 0;
                    }
                    let index = (hap_to_seq[h] as usize) * n_alleles + allele;
                    hap_to_seq[h] = seq_map.get(index).copied().unwrap_or(0);
                }
            }

            steps.push(hap_to_seq);
            value_sizes.push(seq_cnt.max(2));
        }

        ClusterCodedSteps {
            step_starts,
            hap_to_seq: steps,
            value_sizes,
        }
    }

    pub fn n_steps(&self) -> usize {
        self.step_starts.len()
    }

    pub fn step_start(&self, step: usize) -> usize {
        self.step_starts[step]
    }
}

impl ImpIbs {
    pub fn new(
        coded_steps: Arc<ClusterCodedSteps>,
        n_steps_to_merge: usize,
        n_haps_per_step: usize,
        n_ref_haps: usize,
        n_targ_haps: usize,
        seed: u64,
    ) -> Self {
        let mut ibs_haps = Vec::with_capacity(coded_steps.n_steps());
        for step_idx in 0..coded_steps.n_steps() {
            let ibs = compute_ibs_haps(
                &coded_steps,
                step_idx,
                n_steps_to_merge,
                n_haps_per_step,
                n_ref_haps,
                n_targ_haps,
                seed,
            );
            ibs_haps.push(ibs);
        }

        Self {
            coded_steps,
            ibs_haps,
        }
    }

    pub fn ibs_haps(&self, targ_hap: usize, step: usize) -> &[u32] {
        &self.ibs_haps[step][targ_hap]
    }
}

/// Compute step boundary indices based on genetic distance.
///
/// Steps are non-overlapping genetic distance windows. Each step covers
/// approximately `step_cm` centiMorgans of genetic distance, with the first
/// step starting at half the step size to center the first marker.
///
/// # Arguments
/// * `pos` - Genetic positions (cM) for each cluster
/// * `step_cm` - Target step size in centiMorgans
///
/// # Returns
/// Vector of starting cluster indices for each step
fn compute_step_starts(pos: &[f64], step_cm: f64) -> Vec<usize> {
    if pos.is_empty() {
        return Vec::new();
    }
    let mut starts = Vec::new();
    starts.push(0);
    // First step boundary at half the step size to center first cluster
    let mut next_pos = pos[0] + step_cm / 2.0;
    let mut index = next_index(pos, 0, next_pos);
    while index < pos.len() {
        starts.push(index);
        // Subsequent steps at full step_cm intervals
        next_pos = pos[index] + step_cm;
        index = next_index(pos, index, next_pos);
    }
    starts
}

/// Binary search for the first index where pos[index] >= target.
///
/// Used to find step boundaries based on genetic position thresholds.
fn next_index(pos: &[f64], start: usize, target: f64) -> usize {
    let mut lo = start;
    let mut hi = pos.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if pos[mid] < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

/// Compute IBS haplotype sets for a single step by recursive partitioning.
///
/// This implements Beagle's IBS set construction algorithm:
/// 1. Start with all haplotypes in one partition
/// 2. Recursively split by sequence ID across merged steps
/// 3. When a partition has few enough reference haps (<= n_haps_per_step),
///    assign those as the IBS set for all target haps in that partition
/// 4. Large partitions get randomly subsampled for stability
///
/// # Arguments
/// * `coded_steps` - The coded step data
/// * `index` - Which step to compute IBS sets for
/// * `n_steps_to_merge` - Number of subsequent steps to merge for finer partitioning
/// * `n_haps_per_step` - Maximum IBS set size (larger sets get subsampled)
/// * `n_ref_haps` - Number of reference haplotypes
/// * `n_targ_haps` - Number of target haplotypes
/// * `seed` - RNG seed for reproducible subsampling
fn compute_ibs_haps(
    coded_steps: &ClusterCodedSteps,
    index: usize,
    n_steps_to_merge: usize,
    n_haps_per_step: usize,
    n_ref_haps: usize,
    n_targ_haps: usize,
    seed: u64,
) -> Vec<Vec<u32>> {
    let mut results: Vec<Vec<u32>> = vec![Vec::new(); n_targ_haps];
    let n_steps = coded_steps.n_steps();
    let n_steps_to_merge = n_steps_to_merge.min(n_steps.saturating_sub(index));

        let children = init_partition(coded_steps, index, n_ref_haps);
    let mut next_parents: Vec<Vec<u32>> = Vec::new();
    init_update_results(&children, &mut next_parents, &mut results, n_ref_haps, n_haps_per_step);

    for i in 1..n_steps_to_merge {
        let mut parents = next_parents;
        next_parents = Vec::new();
        let step_idx = index + i;
        for parent in parents.iter_mut() {
            let child_lists = partition(coded_steps, step_idx, parent, n_ref_haps);
            update_results(
                parent,
                child_lists,
                &mut next_parents,
                &mut results,
                n_ref_haps,
                n_haps_per_step,
                seed,
            );
        }
    }

    final_update_results(&next_parents, &mut results, n_ref_haps, n_haps_per_step, seed);
    results
}

fn init_partition(coded_steps: &ClusterCodedSteps, step: usize, n_ref_haps: usize) -> Vec<Vec<u32>> {
    let hap_to_seq = &coded_steps.hap_to_seq[step];
    let value_size = coded_steps.value_sizes[step] as usize;
    let n_haps = hap_to_seq.len();

    let mut seq_lists: Vec<Option<Vec<u32>>> = vec![None; value_size.max(1)];
    let mut children: Vec<Vec<u32>> = Vec::new();

    for h in n_ref_haps..n_haps {
        let seq = hap_to_seq[h] as usize;
        if seq >= seq_lists.len() {
            continue;
        }
        if seq_lists[seq].is_none() {
            seq_lists[seq] = Some(Vec::new());
            children.push(Vec::new());
        }
    }

    for h in 0..n_haps {
        let seq = hap_to_seq[h] as usize;
        if seq < seq_lists.len() {
            if let Some(list) = seq_lists[seq].as_mut() {
                list.push(h as u32);
            }
        }
    }

    let mut out = Vec::new();
    for list in seq_lists.into_iter().flatten() {
        if !list.is_empty() {
            let mut list = list;
            list.sort_unstable();
            out.push(list);
        }
    }
    out
}

fn partition(
    coded_steps: &ClusterCodedSteps,
    step: usize,
    parent: &mut Vec<u32>,
    n_ref_haps: usize,
) -> Vec<Vec<u32>> {
    let hap_to_seq = &coded_steps.hap_to_seq[step];
    let value_size = coded_steps.value_sizes[step] as usize;
    let n_parent_haps = parent.len();

    let mut seq_lists: Vec<Option<Vec<u32>>> = vec![None; value_size.max(1)];
    let mut children: Vec<Vec<u32>> = Vec::new();

    let targ_start = ins_pt(parent, n_ref_haps);
    for idx in targ_start..n_parent_haps {
        let hap = parent[idx] as usize;
        let seq = hap_to_seq[hap] as usize;
        if seq >= seq_lists.len() {
            continue;
        }
        if seq_lists[seq].is_none() {
            seq_lists[seq] = Some(Vec::new());
            children.push(Vec::new());
        }
    }

    for idx in 0..n_parent_haps {
        let hap = parent[idx] as usize;
        let seq = hap_to_seq[hap] as usize;
        if seq < seq_lists.len() {
            if let Some(list) = seq_lists[seq].as_mut() {
                list.push(hap as u32);
            }
        }
    }

    let mut out = Vec::new();
    for list in seq_lists.into_iter().flatten() {
        if !list.is_empty() {
            let mut list = list;
            list.sort_unstable();
            out.push(list);
        }
    }
    out
}

fn init_update_results(
    children: &[Vec<u32>],
    next_parents: &mut Vec<Vec<u32>>,
    results: &mut [Vec<u32>],
    n_ref_haps: usize,
    n_haps_per_step: usize,
) {
    for child in children {
        let n_ref = ins_pt(child, n_ref_haps);
        if n_ref <= n_haps_per_step {
            let ibs_list = child[..n_ref].to_vec();
            set_result(child, n_ref_haps, &ibs_list, results);
        } else {
            next_parents.push(child.clone());
        }
    }
}

fn update_results(
    parent: &mut Vec<u32>,
    children: Vec<Vec<u32>>,
    next_parents: &mut Vec<Vec<u32>>,
    results: &mut [Vec<u32>],
    n_ref_haps: usize,
    n_haps_per_step: usize,
    seed: u64,
) {
    for child in children {
        let n_ref = ins_pt(&child, n_ref_haps);
        if n_ref <= n_haps_per_step {
            let ibs_list = ibs_haps(parent, &child, n_ref, n_ref_haps, n_haps_per_step, seed);
            set_result(&child, n_ref_haps, &ibs_list, results);
        } else {
            next_parents.push(child);
        }
    }
}

fn final_update_results(
    parents: &[Vec<u32>],
    results: &mut [Vec<u32>],
    n_ref_haps: usize,
    n_haps_per_step: usize,
    seed: u64,
) {
    for parent in parents {
        let n_ref = ins_pt(parent, n_ref_haps);
        let mut ibs_list = parent[..n_ref].to_vec();
        if n_haps_per_step < ibs_list.len() {
            let mut rng = JavaRng::new(seed + parent[0] as u64);
            shuffle_prefix(&mut ibs_list, n_haps_per_step, &mut rng);
            ibs_list.truncate(n_haps_per_step);
            ibs_list.sort_unstable();
        }
        set_result(parent, n_ref_haps, &ibs_list, results);
    }
}

fn ibs_haps(
    parent: &[u32],
    child: &[u32],
    n_child_ref: usize,
    n_ref_haps: usize,
    n_haps_per_step: usize,
    seed: u64,
) -> Vec<u32> {
    let mut combined: Vec<u32> = child[..n_child_ref].to_vec();
    let need = n_haps_per_step.saturating_sub(n_child_ref);
    if need == 0 {
        return combined;
    }

    let uniq = uniq_to_parent(parent, child, n_child_ref, n_ref_haps);
    let mut rng = JavaRng::new(seed + parent[0] as u64);
    let subset = random_subset(&uniq, need, &mut rng);
    combined.extend_from_slice(&subset);
    combined.sort_unstable();
    combined
}

fn uniq_to_parent(parent: &[u32], child: &[u32], n_child_ref: usize, n_ref_haps: usize) -> Vec<u32> {
    let n_parent_ref = ins_pt(parent, n_ref_haps);
    let mut uniq = Vec::with_capacity(n_parent_ref);
    if n_child_ref == 0 {
        uniq.extend_from_slice(&parent[..n_parent_ref]);
        return uniq;
    }

    let mut c = 0usize;
    let mut c_val = child[c];
    let last = n_child_ref.saturating_sub(1);

    for p in 0..n_parent_ref {
        let p_val = parent[p];
        while c < last && c_val < p_val {
            c += 1;
            c_val = child[c];
        }
        if p_val != c_val {
            uniq.push(p_val);
        }
    }
    uniq
}

fn random_subset(list: &[u32], size: usize, rng: &mut JavaRng) -> Vec<u32> {
    if list.is_empty() || size == 0 {
        return Vec::new();
    }
    let mut buf = list.to_vec();
    let take = size.min(buf.len());
    for i in 0..take {
        let j = i + rng.next_int(buf.len() - i);
        buf.swap(i, j);
    }
    buf.truncate(take);
    buf
}

fn shuffle_prefix(list: &mut [u32], k: usize, rng: &mut JavaRng) {
    let mut i = 0;
    let n = list.len().min(k);
    while i < n {
        let j = i + rng.next_int(list.len() - i);
        list.swap(i, j);
        i += 1;
    }
}

fn set_result(child: &[u32], n_ref_haps: usize, ibs_list: &[u32], results: &mut [Vec<u32>]) {
    let start = ins_pt(child, n_ref_haps);
    for &hap in &child[start..] {
        let idx = (hap as usize).saturating_sub(n_ref_haps);
        if idx < results.len() {
            results[idx] = ibs_list.to_vec();
        }
    }
}

/// Find insertion point separating reference from target haplotypes in a sorted list.
///
/// Returns the index of the first target haplotype (or list.len() if none).
/// This is used to efficiently separate reference and target haps in partitions.
fn ins_pt(list: &[u32], n_ref_haps: usize) -> usize {
    list.partition_point(|&h| (h as usize) < n_ref_haps)
}
