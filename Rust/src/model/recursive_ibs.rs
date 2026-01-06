//! # Recursive IBS Selection
//!
//! Implements recursive partitioning algorithm for finding IBS (Identity By State)
//! haplotype segments. Based on Java's ImpIbs.java from Beagle.
//!
//! This finds haplotypes that share long IBS segments with a target haplotype
//! by recursively partitioning based on allele sequence indices.
//!
//! ## Algorithm Overview
//!
//! 1. At each step, split haplotypes by their coded sequence index
//! 2. Only create child groups if they contain target haplotypes
//! 3. When a group has <= n_haps_per_step reference haplotypes, store as result
//! 4. Otherwise, continue partitioning through more steps
//! 5. Use random sampling to fill if needed

use crate::data::haplotype::HapIdx;
use crate::data::storage::coded_steps::{CodedStep, RefPanelCoded};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// Recursive IBS finder for imputation state selection.
///
/// Ports Java's `ImpIbs.java` algorithm which uses recursive partitioning
/// on coded steps to find the longest shared haplotype segments.
pub struct RecursiveIbs {
    /// Number of reference haplotypes
    n_ref_haps: usize,
    /// Number of target haplotypes
    n_targ_haps: usize,
    /// Total haplotypes (ref + target)
    n_haps: usize,
    /// Random seed for reproducibility
    seed: u64,
    /// Number of steps to merge for IBS matching
    n_steps_to_merge: usize,
    /// Target number of haplotypes per step
    n_haps_per_step: usize,
    /// Precomputed IBS haplotypes: ibs_haps[step][target_hap] -> Vec<ref_hap>
    ibs_haps: Vec<Vec<Vec<u32>>>,
}

impl RecursiveIbs {
    /// Create a new RecursiveIbs from coded reference panel.
    ///
    /// # Arguments
    /// * `coded_panel` - Dictionary-compressed reference panel with coded steps
    /// * `n_ref_haps` - Number of reference haplotypes
    /// * `n_targ_haps` - Number of target haplotypes
    /// * `seed` - Random seed for reproducibility
    /// * `n_steps_to_merge` - Number of consecutive steps to merge for IBS matching
    /// * `n_haps_per_step` - Target number of IBS haplotypes per step
    pub fn new(
        coded_panel: &RefPanelCoded,
        n_ref_haps: usize,
        n_targ_haps: usize,
        seed: u64,
        n_steps_to_merge: usize,
        n_haps_per_step: usize,
    ) -> Self {
        let n_haps = n_ref_haps + n_targ_haps;
        let n_steps = coded_panel.n_steps();

        // Precompute IBS haplotypes for all steps in parallel
        let ibs_haps: Vec<Vec<Vec<u32>>> = (0..n_steps)
            .map(|step_idx| {
                Self::get_ibs_haps_for_step(
                    coded_panel,
                    step_idx,
                    n_ref_haps,
                    n_targ_haps,
                    n_haps,
                    seed,
                    n_steps_to_merge,
                    n_haps_per_step,
                )
            })
            .collect();

        Self {
            n_ref_haps,
            n_targ_haps,
            n_haps,
            seed,
            n_steps_to_merge,
            n_haps_per_step,
            ibs_haps,
        }
    }

    /// Get IBS haplotypes for a target haplotype at a given step.
    ///
    /// Returns a slice of reference haplotype indices that share IBS segments
    /// with the specified target haplotype starting from the given step.
    ///
    /// # Arguments
    /// * `target_hap` - Target haplotype index (0-indexed relative to target panel)
    /// * `step` - Step index
    ///
    /// # Returns
    /// Slice of reference haplotype indices
    pub fn ibs_haps(&self, target_hap: usize, step: usize) -> &[u32] {
        self.ibs_haps
            .get(step)
            .and_then(|step_haps| step_haps.get(target_hap))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Number of steps
    pub fn n_steps(&self) -> usize {
        self.ibs_haps.len()
    }

    /// Number of target haplotypes
    pub fn n_targ_haps(&self) -> usize {
        self.n_targ_haps
    }

    /// Number of reference haplotypes
    pub fn n_ref_haps(&self) -> usize {
        self.n_ref_haps
    }

    /// Target number of IBS haplotypes per step
    pub fn n_haps_per_step(&self) -> usize {
        self.n_haps_per_step
    }

    /// Get IBS haplotypes for all target haplotypes at a given step.
    ///
    /// This is the core algorithm ported from Java's `ImpIbs.getIbsHaps`.
    fn get_ibs_haps_for_step(
        coded_panel: &RefPanelCoded,
        step_index: usize,
        n_ref_haps: usize,
        n_targ_haps: usize,
        n_haps: usize,
        seed: u64,
        n_steps_to_merge: usize,
        n_haps_per_step: usize,
    ) -> Vec<Vec<u32>> {
        let n_steps = coded_panel.n_steps();
        let actual_steps_to_merge = n_steps_to_merge.min(n_steps - step_index);

        // Initialize results array (one entry per target haplotype)
        let mut results: Vec<Option<Vec<u32>>> = vec![None; n_targ_haps];

        // Initial partition based on first coded step
        let first_step = coded_panel.step(step_index);
        let children = Self::init_partition(first_step, n_ref_haps, n_haps);
        let mut next_parents: Vec<HapList> = Vec::with_capacity(children.len());

        // Process initial partition
        Self::init_update_results(
            &children,
            &mut next_parents,
            &mut results,
            n_ref_haps,
            n_haps_per_step,
        );

        // Continue partitioning through subsequent steps
        for i in 1..actual_steps_to_merge {
            let coded_step = coded_panel.step(step_index + i);
            let init_capacity = n_targ_haps.min(2 * next_parents.len());
            let parents = std::mem::replace(&mut next_parents, Vec::with_capacity(init_capacity));

            for parent in parents {
                let children = Self::partition(&parent, coded_step, n_ref_haps);
                Self::update_results(
                    &parent,
                    children,
                    &mut next_parents,
                    &mut results,
                    n_ref_haps,
                    n_haps_per_step,
                    seed,
                );
            }
        }

        // Final update for remaining parents
        Self::final_update_results(&next_parents, &mut results, n_ref_haps, n_haps_per_step, seed);

        // Convert Option<Vec<u32>> to Vec<u32>, filling empty with empty vec
        results.into_iter().map(|opt| opt.unwrap_or_default()).collect()
    }

    /// Initial partition based on first coded step.
    ///
    /// Ports Java's `ImpIbs.initPartition`.
    /// Creates groups of haplotypes that share the same sequence index,
    /// but only for groups that contain at least one target haplotype.
    fn init_partition(coded_step: &CodedStep, n_ref_haps: usize, n_haps: usize) -> Vec<HapList> {
        let n_patterns = coded_step.n_patterns();
        let mut list: Vec<Option<HapList>> = vec![None; n_patterns];
        let mut children = Vec::new();

        // First pass: identify patterns that have target haplotypes
        for h in n_ref_haps..n_haps {
            let pattern = coded_step.pattern(HapIdx::new(h as u32)) as usize;
            if pattern < n_patterns && list[pattern].is_none() {
                let new_list = HapList::new();
                list[pattern] = Some(new_list);
                children.push(pattern);
            }
        }

        // Second pass: add ALL haplotypes (ref + target) to matching patterns
        for h in 0..n_haps {
            let pattern = coded_step.pattern(HapIdx::new(h as u32)) as usize;
            if pattern < n_patterns {
                if let Some(ref mut hap_list) = list[pattern] {
                    hap_list.add(h as u32);
                }
            }
        }

        // Collect non-empty lists
        children
            .into_iter()
            .filter_map(|idx| list[idx].take())
            .collect()
    }

    /// Partition a parent list into children based on coded step.
    ///
    /// Ports Java's `ImpIbs.partition`.
    /// Only creates child groups that contain target haplotypes.
    fn partition(parent: &HapList, coded_step: &CodedStep, n_ref_haps: usize) -> Vec<HapList> {
        let n_patterns = coded_step.n_patterns();
        let mut list: Vec<Option<HapList>> = vec![None; n_patterns];
        let mut children_patterns = Vec::new();

        let targ_start = parent.ins_pt(n_ref_haps as u32);

        // First pass: identify patterns that have target haplotypes
        for k in targ_start..parent.len() {
            let hap = parent.get(k);
            let pattern = coded_step.pattern(HapIdx::new(hap)) as usize;
            if pattern < n_patterns && list[pattern].is_none() {
                list[pattern] = Some(HapList::new());
                children_patterns.push(pattern);
            }
        }

        // Second pass: add ALL haplotypes from parent to matching patterns
        for k in 0..parent.len() {
            let hap = parent.get(k);
            let pattern = coded_step.pattern(HapIdx::new(hap)) as usize;
            if pattern < n_patterns {
                if let Some(ref mut hap_list) = list[pattern] {
                    hap_list.add(hap);
                }
            }
        }

        // Collect non-empty lists
        children_patterns
            .into_iter()
            .filter_map(|idx| list[idx].take())
            .collect()
    }

    /// Initial update of results from partition.
    ///
    /// Ports Java's `ImpIbs.initUpdateResults`.
    fn init_update_results(
        children: &[HapList],
        next_parents: &mut Vec<HapList>,
        results: &mut [Option<Vec<u32>>],
        n_ref_haps: usize,
        n_haps_per_step: usize,
    ) {
        for child in children {
            let n_ref = child.ins_pt(n_ref_haps as u32);
            if n_ref <= n_haps_per_step {
                // Small enough - store as result
                let ibs_list = child.copy_first(n_ref);
                Self::set_result(child, n_ref, ibs_list, results, n_ref_haps);
            } else {
                // Too large - continue partitioning
                next_parents.push(child.clone());
            }
        }
    }

    /// Update results from partition, using parent for filling.
    ///
    /// Ports Java's `ImpIbs.updateResults`.
    fn update_results(
        parent: &HapList,
        children: Vec<HapList>,
        next_parents: &mut Vec<HapList>,
        results: &mut [Option<Vec<u32>>],
        n_ref_haps: usize,
        n_haps_per_step: usize,
        seed: u64,
    ) {
        for child in children {
            let n_child_ref = child.ins_pt(n_ref_haps as u32);
            if n_child_ref <= n_haps_per_step {
                // Small enough - compute IBS haps and store
                let ibs_list = Self::ibs_haps_from_parent(parent, &child, n_child_ref, n_haps_per_step, seed, n_ref_haps);
                Self::set_result(&child, n_child_ref, ibs_list, results, n_ref_haps);
            } else {
                // Too large - continue partitioning
                next_parents.push(child);
            }
        }
    }

    /// Final update for remaining parents that weren't fully partitioned.
    ///
    /// Ports Java's `ImpIbs.finalUpdateResults`.
    fn final_update_results(
        children: &[HapList],
        results: &mut [Option<Vec<u32>>],
        n_ref_haps: usize,
        n_haps_per_step: usize,
        seed: u64,
    ) {
        for child in children {
            let n_ref = child.ins_pt(n_ref_haps as u32);
            let mut ibs_list = child.copy_first(n_ref);

            if n_haps_per_step < ibs_list.len() {
                // Need to subsample - shuffle and truncate
                let first_hap = child.get(0);
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(first_hap as u64));
                ibs_list.shuffle(&mut rng);
                ibs_list.truncate(n_haps_per_step);
                ibs_list.sort_unstable();
            }

            Self::set_result(child, n_ref, ibs_list, results, n_ref_haps);
        }
    }

    /// Compute IBS haplotypes by combining child refs with sampled parent refs.
    ///
    /// Ports Java's `ImpIbs.ibsHaps`.
    fn ibs_haps_from_parent(
        parent: &HapList,
        child: &HapList,
        n_child_ref: usize,
        n_haps_per_step: usize,
        seed: u64,
        n_ref_haps: usize,
    ) -> Vec<u32> {
        let mut combined = Vec::with_capacity(n_haps_per_step);

        // Add all child ref haplotypes
        for j in 0..n_child_ref {
            combined.push(child.get(j));
        }

        // Fill remaining with random sample from parent refs unique to parent
        let size = n_haps_per_step.saturating_sub(n_child_ref);
        if size > 0 {
            let first_hap = parent.get(0);
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed.wrapping_add(first_hap as u64));
            let uniq_to_parent = Self::uniq_to_parent(parent, child, n_child_ref, n_ref_haps);
            let rand_subset = Self::random_subset(&uniq_to_parent, size, &mut rng);
            combined.extend(rand_subset);
        }

        combined.sort_unstable();
        combined
    }

    /// Find reference haplotypes unique to parent (not in child).
    ///
    /// Ports Java's `ImpIbs.uniqToParent`.
    fn uniq_to_parent(
        parent: &HapList,
        child: &HapList,
        n_child_ref: usize,
        n_ref_haps: usize,
    ) -> Vec<u32> {
        if n_child_ref == 0 {
            // All parent refs are unique
            let n_parent_ref = parent.ins_pt(n_ref_haps as u32);
            return (0..n_parent_ref).map(|i| parent.get(i)).collect();
        }

        let n_child_ref_m1 = n_child_ref - 1;
        let n_parent_ref = parent.ins_pt(n_ref_haps as u32);
        let mut uniq = Vec::with_capacity(parent.len());

        let mut c = 0;
        let mut c_val = child.get(c);

        for p in 0..n_parent_ref {
            let p_val = parent.get(p);
            while c_val < p_val && c < n_child_ref_m1 {
                c += 1;
                c_val = child.get(c);
            }
            if p_val != c_val {
                uniq.push(p_val);
            }
        }

        uniq
    }

    /// Random subset of a list.
    ///
    /// Ports Java's `ImpIbs.randomSubset`.
    fn random_subset<R: Rng>(list: &[u32], size: usize, rng: &mut R) -> Vec<u32> {
        let actual_size = size.min(list.len());
        if actual_size == 0 {
            return Vec::new();
        }

        let mut arr = list.to_vec();
        for j in 0..actual_size {
            let x = rng.random_range(0..(arr.len() - j) as u32) as usize;
            arr.swap(j, j + x);
        }
        arr.truncate(actual_size);
        arr
    }

    /// Set result for all target haplotypes in the child list.
    ///
    /// Ports Java's `ImpIbs.setResult`.
    fn set_result(
        child: &HapList,
        first_targ_index: usize,
        ibs_haps: Vec<u32>,
        results: &mut [Option<Vec<u32>>],
        n_ref_haps: usize,
    ) {
        for j in first_targ_index..child.len() {
            let targ_hap = child.get(j) as usize - n_ref_haps;
            if targ_hap < results.len() && results[targ_hap].is_none() {
                results[targ_hap] = Some(ibs_haps.clone());
            }
        }
    }
}

/// A sorted list of haplotype indices with binary search support.
///
/// Mimics Java's IntList used in ImpIbs.
#[derive(Clone, Debug, Default)]
struct HapList {
    haps: Vec<u32>,
}

impl HapList {
    fn new() -> Self {
        Self { haps: Vec::new() }
    }

    fn add(&mut self, hap: u32) {
        self.haps.push(hap);
    }

    fn get(&self, index: usize) -> u32 {
        self.haps[index]
    }

    fn len(&self) -> usize {
        self.haps.len()
    }

    /// Find insertion point for a value (first index where haps[i] >= val).
    ///
    /// Equivalent to Java's IntList.binarySearch behavior for insertion point.
    fn ins_pt(&self, val: u32) -> usize {
        match self.haps.binary_search(&val) {
            Ok(idx) => idx,
            Err(idx) => idx,
        }
    }

    /// Copy first n elements.
    fn copy_first(&self, n: usize) -> Vec<u32> {
        self.haps[..n.min(self.haps.len())].to_vec()
    }
}

// ============================================================================
// Builder for easier construction
// ============================================================================

/// Configuration for RecursiveIbs.
#[derive(Clone, Debug)]
pub struct RecursiveIbsConfig {
    /// Number of steps to merge for IBS matching (default: 6)
    pub n_steps_to_merge: usize,
    /// Target number of haplotypes per step (default: 8)
    pub n_haps_per_step: usize,
    /// Random seed (default: 0)
    pub seed: u64,
}

impl Default for RecursiveIbsConfig {
    fn default() -> Self {
        Self {
            n_steps_to_merge: 6,
            n_haps_per_step: 8,
            seed: 0,
        }
    }
}

impl RecursiveIbsConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_n_steps_to_merge(mut self, n: usize) -> Self {
        self.n_steps_to_merge = n;
        self
    }

    pub fn with_n_haps_per_step(mut self, n: usize) -> Self {
        self.n_haps_per_step = n;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::storage::coded_steps::CodedStep;
    use crate::data::storage::GenotypeMatrix;
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Allele, Marker, Markers};
    use crate::data::ChromIdx;
    use crate::data::storage::GenotypeColumn;
    use std::sync::Arc;

    fn make_test_matrix(n_haps: usize, n_markers: usize) -> GenotypeMatrix {
        let n_samples = n_haps / 2;
        let samples = Arc::new(Samples::from_ids(
            (0..n_samples).map(|i| format!("S{}", i)).collect(),
        ));

        let mut markers = Markers::new();
        markers.add_chrom("chr1");

        let mut columns = Vec::new();
        for i in 0..n_markers {
            let m = Marker::new(
                ChromIdx::new(0),
                (i * 1000 + 100) as u32,
                None,
                Allele::Base(0),
                vec![Allele::Base(1)],
            );
            markers.push(m);

            // Create varied allele patterns
            let alleles: Vec<u8> = (0..n_haps)
                .map(|h| ((h + i) % 2) as u8)
                .collect();
            columns.push(GenotypeColumn::from_alleles(&alleles, 2));
        }

        GenotypeMatrix::new_unphased(markers, columns, samples)
    }

    #[test]
    fn test_hap_list_basic() {
        let mut list = HapList::new();
        list.add(0);
        list.add(5);
        list.add(10);
        list.add(15);

        assert_eq!(list.len(), 4);
        assert_eq!(list.get(0), 0);
        assert_eq!(list.get(1), 5);
        assert_eq!(list.get(2), 10);
        assert_eq!(list.get(3), 15);
    }

    #[test]
    fn test_hap_list_ins_pt() {
        let mut list = HapList::new();
        for h in [0, 1, 2, 5, 10, 15] {
            list.add(h);
        }

        // Insertion point for value 3 (between 2 and 5)
        assert_eq!(list.ins_pt(3), 3);
        // Insertion point for value 5 (exact match)
        assert_eq!(list.ins_pt(5), 3);
        // Insertion point for value 0
        assert_eq!(list.ins_pt(0), 0);
        // Insertion point for value 20 (beyond end)
        assert_eq!(list.ins_pt(20), 6);
    }

    #[test]
    fn test_hap_list_copy_first() {
        let mut list = HapList::new();
        for h in 0..10 {
            list.add(h);
        }

        let first_5 = list.copy_first(5);
        assert_eq!(first_5, vec![0, 1, 2, 3, 4]);

        let first_20 = list.copy_first(20);
        assert_eq!(first_20.len(), 10);
    }

    #[test]
    fn test_init_partition() {
        // Create a simple coded step with known patterns
        let gt = make_test_matrix(10, 3);
        let step = CodedStep::new(&gt, 0, 3);

        let n_ref_haps = 6;
        let n_haps = 10;

        let children = RecursiveIbs::init_partition(&step, n_ref_haps, n_haps);

        // Should have created partitions containing target haplotypes
        assert!(!children.is_empty());

        // Each partition should contain both ref and target haps with same pattern
        for child in &children {
            let n_ref = child.ins_pt(n_ref_haps as u32);
            let n_targ = child.len() - n_ref;
            // Must have at least one target hap (that's how we create partitions)
            assert!(n_targ > 0, "Partition should have target haplotypes");
        }
    }

    #[test]
    fn test_partition() {
        let gt = make_test_matrix(10, 6);
        let step1 = CodedStep::new(&gt, 0, 3);
        let step2 = CodedStep::new(&gt, 3, 6);

        let n_ref_haps = 6;
        let n_haps = 10;

        // Get initial partitions
        let initial = RecursiveIbs::init_partition(&step1, n_ref_haps, n_haps);
        assert!(!initial.is_empty());

        // Further partition the first group
        let children = RecursiveIbs::partition(&initial[0], &step2, n_ref_haps);

        // Children should be subsets of parent
        let parent_len: usize = initial[0].len();
        let children_total: usize = children.iter().map(|c| c.len()).sum();
        assert!(children_total <= parent_len);
    }

    #[test]
    fn test_random_subset() {
        let list: Vec<u32> = (0..100).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let subset = RecursiveIbs::random_subset(&list, 10, &mut rng);
        assert_eq!(subset.len(), 10);

        // All elements should be from original list
        for &x in &subset {
            assert!(list.contains(&x));
        }

        // Request more than available
        let subset_large = RecursiveIbs::random_subset(&list, 200, &mut rng);
        assert_eq!(subset_large.len(), 100);
    }

    #[test]
    fn test_uniq_to_parent() {
        let mut parent = HapList::new();
        for h in [0, 1, 2, 3, 4, 10, 11] {
            parent.add(h);
        }

        let mut child = HapList::new();
        for h in [1, 3, 10] {
            child.add(h);
        }

        let n_ref_haps = 5; // haps 0-4 are ref, 10-11 are target
        let n_child_ref = 2; // 1 and 3 are ref in child

        let uniq = RecursiveIbs::uniq_to_parent(&parent, &child, n_child_ref, n_ref_haps);

        // Should contain parent refs not in child: 0, 2, 4
        assert_eq!(uniq.len(), 3);
        assert!(uniq.contains(&0));
        assert!(uniq.contains(&2));
        assert!(uniq.contains(&4));
        assert!(!uniq.contains(&1));
        assert!(!uniq.contains(&3));
    }

    #[test]
    fn test_recursive_ibs_construction() {
        // Create test data
        let gt = make_test_matrix(20, 30);
        let step_starts: Vec<usize> = (0..10).map(|i| i * 3).collect();
        let coded_panel = RefPanelCoded::new(&gt, &step_starts);

        let n_ref_haps = 12;
        let n_targ_haps = 8;

        let config = RecursiveIbsConfig::default();
        let ibs = RecursiveIbs::new(
            &coded_panel,
            n_ref_haps,
            n_targ_haps,
            config.seed,
            config.n_steps_to_merge,
            config.n_haps_per_step,
        );

        assert_eq!(ibs.n_steps(), 10);
        assert_eq!(ibs.n_targ_haps(), 8);
        assert_eq!(ibs.n_ref_haps(), 12);

        // Check that we get IBS haplotypes for each target/step
        for step in 0..ibs.n_steps() {
            for targ in 0..n_targ_haps {
                let haps = ibs.ibs_haps(targ, step);
                // All returned haps should be valid ref haps
                for &h in haps {
                    assert!(
                        (h as usize) < n_ref_haps,
                        "IBS hap {} should be < n_ref_haps {}",
                        h,
                        n_ref_haps
                    );
                }
            }
        }
    }

    #[test]
    fn test_recursive_ibs_deterministic() {
        let gt = make_test_matrix(20, 30);
        let step_starts: Vec<usize> = (0..10).map(|i| i * 3).collect();
        let coded_panel = RefPanelCoded::new(&gt, &step_starts);

        let n_ref_haps = 12;
        let n_targ_haps = 8;

        // Same seed should give same results
        let ibs1 = RecursiveIbs::new(&coded_panel, n_ref_haps, n_targ_haps, 42, 6, 8);
        let ibs2 = RecursiveIbs::new(&coded_panel, n_ref_haps, n_targ_haps, 42, 6, 8);

        for step in 0..ibs1.n_steps() {
            for targ in 0..n_targ_haps {
                assert_eq!(
                    ibs1.ibs_haps(targ, step),
                    ibs2.ibs_haps(targ, step),
                    "Results should be deterministic for same seed"
                );
            }
        }
    }

    #[test]
    fn test_recursive_ibs_different_seeds() {
        let gt = make_test_matrix(40, 30);
        let step_starts: Vec<usize> = (0..10).map(|i| i * 3).collect();
        let coded_panel = RefPanelCoded::new(&gt, &step_starts);

        let n_ref_haps = 30;
        let n_targ_haps = 10;

        let ibs1 = RecursiveIbs::new(&coded_panel, n_ref_haps, n_targ_haps, 42, 6, 8);
        let ibs2 = RecursiveIbs::new(&coded_panel, n_ref_haps, n_targ_haps, 123, 6, 8);

        // With different seeds, verify both produce valid results
        // (Note: results may or may not differ depending on whether random sampling is needed)
        for step in 0..ibs1.n_steps() {
            for targ in 0..n_targ_haps {
                let haps1 = ibs1.ibs_haps(targ, step);
                let haps2 = ibs2.ibs_haps(targ, step);
                // Both should produce valid ref haplotypes
                for &h in haps1 {
                    assert!((h as usize) < n_ref_haps);
                }
                for &h in haps2 {
                    assert!((h as usize) < n_ref_haps);
                }
            }
        }
    }

    #[test]
    fn test_config_builder() {
        let config = RecursiveIbsConfig::new()
            .with_n_steps_to_merge(10)
            .with_n_haps_per_step(16)
            .with_seed(12345);

        assert_eq!(config.n_steps_to_merge, 10);
        assert_eq!(config.n_haps_per_step, 16);
        assert_eq!(config.seed, 12345);
    }

    #[test]
    fn test_edge_case_empty_targets() {
        let gt = make_test_matrix(10, 15);
        let step_starts: Vec<usize> = (0..5).map(|i| i * 3).collect();
        let coded_panel = RefPanelCoded::new(&gt, &step_starts);

        // All haplotypes are reference
        let ibs = RecursiveIbs::new(&coded_panel, 10, 0, 0, 6, 8);

        assert_eq!(ibs.n_targ_haps(), 0);
        assert_eq!(ibs.n_steps(), 5);
    }

    #[test]
    fn test_edge_case_few_refs() {
        let gt = make_test_matrix(10, 15);
        let step_starts: Vec<usize> = (0..5).map(|i| i * 3).collect();
        let coded_panel = RefPanelCoded::new(&gt, &step_starts);

        // Very few reference haplotypes
        let ibs = RecursiveIbs::new(&coded_panel, 2, 8, 0, 6, 8);

        // Should still work, just with fewer IBS haps available
        for step in 0..ibs.n_steps() {
            for targ in 0..8 {
                let haps = ibs.ibs_haps(targ, step);
                // Returned haps should be <= n_ref_haps
                assert!(haps.len() <= 2);
            }
        }
    }

    #[test]
    fn test_n_haps_per_step_limit() {
        let gt = make_test_matrix(50, 30);
        let step_starts: Vec<usize> = (0..10).map(|i| i * 3).collect();
        let coded_panel = RefPanelCoded::new(&gt, &step_starts);

        let n_ref_haps = 40;
        let n_targ_haps = 10;
        let n_haps_per_step = 5;

        let ibs = RecursiveIbs::new(&coded_panel, n_ref_haps, n_targ_haps, 42, 6, n_haps_per_step);

        // Check that we don't exceed n_haps_per_step
        for step in 0..ibs.n_steps() {
            for targ in 0..n_targ_haps {
                let haps = ibs.ibs_haps(targ, step);
                assert!(
                    haps.len() <= n_haps_per_step,
                    "Step {} targ {}: got {} haps, expected <= {}",
                    step,
                    targ,
                    haps.len(),
                    n_haps_per_step
                );
            }
        }
    }
}
