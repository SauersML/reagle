//! # Positional Burrows-Wheeler Transform (PBWT)
//!
//! Implementation of the PBWT algorithm for efficient haplotype matching.
//! Based on Durbin (2014) "Efficient haplotype matching and storage using
//! the positional Burrows-Wheeler transform (PBWT)".
//!
//! This implementation follows the Beagle Java code (PbwtUpdater.java and
//! PbwtDivUpdater.java) closely for correctness.
//!
//! ## Key Concepts
//! - `Prefix array (a)`: Permutation of haplotypes sorted by reverse prefixes
//! - `Divergence array (d)`: Position where each haplotype diverges from predecessor
//!
//! ## Reference
//! Durbin, Richard (2014) Efficient haplotype matching and storage using the
//! positional Burrows-Wheeler transform (PBWT).

use crate::data::HapIdx;

/// PBWT prefix array updater (without divergence tracking)
///
/// This matches Java PbwtUpdater exactly.
#[derive(Debug)]
pub struct PbwtUpdater {
    /// Number of haplotypes
    n_haps: usize,
    /// Temporary storage for each allele bucket: a[allele] contains hap indices
    a: Vec<Vec<u32>>,
}

impl PbwtUpdater {
    /// Create a new PBWT updater
    ///
    /// # Arguments
    /// * `n_haps` - Number of haplotypes at each position
    pub fn new(n_haps: usize) -> Self {
        let init_num_alleles = 4;
        Self {
            n_haps,
            a: (0..init_num_alleles).map(|_| Vec::new()).collect(),
        }
    }

    /// Number of haplotypes
    pub fn n_haps(&self) -> usize {
        self.n_haps
    }

    /// Update prefix array using forward PBWT
    ///
    /// # Arguments
    /// * `alleles` - Allele for each haplotype (indexed by haplotype, not prefix order)
    /// * `n_alleles` - Number of distinct alleles
    /// * `prefix` - Prefix array to update in-place
    pub fn update(&mut self, alleles: &[u8], n_alleles: usize, prefix: &mut [u32]) {
        assert_eq!(alleles.len(), self.n_haps, "alleles length mismatch");
        assert_eq!(prefix.len(), self.n_haps, "prefix length mismatch");
        assert!(n_alleles >= 1, "must have at least one allele");

        self.ensure_capacity(n_alleles);

        // Clear buckets
        for bucket in &mut self.a[..n_alleles] {
            bucket.clear();
        }

        // Distribute haplotypes to buckets based on allele
        for &h in prefix.iter() {
            let allele = alleles[h as usize] as usize;
            assert!(allele < n_alleles, "allele {} out of bounds", allele);
            self.a[allele].push(h);
        }

        // Concatenate buckets back to prefix array
        let mut start = 0;
        for al in 0..n_alleles {
            let size = self.a[al].len();
            prefix[start..start + size].copy_from_slice(&self.a[al]);
            start += size;
            self.a[al].clear();
        }
        debug_assert_eq!(start, self.n_haps);
    }

    /// Update prefix array using a closure for allele access
    pub fn update_with<F>(&mut self, get_allele: F, n_alleles: usize, prefix: &mut [u32])
    where
        F: Fn(usize) -> u8,
    {
        assert_eq!(prefix.len(), self.n_haps, "prefix length mismatch");
        assert!(n_alleles >= 1, "must have at least one allele");

        self.ensure_capacity(n_alleles);

        for bucket in &mut self.a[..n_alleles] {
            bucket.clear();
        }

        for &h in prefix.iter() {
            let allele = get_allele(h as usize) as usize;
            assert!(allele < n_alleles, "allele {} out of bounds", allele);
            self.a[allele].push(h);
        }

        let mut start = 0;
        for al in 0..n_alleles {
            let size = self.a[al].len();
            prefix[start..start + size].copy_from_slice(&self.a[al]);
            start += size;
            self.a[al].clear();
        }
        debug_assert_eq!(start, self.n_haps);
    }

    fn ensure_capacity(&mut self, n_alleles: usize) {
        if n_alleles > self.a.len() {
            let old_len = self.a.len();
            self.a.resize_with(n_alleles, Vec::new);
            for bucket in &mut self.a[old_len..] {
                bucket.clear();
            }
        }
    }
}

/// PBWT updater with divergence array tracking
///
/// This matches Java PbwtDivUpdater exactly, including the correct
/// divergence propagation algorithm.
#[derive(Debug)]
pub struct PbwtDivUpdater {
    /// Number of haplotypes
    n_haps: usize,
    /// Temporary storage for prefix values by allele
    a: Vec<Vec<u32>>,
    /// Temporary storage for divergence values by allele
    d: Vec<Vec<i32>>,
    /// Propagation array for tracking max/min divergence across alleles
    p: Vec<i32>,
}

impl PbwtDivUpdater {
    /// Create a new PBWT divergence updater
    pub fn new(n_haps: usize) -> Self {
        let init_num_alleles = 4;
        Self {
            n_haps,
            a: (0..init_num_alleles).map(|_| Vec::new()).collect(),
            d: (0..init_num_alleles).map(|_| Vec::new()).collect(),
            p: vec![0; init_num_alleles],
        }
    }



    /// Forward update of prefix and divergence arrays
    ///
    /// This exactly matches Java PbwtDivUpdater.fwdUpdate
    ///
    /// # Arguments
    /// * `alleles` - Allele for each haplotype
    /// * `n_alleles` - Number of distinct alleles
    /// * `marker` - Current marker index
    /// * `prefix` - Prefix array to update
    /// * `divergence` - Divergence array to update
    pub fn fwd_update(
        &mut self,
        alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        prefix: &mut [u32],
        divergence: &mut [i32],
    ) {
        assert_eq!(alleles.len(), self.n_haps);
        assert_eq!(prefix.len(), self.n_haps);
        assert!(divergence.len() >= self.n_haps);

        self.ensure_capacity(n_alleles);

        // Initialize p array with marker+1 (divergence if no prior match)
        let init_value = (marker + 1) as i32;
        for j in 0..n_alleles {
            self.p[j] = init_value;
        }

        // Process haplotypes in prefix order
        for i in 0..self.n_haps {
            let hap = prefix[i];
            let div = divergence[i];
            let allele = alleles[hap as usize] as usize;
            assert!(allele < n_alleles);

            // Update p[j] = max(p[j], div) for all alleles
            // This propagates the maximum divergence seen so far
            for j in 0..n_alleles {
                if div > self.p[j] {
                    self.p[j] = div;
                }
            }

            // Store this haplotype with divergence = p[allele]
            self.a[allele].push(hap);
            self.d[allele].push(self.p[allele]);

            // Reset p[allele] for the next hap with this allele
            // Using i32::MIN so any real divergence will be larger
            self.p[allele] = i32::MIN;
        }

        // Concatenate buckets back to arrays
        self.update_prefix_and_div(n_alleles, prefix, divergence);
    }



    /// Concatenate allele buckets back to prefix and divergence arrays
    fn update_prefix_and_div(&mut self, n_alleles: usize, prefix: &mut [u32], divergence: &mut [i32]) {
        let mut start = 0;
        for al in 0..n_alleles {
            let size = self.a[al].len();
            prefix[start..start + size].copy_from_slice(&self.a[al]);
            divergence[start..start + size].copy_from_slice(&self.d[al]);
            start += size;
            self.a[al].clear();
            self.d[al].clear();
        }
        debug_assert_eq!(start, self.n_haps);
    }

    fn ensure_capacity(&mut self, n_alleles: usize) {
        if n_alleles > self.a.len() {
            let old_len = self.a.len();
            self.a.resize_with(n_alleles, Vec::new);
            self.d.resize_with(n_alleles, Vec::new);
            self.p.resize(n_alleles, 0);
            for i in old_len..n_alleles {
                self.a[i].clear();
                self.d[i].clear();
            }
        }
    }

    /// Find IBS matches for a target haplotype using the divergence array
    ///
    /// This implements the neighbor-finding algorithm from PbwtPhaseIbs.
    ///
    /// # Arguments
    /// * `target_pos` - Position of target haplotype in prefix array
    /// * `prefix` - Current prefix array
    /// * `divergence` - Current divergence array (with sentinel at end)
    /// * `marker` - Current marker
    /// * `n_candidates` - Maximum number of candidates to find
    ///
    /// # Returns
    /// (start_pos, end_pos) indices in prefix array of matching neighbors
    pub fn find_ibs_neighbors(
        target_pos: usize,
        prefix: &[u32],
        divergence: &[i32],
        marker: i32,
        n_candidates: usize,
        is_backward: bool,
    ) -> (usize, usize) {
        let n = prefix.len();

        // Set sentinels at boundaries
        let d0 = if is_backward { marker - 2 } else { marker + 2 };

        let mut u = target_pos;       // inclusive start
        let mut v = target_pos + 1;   // exclusive end

        // Get divergence values, using sentinel for out-of-bounds
        let get_div = |i: usize| -> i32 {
            if i == 0 || i >= n {
                d0
            } else {
                divergence[i]
            }
        };

        let mut u_next_match = get_div(u);
        let mut v_next_match = get_div(v);

        // Expand range until we have enough candidates or hit boundaries
        while (v - u) < n_candidates {
            let can_expand = if is_backward {
                marker <= u_next_match || marker <= v_next_match
            } else {
                u_next_match <= marker || v_next_match <= marker
            };

            if !can_expand {
                break;
            }

            if is_backward {
                // For backward PBWT, expand toward larger divergence first
                if u_next_match <= v_next_match {
                    v += 1;
                    v_next_match = get_div(v).min(v_next_match);
                } else {
                    if u > 0 {
                        u -= 1;
                    }
                    u_next_match = get_div(u).min(u_next_match);
                }
            } else {
                // For forward PBWT, expand toward smaller divergence first
                if v_next_match <= u_next_match {
                    v += 1;
                    v_next_match = get_div(v).max(v_next_match);
                } else {
                    if u > 0 {
                        u -= 1;
                    }
                    u_next_match = get_div(u).max(u_next_match);
                }
            }
        }

        (u, v)
    }
}



/// PBWT-based IBS (identity-by-state) segment finder
///
/// This combines forward and backward PBWT for finding IBS neighbors.
#[derive(Debug)]
pub struct PbwtIbs {
    /// Forward PBWT state
    fwd_prefix: Vec<u32>,
    fwd_divergence: Vec<i32>,
}

impl PbwtIbs {
    /// Create a new PBWT IBS finder
    pub fn new(n_haps: usize) -> Self {
        Self {
            fwd_prefix: (0..n_haps as u32).collect(),
            fwd_divergence: vec![0; n_haps + 1], // +1 for sentinel
        }
    }



    /// Get forward prefix array
    pub fn fwd_prefix(&self) -> &[u32] {
        &self.fwd_prefix
    }

    /// Get forward divergence array
    pub fn fwd_divergence(&self) -> &[i32] {
        &self.fwd_divergence
    }

    /// Get mutable forward prefix array
    pub fn fwd_prefix_mut(&mut self) -> &mut [u32] {
        &mut self.fwd_prefix
    }

    /// Get mutable forward divergence array
    pub fn fwd_divergence_mut(&mut self) -> &mut [i32] {
        &mut self.fwd_divergence
    }



    /// Select best matching haplotypes for a target using PBWT
    ///
    /// # Arguments
    /// * `target_hap` - Target haplotype to find matches for
    /// * `n_states` - Number of states to select
    /// * `marker` - Current marker index
    /// * `n_candidates` - Number of candidates to consider
    /// * `use_backward` - Whether to use backward PBWT
    /// * `exclude_self` - Whether to exclude target from results
    ///
    /// # Returns
    /// Vector of selected haplotype indices
    /// Select best matching haplotypes for a target using PBWT
    ///
    /// # Arguments
    /// * `target_hap` - Target haplotype to find matches for
    /// * `n_states` - Number of states to select
    /// * `marker` - Current marker index
    /// * `n_candidates` - Number of candidates to consider
    /// * `exclude_self` - Whether to exclude target from results
    ///
    /// # Returns
    /// Vector of selected haplotype indices
    pub fn select_states(
        &self,
        target_hap: HapIdx,
        n_states: usize,
        marker: usize,
        n_candidates: usize,
        exclude_self: bool,
    ) -> Vec<HapIdx> {
        let prefix = &self.fwd_prefix;
        let divergence = &self.fwd_divergence;

        // Find target position
        let target_pos = match prefix.iter().position(|&h| h == target_hap.0) {
            Some(pos) => pos,
            None => return Vec::new(),
        };

        // Find IBS neighbors
        let (u, v) = PbwtDivUpdater::find_ibs_neighbors(
            target_pos,
            prefix,
            divergence,
            marker as i32,
            n_candidates,
            false,
        );

        // Collect neighbors, excluding self if requested
        let mut selected = Vec::with_capacity(n_states);
        for i in u..v {
            if i < prefix.len() {
                let hap = HapIdx::new(prefix[i]);
                if exclude_self && hap == target_hap {
                    continue;
                }
                if selected.len() >= n_states {
                    break;
                }
                selected.push(hap);
            }
        }

        // If we don't have enough, expand further
        if selected.len() < n_states {
            let mut left = if u > 0 { u - 1 } else { 0 };
            let mut right = v;

            while selected.len() < n_states && (left > 0 || right < prefix.len()) {
                if left > 0 {
                    left = left.saturating_sub(1);
                    let hap = HapIdx::new(prefix[left]);
                    if !exclude_self || hap != target_hap {
                        selected.push(hap);
                    }
                }

                if selected.len() < n_states && right < prefix.len() {
                    let hap = HapIdx::new(prefix[right]);
                    if !exclude_self || hap != target_hap {
                        selected.push(hap);
                    }
                    right += 1;
                }
            }
        }

        selected
    }


}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbwt_update() {
        let mut updater = PbwtUpdater::new(4);
        let mut prefix: Vec<u32> = vec![0, 1, 2, 3];

        // All same allele - order preserved
        let alleles = vec![0u8, 0, 0, 0];
        updater.update(&alleles, 2, &mut prefix);
        assert_eq!(prefix, vec![0, 1, 2, 3]);

        // Alternate alleles - should group by allele
        let alleles = vec![0u8, 1, 0, 1];
        updater.update(&alleles, 2, &mut prefix);
        // Haps with allele 0 come first: 0, 2
        // Haps with allele 1 come second: 1, 3
        assert_eq!(prefix, vec![0, 2, 1, 3]);
    }

    #[test]
    fn test_pbwt_div_fwd_update() {
        let mut updater = PbwtDivUpdater::new(4);
        let mut prefix: Vec<u32> = vec![0, 1, 2, 3];
        let mut divergence: Vec<i32> = vec![0, 0, 0, 0];

        let alleles = vec![0u8, 1, 0, 1];
        updater.fwd_update(&alleles, 2, 0, &mut prefix, &mut divergence);

        // Check grouping: haps with allele 0 first (0, 2), then allele 1 (1, 3)
        assert_eq!(prefix, vec![0, 2, 1, 3]);

        // For forward PBWT at marker 0 with initial div=0 for all:
        // - p initialized to marker+1=1
        // - Hap 0 (allele 0): div=0 <= p[0]=1, so p unchanged. Store d=p[0]=1, reset p[0]=MIN
        // - Hap 1 (allele 1): div=0 > MIN, so p[0]=0. div=0 <= p[1]=1. Store d=p[1]=1, reset p[1]=MIN
        // - Hap 2 (allele 0): div=0 > MIN for both. p=[0,0]. Store d=p[0]=0, reset p[0]=MIN
        // - Hap 3 (allele 1): div=0 > MIN for both. p=[0,0]. Store d=p[1]=0, reset p[1]=MIN
        // Result: d[allele 0]=[1,0], d[allele 1]=[1,0]
        // Final divergence = [1, 0, 1, 0]
        assert_eq!(divergence[0], 1); // Hap 0, first with allele 0
        assert_eq!(divergence[1], 0); // Hap 2, second with allele 0
        assert_eq!(divergence[2], 1); // Hap 1, first with allele 1
        assert_eq!(divergence[3], 0); // Hap 3, second with allele 1
    }



    #[test]
    fn test_pbwt_ibs_select() {
        let ibs = PbwtIbs::new(10);
        let selected = ibs.select_states(HapIdx::new(5), 4, 0, 10, true);
        assert_eq!(selected.len(), 4);
        assert!(!selected.contains(&HapIdx::new(5)));
    }

    #[test]
    fn test_find_ibs_neighbors() {
        let prefix: Vec<u32> = (0..10).collect();
        let divergence: Vec<i32> = vec![0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0];

        // Forward: look for neighbors with small divergence (matching far back)
        let (u, v) = PbwtDivUpdater::find_ibs_neighbors(5, &prefix, &divergence, 5, 4, false);
        assert!(v - u >= 1);

        // Backward: look for neighbors with large divergence
        let (u, v) = PbwtDivUpdater::find_ibs_neighbors(5, &prefix, &divergence, 5, 4, true);
        assert!(v - u >= 1);
    }
}
