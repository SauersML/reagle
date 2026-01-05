//! # Positional Burrows-Wheeler Transform (PBWT)
//!
//! Implementation of the PBWT algorithm for efficient haplotype matching.
//! Based on Durbin (2014) "Efficient haplotype matching and storage using
//! the positional Burrows-Wheeler transform (PBWT)".
//!
//! ## Key Concepts
//! - **Prefix array**: Permutation of haplotypes sorted by reverse prefixes
//! - **Divergence array**: Position where each haplotype diverges from predecessor
//!
//! ## Usage
//! ```ignore
//! let mut updater = PbwtUpdater::new(n_haps);
//! for marker in 0..n_markers {
//!     let alleles = get_alleles(marker);
//!     updater.update(&alleles, n_alleles, &mut prefix);
//! }
//! ```

use crate::data::HapIdx;

/// PBWT prefix array updater (without divergence tracking)
#[derive(Debug)]
pub struct PbwtUpdater {
    /// Number of haplotypes
    n_haps: usize,
    /// Temporary storage for each allele bucket
    buckets: Vec<Vec<u32>>,
}

impl PbwtUpdater {
    /// Create a new PBWT updater
    pub fn new(n_haps: usize) -> Self {
        Self {
            n_haps,
            buckets: vec![Vec::new(); 4], // Start with 4 alleles
        }
    }

    /// Number of haplotypes
    pub fn n_haps(&self) -> usize {
        self.n_haps
    }

    /// Update prefix array for one marker
    ///
    /// # Arguments
    /// * `alleles` - Allele for each haplotype (indexed by haplotype, not prefix order)
    /// * `n_alleles` - Number of distinct alleles
    /// * `prefix` - Prefix array to update in-place
    pub fn update(&mut self, alleles: &[u8], n_alleles: usize, prefix: &mut [u32]) {
        debug_assert_eq!(alleles.len(), self.n_haps);
        debug_assert_eq!(prefix.len(), self.n_haps);

        // Ensure we have enough buckets
        if n_alleles > self.buckets.len() {
            self.buckets.resize(n_alleles, Vec::new());
        }

        // Clear buckets
        for bucket in &mut self.buckets[..n_alleles] {
            bucket.clear();
        }

        // Distribute haplotypes to buckets based on allele
        for &hap in prefix.iter() {
            let allele = alleles[hap as usize] as usize;
            if allele < n_alleles {
                self.buckets[allele].push(hap);
            }
        }

        // Concatenate buckets back to prefix array
        let mut idx = 0;
        for bucket in &self.buckets[..n_alleles] {
            for &hap in bucket {
                prefix[idx] = hap;
                idx += 1;
            }
        }
    }

    /// Update prefix array using an IntArray-like accessor
    pub fn update_with<F>(&mut self, get_allele: F, n_alleles: usize, prefix: &mut [u32])
    where
        F: Fn(usize) -> u8,
    {
        debug_assert_eq!(prefix.len(), self.n_haps);

        // Ensure we have enough buckets
        if n_alleles > self.buckets.len() {
            self.buckets.resize(n_alleles, Vec::new());
        }

        // Clear buckets
        for bucket in &mut self.buckets[..n_alleles] {
            bucket.clear();
        }

        // Distribute haplotypes to buckets
        for &hap in prefix.iter() {
            let allele = get_allele(hap as usize) as usize;
            if allele < n_alleles {
                self.buckets[allele].push(hap);
            }
        }

        // Concatenate buckets back
        let mut idx = 0;
        for bucket in &self.buckets[..n_alleles] {
            for &hap in bucket {
                prefix[idx] = hap;
                idx += 1;
            }
        }
    }
}

/// PBWT updater with divergence array tracking
#[derive(Debug)]
pub struct PbwtDivUpdater {
    /// Number of haplotypes
    n_haps: usize,
    /// Temporary storage for each allele bucket (hap, div pairs)
    buckets: Vec<Vec<(u32, u32)>>,
}

impl PbwtDivUpdater {
    /// Create a new PBWT divergence updater
    pub fn new(n_haps: usize) -> Self {
        Self {
            n_haps,
            buckets: vec![Vec::new(); 4],
        }
    }

    /// Update prefix and divergence arrays for one marker
    ///
    /// # Arguments
    /// * `alleles` - Allele for each haplotype
    /// * `n_alleles` - Number of distinct alleles
    /// * `marker` - Current marker index (for divergence update)
    /// * `prefix` - Prefix array to update
    /// * `divergence` - Divergence array to update
    pub fn update(
        &mut self,
        alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        prefix: &mut [u32],
        divergence: &mut [u32],
    ) {
        debug_assert_eq!(alleles.len(), self.n_haps);
        debug_assert_eq!(prefix.len(), self.n_haps);
        debug_assert_eq!(divergence.len(), self.n_haps);

        // Ensure we have enough buckets
        if n_alleles > self.buckets.len() {
            self.buckets.resize(n_alleles, Vec::new());
        }

        // Clear buckets
        for bucket in &mut self.buckets[..n_alleles] {
            bucket.clear();
        }

        // Track minimum divergence seen so far for each allele
        let mut min_div = vec![marker as u32 + 1; n_alleles];

        // Distribute haplotypes to buckets
        for i in 0..self.n_haps {
            let hap = prefix[i];
            let div = divergence[i];
            let allele = alleles[hap as usize] as usize;

            if allele < n_alleles {
                // Update divergence: max of current div and min_div for this allele
                let new_div = div.max(min_div[allele]);
                self.buckets[allele].push((hap, new_div));

                // Update min_div for this allele
                min_div[allele] = min_div[allele].min(div);
            }
        }

        // Concatenate buckets back
        let mut idx = 0;
        for bucket in &self.buckets[..n_alleles] {
            for &(hap, div) in bucket {
                prefix[idx] = hap;
                divergence[idx] = div;
                idx += 1;
            }
        }
    }

    /// Find long matches ending at current position
    ///
    /// Returns indices in prefix order of haplotypes that match the query
    /// for at least `min_length` markers.
    pub fn find_matches(
        &self,
        _query_allele: u8,
        prefix: &[u32],
        divergence: &[u32],
        marker: usize,
        min_length: usize,
        query_hap: Option<HapIdx>,
    ) -> Vec<usize> {
        let mut matches = Vec::new();
        let threshold = if marker >= min_length {
            (marker - min_length + 1) as u32
        } else {
            0
        };

        for (i, (&hap, &div)) in prefix.iter().zip(divergence.iter()).enumerate() {
            // Skip self if specified
            if let Some(qh) = query_hap {
                if hap == qh.0 {
                    continue;
                }
            }

            // Check if this haplotype matches for long enough
            if div <= threshold {
                matches.push(i);
            }
        }

        matches
    }
}

/// PBWT-based IBS (identity-by-state) segment finder
#[derive(Debug)]
pub struct PbwtIbs {
    /// Forward PBWT state
    fwd_prefix: Vec<u32>,
    fwd_divergence: Vec<u32>,

    /// Backward PBWT state
    bwd_prefix: Vec<u32>,
    bwd_divergence: Vec<u32>,

    /// Number of haplotypes
    n_haps: usize,
}

impl PbwtIbs {
    /// Create a new PBWT IBS finder
    pub fn new(n_haps: usize) -> Self {
        Self {
            fwd_prefix: (0..n_haps as u32).collect(),
            fwd_divergence: vec![0; n_haps],
            bwd_prefix: (0..n_haps as u32).collect(),
            bwd_divergence: vec![0; n_haps],
            n_haps,
        }
    }

    /// Reset to initial state
    pub fn reset(&mut self) {
        for (i, p) in self.fwd_prefix.iter_mut().enumerate() {
            *p = i as u32;
        }
        self.fwd_divergence.fill(0);
        for (i, p) in self.bwd_prefix.iter_mut().enumerate() {
            *p = i as u32;
        }
        self.bwd_divergence.fill(0);
    }

    /// Get forward prefix array
    pub fn fwd_prefix(&self) -> &[u32] {
        &self.fwd_prefix
    }

    /// Get forward divergence array
    pub fn fwd_divergence(&self) -> &[u32] {
        &self.fwd_divergence
    }

    /// Get mutable forward prefix array
    pub fn fwd_prefix_mut(&mut self) -> &mut [u32] {
        &mut self.fwd_prefix
    }

    /// Get mutable forward divergence array
    pub fn fwd_divergence_mut(&mut self) -> &mut [u32] {
        &mut self.fwd_divergence
    }

    /// Select best matching haplotypes for a target using PBWT
    ///
    /// Returns indices of the `n_states` best matching haplotypes
    pub fn select_states(
        &self,
        target_hap: HapIdx,
        n_states: usize,
        exclude_self: bool,
    ) -> Vec<HapIdx> {
        // Find position of target in prefix array
        let target_pos = self
            .fwd_prefix
            .iter()
            .position(|&h| h == target_hap.0)
            .unwrap_or(0);

        let mut selected = Vec::with_capacity(n_states);
        let mut left = target_pos.saturating_sub(1);
        let mut right = target_pos + 1;

        // Expand outward from target position
        while selected.len() < n_states && (left > 0 || right < self.n_haps) {
            // Add from left
            if left > 0 {
                left -= 1;
                let hap = HapIdx::new(self.fwd_prefix[left]);
                if !exclude_self || hap != target_hap {
                    selected.push(hap);
                }
            }

            // Add from right
            if right < self.n_haps && selected.len() < n_states {
                let hap = HapIdx::new(self.fwd_prefix[right]);
                if !exclude_self || hap != target_hap {
                    selected.push(hap);
                }
                right += 1;
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
    fn test_pbwt_div_update() {
        let mut updater = PbwtDivUpdater::new(4);
        let mut prefix: Vec<u32> = vec![0, 1, 2, 3];
        let mut divergence: Vec<u32> = vec![0, 0, 0, 0];

        let alleles = vec![0u8, 1, 0, 1];
        updater.update(&alleles, 2, 0, &mut prefix, &mut divergence);

        // Check grouping
        assert_eq!(prefix, vec![0, 2, 1, 3]);

        // Divergence should be updated for haps that changed groups
        // First in each group keeps divergence, others get marker+1
    }

    #[test]
    fn test_pbwt_ibs_select() {
        let ibs = PbwtIbs::new(10);
        let selected = ibs.select_states(HapIdx::new(5), 4, true);
        assert_eq!(selected.len(), 4);
        assert!(!selected.contains(&HapIdx::new(5)));
    }
}