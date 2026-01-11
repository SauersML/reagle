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

/// PBWT updater with divergence array tracking
///
/// This optimized implementation uses flat arrays and Counting Sort
/// to avoid heap allocations during updates.
#[derive(Debug)]
pub struct PbwtDivUpdater {
    /// Number of haplotypes
    n_haps: usize,
    /// Current prefix array (flat)
    a: Vec<u32>,
    /// Current divergence array (flat)
    d: Vec<i32>,
    /// Scratch prefix array for double buffering
    scratch_a: Vec<u32>,
    /// Scratch divergence array for double buffering
    scratch_d: Vec<i32>,
    /// Propagation array for tracking max/min divergence across alleles
    p: Vec<i32>,
    /// Helper for counting sort: counts per allele
    counts: Vec<usize>,
    /// Helper for counting sort: starting offset per allele
    offsets: Vec<usize>,
}

impl PbwtDivUpdater {
    /// Create a new PBWT divergence updater
    pub fn new(n_haps: usize) -> Self {
        let max_alleles = 256; // Max u8 alleles
        Self {
            n_haps,
            a: Vec::new(), // Will be initialized on first use
            d: Vec::new(),
            scratch_a: Vec::new(),
            scratch_d: Vec::new(),
            p: vec![0; max_alleles],
            counts: vec![0; max_alleles],
            offsets: vec![0; max_alleles + 1],
        }
    }

    fn ensure_capacity(&mut self, n_alleles: usize) {
        if self.p.len() < n_alleles {
            self.p.resize(n_alleles, 0);
            self.counts.resize(n_alleles, 0);
            self.offsets.resize(n_alleles + 1, 0);
        }
        
        if self.scratch_a.len() < self.n_haps {
             self.scratch_a.resize(self.n_haps, 0);
             self.scratch_d.resize(self.n_haps, 0);
             // Also initialize 'a' and 'd' if empty (lazy init)
             if self.a.is_empty() {
                 self.a.resize(self.n_haps, 0);
                 self.d.resize(self.n_haps, 0);
             }
        }
    }

    /// Forward update of prefix and divergence arrays
    ///
    /// Uses In-Place Counting Sort to remove allocations.
    ///
    /// # Forward PBWT Semantics (vs Backward)
    ///
    /// - **Forward PBWT** (markers 0 → M-1): divergence[i] = marker where match STARTS
    ///   - Small divergence = long match (started far back)
    ///   - Uses MAX propagation: latest divergence point limits the match
    ///   - Reset to MIN_VALUE after output
    ///
    /// - **Backward PBWT** (markers M-1 → 0): divergence[i] = marker where match ENDS
    ///   - Small divergence = short match (ends soon)
    ///   - Uses MIN propagation: earliest end point limits the match
    ///   - Reset to MAX_VALUE after output
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

        // 1. Count frequencies of each allele (Counting Sort Phase 1)
        self.counts[..n_alleles].fill(0);

        for i in 0..self.n_haps {
            let hap = prefix[i];
            let allele = alleles[hap as usize] as usize;
            let allele = if allele >= n_alleles { 0 } else { allele };
            self.counts[allele] += 1;
        }

        // 2. Compute Offsets (Counting Sort Phase 2)
        let mut running = 0;
        for i in 0..n_alleles {
            self.offsets[i] = running;
            running += self.counts[i];
        }

        // 3. Initialize p array and reset counts for scatter pass
        let init_value = (marker + 1) as i32;
        self.counts[..n_alleles].fill(0);
        self.p[..n_alleles].fill(init_value);

        // 4. Scatter to scratch buffers with p propagation (Counting Sort Phase 3)
        // OPTIMIZED: Instead of O(n_alleles) inner loop, track a running global max.
        // p[allele] = max(init_value, max_div_seen_since_last_output_of_allele)
        // This works because the "propagation" means: for each allele, the output
        // divergence is the maximum of:
        //   - init_value (marker + 1)
        //   - all divergence values seen since the last haplotype of that allele was output
        // We can track this with a single running max that gets reset per-allele on output.
        let mut running_max = i32::MIN;

        for i in 0..self.n_haps {
            let hap = prefix[i];
            let div = divergence[i];
            let allele = alleles[hap as usize] as usize;
            let allele = if allele >= n_alleles { 0 } else { allele };

            // Update running max with current divergence
            running_max = running_max.max(div);

            // p[allele] should be max of init_value and all divs since last output
            // We update it with the running_max
            self.p[allele] = self.p[allele].max(running_max);

            let base = self.offsets[allele];
            let offset = self.counts[allele];
            let pos = base + offset;

            self.scratch_a[pos] = hap;
            self.scratch_d[pos] = self.p[allele];

            // Reset p for this allele after output (so next time it starts fresh)
            self.p[allele] = i32::MIN;

            self.counts[allele] += 1;
        }

        // 5. Copy back
        prefix.copy_from_slice(&self.scratch_a[..self.n_haps]);
        divergence[..self.n_haps].copy_from_slice(&self.scratch_d[..self.n_haps]);
    }

    /// Backward update of prefix and divergence arrays
    ///
    /// Uses In-Place Counting Sort.
    ///
    /// # Backward PBWT Semantics (vs Forward)
    ///
    /// - **Forward PBWT** (markers 0 → M-1): divergence[i] = marker where match STARTS
    ///   - Small divergence = long match (started far back)
    ///   - Uses MAX propagation: latest divergence point limits the match
    ///   - Reset to MIN_VALUE after output
    ///
    /// - **Backward PBWT** (markers M-1 → 0): divergence[i] = marker where match ENDS
    ///   - Small divergence = short match (ends soon)
    ///   - Uses MIN propagation: earliest end point limits the match
    ///   - Reset to MAX_VALUE after output
    ///
    /// This matches the Java Beagle implementation in PbwtDivUpdater.bwdUpdate.
    pub fn bwd_update(
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

        // 1. Initialize p array - Backward uses (marker - 1) as init value
        let init_value = (marker as i32) - 1;

        // 2. Count frequencies
        self.counts[..n_alleles].fill(0);
        for i in 0..self.n_haps {
            let hap = prefix[i];
            let allele = alleles[hap as usize] as usize;
            let allele = if allele >= n_alleles { 0 } else { allele };
            self.counts[allele] += 1;
        }

        // 3. Compute Offsets
        let mut running = 0;
        for i in 0..n_alleles {
            self.offsets[i] = running;
            running += self.counts[i];
        }

        // 4. Scatter with MIN propagation for backward PBWT
        // OPTIMIZED: Instead of O(n_alleles) inner loop, track a running global min.
        // p[allele] = min(init_value, min_div_seen_since_last_output_of_allele)
        self.counts[..n_alleles].fill(0);
        self.p[..n_alleles].fill(init_value);
        let mut running_min = i32::MAX;

        for i in 0..self.n_haps {
            let hap = prefix[i];
            let div = divergence[i];
            let allele = alleles[hap as usize] as usize;
            let allele = if allele >= n_alleles { 0 } else { allele };

            // Update running min with current divergence
            running_min = running_min.min(div);

            // p[allele] should be min of init_value and all divs since last output
            self.p[allele] = self.p[allele].min(running_min);

            let base = self.offsets[allele];
            let offset = self.counts[allele];
            let pos = base + offset;

            self.scratch_a[pos] = hap;
            self.scratch_d[pos] = self.p[allele];

            // Reset to MAX so next haplotype takes its own divergence
            self.p[allele] = i32::MAX;
            self.counts[allele] += 1;
        }

        // 5. Copy back
        prefix.copy_from_slice(&self.scratch_a[..self.n_haps]);
        divergence[..self.n_haps].copy_from_slice(&self.scratch_d[..self.n_haps]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
