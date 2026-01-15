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

/// Snapshot of PBWT state for window handoff
///
/// Contains the prefix array and divergence array at a specific marker position.
/// Used to initialize PBWT in the next window without recomputation.
#[derive(Debug, Clone)]
pub struct PbwtState {
    /// Prefix array (ppa): current haplotype sort order
    pub ppa: Vec<u32>,
    /// Divergence array: positions where haplotypes diverge
    pub div: Vec<i32>,
    /// Marker position this state corresponds to
    pub marker_pos: usize,
}

impl PbwtState {
    /// Create a new PBWT state snapshot
    pub fn new(ppa: Vec<u32>, div: Vec<i32>, marker_pos: usize) -> Self {
        Self {
            ppa,
            div,
            marker_pos,
        }
    }

    /// Extract state from a PbwtDivUpdater at a given marker
    pub fn from_updater(updater: &PbwtDivUpdater, marker_pos: usize) -> Self {
        Self {
            ppa: updater.a.clone(),
            div: updater.d.clone(),
            marker_pos,
        }
    }
}

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
    /// Pre-permuted alleles: permuted_alleles[i] = alleles[prefix[i]]
    /// Converts random-access gather to sequential access for counting and scatter
    permuted_alleles: Vec<u8>,
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
            permuted_alleles: Vec::new(),
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
             self.permuted_alleles.resize(self.n_haps, 0);
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
    /// # Missing Data Handling
    ///
    /// Missing data (allele >= n_alleles, typically 255) is placed in its own bin
    /// at index n_alleles. This prevents reference bias that would occur from
    /// grouping missing data with the reference allele.
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

        // Use n_alleles + 1 bins: 0..n_alleles for valid alleles, n_alleles for missing
        let n_bins = n_alleles + 1;
        self.ensure_capacity(n_bins);

        // 0. Pre-permute alleles: gather alleles[prefix[i]] into contiguous buffer.
        // This converts subsequent random-access patterns to sequential access.
        // Single gather pass here enables two sequential passes below.
        for i in 0..self.n_haps {
            self.permuted_alleles[i] = alleles[prefix[i] as usize];
        }

        // 1. Count frequencies of each allele (Counting Sort Phase 1) - now sequential access
        self.counts[..n_bins].fill(0);

        if n_alleles == 2 {
            // Fast path for biallelic sites - single pass counting
            // Computes sum of bytes AND detects non-0/1 values simultaneously
            let data = &self.permuted_alleles[..self.n_haps];

            // Single-pass: sum all bytes and OR-reduce to detect values > 1
            // For pure 0/1 data, sum = count of 1s, and any_high will be 0 or 1
            // For data with missing (>1), any_high will have bits 1-7 set
            let mut sum = 0usize;
            let mut any_high = 0u8;

            // Process in chunks for autovectorization
            let chunks = data.chunks_exact(32);
            let remainder = chunks.remainder();

            for chunk in chunks {
                for &b in chunk {
                    sum += b as usize;
                    any_high |= b;
                }
            }
            for &b in remainder {
                sum += b as usize;
                any_high |= b;
            }

            // any_high > 1 means some byte had value > 1 (missing data)
            if any_high <= 1 {
                // Pure 0/1 data - sum equals count of 1s
                self.counts[0] = self.n_haps - sum;
                self.counts[1] = sum;
                self.counts[n_alleles] = 0;
            } else {
                // Has missing data - need detailed count
                // This path is rare (<1% of markers) so extra scan is acceptable
                let mut count1 = 0usize;
                let mut count_miss = 0usize;

                let chunks = data.chunks_exact(32);
                let remainder = chunks.remainder();

                for chunk in chunks {
                    for &b in chunk {
                        count1 += (b == 1) as usize;
                        count_miss += (b > 1) as usize;
                    }
                }
                for &b in remainder {
                    count1 += (b == 1) as usize;
                    count_miss += (b > 1) as usize;
                }

                self.counts[0] = self.n_haps - count1 - count_miss;
                self.counts[1] = count1;
                self.counts[n_alleles] = count_miss;
            }
        } else {
            // General path for multiallelic
            for i in 0..self.n_haps {
                let allele = self.permuted_alleles[i] as usize;
                // Map missing/invalid alleles to the dedicated missing bin
                let bin = if allele >= n_alleles { n_alleles } else { allele };
                self.counts[bin] += 1;
            }
        }

        // 2. Compute Offsets (Counting Sort Phase 2)
        let mut running = 0;
        for i in 0..n_bins {
            self.offsets[i] = running;
            running += self.counts[i];
        }

        // 3. Check if there's any missing data (before resetting counts)
        // This lets us use a faster 2-bin path when there's no missing data
        let has_missing = self.counts[n_alleles] > 0;

        // 4. Initialize p array and reset counts for scatter pass
        let init_value = (marker + 1) as i32;
        self.counts[..n_bins].fill(0);
        self.p[..n_bins].fill(init_value);

        // 5. Scatter to scratch buffers with p propagation (Counting Sort Phase 3)
        // Now uses permuted_alleles for sequential access instead of random gather
        if n_alleles == 2 && !has_missing {
            // Fast biallelic path when no missing data - only 2 bins needed
            // This is the common case and avoids the 3rd comparison per haplotype
            let mut p0 = init_value;
            let mut p1 = init_value;
            let base0 = self.offsets[0];
            let base1 = self.offsets[1];
            let mut count0 = 0usize;
            let mut count1 = 0usize;

            for i in 0..self.n_haps {
                let hap = prefix[i];
                let div = divergence[i];
                let allele = self.permuted_alleles[i]; // Sequential access

                // Propagate max to both bins
                if div > p0 { p0 = div; }
                if div > p1 { p1 = div; }

                if allele == 0 {
                    let pos = base0 + count0;
                    self.scratch_a[pos] = hap;
                    self.scratch_d[pos] = p0;
                    p0 = i32::MIN;
                    count0 += 1;
                } else {
                    let pos = base1 + count1;
                    self.scratch_a[pos] = hap;
                    self.scratch_d[pos] = p1;
                    p1 = i32::MIN;
                    count1 += 1;
                }
            }
        } else if n_alleles == 2 {
            // Biallelic path with missing data - needs 3 bins
            let mut p0 = init_value;
            let mut p1 = init_value;
            let mut p_miss = init_value;
            let base0 = self.offsets[0];
            let base1 = self.offsets[1];
            let base_miss = self.offsets[2];
            let mut count0 = 0usize;
            let mut count1 = 0usize;
            let mut count_miss = 0usize;

            for i in 0..self.n_haps {
                let hap = prefix[i];
                let div = divergence[i];
                let allele = self.permuted_alleles[i]; // Sequential access

                // Propagate max to all bins - this is essential for correctness
                // The divergence must propagate through all allele bins
                if div > p0 { p0 = div; }
                if div > p1 { p1 = div; }
                if div > p_miss { p_miss = div; }

                match allele {
                    0 => {
                        let pos = base0 + count0;
                        self.scratch_a[pos] = hap;
                        self.scratch_d[pos] = p0;
                        p0 = i32::MIN;
                        count0 += 1;
                    }
                    1 => {
                        let pos = base1 + count1;
                        self.scratch_a[pos] = hap;
                        self.scratch_d[pos] = p1;
                        p1 = i32::MIN;
                        count1 += 1;
                    }
                    _ => {
                        // Missing or invalid allele
                        let pos = base_miss + count_miss;
                        self.scratch_a[pos] = hap;
                        self.scratch_d[pos] = p_miss;
                        p_miss = i32::MIN;
                        count_miss += 1;
                    }
                }
            }
        } else {
            // General multiallelic path
            for i in 0..self.n_haps {
                let hap = prefix[i];
                let div = divergence[i];
                let allele = self.permuted_alleles[i] as usize; // Sequential access
                // Map missing/invalid alleles to the dedicated missing bin
                let bin = if allele >= n_alleles { n_alleles } else { allele };

                // Update p (Max Divergence Propagation) for ALL bins
                for j in 0..n_bins {
                    if div > self.p[j] {
                        self.p[j] = div;
                    }
                }

                let base = self.offsets[bin];
                let offset = self.counts[bin];
                let pos = base + offset;

                self.scratch_a[pos] = hap;
                self.scratch_d[pos] = self.p[bin];

                // Reset p for this bin after output
                self.p[bin] = i32::MIN;

                self.counts[bin] += 1;
            }
        }

        // 6. Copy back
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
    /// # Missing Data Handling
    ///
    /// Missing data (allele >= n_alleles, typically 255) is placed in its own bin
    /// at index n_alleles. This prevents reference bias that would occur from
    /// grouping missing data with the reference allele.
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

        // Use n_alleles + 1 bins: 0..n_alleles for valid alleles, n_alleles for missing
        let n_bins = n_alleles + 1;
        self.ensure_capacity(n_bins);

        // 0. Pre-permute alleles: gather alleles[prefix[i]] into contiguous buffer.
        // This converts subsequent random-access patterns to sequential access.
        for i in 0..self.n_haps {
            self.permuted_alleles[i] = alleles[prefix[i] as usize];
        }

        // 1. Initialize p array for backward PBWT
        //
        // Java uses marker-1 for initialization. This correctly handles allele boundaries:
        // when two adjacent haplotypes in sorted order have different alleles at marker m,
        // their match must end at m-1 (they differ at m). Using i32::MAX would incorrectly
        // suggest they match indefinitely.
        let init_value = (marker as i32) - 1;

        // 2. Count frequencies - now sequential access via permuted_alleles
        self.counts[..n_bins].fill(0);

        if n_alleles == 2 {
            // Fast path for biallelic sites - single pass counting
            let data = &self.permuted_alleles[..self.n_haps];

            // Single-pass: sum all bytes and OR-reduce to detect values > 1
            let mut sum = 0usize;
            let mut any_high = 0u8;

            let chunks = data.chunks_exact(32);
            let remainder = chunks.remainder();

            for chunk in chunks {
                for &b in chunk {
                    sum += b as usize;
                    any_high |= b;
                }
            }
            for &b in remainder {
                sum += b as usize;
                any_high |= b;
            }

            if any_high <= 1 {
                // Pure 0/1 data - sum equals count of 1s
                self.counts[0] = self.n_haps - sum;
                self.counts[1] = sum;
                self.counts[n_alleles] = 0;
            } else {
                // Has missing data - need detailed count
                let mut count1 = 0usize;
                let mut count_miss = 0usize;

                let chunks = data.chunks_exact(32);
                let remainder = chunks.remainder();

                for chunk in chunks {
                    for &b in chunk {
                        count1 += (b == 1) as usize;
                        count_miss += (b > 1) as usize;
                    }
                }
                for &b in remainder {
                    count1 += (b == 1) as usize;
                    count_miss += (b > 1) as usize;
                }

                self.counts[0] = self.n_haps - count1 - count_miss;
                self.counts[1] = count1;
                self.counts[n_alleles] = count_miss;
            }
        } else {
            // General path for multiallelic
            for i in 0..self.n_haps {
                let allele = self.permuted_alleles[i] as usize;
                let bin = if allele >= n_alleles { n_alleles } else { allele };
                self.counts[bin] += 1;
            }
        }

        // 3. Compute Offsets
        let mut running = 0;
        for i in 0..n_bins {
            self.offsets[i] = running;
            running += self.counts[i];
        }

        // 4. Check if there's any missing data (before resetting counts)
        let has_missing = self.counts[n_alleles] > 0;

        // 5. Scatter with MIN propagation for backward PBWT
        // p[j] tracks the minimum divergence seen since last output for allele j
        // Now uses permuted_alleles for sequential access
        self.counts[..n_bins].fill(0);
        self.p[..n_bins].fill(init_value);

        if n_alleles == 2 && !has_missing {
            // Fast biallelic path when no missing data - only 2 bins needed
            let mut p0 = init_value;
            let mut p1 = init_value;
            let base0 = self.offsets[0];
            let base1 = self.offsets[1];
            let mut count0 = 0usize;
            let mut count1 = 0usize;

            for i in 0..self.n_haps {
                let hap = prefix[i];
                let div = divergence[i];
                let allele = self.permuted_alleles[i]; // Sequential access

                // Propagate min to both bins (backward PBWT)
                if div < p0 { p0 = div; }
                if div < p1 { p1 = div; }

                if allele == 0 {
                    let pos = base0 + count0;
                    self.scratch_a[pos] = hap;
                    self.scratch_d[pos] = p0;
                    p0 = i32::MAX;
                    count0 += 1;
                } else {
                    let pos = base1 + count1;
                    self.scratch_a[pos] = hap;
                    self.scratch_d[pos] = p1;
                    p1 = i32::MAX;
                    count1 += 1;
                }
            }
        } else if n_alleles == 2 {
            // Biallelic path with missing data - needs 3 bins
            let mut p0 = init_value;
            let mut p1 = init_value;
            let mut p_miss = init_value;
            let base0 = self.offsets[0];
            let base1 = self.offsets[1];
            let base_miss = self.offsets[2];
            let mut count0 = 0usize;
            let mut count1 = 0usize;
            let mut count_miss = 0usize;

            for i in 0..self.n_haps {
                let hap = prefix[i];
                let div = divergence[i];
                let allele = self.permuted_alleles[i]; // Sequential access

                // Propagate min to all bins (backward PBWT)
                if div < p0 { p0 = div; }
                if div < p1 { p1 = div; }
                if div < p_miss { p_miss = div; }

                match allele {
                    0 => {
                        let pos = base0 + count0;
                        self.scratch_a[pos] = hap;
                        self.scratch_d[pos] = p0;
                        p0 = i32::MAX;
                        count0 += 1;
                    }
                    1 => {
                        let pos = base1 + count1;
                        self.scratch_a[pos] = hap;
                        self.scratch_d[pos] = p1;
                        p1 = i32::MAX;
                        count1 += 1;
                    }
                    _ => {
                        // Missing or invalid allele
                        let pos = base_miss + count_miss;
                        self.scratch_a[pos] = hap;
                        self.scratch_d[pos] = p_miss;
                        p_miss = i32::MAX;
                        count_miss += 1;
                    }
                }
            }
        } else {
            // General multiallelic path
            for i in 0..self.n_haps {
                let hap = prefix[i];
                let div = divergence[i];
                let allele = self.permuted_alleles[i] as usize; // Sequential access
                // Map missing/invalid alleles to the dedicated missing bin
                let bin = if allele >= n_alleles { n_alleles } else { allele };

                // Update p: min(p, div) for backward PBWT
                // Smaller divergence = earlier end point = shorter match
                // We propagate the minimum to find the "worst case" match length
                for j in 0..n_bins {
                    if div < self.p[j] {
                        self.p[j] = div;
                    }
                }

                let base = self.offsets[bin];
                let offset = self.counts[bin];
                let pos = base + offset;

                self.scratch_a[pos] = hap;
                self.scratch_d[pos] = self.p[bin];

                // Reset to MAX so next haplotype takes its own divergence
                self.p[bin] = i32::MAX;
                self.counts[bin] += 1;
            }
        }

        // 6. Copy back
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

    /// Test PBWT divergence propagation across multiple markers.
    /// This tests the CRITIC's claim that "PBWT forgets history at every step".
    /// If that were true, divergence would reset to marker+1 at each step,
    /// and we couldn't find matches longer than 1 marker.
    #[test]
    fn test_pbwt_multi_marker_divergence_propagation() {
        let mut updater = PbwtDivUpdater::new(4);
        let mut prefix: Vec<u32> = vec![0, 1, 2, 3];
        let mut divergence: Vec<i32> = vec![0, 0, 0, 0]; // Match started at marker 0

        // Haplotypes:
        // Hap 0: [0, 0, 0]  - all allele 0
        // Hap 1: [0, 0, 0]  - all allele 0 (matches hap 0)
        // Hap 2: [1, 1, 1]  - all allele 1
        // Hap 3: [1, 1, 1]  - all allele 1 (matches hap 2)

        // Marker 0
        let alleles_m0 = vec![0u8, 0, 1, 1];
        updater.fwd_update(&alleles_m0, 2, 0, &mut prefix, &mut divergence);
        // After m0: prefix=[0,1,2,3], divergence=[1,0,1,0]
        // Haps 0,1 grouped (allele 0), 2,3 grouped (allele 1)

        // Marker 1
        let alleles_m1 = vec![0u8, 0, 1, 1];
        updater.fwd_update(&alleles_m1, 2, 1, &mut prefix, &mut divergence);
        // After m1: same grouping, divergence should propagate
        // If CRITIC were right, divergence would be [2,2,2,2] (marker+1)
        // But with correct propagation, hap 1's divergence carries forward from m0

        // Marker 2
        let alleles_m2 = vec![0u8, 0, 1, 1];
        updater.fwd_update(&alleles_m2, 2, 2, &mut prefix, &mut divergence);

        // Key assertion: If divergence propagates correctly, hap 1 should have
        // divergence value propagated from marker 0 (when it was first grouped with hap 0).
        // If CRITIC were right, all divergences would be marker+1 = 3.

        // The second hap in each allele group should have LOW divergence (0 or 1),
        // indicating a match that started early.
        assert!(
            divergence[1] < 3,
            "PBWT divergence NOT propagating! Second hap in group has div={}, expected < 3",
            divergence[1]
        );
        assert!(
            divergence[3] < 3,
            "PBWT divergence NOT propagating! Fourth hap in group has div={}, expected < 3",
            divergence[3]
        );

        // First hap in each group can have marker+1 (no predecessor with same allele yet)
        // That's expected behavior, not a bug.
    }

    /// Test that matches are detected when haplotypes share a long identical segment.
    /// This tests the core PBWT functionality for finding IBS matches.
    #[test]
    fn test_pbwt_long_match_detection() {
        let mut updater = PbwtDivUpdater::new(4);
        let mut prefix: Vec<u32> = vec![0, 1, 2, 3];
        let mut divergence: Vec<i32> = vec![0, 0, 0, 0];

        // Haplotypes sharing a 5-marker segment:
        // Hap 0: [0, 0, 0, 0, 0]
        // Hap 1: [0, 0, 0, 0, 0]  - matches hap 0 from start
        // Hap 2: [1, 0, 0, 0, 0]  - matches haps 0,1 from marker 1
        // Hap 3: [1, 1, 0, 0, 0]  - matches haps 0,1,2 from marker 2

        for marker in 0..5 {
            let alleles = vec![
                0u8,                                 // hap 0: always 0
                0,                                   // hap 1: always 0
                if marker < 1 { 1 } else { 0 },      // hap 2: 1 at m0
                if marker < 2 { 1 } else { 0 },      // hap 3: 1 at m0,m1
            ];
            updater.fwd_update(&alleles, 2, marker, &mut prefix, &mut divergence);
        }

        // After marker 4:
        // Haps 0,1 have been together since marker 0 -> one should have low divergence
        // Hap 2 joined allele-0 group at marker 1 -> divergence should be ~1
        // Hap 3 joined allele-0 group at marker 2 -> divergence should be ~2

        // Find hap 1 in the sorted prefix array
        let hap1_pos = prefix.iter().position(|&h| h == 1).unwrap();
        let hap1_div = divergence[hap1_pos];

        // Hap 1 should have a match starting early (div close to 0 or 1)
        assert!(
            hap1_div <= 2,
            "Hap 1 should have long match with hap 0, but divergence is {} (expected <= 2)",
            hap1_div
        );
    }

    /// Test that missing data (255) is handled correctly without reference bias.
    ///
    /// This tests the fix for the CRITIC-identified bug where missing data was
    /// being mapped to REF (allele 0), creating systematic reference bias.
    /// Missing data should be placed in its own bin, not grouped with REF or ALT.
    #[test]
    fn test_pbwt_missing_data_no_reference_bias() {
        let mut updater = PbwtDivUpdater::new(4);
        let mut prefix: Vec<u32> = vec![0, 1, 2, 3];
        let mut divergence: Vec<i32> = vec![0, 0, 0, 0];

        // Haplotypes:
        // Hap 0: REF (0)
        // Hap 1: ALT (1)
        // Hap 2: MISSING (255)
        // Hap 3: REF (0)
        let alleles = vec![0u8, 1, 255, 0];
        updater.fwd_update(&alleles, 2, 0, &mut prefix, &mut divergence);

        // With the fix: missing (255) goes to bin 2 (separate from REF and ALT)
        // Sorted order should be: [REF haps (0,3), ALT haps (1), MISSING haps (2)]
        // = [0, 3, 1, 2] or similar grouping

        // Key assertion: hap 2 (MISSING) should NOT be grouped with hap 0 (REF)
        // Find positions of hap 0 and hap 2 in sorted array
        let hap0_pos = prefix.iter().position(|&h| h == 0).unwrap();
        let hap2_pos = prefix.iter().position(|&h| h == 2).unwrap();
        let hap1_pos = prefix.iter().position(|&h| h == 1).unwrap();
        let hap3_pos = prefix.iter().position(|&h| h == 3).unwrap();

        // REF haps (0 and 3) should be adjacent (grouped together)
        assert!(
            (hap0_pos as i32 - hap3_pos as i32).abs() == 1,
            "REF haps 0 and 3 should be adjacent in PBWT. Positions: hap0={}, hap3={}",
            hap0_pos, hap3_pos
        );

        // MISSING hap (2) should NOT be adjacent to REF haps
        // If the bug existed, hap 2 would be in the REF group
        let hap2_near_ref = (hap2_pos as i32 - hap0_pos as i32).abs() == 1
            || (hap2_pos as i32 - hap3_pos as i32).abs() == 1;

        // hap 2 should be in its own bin at the end, not adjacent to REF
        // Actually, the order is: REF bin, ALT bin, MISSING bin
        // So hap 2 should be at the end (position 3)
        assert!(
            !hap2_near_ref || hap1_pos < hap2_pos,
            "MISSING hap 2 should be in separate bin from REF. \
             Positions: hap0={}, hap1={}, hap2={}, hap3={}. \
             If hap2 is adjacent to REF, reference bias exists!",
            hap0_pos, hap1_pos, hap2_pos, hap3_pos
        );
    }

    /// Test backward PBWT also handles missing data correctly.
    #[test]
    fn test_pbwt_bwd_missing_data_no_reference_bias() {
        let mut updater = PbwtDivUpdater::new(4);
        let mut prefix: Vec<u32> = vec![0, 1, 2, 3];
        let mut divergence: Vec<i32> = vec![10, 10, 10, 10]; // High initial values for backward

        // Haplotypes:
        // Hap 0: REF (0)
        // Hap 1: ALT (1)
        // Hap 2: MISSING (255)
        // Hap 3: REF (0)
        let alleles = vec![0u8, 1, 255, 0];
        let marker = 5; // Use marker 5 so init_value = 4
        updater.bwd_update(&alleles, 2, marker, &mut prefix, &mut divergence);

        // Same logic as forward: MISSING should be in its own bin
        let hap0_pos = prefix.iter().position(|&h| h == 0).unwrap();
        let hap2_pos = prefix.iter().position(|&h| h == 2).unwrap();
        let hap3_pos = prefix.iter().position(|&h| h == 3).unwrap();

        // REF haps (0 and 3) should be adjacent
        assert!(
            (hap0_pos as i32 - hap3_pos as i32).abs() == 1,
            "REF haps 0 and 3 should be adjacent in backward PBWT. Positions: hap0={}, hap3={}",
            hap0_pos, hap3_pos
        );

        // MISSING hap should be separate from REF group
        let hap2_near_ref = (hap2_pos as i32 - hap0_pos as i32).abs() == 1
            || (hap2_pos as i32 - hap3_pos as i32).abs() == 1;
        assert!(
            !hap2_near_ref,
            "MISSING hap 2 should NOT be adjacent to REF haps in backward PBWT. Position: {}",
            hap2_pos
        );
    }
}
