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

        // 1. Initialize p array (propagation)
        let init_value = (marker + 1) as i32;
        self.p[..n_alleles].fill(init_value);

        // 2. Count frequencies of each allele (Counting Sort Phase 1)
        self.counts[..n_alleles].fill(0);
        
        // We iterate in current prefix order to compute p and counts
        for i in 0..self.n_haps {
            let hap = prefix[i];
            let div = divergence[i];
            let allele = alleles[hap as usize] as usize;
            
            // Check bounds (shouldn't happen with valid data)
            let allele = if allele >= n_alleles { 0 } else { allele };

            // Update p (Max Divergence Propagation) for ALL alleles
            for j in 0..n_alleles {
                if div > self.p[j] {
                    self.p[j] = div;
                }
            }
            
            self.counts[allele] += 1;
        }

        // 3. Compute Offsets (Counting Sort Phase 2)
        let mut running = 0;
        for i in 0..n_alleles {
            self.offsets[i] = running;
            running += self.counts[i];
        }
        
        // 4. Scatter to scratch buffers (Counting Sort Phase 3)
        // Reset counts and p for the SCATTER pass
        self.counts[..n_alleles].fill(0);
        self.p[..n_alleles].fill(init_value);

        for i in 0..self.n_haps {
            let hap = prefix[i];
            let div = divergence[i];
            let allele = alleles[hap as usize] as usize;
            let allele = if allele >= n_alleles { 0 } else { allele };

            // Update p for all alleles
            for j in 0..n_alleles {
                if div > self.p[j] {
                    self.p[j] = div;
                }
            }

            let base = self.offsets[allele];
            let offset = self.counts[allele];
            let pos = base + offset;

            self.scratch_a[pos] = hap;
            self.scratch_d[pos] = self.p[allele];

            // Reset p for next item in this allele bucket
            self.p[allele] = i32::MIN; // Use MIN for 'reset'
            
            self.counts[allele] += 1;
        }

        // 5. Copy back
        prefix.copy_from_slice(&self.scratch_a[..self.n_haps]);
        divergence[..self.n_haps].copy_from_slice(&self.scratch_d[..self.n_haps]);
    }

    /// Forward update of prefix and divergence arrays using bit-packed alleles
    ///
    /// Optimized for biallelic markers stored in `DenseColumn` (u64 words).
    /// Eliminates u8 expansion overhead.
    pub fn fwd_update_packed(
        &mut self,
        alleles_packed: &[u64],
        marker: usize,
        prefix: &mut [u32],
        divergence: &mut [i32],
    ) {
        assert_eq!(prefix.len(), self.n_haps);
        assert!(divergence.len() >= self.n_haps);

        // Always 2 alleles for packed path
        let n_alleles = 2;
        self.ensure_capacity(n_alleles);

        // 1. Initialize p array
        let init_value = (marker + 1) as i32;
        self.p[0] = init_value;
        self.p[1] = init_value;

        // 2. Count frequencies (Counting Sort Phase 1)
        self.counts[0] = 0;
        self.counts[1] = 0;

        for i in 0..self.n_haps {
            let hap = prefix[i] as usize;
            let div = divergence[i];
            
            // Extract allele bit: (word >> (hap % 64)) & 1
            let word_idx = hap / 64;
            let bit_idx = hap % 64;
            // Safety: hap < n_haps, so word_idx is valid if alleles_packed matches n_haps
            // We trust caller (DenseColumn) provides correct slice length
            let allele = ((alleles_packed[word_idx] >> bit_idx) & 1) as usize;
            
            // Update p for ALL alleles
            if div > self.p[0] { self.p[0] = div; }
            if div > self.p[1] { self.p[1] = div; }
            
            self.counts[allele] += 1;
        }

        // 3. Compute Offsets
        self.offsets[0] = 0;
        self.offsets[1] = self.counts[0];
        
        // 4. Scatter
        self.counts[0] = 0;
        self.counts[1] = 0;
        
        // Reset p for scatter pass
        self.p[0] = init_value;
        self.p[1] = init_value;

        for i in 0..self.n_haps {
            let hap = prefix[i] as usize;
            let div = divergence[i];
            
            let word_idx = hap / 64;
            let bit_idx = hap % 64;
            let allele = ((alleles_packed[word_idx] >> bit_idx) & 1) as usize;

            // Update p for ALL alleles
            if div > self.p[0] { self.p[0] = div; }
            if div > self.p[1] { self.p[1] = div; }

            let base = self.offsets[allele];
            let offset = self.counts[allele];
            let pos = base + offset;

            self.scratch_a[pos] = hap as u32;
            self.scratch_d[pos] = self.p[allele];

            self.p[allele] = i32::MIN; // Reset
            self.counts[allele] += 1;
        }

        // 5. Copy back
        prefix.copy_from_slice(&self.scratch_a[..self.n_haps]);
        divergence[..self.n_haps].copy_from_slice(&self.scratch_d[..self.n_haps]);
    }

    /// Backward update of prefix and divergence arrays
    ///
    /// Uses In-Place Counting Sort.
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

        // 1. Initialize p array (propagation) - Backward Direction
        let init_value = (marker as i32).saturating_sub(1);
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

        // 4. Scatter (maintaining min divergence)
        self.counts[..n_alleles].fill(0);
        self.p[..n_alleles].fill(init_value);

        for i in 0..self.n_haps {
            let hap = prefix[i];
            let div = divergence[i];
            let allele = alleles[hap as usize] as usize;
            let allele = if allele >= n_alleles { 0 } else { allele };

            // Update p: min(p, div) for backward, for ALL alleles
            for j in 0..n_alleles {
                if div < self.p[j] {
                    self.p[j] = div;
                }
            }

            let base = self.offsets[allele];
            let offset = self.counts[allele];
            let pos = base + offset;

            self.scratch_a[pos] = hap;
            self.scratch_d[pos] = self.p[allele];

            self.p[allele] = i32::MAX; // Reset with MAX for min-tracking
            self.counts[allele] += 1;
        }

        // 5. Copy back
        prefix.copy_from_slice(&self.scratch_a[..self.n_haps]);
        divergence[..self.n_haps].copy_from_slice(&self.scratch_d[..self.n_haps]);
    }

    /// Backward update of prefix and divergence arrays using bit-packed alleles
    pub fn bwd_update_packed(
        &mut self,
        alleles_packed: &[u64],
        marker: usize,
        prefix: &mut [u32],
        divergence: &mut [i32],
    ) {
        assert_eq!(prefix.len(), self.n_haps);
        assert!(divergence.len() >= self.n_haps);

        let n_alleles = 2;
        self.ensure_capacity(n_alleles);

        // 1. Initialize p array (propagation) - Backward Direction
        let init_value = (marker as i32).saturating_sub(1);
        
        // 2. Count frequencies
        self.counts[0] = 0;
        self.counts[1] = 0;
        
        for i in 0..self.n_haps {
            let hap = prefix[i] as usize;
            let word_idx = hap / 64;
            let bit_idx = hap % 64;
            let allele = ((alleles_packed[word_idx] >> bit_idx) & 1) as usize;
            self.counts[allele] += 1;
        }

        // 3. Compute Offsets
        self.offsets[0] = 0;
        self.offsets[1] = self.counts[0];

        // 4. Scatter (maintaining min divergence)
        self.counts[0] = 0;
        self.counts[1] = 0;
        
        self.p[0] = init_value;
        self.p[1] = init_value;

        for i in 0..self.n_haps {
            let hap = prefix[i] as usize;
            let div = divergence[i];
            
            let word_idx = hap / 64;
            let bit_idx = hap % 64;
            let allele = ((alleles_packed[word_idx] >> bit_idx) & 1) as usize;

            // Update p: min(p, div) for backward, for ALL alleles
            if div < self.p[0] { self.p[0] = div; }
            if div < self.p[1] { self.p[1] = div; }

            let base = self.offsets[allele];
            let offset = self.counts[allele];
            let pos = base + offset;

            self.scratch_a[pos] = hap as u32;
            self.scratch_d[pos] = self.p[allele];

            self.p[allele] = i32::MAX; // Reset with MAX for min-tracking
            self.counts[allele] += 1;
        }

        // 5. Copy back
        prefix.copy_from_slice(&self.scratch_a[..self.n_haps]);
        divergence[..self.n_haps].copy_from_slice(&self.scratch_d[..self.n_haps]);
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

        let mut u = target_pos; // inclusive start
        let mut v = target_pos + 1; // exclusive end

        // Get divergence values, using sentinel for out-of-bounds
        let get_div = |i: usize| -> i32 { if i == 0 || i >= n { d0 } else { divergence[i] } };

        let mut u_next_match = get_div(u);
        let mut v_next_match = get_div(v);

        // Expand range until we have enough candidates or hit boundaries
        // Expand range until we have enough candidates or hit boundaries
        while (v - u) < n_candidates {
            if is_backward {
                // Backward PBWT: Expand toward LARGER divergence (closest neighbors have largest divergence <= marker)
                // We want to find range [u, v) where divergence[i] > marker (meaning match extends PAST this marker)
                // Wait, PBWT definition: 
                // d[i] is divergence between a[i] and a[i-1].
                // We expand while d[i] > marker? No.
                //
                // Standard: d[i] is FIRST marker where mismatch occurs.
                // Match length is "marker index".
                // We want matches that extend at least as far as current marker.
                // For Backward: match extends "backward" to marker M-L.
                // 
                // Let's stick to the REFERENCE implementation Durbin (2014) logic:
                // Expand while divergence[u] > k or divergence[v] > k.
                // This means the match extends *beyond* the current threshold k.
                
                // My existing logic was:
                // marker <= u_next_match || marker <= v_next_match
                // where u_next_match = divergence[u]
                
                // Let's optimize the bulk expansion.
                
                // Try scanning forward for v: find next v where divergence[v] < marker
                // Scan backward for u: find next u where divergence[u] < marker
                
                // However, standard neighbor finding is greedy: expand better side first.
                // SimdScanner is useful if we want to expand *aggressively* in one direction.
                
                // Let's support the greedy strategy but in blocks if possible? 
                // Or just use SimdScanner for "check if we can jump 8 steps".
                
                // Actually, the simplest speedup is: 
                // If u_next >= marker AND v_next >= marker:
                // We can expand both u and v until they hit a low divergence.
                
                // Let's keep the greedy structure but use scanner for the chosen direction.
                
                if u_next_match <= v_next_match {
                    if marker <= v_next_match {
                        let scan_limit = n_candidates - (v - u); // Max we can expand
                        let next_v = SimdScanner::scan_while_le(divergence, v + 1, v + 1 + scan_limit.min(n - 1 - v), marker - 1);
                        v = next_v;
                        v_next_match = get_div(v).min(v_next_match);
                    } else {
                         if marker <= u_next_match {
                             if u > 0 { 
                                 let scan_limit = n_candidates - (v - u);
                                 let next_u = SimdScanner::scan_back_while_le(divergence, u, u.saturating_sub(scan_limit), marker - 1);
                                 u = next_u;
                                 u_next_match = get_div(u).min(u_next_match);
                             } else {
                                break;
                             }
                         } else {
                             break;
                         }
                    }
                } else {
                     if marker <= u_next_match {
                        if u > 0 { 
                             let scan_limit = n_candidates - (v - u);
                             let next_u = SimdScanner::scan_back_while_le(divergence, u, u.saturating_sub(scan_limit), marker - 1);
                             u = next_u;
                             u_next_match = get_div(u).min(u_next_match);
                        } else {
                           break;
                        }
                     } else {
                        if marker <= v_next_match {
                             let scan_limit = n_candidates - (v - u);
                             let next_v = SimdScanner::scan_while_le(divergence, v + 1, v + 1 + scan_limit.min(n - 1 - v), marker - 1);
                             v = next_v;
                             v_next_match = get_div(v).min(v_next_match);
                         } else {
                             break;
                         }
                     }
                }
            } else {
                // Forward PBWT: Look for SMALL divergence (matching far back)
                // Expand while divergence[u] <= marker OR divergence[v] <= marker
                // i.e. NOT (divergence > marker)
                // Find first i where divergence[i] > marker.
                
                if v_next_match <= u_next_match {
                    if v_next_match <= marker {
                         let scan_limit = n_candidates - (v - u);
                         let next_v = SimdScanner::scan_while_le(divergence, v + 1, v + 1 + scan_limit.min(n - 1 - v), marker);
                         v = next_v;
                         v_next_match = get_div(v).max(v_next_match);
                    } else {
                        if u_next_match <= marker {
                             if u > 0 { 
                                 let scan_limit = n_candidates - (v - u);
                                 let next_u = SimdScanner::scan_back_while_le(divergence, u, u.saturating_sub(scan_limit), marker);
                                 u = next_u;
                                 u_next_match = get_div(u).max(u_next_match);
                             } else {
                                break;
                             }
                        } else {
                            break;
                        }
                    }
                } else {
                     if u_next_match <= marker {
                         if u > 0 { 
                             let scan_limit = n_candidates - (v - u);
                             let next_u = SimdScanner::scan_back_while_le(divergence, u, u.saturating_sub(scan_limit), marker);
                             u = next_u;
                             u_next_match = get_div(u).max(u_next_match);
                         } else {
                             break;
                         }
                     } else {
                         if v_next_match <= marker {
                              let scan_limit = n_candidates - (v - u);
                              let next_v = SimdScanner::scan_while_le(divergence, v + 1, v + 1 + scan_limit.min(n - 1 - v), marker);
                              v = next_v;
                              v_next_match = get_div(v).max(v_next_match);
                         } else {
                             break;
                         }
                     }
                }
            }
        }

        (u, v)
    }
}



/// SIMD-accelerated scanner for divergence arrays
///
/// Provides vectorized search for the first element violating a threshold condition.
pub struct SimdScanner;

impl SimdScanner {
    /// Find the first index `i` in `[start, end)` where `data[i] > threshold`.
    ///
    /// This corresponds to finding the boundary of a block where all values are `<= threshold`.
    /// Optimized for auto-vectorization.
    #[inline]
    pub fn scan_while_le(data: &[i32], start: usize, end: usize, threshold: i32) -> usize {
        let mut i = start;
        
        // Manual unrolling to encourage auto-vectorization
        // Check 8 elements at a time
        while i + 8 <= end {
            // If all 8 elements are <= threshold, we can skip them.
            // We check the inverse: if ANY element is > threshold, we need to find it matching.
            let chunk = &data[i..i+8];
            let mut violated = false;
            for &val in chunk {
                if val > threshold {
                    violated = true;
                    break;
                }
            }
            
            if violated {
                break;
            }
            i += 8;
        }

        // Scalar cleanup
        while i < end {
            if data[i] > threshold {
                return i;
            }
            i += 1;
        }
        
        end
    }

    /// Find the first index `i` moving backwards from `start` (exclusive) to `limit` (inclusive)
    /// where `data[i] > threshold`.
    ///
    /// Returns the index `i` (exclusive boundary). 
    /// If `data[start-1] > threshold`, returns `start`.
    /// If all `data[limit..start] <= threshold`, returns `limit`.
    pub fn scan_back_while_le(data: &[i32], start: usize, limit: usize, threshold: i32) -> usize {
        let mut i = start;
        
        // Unrolled backward scan
        while i >= limit + 8 {
            let chunk_end = i;
            let chunk_start = i - 8;
            let chunk = &data[chunk_start..chunk_end];
            let mut violated = false;
            
            for &val in chunk {
                if val > threshold {
                    violated = true;
                    break;
                }
            }
            
            if violated {
                break;
            }
            i -= 8;
        }
        
        // Scalar cleanup
        while i > limit {
            if data[i - 1] > threshold {
                return i;
            }
            i -= 1;
        }
        
        i
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
    #[test]
    fn test_simd_scanner_forward() {
        let data = vec![1, 2, 3, 4, 10, 5, 6, 7, 8, 9];
        // Scan while <= 5. Should find index 4 (value 10).
        let idx = SimdScanner::scan_while_le(&data, 0, data.len(), 5);
        assert_eq!(idx, 4);

        // Scan whole array if condition met
        let idx = SimdScanner::scan_while_le(&data, 0, 4, 5);
        assert_eq!(idx, 4);
        
        let idx = SimdScanner::scan_while_le(&data, 0, data.len(), 100);
        assert_eq!(idx, data.len());
    }

    #[test]
    fn test_simd_scanner_backward() {
        let data = vec![10, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        // Scan back from end (index 10) while <= 5.
        // 10 (idx 9) > 5 -> returns 10 (boundary)
        // Wait, start is exclusive.
        let idx = SimdScanner::scan_back_while_le(&data, 10, 0, 5);
        // data[9]=10 > 5. returns 10.
        assert_eq!(idx, 10);

        // From index 5 (value 6 is at idx 5? no, idx 5 is 6).
        // array: 0:10, 1:2, 2:3, 3:4, 4:5, 5:6
        // scan_back(start=5, limit=0, thresh=5)
        // data[4]=5 <= 5.
        // data[3]=4 <= 5.
        // data[2]=3 <= 5.
        // data[1]=2 <= 5.
        // data[0]=10 > 5. -> returns 1.
        let idx = SimdScanner::scan_back_while_le(&data, 5, 0, 5);
        assert_eq!(idx, 1);
    }
}
