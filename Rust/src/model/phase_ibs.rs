use crate::data::haplotype::SampleIdx;
use crate::model::ibs2::Ibs2;
use crate::model::pbwt::PbwtDivUpdater;

/// Manages the global PBWT state for the entire cohort.
/// This allows O(N*M) total complexity instead of O(N^2*M).
pub struct GlobalPhaseIbs {
    /// Permutation Prefix Array (a[i] = hap_idx)
    ppa: Vec<u32>,
    /// Divergence array (d[i] = start of match)
    div: Vec<i32>,
    /// Inverse PPA (opa[hap_idx] = sorted_pos)
    opa: Vec<u32>,
    /// PBWT Updater
    updater: PbwtDivUpdater,
    /// Number of haplotypes
    n_haps: usize,
}

impl GlobalPhaseIbs {
    pub fn new(n_haps: usize) -> Self {
        // Initial state: identity permutation, 0 divergence
        let ppa: Vec<u32> = (0..n_haps as u32).collect();
        let div: Vec<i32> = vec![0; n_haps + 1]; // +1 for boundary
        let opa: Vec<u32> = (0..n_haps as u32).collect();

        Self {
            ppa,
            div,
            opa,
            updater: PbwtDivUpdater::new(n_haps),
            n_haps,
        }
    }

    /// Advance the PBWT state to the next marker using the given alleles.
    /// Alleles must be finalized (phased) for this marker.
    pub fn advance(&mut self, alleles: &[u8], marker_idx: usize) {
        self.updater.fwd_update(alleles, 2, marker_idx, &mut self.ppa, &mut self.div);
        
        // Rebuild OPA (Inverse PPA)
        for (sorted_pos, &hap_idx) in self.ppa.iter().enumerate() {
            self.opa[hap_idx as usize] = sorted_pos as u32;
        }
    }
    
    /// Advance the PBWT state using packed alleles (u64 words)
    pub fn advance_packed(&mut self, alleles_packed: &[u64], marker_idx: usize) {
        self.updater.fwd_update_packed(alleles_packed, marker_idx, &mut self.ppa, &mut self.div);
        
        // Rebuild OPA (Inverse PPA)
        for (sorted_pos, &hap_idx) in self.ppa.iter().enumerate() {
            self.opa[hap_idx as usize] = sorted_pos as u32;
        }
    }

    /// Find best neighbors for a haplotype using the current PBWT state.
    ///
    /// # Arguments
    /// * `hap_idx` - The haplotype to find neighbors for
    /// * `marker_idx` - Current marker index
    /// * `ibs2` - IBS2 structure for filtering/prioritizing
    /// * `n_candidates` - Number of candidates to check on each side
    /// Find best neighbors for a haplotype using the current PBWT state.
    ///
    /// # Arguments
    /// * `hap_idx` - The haplotype to find neighbors for
    /// * `marker_idx` - Current marker index
    /// * `ibs2` - IBS2 structure for filtering/prioritizing
    /// * `n_candidates` - Number of candidates to check on each side
    pub fn find_neighbors(
        &self,
        hap_idx: u32,
        marker_idx: usize,
        ibs2: &Ibs2,
        n_candidates: usize,
    ) -> Vec<u32> {
        let mut neighbors = Vec::with_capacity(n_candidates * 2 + 10);
        self.find_neighbors_buf(hap_idx, marker_idx, ibs2, n_candidates, &mut neighbors);
        neighbors
    }

    /// Find best neighbors writing into a reusable buffer
    ///
    /// Uses PBWT divergence array to select neighbors with actual IBS matches.
    /// Only selects neighbors whose divergence value indicates a match extending
    /// to or past the current marker position (i.e., divergence <= marker_idx).
    pub fn find_neighbors_buf(
        &self,
        hap_idx: u32,
        marker_idx: usize,
        ibs2: &Ibs2,
        n_candidates: usize,
        neighbors: &mut Vec<u32>,
    ) {
        neighbors.clear();
        let sample = SampleIdx::new(hap_idx / 2);
        let marker_i32 = marker_idx as i32;

        // 1. Add guaranteed IBS2 matches
        // IBS2 segments indicate shared long-range haplotype blocks (potentially unphased)
        // We include both haplotypes of the IBS2 partner as candidates
        let segments = ibs2.segments(sample);
        for seg in segments {
            if seg.contains(marker_idx) {
                let other_s = seg.other_sample;
                if other_s != sample {
                    let h1 = other_s.hap1().0;
                    let h2 = other_s.hap2().0;
                    neighbors.push(h1);
                    neighbors.push(h2);
                }
            }
        }

        // 2. Add PBWT neighbors using divergence array for IBS validation
        // Get sorted position of the target haplotype
        let sorted_pos = self.opa[hap_idx as usize] as usize;

        // In forward PBWT, divergence[i] indicates where the match between
        // prefix[i] and prefix[i-1] begins. A haplotype at position i is IBS
        // with the target (at sorted_pos) if the divergence values between them
        // are all <= marker_idx (meaning the match extends at least to marker_idx).

        // Check "up" (decreasing sorted_pos)
        // Track the maximum divergence seen as we expand upward
        let mut u = sorted_pos;
        let mut count = 0;
        let mut max_div_u = i32::MIN;

        while count < n_candidates && u > 0 {
            // Divergence at position u tells us where prefix[u] diverges from prefix[u-1]
            let div = self.div.get(u).copied().unwrap_or(i32::MAX);
            max_div_u = max_div_u.max(div);

            // Only accept neighbors whose match extends to current marker
            // (divergence <= marker_idx means match started at or before marker_idx)
            if max_div_u > marker_i32 {
                // No more valid IBS matches in this direction
                break;
            }

            u -= 1;
            let neighbor_hap = self.ppa[u];

            if neighbor_hap == hap_idx { continue; }

            let other_sample_idx = neighbor_hap / 2;
            if sample.0 == other_sample_idx {
                continue;
            }

            neighbors.push(neighbor_hap);
            count += 1;
        }

        // Check "down" (increasing sorted_pos)
        // Track the maximum divergence seen as we expand downward
        let mut v = sorted_pos + 1;
        count = 0;
        let mut max_div_v = i32::MIN;

        while count < n_candidates && v < self.n_haps {
            // Divergence at position v tells us where prefix[v] diverges from prefix[v-1]
            let div = self.div.get(v).copied().unwrap_or(i32::MAX);
            max_div_v = max_div_v.max(div);

            // Only accept neighbors whose match extends to current marker
            if max_div_v > marker_i32 {
                // No more valid IBS matches in this direction
                break;
            }

            let neighbor_hap = self.ppa[v];

            if neighbor_hap == hap_idx {
                v += 1;
                continue;
            }

            let other_sample_idx = neighbor_hap / 2;
            if sample.0 == other_sample_idx {
                v += 1;
                continue;
            }

            neighbors.push(neighbor_hap);
            v += 1;
            count += 1;
        }

        // If we didn't find enough neighbors with IBS matches, fall back to
        // nearest neighbors in PBWT order (without strict divergence check)
        // This ensures we always return some candidates
        if neighbors.len() < n_candidates {
            u = sorted_pos;
            v = sorted_pos + 1;
            let target_count = n_candidates.min(self.n_haps.saturating_sub(1));

            while neighbors.len() < target_count && (u > 0 || v < self.n_haps) {
                // Expand in direction with smaller divergence first (greedy)
                let div_u = if u > 0 { self.div.get(u).copied().unwrap_or(i32::MAX) } else { i32::MAX };
                let div_v = if v < self.n_haps { self.div.get(v).copied().unwrap_or(i32::MAX) } else { i32::MAX };

                if div_u <= div_v && u > 0 {
                    u -= 1;
                    let neighbor_hap = self.ppa[u];
                    if neighbor_hap != hap_idx && neighbor_hap / 2 != sample.0 && !neighbors.contains(&neighbor_hap) {
                        neighbors.push(neighbor_hap);
                    }
                } else if v < self.n_haps {
                    let neighbor_hap = self.ppa[v];
                    if neighbor_hap != hap_idx && neighbor_hap / 2 != sample.0 && !neighbors.contains(&neighbor_hap) {
                        neighbors.push(neighbor_hap);
                    }
                    v += 1;
                } else {
                    break;
                }
            }
        }
    }
}
