use crate::data::haplotype::SampleIdx;
use crate::model::ibs2::Ibs2;
use crate::model::pbwt::PbwtDivUpdater;

/// Default backoff distance in cM for fuzzy PBWT neighbor matching.
/// This allows finding neighbors that diverged recently (within this genetic distance),
/// improving state selection in regions where exact IBS matches are rare or broken by errors.
/// Value of 0.3 cM matches Java Beagle's default maxBackoffSteps behavior.
pub const DEFAULT_PBWT_BACKOFF_CM: f64 = 0.3;

/// Manages bidirectional PBWT state for HMM state selection.
///
/// Stores both forward and backward PBWT arrays at each marker to enable
/// selecting haplotypes that match well both upstream and downstream.
/// This is critical for phasing accuracy around recombination hotspots.
///
/// ## Subset Support
///
/// When built for a marker subset (e.g., high-frequency markers in Stage 1),
/// the PBWT operates in subset index space (0..n_subset), but IBS2 segments
/// use global marker indices. The `subset_to_global` mapping handles this
/// coordinate space conversion automatically in `find_neighbors`.
pub struct BidirectionalPhaseIbs {
    /// Forward divergence at each marker: fwd_div[m] = divergence array after processing markers 0..=m
    fwd_div: Vec<Vec<i32>>,
    /// Forward prefix array at each marker
    fwd_ppa: Vec<Vec<u32>>,
    /// Backward divergence at each marker: bwd_div[m] = divergence array after processing markers m..n_markers
    bwd_div: Vec<Vec<i32>>,
    /// Backward prefix array at each marker
    bwd_ppa: Vec<Vec<u32>>,
    n_haps: usize,
    n_markers: usize,
    /// Pre-computed forward backoff limits: fwd_backoff_limit[m] = earliest marker within backoff distance
    fwd_backoff_limit: Vec<i32>,
    /// Pre-computed backward backoff limits: bwd_backoff_limit[m] = latest marker within backoff distance
    bwd_backoff_limit: Vec<i32>,
    /// Optional mapping from subset marker index to global marker index.
    /// When Some, IBS2 lookups use the mapped global index.
    /// When None (full chromosome), marker indices are used directly.
    subset_to_global: Option<Vec<usize>>,
}

impl BidirectionalPhaseIbs {
    /// Build bidirectional PBWT from genotype data with genetic positions for backoff
    ///
    /// # Arguments
    /// * `alleles` - Allele data per marker
    /// * `n_haps` - Number of haplotypes
    /// * `n_markers` - Number of markers
    /// * `gen_positions` - Optional genetic positions (cM) for computing backoff limits
    pub fn build(
        alleles: &[Vec<u8>],
        n_haps: usize,
        n_markers: usize,
        gen_positions: Option<&[f64]>,
    ) -> Self {
        let mut fwd_div = Vec::with_capacity(n_markers);
        let mut fwd_ppa = Vec::with_capacity(n_markers);
        let mut bwd_div = vec![Vec::new(); n_markers];
        let mut bwd_ppa = vec![Vec::new(); n_markers];

        let mut updater = PbwtDivUpdater::new(n_haps);

        let mut ppa: Vec<u32> = (0..n_haps as u32).collect();
        let mut div: Vec<i32> = vec![0; n_haps + 1];

        for m in 0..n_markers {
            updater.fwd_update(&alleles[m], 2, m, &mut ppa, &mut div);
            fwd_ppa.push(ppa.clone());
            fwd_div.push(div[..n_haps].to_vec());
        }

        ppa = (0..n_haps as u32).collect();
        div = vec![n_markers as i32; n_haps + 1];

        for m in (0..n_markers).rev() {
            updater.bwd_update(&alleles[m], 2, m, &mut ppa, &mut div);
            bwd_ppa[m] = ppa.clone();
            bwd_div[m] = div[..n_haps].to_vec();
        }

        // Compute backoff limits based on genetic positions
        let (fwd_backoff_limit, bwd_backoff_limit) =
            Self::compute_backoff_limits(gen_positions, n_markers, DEFAULT_PBWT_BACKOFF_CM);

        Self {
            fwd_div,
            fwd_ppa,
            bwd_div,
            bwd_ppa,
            n_haps,
            n_markers,
            fwd_backoff_limit,
            bwd_backoff_limit,
            subset_to_global: None,
        }
    }

    /// Build bidirectional PBWT for a marker subset with global index mapping
    ///
    /// This variant stores the subset-to-global marker mapping so that IBS2
    /// lookups (which use global indices) work correctly when the PBWT is
    /// built on a marker subset (e.g., high-frequency markers in Stage 1).
    ///
    /// # Arguments
    /// * `alleles` - Allele data per subset marker
    /// * `n_haps` - Number of haplotypes
    /// * `n_markers` - Number of markers in subset
    /// * `gen_positions` - Optional genetic positions (cM) for computing backoff limits
    /// * `subset_to_global` - Mapping from subset index to global marker index
    pub fn build_for_subset(
        alleles: &[Vec<u8>],
        n_haps: usize,
        n_markers: usize,
        gen_positions: Option<&[f64]>,
        subset_to_global: &[usize],
    ) -> Self {
        let mut result = Self::build(alleles, n_haps, n_markers, gen_positions);
        result.subset_to_global = Some(subset_to_global.to_vec());
        result
    }

    /// Compute backoff limits for each marker based on genetic distance
    ///
    /// For forward PBWT: fwd_backoff_limit[m] = earliest marker within backoff_cm of m
    /// For backward PBWT: bwd_backoff_limit[m] = latest marker within backoff_cm of m
    fn compute_backoff_limits(
        gen_positions: Option<&[f64]>,
        n_markers: usize,
        backoff_cm: f64,
    ) -> (Vec<i32>, Vec<i32>) {
        if n_markers == 0 {
            return (Vec::new(), Vec::new());
        }

        match gen_positions {
            Some(gp) if gp.len() == n_markers => {
                let mut fwd_limit = vec![0i32; n_markers];
                let mut bwd_limit = vec![(n_markers - 1) as i32; n_markers];

                for m in 0..n_markers {
                    let pos_m = gp[m];

                    // Forward: find earliest marker within backoff distance
                    let mut earliest = m;
                    while earliest > 0 && (pos_m - gp[earliest - 1]) < backoff_cm {
                        earliest -= 1;
                    }
                    fwd_limit[m] = earliest as i32;

                    // Backward: find latest marker within backoff distance
                    let mut latest = m;
                    while latest < n_markers - 1 && (gp[latest + 1] - pos_m) < backoff_cm {
                        latest += 1;
                    }
                    bwd_limit[m] = latest as i32;
                }

                (fwd_limit, bwd_limit)
            }
            _ => {
                // No genetic positions: use marker-based approximation
                // Assume ~1 cM per 1000 markers as rough estimate, so 0.3 cM ≈ 300 markers
                // But be conservative with a smaller value to avoid over-relaxation
                let default_backoff_markers = 50i32;
                let fwd = (0..n_markers)
                    .map(|m| (m as i32 - default_backoff_markers).max(0))
                    .collect();
                let bwd = (0..n_markers)
                    .map(|m| (m as i32 + default_backoff_markers).min((n_markers - 1) as i32))
                    .collect();
                (fwd, bwd)
            }
        }
    }

    /// Find neighbors at a marker using both forward and backward PBWT
    ///
    /// When built with `build_for_subset`, automatically converts the subset
    /// marker index to global space for IBS2 segment lookups.
    pub fn find_neighbors(
        &self,
        hap_idx: u32,
        marker_idx: usize,
        ibs2: &Ibs2,
        n_candidates: usize,
    ) -> Vec<u32> {
        let mut neighbors = Vec::with_capacity(n_candidates * 2 + 10);
        let sample = SampleIdx::new(hap_idx / 2);

        // Convert marker index to global space for IBS2 lookup
        // IBS2 segments use global marker indices, but when built for a subset,
        // marker_idx is in subset space. The mapping handles this conversion.
        let ibs2_marker_idx = self
            .subset_to_global
            .as_ref()
            .and_then(|mapping| mapping.get(marker_idx).copied())
            .unwrap_or(marker_idx);

        for seg in ibs2.segments(sample) {
            if seg.contains(ibs2_marker_idx) {
                let other_s = seg.other_sample;
                if other_s != sample {
                    neighbors.push(other_s.hap1().0);
                    neighbors.push(other_s.hap2().0);
                }
            }
        }

        let fwd_neighbors = self.find_fwd_neighbors(hap_idx, marker_idx, n_candidates);
        let bwd_neighbors = self.find_bwd_neighbors(hap_idx, marker_idx, n_candidates);

        for h in fwd_neighbors {
            if h != hap_idx && h / 2 != sample.0 && !neighbors.contains(&h) {
                neighbors.push(h);
            }
        }

        for h in bwd_neighbors {
            if h != hap_idx && h / 2 != sample.0 && !neighbors.contains(&h) {
                neighbors.push(h);
            }
        }

        neighbors
    }

    fn find_fwd_neighbors(&self, hap_idx: u32, marker_idx: usize, n_candidates: usize) -> Vec<u32> {
        if marker_idx >= self.n_markers {
            return Vec::new();
        }

        let ppa = &self.fwd_ppa[marker_idx];
        let div = &self.fwd_div[marker_idx];

        let sorted_pos = ppa.iter().position(|&h| h == hap_idx).unwrap_or(0);
        let marker_i32 = marker_idx as i32;

        // Backoff limit: accept divergences that occurred after this marker
        // This allows "fuzzy" matches where sequences diverged recently
        let backoff_limit = self.fwd_backoff_limit[marker_idx];

        let mut result = Vec::with_capacity(n_candidates);

        let mut u = sorted_pos;
        let mut max_div = i32::MIN;
        while result.len() < n_candidates / 2 && u > 0 {
            max_div = max_div.max(div.get(u).copied().unwrap_or(i32::MAX));
            // Allow backoff: accept if divergence is within backoff limit
            // Original: break if max_div > marker_i32 (exact match only)
            // With backoff: break if max_div > marker_i32 AND max_div < backoff_limit
            if max_div > marker_i32 && max_div < backoff_limit {
                break;
            }
            u -= 1;
            let h = ppa[u];
            if h != hap_idx {
                result.push(h);
            }
        }

        let mut v = sorted_pos + 1;
        max_div = i32::MIN;
        while result.len() < n_candidates && v < self.n_haps {
            max_div = max_div.max(div.get(v).copied().unwrap_or(i32::MAX));
            // Allow backoff: same logic as above
            if max_div > marker_i32 && max_div < backoff_limit {
                break;
            }
            result.push(ppa[v]);
            v += 1;
        }

        result
    }

    fn find_bwd_neighbors(&self, hap_idx: u32, marker_idx: usize, n_candidates: usize) -> Vec<u32> {
        if marker_idx >= self.n_markers {
            return Vec::new();
        }

        let ppa = &self.bwd_ppa[marker_idx];
        let div = &self.bwd_div[marker_idx];

        let sorted_pos = ppa.iter().position(|&h| h == hap_idx).unwrap_or(0);
        let marker_i32 = marker_idx as i32;

        // Backoff limit: accept divergences that occurred before this marker
        // For backward PBWT, divergence represents where match ENDS
        let backoff_limit = self.bwd_backoff_limit[marker_idx];

        let mut result = Vec::with_capacity(n_candidates);

        let mut u = sorted_pos;
        let mut min_div = i32::MAX;
        while result.len() < n_candidates / 2 && u > 0 {
            min_div = min_div.min(div.get(u).copied().unwrap_or(0));
            // Allow backoff: accept if divergence is within backoff limit
            // Original: break if min_div < marker_i32 (exact match only)
            // With backoff: break if min_div < marker_i32 AND min_div > backoff_limit
            if min_div < marker_i32 && min_div > backoff_limit {
                break;
            }
            u -= 1;
            let h = ppa[u];
            if h != hap_idx {
                result.push(h);
            }
        }

        let mut v = sorted_pos + 1;
        min_div = i32::MAX;
        while result.len() < n_candidates && v < self.n_haps {
            min_div = min_div.min(div.get(v).copied().unwrap_or(0));
            // Allow backoff: same logic as above
            if min_div < marker_i32 && min_div > backoff_limit {
                break;
            }
            result.push(ppa[v]);
            v += 1;
        }

        result
    }
}
