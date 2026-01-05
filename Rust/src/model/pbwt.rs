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
//! Bioinformatics 30(9):1266-1272. doi: 10.1093/bioinformatics/btu014

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

    /// Number of haplotypes
    pub fn n_haps(&self) -> usize {
        self.n_haps
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

    /// Backward update of prefix and divergence arrays
    ///
    /// This exactly matches Java PbwtDivUpdater.bwdUpdate
    ///
    /// # Arguments
    /// * `alleles` - Allele for each haplotype
    /// * `n_alleles` - Number of distinct alleles
    /// * `marker` - Current marker index
    /// * `prefix` - Prefix array to update
    /// * `divergence` - Divergence array to update
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

        // Initialize p array with marker-1 (divergence if no prior match)
        let init_value = marker as i32 - 1;
        for j in 0..n_alleles {
            self.p[j] = init_value;
        }

        // Process haplotypes in prefix order
        for i in 0..self.n_haps {
            let hap = prefix[i];
            let div = divergence[i];
            let allele = alleles[hap as usize] as usize;
            assert!(allele < n_alleles);

            // Update p[j] = min(p[j], div) for all alleles
            // This propagates the minimum divergence seen so far
            for j in 0..n_alleles {
                if div < self.p[j] {
                    self.p[j] = div;
                }
            }

            // Store this haplotype with divergence = p[allele]
            self.a[allele].push(hap);
            self.d[allele].push(self.p[allele]);

            // Reset p[allele] for the next hap with this allele
            // Using i32::MAX so any real divergence will be smaller
            self.p[allele] = i32::MAX;
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

/// Simple divergence updater for u32 divergence arrays (backward compatibility)
#[derive(Debug)]
pub struct SimpleDivUpdater {
    n_haps: usize,
    buckets: Vec<Vec<(u32, u32)>>,
}

impl SimpleDivUpdater {
    pub fn new(n_haps: usize) -> Self {
        Self {
            n_haps,
            buckets: vec![Vec::new(); 4],
        }
    }

    #[allow(unused_variables)]
    pub fn update(
        &mut self,
        alleles: &[u8],
        n_alleles: usize,
        marker: usize,
        prefix: &mut [u32],
        divergence: &mut [u32],
    ) {
        if n_alleles > self.buckets.len() {
            self.buckets.resize(n_alleles, Vec::new());
        }

        for bucket in &mut self.buckets[..n_alleles] {
            bucket.clear();
        }

        for i in 0..self.n_haps {
            let hap = prefix[i];
            let div = divergence[i];
            let allele = alleles[hap as usize] as usize;
            if allele < n_alleles {
                self.buckets[allele].push((hap, div));
            }
        }

        let mut idx = 0;
        for bucket in &self.buckets[..n_alleles] {
            for &(hap, div) in bucket {
                prefix[idx] = hap;
                divergence[idx] = div;
                idx += 1;
            }
        }
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

    /// Backward PBWT state
    bwd_prefix: Vec<u32>,
    bwd_divergence: Vec<i32>,

    /// Number of haplotypes
    n_haps: usize,

    /// Updater for forward direction
    fwd_updater: PbwtDivUpdater,

    /// Updater for backward direction
    bwd_updater: PbwtDivUpdater,
}

impl PbwtIbs {
    /// Create a new PBWT IBS finder
    pub fn new(n_haps: usize) -> Self {
        Self {
            fwd_prefix: (0..n_haps as u32).collect(),
            fwd_divergence: vec![0; n_haps + 1], // +1 for sentinel
            bwd_prefix: (0..n_haps as u32).collect(),
            bwd_divergence: vec![0; n_haps + 1],
            n_haps,
            fwd_updater: PbwtDivUpdater::new(n_haps),
            bwd_updater: PbwtDivUpdater::new(n_haps),
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

    /// Update forward PBWT state
    pub fn fwd_update(&mut self, alleles: &[u8], n_alleles: usize, marker: usize) {
        self.fwd_updater.fwd_update(
            alleles,
            n_alleles,
            marker,
            &mut self.fwd_prefix,
            &mut self.fwd_divergence[..self.n_haps],
        );
    }

    /// Update backward PBWT state
    pub fn bwd_update(&mut self, alleles: &[u8], n_alleles: usize, marker: usize) {
        self.bwd_updater.bwd_update(
            alleles,
            n_alleles,
            marker,
            &mut self.bwd_prefix,
            &mut self.bwd_divergence[..self.n_haps],
        );
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

    /// Find position of a haplotype in the prefix array
    pub fn find_position(&self, hap: HapIdx, use_backward: bool) -> Option<usize> {
        let prefix = if use_backward {
            &self.bwd_prefix
        } else {
            &self.fwd_prefix
        };
        prefix.iter().position(|&h| h == hap.0)
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
    pub fn select_states(
        &self,
        target_hap: HapIdx,
        n_states: usize,
        marker: usize,
        n_candidates: usize,
        use_backward: bool,
        exclude_self: bool,
    ) -> Vec<HapIdx> {
        let (prefix, divergence) = if use_backward {
            (&self.bwd_prefix, &self.bwd_divergence)
        } else {
            (&self.fwd_prefix, &self.fwd_divergence)
        };

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
            use_backward,
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

    /// Select IBS haplotype for a target within the reference panel
    ///
    /// Returns the best IBS match or -1 if none found (matching Java interface)
    pub fn select_ibs_hap(
        &self,
        target_hap: HapIdx,
        marker: usize,
        n_candidates: usize,
        use_backward: bool,
    ) -> i32 {
        let matches = self.select_states(target_hap, 1, marker, n_candidates, use_backward, true);
        matches.first().map(|h| h.0 as i32).unwrap_or(-1)
    }

    /// Select IBS haplotypes for an external target (not in the reference panel)
    ///
    /// This is used during imputation where the target haplotype is from the
    /// target panel, not the reference panel. We find the position where the
    /// target allele sequence would sort in the PBWT and expand from there.
    ///
    /// # Arguments
    /// * `target_allele` - The target's allele at the current marker
    /// * `n_states` - Number of states to select
    /// * `use_backward` - Whether to use backward PBWT
    pub fn select_states_for_external_target(
        &self,
        _target_allele: u8,
        n_states: usize,
        use_backward: bool,
    ) -> Vec<HapIdx> {
        let prefix = if use_backward {
            &self.bwd_prefix
        } else {
            &self.fwd_prefix
        };

        if prefix.is_empty() || n_states == 0 {
            return Vec::new();
        }

        // Find the first position in the prefix array where a haplotype has
        // the same allele as the target. In PBWT, haplotypes are sorted by
        // their allele sequence prefixes, so haplotypes with the same allele
        // at the current position are grouped together.
        let _start_pos: Option<usize> = None;
        let _end_pos: usize = 0;

        // Since we don't have direct access to alleles here, use the middle
        // as a starting point and expand. The caller should use this method
        // after updating PBWT with alleles that can be matched.
        let mid = prefix.len() / 2;

        // Expand from middle to collect n_states haplotypes
        let mut selected = Vec::with_capacity(n_states);
        let mut left = mid;
        let mut right = mid;

        // Add middle first
        if mid < prefix.len() {
            selected.push(HapIdx::new(prefix[mid]));
            right += 1;
        }

        // Expand alternating left and right
        while selected.len() < n_states && (left > 0 || right < prefix.len()) {
            if left > 0 {
                left -= 1;
                selected.push(HapIdx::new(prefix[left]));
            }
            if selected.len() < n_states && right < prefix.len() {
                selected.push(HapIdx::new(prefix[right]));
                right += 1;
            }
        }

        selected
    }

    /// Find position in prefix array where target allele would sort
    ///
    /// Uses the allele to find the region of haplotypes with matching allele
    pub fn find_allele_region(
        &self,
        alleles: &[u8],
        target_allele: u8,
        use_backward: bool,
    ) -> (usize, usize) {
        let prefix = if use_backward {
            &self.bwd_prefix
        } else {
            &self.fwd_prefix
        };

        if prefix.is_empty() {
            return (0, 0);
        }

        let mut start = None;
        let mut end = 0;

        for (i, &hap_idx) in prefix.iter().enumerate() {
            let hap_allele = alleles.get(hap_idx as usize).copied().unwrap_or(255);
            if hap_allele == target_allele {
                if start.is_none() {
                    start = Some(i);
                }
                end = i + 1;
            }
        }

        (start.unwrap_or(0), end)
    }
}

/// Bidirectional PBWT phasing result for a single sample
///
/// This represents the phase consensus from forward and backward PBWT sweeps.
/// Sites where both directions agree are "confident", others need HMM resolution.
#[derive(Clone, Debug)]
pub struct BidirectionalPhaseResult {
    /// Phase assignments for each heterozygous marker (true = swap alleles)
    pub phase_decisions: Vec<bool>,
    /// Confidence for each decision (true = forward and backward agreed)
    pub confident: Vec<bool>,
    /// Marker indices that are heterozygous
    pub het_markers: Vec<usize>,
}

impl BidirectionalPhaseResult {
    /// Create a new result with the given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            phase_decisions: Vec::with_capacity(capacity),
            confident: Vec::with_capacity(capacity),
            het_markers: Vec::with_capacity(capacity),
        }
    }

    /// Add a phase decision
    pub fn add(&mut self, marker: usize, decision: bool, confident: bool) {
        self.het_markers.push(marker);
        self.phase_decisions.push(decision);
        self.confident.push(confident);
    }

    /// Get number of heterozygous sites
    pub fn n_hets(&self) -> usize {
        self.het_markers.len()
    }

    /// Get number of confident phase calls
    pub fn n_confident(&self) -> usize {
        self.confident.iter().filter(|&&c| c).count()
    }
}

/// Run bidirectional PBWT phasing to establish initial phase consensus
///
/// This implements the PbwtPhaser approach from Java:
/// 1. Run forward PBWT sweep, recording IBS matches at each marker
/// 2. Run backward PBWT sweep, recording IBS matches at each marker
/// 3. At each heterozygous site, check if forward and backward agree on phase
/// 4. Mark sites as "confident" where they agree
///
/// # Arguments
/// * `alleles` - Genotype matrix [marker][haplotype]
/// * `n_markers` - Number of markers
/// * `n_haps` - Number of haplotypes
/// * `n_alleles_per_marker` - Number of alleles at each marker
///
/// # Returns
/// Phase results for each sample, with confidence flags
pub fn run_bidirectional_pbwt_phasing(
    get_allele: impl Fn(usize, usize) -> u8 + Sync,
    n_markers: usize,
    n_haps: usize,
    n_alleles_per_marker: &[usize],
) -> Vec<BidirectionalPhaseResult> {
    let n_samples = n_haps / 2;
    let mut results = Vec::with_capacity(n_samples);

    // Build forward PBWT and record IBS matches
    let fwd_matches = build_pbwt_matches(&get_allele, n_markers, n_haps, n_alleles_per_marker, false);

    // Build backward PBWT and record IBS matches
    let bwd_matches = build_pbwt_matches(&get_allele, n_markers, n_haps, n_alleles_per_marker, true);

    // Compare forward and backward for each sample
    for s in 0..n_samples {
        let hap1 = s * 2;
        let hap2 = hap1 + 1;

        let mut result = BidirectionalPhaseResult::new(n_markers / 10);

        for m in 0..n_markers {
            let a1 = get_allele(m, hap1);
            let a2 = get_allele(m, hap2);

            // Skip homozygous sites
            if a1 == a2 {
                continue;
            }

            // Get IBS match alleles from forward direction
            let fwd_ibs = fwd_matches.get(m).and_then(|mm| mm.get(&hap1)).copied();
            let bwd_ibs = bwd_matches.get(m).and_then(|mm| mm.get(&hap1)).copied();

            // Determine phase from each direction
            // If IBS match allele == a1, keep phase; if == a2, swap
            let fwd_swap = fwd_ibs.map(|ibs_allele| ibs_allele == a2);
            let bwd_swap = bwd_ibs.map(|ibs_allele| ibs_allele == a2);

            match (fwd_swap, bwd_swap) {
                (Some(fwd), Some(bwd)) if fwd == bwd => {
                    // Both directions agree
                    result.add(m, fwd, true);
                }
                (Some(fwd), Some(_)) => {
                    // Directions disagree - use forward but mark as uncertain
                    result.add(m, fwd, false);
                }
                (Some(fwd), None) => {
                    // Only forward available
                    result.add(m, fwd, false);
                }
                (None, Some(bwd)) => {
                    // Only backward available
                    result.add(m, bwd, false);
                }
                (None, None) => {
                    // No information - random phase
                    result.add(m, false, false);
                }
            }
        }

        results.push(result);
    }

    results
}

/// Build PBWT and record IBS matches at each marker
///
/// Returns a map: marker -> (haplotype -> IBS match allele)
fn build_pbwt_matches(
    get_allele: impl Fn(usize, usize) -> u8,
    n_markers: usize,
    n_haps: usize,
    n_alleles_per_marker: &[usize],
    backward: bool,
) -> Vec<std::collections::HashMap<usize, u8>> {
    use std::collections::HashMap;

    let mut matches: Vec<HashMap<usize, u8>> = vec![HashMap::new(); n_markers];

    let mut prefix: Vec<u32> = (0..n_haps as u32).collect();
    let mut divergence: Vec<i32> = vec![0; n_haps + 1];
    let mut updater = PbwtDivUpdater::new(n_haps);

    // Create marker ordering
    let marker_order: Vec<usize> = if backward {
        (0..n_markers).rev().collect()
    } else {
        (0..n_markers).collect()
    };

    // Collect alleles buffer
    let mut alleles = vec![0u8; n_haps];

    for &m in &marker_order {
        // Fill alleles
        for h in 0..n_haps {
            alleles[h] = get_allele(m, h);
        }

        let n_alleles = n_alleles_per_marker.get(m).copied().unwrap_or(2).max(2);

        // Record IBS matches before update (at current position)
        for (pos, &hap) in prefix.iter().enumerate() {
            let hap_idx = hap as usize;

            // Find nearest neighbor (prefer same allele, then adjacent in prefix order)
            let neighbor = if pos > 0 {
                prefix[pos - 1] as usize
            } else if pos + 1 < n_haps {
                prefix[pos + 1] as usize
            } else {
                continue;
            };

            // Record the IBS match allele (what allele the neighbor has)
            matches[m].insert(hap_idx, alleles[neighbor]);
        }

        // Update PBWT
        if backward {
            updater.bwd_update(&alleles, n_alleles, m, &mut prefix, &mut divergence);
        } else {
            updater.fwd_update(&alleles, n_alleles, m, &mut prefix, &mut divergence);
        }
    }

    matches
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
    fn test_pbwt_div_bwd_update() {
        let mut updater = PbwtDivUpdater::new(4);
        let mut prefix: Vec<u32> = vec![0, 1, 2, 3];
        let mut divergence: Vec<i32> = vec![10, 10, 10, 10];

        let alleles = vec![0u8, 1, 0, 1];
        updater.bwd_update(&alleles, 2, 5, &mut prefix, &mut divergence);

        // Check grouping
        assert_eq!(prefix, vec![0, 2, 1, 3]);
    }

    #[test]
    fn test_pbwt_ibs_select() {
        let ibs = PbwtIbs::new(10);
        let selected = ibs.select_states(HapIdx::new(5), 4, 0, 10, false, true);
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
