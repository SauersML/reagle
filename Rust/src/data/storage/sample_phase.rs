//! # Sample Phase State Tracking
//!
//! Tracks the phase state of each marker cluster for a single sample.
//! Based on Java's SamplePhase.java from Beagle.
//!
//! Supports multiallelic markers by storing full byte values (0-254 for alleles,
//! 255 for missing). This matches Java Beagle's approach.

/// Status of a genotype cluster
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ClusterStatus {
    /// Missing genotype data
    Missing = 0,
    /// Masked heterozygote (excluded from phasing)
    Masked = 1,
    /// Homozygous genotype
    Homozygous = 2,
    /// Phase has been determined with high confidence
    Phased = 3,
    /// Phase is not yet determined
    Unphased = 4,
}

impl ClusterStatus {
    /// Number of status variants (for array indexing)
    pub const COUNT: usize = 5;
}

/// Phase state tracking for a single sample
///
/// Uses byte storage (1 byte per allele) to support multiallelic markers.
/// Allele values: 0 = REF, 1-254 = ALT alleles, 255 = missing
#[derive(Clone, Debug)]
pub struct SamplePhase {
    /// Sample index
    sample: u32,
    /// Number of markers
    n_markers: usize,
    /// Alleles on first haplotype (byte per marker for multiallelic support)
    hap1: Vec<u8>,
    /// Alleles on second haplotype
    hap2: Vec<u8>,
    /// Status of each marker
    status: Vec<ClusterStatus>,
    /// Count of each status type for quick access
    status_counts: [usize; ClusterStatus::COUNT],
}

impl SamplePhase {
    /// Creates a new SamplePhase from allele data.
    ///
    /// # Arguments
    /// * `sample` - Sample index
    /// * `n_markers` - Number of markers
    /// * `hap1_alleles` - Alleles on first haplotype (0-254 for alleles, 255 for missing)
    /// * `hap2_alleles` - Alleles on second haplotype
    /// * `unphased_hets` - Sorted indices of markers that are unphased heterozygotes
    /// * `missing` - Sorted indices of markers with missing data
    ///
    /// # Panics
    /// Panics if allele slices don't match n_markers or indices are invalid.
    pub fn new(
        sample: u32,
        n_markers: usize,
        hap1_alleles: &[u8],
        hap2_alleles: &[u8],
        unphased_hets: &[usize],
        missing: &[usize],
    ) -> Self {
        assert_eq!(hap1_alleles.len(), n_markers, "hap1 length mismatch");
        assert_eq!(hap2_alleles.len(), n_markers, "hap2 length mismatch");

        // Copy alleles directly - preserves multiallelic values
        let hap1: Vec<u8> = hap1_alleles.to_vec();
        let hap2: Vec<u8> = hap2_alleles.to_vec();

        let mut status = Vec::with_capacity(n_markers);
        let mut status_counts = [0usize; ClusterStatus::COUNT];

        let mut missing_idx = 0;
        let mut unphased_idx = 0;

        for m in 0..n_markers {
            let is_missing = missing_idx < missing.len() && missing[missing_idx] == m;
            let is_unphased = unphased_idx < unphased_hets.len() && unphased_hets[unphased_idx] == m;

            if is_missing {
                missing_idx += 1;
            }
            if is_unphased {
                unphased_idx += 1;
            }

            let a1 = hap1_alleles[m];
            let a2 = hap2_alleles[m];

            let st = Self::determine_status(is_missing, is_unphased, a1, a2);
            status_counts[st as usize] += 1;
            status.push(st);
        }

        Self {
            sample,
            n_markers,
            hap1,
            hap2,
            status,
            status_counts,
        }
    }

    /// Determine cluster status from genotype properties
    #[inline]
    fn determine_status(is_missing: bool, is_unphased: bool, a1: u8, a2: u8) -> ClusterStatus {
        if is_missing {
            ClusterStatus::Missing
        } else if a1 == a2 {
            ClusterStatus::Homozygous
        } else if is_unphased {
            ClusterStatus::Unphased
        } else {
            ClusterStatus::Phased
        }
    }

    /// Returns the sample index.
    #[inline]
    pub fn sample(&self) -> u32 {
        self.sample
    }

    /// Returns the number of markers.
    #[inline]
    pub fn n_markers(&self) -> usize {
        self.n_markers
    }

    /// Returns the allele on haplotype 1 for the specified marker.
    /// Values: 0 = REF, 1-254 = ALT alleles, 255 = missing
    #[inline]
    pub fn allele1(&self, marker: usize) -> u8 {
        self.hap1[marker]
    }

    /// Returns the allele on haplotype 2 for the specified marker.
    /// Values: 0 = REF, 1-254 = ALT alleles, 255 = missing
    #[inline]
    pub fn allele2(&self, marker: usize) -> u8 {
        self.hap2[marker]
    }

    /// Returns the cluster status for the specified marker.
    #[inline]
    pub fn status(&self, marker: usize) -> ClusterStatus {
        self.status[marker]
    }

    /// Returns true if the marker is phased.
    #[inline]
    pub fn is_phased(&self, marker: usize) -> bool {
        self.status[marker] == ClusterStatus::Phased
    }

    /// Returns true if the marker is an unphased heterozygote.
    #[inline]
    pub fn is_unphased(&self, marker: usize) -> bool {
        self.status[marker] == ClusterStatus::Unphased
    }

    /// Returns true if the marker is a heterozygote (phased or unphased).
    #[inline]
    pub fn is_het(&self, marker: usize) -> bool {
        matches!(
            self.status[marker],
            ClusterStatus::Phased | ClusterStatus::Unphased | ClusterStatus::Masked
        )
    }

    /// Attempt to set phase for an unphased marker.
    ///
    /// Only affects markers with Unphased status. If `lr` (likelihood ratio)
    /// exceeds `threshold`, the marker is marked as Phased.
    ///
    /// # Arguments
    /// * `marker` - Marker index
    /// * `swap` - If true, swap haplotype alleles before phasing
    /// * `lr` - Likelihood ratio (log scale or linear, depending on caller)
    /// * `threshold` - Minimum LR required to confirm phase
    ///
    /// # Returns
    /// `true` if phase was set or updated, `false` if marker was not Unphased.
    pub fn try_phase(&mut self, marker: usize, swap: bool, lr: f64, threshold: f64) -> bool {
        if self.status[marker] != ClusterStatus::Unphased {
            return false;
        }

        if swap {
            let tmp = self.hap1[marker];
            self.hap1[marker] = self.hap2[marker];
            self.hap2[marker] = tmp;
        }

        if lr >= threshold {
            self.mark_phased(marker);
        }

        true
    }

    /// Swap alleles between haplotypes in the specified range.
    ///
    /// Only swaps markers that are currently Unphased.
    ///
    /// # Arguments
    /// * `start` - Start marker index (inclusive)
    /// * `end` - End marker index (exclusive)
    pub fn swap_haps(&mut self, start: usize, end: usize) {
        for m in start..end {
            if self.status[m] == ClusterStatus::Unphased {
                let tmp = self.hap1[m];
                self.hap1[m] = self.hap2[m];
                self.hap2[m] = tmp;
            }
        }
    }

    /// Swap alleles between haplotypes in the specified range unconditionally.
    ///
    /// Unlike `swap_haps`, this swaps all markers regardless of status.
    ///
    /// # Arguments
    /// * `start` - Start marker index (inclusive)
    /// * `end` - End marker index (exclusive)
    pub fn swap_haps_unchecked(&mut self, start: usize, end: usize) {
        for m in start..end {
            let tmp = self.hap1[m];
            self.hap1[m] = self.hap2[m];
            self.hap2[m] = tmp;
        }
    }

    /// Swap alleles at a single marker unconditionally.
    #[inline]
    pub fn swap_alleles(&mut self, marker: usize) {
        let tmp = self.hap1[marker];
        self.hap1[marker] = self.hap2[marker];
        self.hap2[marker] = tmp;
    }

    /// Mark a marker as phased.
    ///
    /// Only has effect if the marker is currently Unphased.
    pub fn mark_phased(&mut self, marker: usize) {
        if self.status[marker] == ClusterStatus::Unphased {
            self.status_counts[ClusterStatus::Unphased as usize] -= 1;
            self.status_counts[ClusterStatus::Phased as usize] += 1;
            self.status[marker] = ClusterStatus::Phased;
        }
    }

    /// Mark a marker as masked (excluded from phasing).
    ///
    /// Only has effect if the marker is currently Unphased or Phased.
    pub fn mark_masked(&mut self, marker: usize) {
        let old_status = self.status[marker];
        if old_status == ClusterStatus::Unphased || old_status == ClusterStatus::Phased {
            self.status_counts[old_status as usize] -= 1;
            self.status_counts[ClusterStatus::Masked as usize] += 1;
            self.status[marker] = ClusterStatus::Masked;
        }
    }

    /// Returns the count of unphased heterozygotes.
    #[inline]
    pub fn n_unphased(&self) -> usize {
        self.status_counts[ClusterStatus::Unphased as usize]
    }

    /// Returns the count of phased heterozygotes.
    #[inline]
    pub fn n_phased(&self) -> usize {
        self.status_counts[ClusterStatus::Phased as usize]
    }

    /// Returns the count of masked heterozygotes.
    #[inline]
    pub fn n_masked(&self) -> usize {
        self.status_counts[ClusterStatus::Masked as usize]
    }

    /// Returns the count of missing genotypes.
    #[inline]
    pub fn n_missing(&self) -> usize {
        self.status_counts[ClusterStatus::Missing as usize]
    }

    /// Returns the count of homozygous markers.
    #[inline]
    pub fn n_homozygous(&self) -> usize {
        self.status_counts[ClusterStatus::Homozygous as usize]
    }

    /// Returns iterator over indices of unphased heterozygotes.
    pub fn unphased_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.status
            .iter()
            .enumerate()
            .filter(|(_, s)| **s == ClusterStatus::Unphased)
            .map(|(i, _)| i)
    }

    /// Returns iterator over indices of phased heterozygotes.
    pub fn phased_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.status
            .iter()
            .enumerate()
            .filter(|(_, s)| **s == ClusterStatus::Phased)
            .map(|(i, _)| i)
    }

    /// Copy alleles to provided slices.
    pub fn copy_alleles(&self, hap1_out: &mut [u8], hap2_out: &mut [u8]) {
        assert_eq!(hap1_out.len(), self.n_markers);
        assert_eq!(hap2_out.len(), self.n_markers);

        hap1_out.copy_from_slice(&self.hap1);
        hap2_out.copy_from_slice(&self.hap2);
    }

    /// Set allele on haplotype 1.
    /// Values: 0 = REF, 1-254 = ALT alleles, 255 = missing
    #[inline]
    pub fn set_allele1(&mut self, marker: usize, allele: u8) {
        self.hap1[marker] = allele;
    }

    /// Set allele on haplotype 2.
    /// Values: 0 = REF, 1-254 = ALT alleles, 255 = missing
    #[inline]
    pub fn set_allele2(&mut self, marker: usize, allele: u8) {
        self.hap2[marker] = allele;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_basic() {
        let hap1 = vec![0, 1, 0, 1, 0];
        let hap2 = vec![0, 0, 1, 1, 0];
        let unphased = vec![2usize];
        let missing = vec![4usize];

        let sp = SamplePhase::new(0, 5, &hap1, &hap2, &unphased, &missing);

        assert_eq!(sp.sample(), 0);
        assert_eq!(sp.n_markers(), 5);
    }

    #[test]
    fn test_allele_access() {
        let hap1 = vec![0, 1, 0, 1];
        let hap2 = vec![1, 0, 1, 1];

        let sp = SamplePhase::new(0, 4, &hap1, &hap2, &[], &[]);

        assert_eq!(sp.allele1(0), 0);
        assert_eq!(sp.allele1(1), 1);
        assert_eq!(sp.allele2(0), 1);
        assert_eq!(sp.allele2(3), 1);
    }

    #[test]
    fn test_status_determination() {
        // marker 0: hom (0/0)
        // marker 1: phased het (1/0, not in unphased list)
        // marker 2: unphased het (0/1, in unphased list)
        // marker 3: missing
        let hap1 = vec![0, 1, 0, 0];
        let hap2 = vec![0, 0, 1, 0];
        let unphased = vec![2usize];
        let missing = vec![3usize];

        let sp = SamplePhase::new(0, 4, &hap1, &hap2, &unphased, &missing);

        assert_eq!(sp.status(0), ClusterStatus::Homozygous);
        assert_eq!(sp.status(1), ClusterStatus::Phased);
        assert_eq!(sp.status(2), ClusterStatus::Unphased);
        assert_eq!(sp.status(3), ClusterStatus::Missing);

        assert!(!sp.is_phased(0));
        assert!(sp.is_phased(1));
        assert!(!sp.is_phased(2));
        assert!(sp.is_unphased(2));
    }

    #[test]
    fn test_counts() {
        let hap1 = vec![0, 1, 0, 0, 1];
        let hap2 = vec![0, 0, 1, 0, 0];
        let unphased = vec![2usize];
        let missing = vec![3usize];

        let sp = SamplePhase::new(0, 5, &hap1, &hap2, &unphased, &missing);

        assert_eq!(sp.n_homozygous(), 1); // marker 0
        assert_eq!(sp.n_phased(), 2); // markers 1, 4
        assert_eq!(sp.n_unphased(), 1); // marker 2
        assert_eq!(sp.n_missing(), 1); // marker 3
    }

    #[test]
    fn test_try_phase() {
        let hap1 = vec![0, 0];
        let hap2 = vec![0, 1];
        let unphased = vec![1usize];

        let mut sp = SamplePhase::new(0, 2, &hap1, &hap2, &unphased, &[]);

        assert_eq!(sp.n_unphased(), 1);
        assert_eq!(sp.n_phased(), 0);

        // Try phasing homozygous marker - should fail
        assert!(!sp.try_phase(0, false, 10.0, 5.0));

        // Try phasing unphased het with LR below threshold - should update but not mark phased
        assert!(sp.try_phase(1, false, 3.0, 5.0));
        assert_eq!(sp.n_unphased(), 1);

        // Try phasing with swap and LR above threshold
        assert!(sp.try_phase(1, true, 10.0, 5.0));
        assert_eq!(sp.n_unphased(), 0);
        assert_eq!(sp.n_phased(), 1);

        // Alleles should be swapped
        assert_eq!(sp.allele1(1), 1);
        assert_eq!(sp.allele2(1), 0);
    }

    #[test]
    fn test_swap_haps() {
        let hap1 = vec![0, 1, 0];
        let hap2 = vec![1, 0, 1];
        let unphased = vec![0usize, 2];

        let mut sp = SamplePhase::new(0, 3, &hap1, &hap2, &unphased, &[]);

        // Marker 1 is phased het, markers 0 and 2 are unphased
        sp.swap_haps(0, 3);

        // Only unphased markers should be swapped
        assert_eq!(sp.allele1(0), 1); // swapped
        assert_eq!(sp.allele2(0), 0);
        assert_eq!(sp.allele1(1), 1); // NOT swapped (phased)
        assert_eq!(sp.allele2(1), 0);
        assert_eq!(sp.allele1(2), 1); // swapped
        assert_eq!(sp.allele2(2), 0);
    }

    #[test]
    fn test_swap_haps_unchecked() {
        let hap1 = vec![0, 1, 0];
        let hap2 = vec![1, 0, 1];

        let mut sp = SamplePhase::new(0, 3, &hap1, &hap2, &[], &[]);

        sp.swap_haps_unchecked(0, 3);

        // All markers should be swapped
        assert_eq!(sp.allele1(0), 1);
        assert_eq!(sp.allele2(0), 0);
        assert_eq!(sp.allele1(1), 0);
        assert_eq!(sp.allele2(1), 1);
    }

    #[test]
    fn test_mark_phased() {
        let hap1 = vec![0, 0];
        let hap2 = vec![1, 1];
        let unphased = vec![0usize, 1];

        let mut sp = SamplePhase::new(0, 2, &hap1, &hap2, &unphased, &[]);

        assert_eq!(sp.n_unphased(), 2);
        sp.mark_phased(0);
        assert_eq!(sp.n_unphased(), 1);
        assert_eq!(sp.n_phased(), 1);
        assert!(sp.is_phased(0));
    }

    #[test]
    fn test_mark_masked() {
        let hap1 = vec![0, 0];
        let hap2 = vec![1, 1];
        let unphased = vec![0usize, 1];

        let mut sp = SamplePhase::new(0, 2, &hap1, &hap2, &unphased, &[]);

        sp.mark_masked(0);
        assert_eq!(sp.n_unphased(), 1);
        assert_eq!(sp.n_masked(), 1);
        assert_eq!(sp.status(0), ClusterStatus::Masked);
    }

    #[test]
    fn test_iterators() {
        let hap1 = vec![0, 1, 0, 1, 0];
        let hap2 = vec![1, 0, 1, 0, 0];
        let unphased = vec![0usize, 2];

        let sp = SamplePhase::new(0, 5, &hap1, &hap2, &unphased, &[]);

        let unphased_markers: Vec<_> = sp.unphased_iter().collect();
        assert_eq!(unphased_markers, vec![0, 2]);

        let phased_markers: Vec<_> = sp.phased_iter().collect();
        assert_eq!(phased_markers, vec![1, 3]);
    }

    #[test]
    fn test_copy_alleles() {
        let hap1 = vec![0, 1, 0];
        let hap2 = vec![1, 0, 1];

        let sp = SamplePhase::new(0, 3, &hap1, &hap2, &[], &[]);

        let mut out1 = vec![0u8; 3];
        let mut out2 = vec![0u8; 3];
        sp.copy_alleles(&mut out1, &mut out2);

        assert_eq!(out1, hap1);
        assert_eq!(out2, hap2);
    }

    #[test]
    fn test_set_alleles() {
        let hap1 = vec![0, 0, 0];
        let hap2 = vec![0, 0, 0];

        let mut sp = SamplePhase::new(0, 3, &hap1, &hap2, &[], &[]);

        sp.set_allele1(1, 1);
        sp.set_allele2(2, 1);

        assert_eq!(sp.allele1(0), 0);
        assert_eq!(sp.allele1(1), 1);
        assert_eq!(sp.allele2(2), 1);
    }

    #[test]
    fn test_is_het() {
        let hap1 = vec![0, 0, 0, 0];
        let hap2 = vec![0, 1, 1, 1];
        let unphased = vec![2usize];

        let mut sp = SamplePhase::new(0, 4, &hap1, &hap2, &unphased, &[]);

        assert!(!sp.is_het(0)); // homozygous
        assert!(sp.is_het(1)); // phased het
        assert!(sp.is_het(2)); // unphased het

        sp.mark_masked(3);
        assert!(sp.is_het(3)); // masked het
    }

    #[test]
    fn test_multiallelic_support() {
        // Test that multiallelic alleles (2, 3, etc.) are preserved correctly
        let hap1 = vec![0, 1, 2, 3, 255]; // REF, ALT1, ALT2, ALT3, missing
        let hap2 = vec![2, 0, 1, 2, 255]; // ALT2, REF, ALT1, ALT2, missing
        let unphased = vec![1usize, 2, 3];
        let missing = vec![4usize];

        let sp = SamplePhase::new(0, 5, &hap1, &hap2, &unphased, &missing);

        // Verify alleles are preserved exactly
        assert_eq!(sp.allele1(0), 0);
        assert_eq!(sp.allele2(0), 2);
        assert_eq!(sp.allele1(1), 1);
        assert_eq!(sp.allele2(1), 0);
        assert_eq!(sp.allele1(2), 2); // ALT2 preserved (not coerced to 1)
        assert_eq!(sp.allele2(2), 1);
        assert_eq!(sp.allele1(3), 3); // ALT3 preserved (not coerced to 1)
        assert_eq!(sp.allele2(3), 2);
        assert_eq!(sp.allele1(4), 255); // Missing preserved
        assert_eq!(sp.allele2(4), 255);

        // Test copy_alleles preserves values
        let mut out1 = vec![0u8; 5];
        let mut out2 = vec![0u8; 5];
        sp.copy_alleles(&mut out1, &mut out2);
        assert_eq!(out1, hap1);
        assert_eq!(out2, hap2);
    }

    #[test]
    fn test_multiallelic_swap() {
        let hap1 = vec![0, 2, 3]; // REF, ALT2, ALT3
        let hap2 = vec![1, 0, 2]; // ALT1, REF, ALT2
        let unphased = vec![0usize, 1, 2];

        let mut sp = SamplePhase::new(0, 3, &hap1, &hap2, &unphased, &[]);

        // Swap all haps
        sp.swap_haps(0, 3);

        // Verify alleles are swapped correctly
        assert_eq!(sp.allele1(0), 1);
        assert_eq!(sp.allele2(0), 0);
        assert_eq!(sp.allele1(1), 0);
        assert_eq!(sp.allele2(1), 2); // ALT2 preserved
        assert_eq!(sp.allele1(2), 2); // ALT2 preserved
        assert_eq!(sp.allele2(2), 3); // ALT3 preserved
    }

    #[test]
    fn test_set_multiallelic() {
        let hap1 = vec![0, 0, 0];
        let hap2 = vec![0, 0, 0];

        let mut sp = SamplePhase::new(0, 3, &hap1, &hap2, &[], &[]);

        // Set multiallelic values
        sp.set_allele1(0, 2); // ALT2
        sp.set_allele1(1, 3); // ALT3
        sp.set_allele2(2, 255); // Missing

        assert_eq!(sp.allele1(0), 2);
        assert_eq!(sp.allele1(1), 3);
        assert_eq!(sp.allele2(2), 255);
    }
}
