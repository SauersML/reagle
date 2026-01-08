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
    /// Homozygous genotype
    Homozygous = 1,
    /// Phase has been determined with high confidence
    Phased = 2,
    /// Phase is not yet determined
    Unphased = 3,
}

impl ClusterStatus {
    /// Number of status variants (for array indexing)
    pub const COUNT: usize = 4;
}

/// Phase state tracking for a single sample
///
/// Uses byte storage (1 byte per allele) to support multiallelic markers.
/// Allele values: 0 = REF, 1-254 = ALT alleles, 255 = missing
#[derive(Clone, Debug)]
pub struct SamplePhase {
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
    /// * `sample` - Sample index (unused, kept for API compatibility)
    /// * `n_markers` - Number of markers
    /// * `hap1_alleles` - Alleles on first haplotype (0-254 for alleles, 255 for missing)
    /// * `hap2_alleles` - Alleles on second haplotype
    /// * `unphased_hets` - Sorted indices of markers that are unphased heterozygotes
    /// * `missing` - Sorted indices of markers with missing data
    ///
    /// # Panics
    /// Panics if allele slices don't match n_markers or indices are invalid.
    #[allow(unused_variables)]
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

    /// Returns true if the marker is an unphased heterozygote.
    #[inline]
    pub fn is_unphased(&self, marker: usize) -> bool {
        self.status[marker] == ClusterStatus::Unphased
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
                std::mem::swap(&mut self.hap1[m], &mut self.hap2[m]);
            }
        }
    }

    /// Swap alleles at a single marker unconditionally.
    #[inline]
    pub fn swap_alleles(&mut self, marker: usize) {
        std::mem::swap(&mut self.hap1[marker], &mut self.hap2[marker]);
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

    /// Returns true if the marker has missing genotype data.
    #[inline]
    pub fn is_missing(&self, marker: usize) -> bool {
        self.status[marker] == ClusterStatus::Missing
    }

    /// Set imputed alleles for a missing marker.
    ///
    /// Sets the alleles and changes status from Missing to Phased.
    /// Only has effect if the marker is currently Missing.
    pub fn set_imputed(&mut self, marker: usize, a1: u8, a2: u8) {
        if self.status[marker] == ClusterStatus::Missing {
            self.hap1[marker] = a1;
            self.hap2[marker] = a2;
            self.status_counts[ClusterStatus::Missing as usize] -= 1;
            self.status_counts[ClusterStatus::Phased as usize] += 1;
            self.status[marker] = ClusterStatus::Phased;
        }
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

        assert_eq!(sp.hap1.len(), 5);
        assert_eq!(sp.hap2.len(), 5);
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
    fn test_is_unphased() {
        let hap1 = vec![0, 1, 0, 0];
        let hap2 = vec![0, 0, 1, 0];
        let unphased = vec![2usize];
        let missing = vec![3usize];

        let sp = SamplePhase::new(0, 4, &hap1, &hap2, &unphased, &missing);

        assert!(!sp.is_unphased(0)); // homozygous
        assert!(!sp.is_unphased(1)); // phased
        assert!(sp.is_unphased(2)); // unphased
        assert!(!sp.is_unphased(3)); // missing
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
    fn test_mark_phased() {
        let hap1 = vec![0, 0];
        let hap2 = vec![1, 1];
        let unphased = vec![0usize, 1];

        let mut sp = SamplePhase::new(0, 2, &hap1, &hap2, &unphased, &[]);

        assert!(sp.is_unphased(0));
        sp.mark_phased(0);
        assert!(!sp.is_unphased(0));
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
        assert_eq!(sp.allele1(2), 2); // ALT2 preserved
        assert_eq!(sp.allele1(3), 3); // ALT3 preserved
        assert_eq!(sp.allele1(4), 255); // Missing preserved
    }

    #[test]
    fn test_swap_alleles() {
        let hap1 = vec![0, 2, 3];
        let hap2 = vec![1, 0, 2];

        let mut sp = SamplePhase::new(0, 3, &hap1, &hap2, &[], &[]);

        sp.swap_alleles(1);

        assert_eq!(sp.allele1(1), 0);
        assert_eq!(sp.allele2(1), 2);
    }
}
