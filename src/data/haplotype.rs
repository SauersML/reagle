//! # Haplotype and Sample Definitions
//!
//! Sample and haplotype index types. Replaces `vcf/Samples.java`.

use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// Zero-cost newtype for sample indices
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub struct SampleIdx(pub u32);

impl SampleIdx {
    pub fn new(idx: u32) -> Self {
        Self(idx)
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Get the first haplotype index for this sample (assumes diploid)
    pub fn hap1(self) -> HapIdx {
        HapIdx::new(self.0 * 2)
    }

    /// Get the second haplotype index for this sample (assumes diploid)
    pub fn hap2(self) -> HapIdx {
        HapIdx::new(self.0 * 2 + 1)
    }
}

impl From<u32> for SampleIdx {
    fn from(idx: u32) -> Self {
        Self(idx)
    }
}

impl From<usize> for SampleIdx {
    fn from(idx: usize) -> Self {
        Self(idx as u32)
    }
}

impl From<SampleIdx> for usize {
    fn from(idx: SampleIdx) -> usize {
        idx.0 as usize
    }
}

/// Zero-cost newtype for haplotype indices
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub struct HapIdx(pub u32);

impl HapIdx {
    pub fn new(idx: u32) -> Self {
        Self(idx)
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Get the sample index for this haplotype (assumes diploid)
    pub fn sample(self) -> SampleIdx {
        SampleIdx::new(self.0 / 2)
    }

    /// Check if this is the first haplotype of the sample (hap index 0)
    pub fn is_first(self) -> bool {
        self.0 % 2 == 0
    }

    /// Check if this is the second haplotype of the sample (hap index 1)
    pub fn is_second(self) -> bool {
        self.0 % 2 == 1
    }

    /// Get the other haplotype for this sample
    pub fn other(self) -> HapIdx {
        if self.is_first() {
            HapIdx::new(self.0 + 1)
        } else {
            HapIdx::new(self.0 - 1)
        }
    }
}

impl From<u32> for HapIdx {
    fn from(idx: u32) -> Self {
        Self(idx)
    }
}

impl From<usize> for HapIdx {
    fn from(idx: usize) -> Self {
        Self(idx as u32)
    }
}

impl From<HapIdx> for usize {
    fn from(idx: HapIdx) -> usize {
        idx.0 as usize
    }
}

/// A collection of samples
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Samples {
    /// Sample IDs
    ids: Vec<Arc<str>>,
    /// Whether each sample is diploid (true) or haploid (false)
    is_diploid: Vec<bool>,
    /// Map from sample ID to index for fast lookup
    id_to_idx: HashMap<Arc<str>, SampleIdx>,
    /// Cumulative haplotype offsets for each sample (precomputed for O(1) lookup)
    /// hap_offset[i] = number of haplotypes from samples 0..i
    #[serde(skip)]
    hap_offset: Vec<usize>,
}

impl Samples {
    /// Create from a vector of sample IDs (all diploid)
    pub fn from_ids(ids: Vec<String>) -> Self {
        let is_diploid = vec![true; ids.len()];
        Self::from_ids_with_ploidy(ids, is_diploid)
    }

    /// Create from sample IDs with explicit ploidy per sample
    ///
    /// # Arguments
    /// * `ids` - Sample identifiers
    /// * `is_diploid` - Whether each sample is diploid (true) or haploid (false)
    pub fn from_ids_with_ploidy(ids: Vec<String>, is_diploid: Vec<bool>) -> Self {
        assert_eq!(ids.len(), is_diploid.len(), "ids and is_diploid must have same length");

        let ids: Vec<Arc<str>> = ids.into_iter().map(|s| s.into()).collect();
        let id_to_idx = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), SampleIdx::new(i as u32)))
            .collect();

        // Compute cumulative haplotype offsets for O(1) lookup
        let hap_offset = Self::compute_hap_offsets(&is_diploid);

        Self {
            ids,
            is_diploid,
            id_to_idx,
            hap_offset,
        }
    }

    /// Compute cumulative haplotype offsets from ploidy vector
    fn compute_hap_offsets(is_diploid: &[bool]) -> Vec<usize> {
        let mut offsets = Vec::with_capacity(is_diploid.len() + 1);
        offsets.push(0);
        let mut cumulative = 0usize;
        for &diploid in is_diploid {
            cumulative += if diploid { 2 } else { 1 };
            offsets.push(cumulative);
        }
        offsets
    }

    /// Number of samples
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Number of haplotypes in storage (always 2 per sample for compatibility)
    ///
    /// Note: Storage always allocates 2 slots per sample even for haploid samples
    /// (the allele is duplicated). Use `is_diploid()` to check actual ploidy.
    pub fn n_haps(&self) -> usize {
        self.ids.len() * 2
    }

    /// Number of true haplotypes accounting for ploidy
    ///
    /// Returns 2 for diploid samples, 1 for haploid samples.
    /// Use this for statistical calculations, not for storage indexing.
    pub fn n_true_haps(&self) -> usize {
        self.hap_offset.last().copied().unwrap_or(0)
    }

    /// Check if a sample is diploid
    pub fn is_diploid(&self, idx: SampleIdx) -> bool {
        self.is_diploid.get(idx.as_usize()).copied().unwrap_or(true)
    }

    /// Get all sample IDs
    pub fn ids(&self) -> &[Arc<str>] {
        &self.ids
    }
}

impl std::ops::Index<SampleIdx> for Samples {
    type Output = str;

    fn index(&self, idx: SampleIdx) -> &Self::Output {
        &self.ids[idx.as_usize()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_hap_indices() {
        let sample = SampleIdx::new(5);
        assert_eq!(sample.hap1(), HapIdx::new(10));
        assert_eq!(sample.hap2(), HapIdx::new(11));
    }

    #[test]
    fn test_hap_sample_index() {
        let hap = HapIdx::new(11);
        assert_eq!(hap.sample(), SampleIdx::new(5));
        assert!(hap.is_second());
        assert!(!hap.is_first());
    }

    #[test]
    fn test_samples_n_haps() {
        let samples = Samples::from_ids(vec!["A".to_string(), "B".to_string(), "C".to_string()]);
        assert_eq!(samples.len(), 3);
        assert_eq!(samples.n_haps(), 6);
    }

    #[test]
    fn test_samples_mixed_ploidy() {
        // Create samples with mixed ploidy: diploid, haploid, diploid
        let samples = Samples::from_ids_with_ploidy(
            vec!["A".to_string(), "B".to_string(), "C".to_string()],
            vec![true, false, true], // A: diploid, B: haploid, C: diploid
        );
        assert_eq!(samples.len(), 3);
        // n_haps() returns storage-based count (always 2 per sample)
        assert_eq!(samples.n_haps(), 6);
        // n_true_haps() returns actual count accounting for ploidy
        assert_eq!(samples.n_true_haps(), 5); // 2 + 1 + 2 = 5

        // Check ploidy
        assert!(samples.is_diploid(SampleIdx::new(0)));
        assert!(!samples.is_diploid(SampleIdx::new(1)));
        assert!(samples.is_diploid(SampleIdx::new(2)));
    }
}
