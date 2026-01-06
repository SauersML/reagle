//! # Haplotype and Sample Definitions
//!
//! Sample and haplotype index types. Replaces `vcf/Samples.java`.

use std::collections::HashMap;
use std::sync::Arc;

/// Zero-cost newtype for sample indices
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
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
#[derive(Clone, Debug, Default)]
pub struct Samples {
    /// Sample IDs
    ids: Vec<Arc<str>>,
    /// Whether each sample is diploid (true) or haploid (false)
    is_diploid: Vec<bool>,
    /// Map from sample ID to index for fast lookup
    id_to_idx: HashMap<Arc<str>, SampleIdx>,
}

impl Samples {
    /// Create from a vector of sample IDs (all diploid)
    pub fn from_ids(ids: Vec<String>) -> Self {
        let ids: Vec<Arc<str>> = ids.into_iter().map(|s| s.into()).collect();
        let is_diploid = vec![true; ids.len()];
        let id_to_idx = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id.clone(), SampleIdx::new(i as u32)))
            .collect();

        Self {
            ids,
            is_diploid,
            id_to_idx,
        }
    }

    /// Number of samples
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Number of haplotypes (2 per diploid sample, 1 per haploid)
    pub fn n_haps(&self) -> usize {
        self.is_diploid.iter().map(|&d| if d { 2 } else { 1 }).sum()
    }

    /// Get sample index by ID
    pub fn index_of(&self, id: &str) -> Option<SampleIdx> {
        self.id_to_idx.get(id).copied()
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
    fn test_samples_lookup() {
        let samples = Samples::from_ids(vec!["A".to_string(), "B".to_string()]);
        assert_eq!(samples.index_of("A"), Some(SampleIdx::new(0)));
        assert_eq!(samples.index_of("B"), Some(SampleIdx::new(1)));
        assert_eq!(samples.index_of("C"), None);
    }
}
