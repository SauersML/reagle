//! # IBS2 Segment Detection
//!
//! Identifies segments where samples share both haplotypes (IBS2).
//! This matches Java `phase/Ibs2.java` and `phase/Ibs2Sets.java`.
//!
//! IBS2 segments are used to:
//! - Enforce phase consistency between related samples
//! - Speed up phasing in regions of high relatedness
//! - Prune HMM search space

use std::collections::HashMap;

use crate::data::genetic_map::GeneticMap;
use crate::data::haplotype::SampleIdx;
use crate::data::marker::MarkerIdx;
use crate::data::storage::GenotypeMatrix;

/// Minimum IBS2 segment length in cM
const MIN_IBS2_CM: f64 = 2.0;

/// Maximum gap to merge IBS2 segments in cM
const MAX_IBS2_GAP_CM: f64 = 4.0;

/// A segment where two samples are IBS2
#[derive(Clone, Debug)]
pub struct Ibs2Segment {
    /// The other sample in the IBS2 relationship
    pub other_sample: SampleIdx,
    /// Start marker (inclusive)
    pub start: usize,
    /// End marker (inclusive)
    pub incl_end: usize,
}

impl Ibs2Segment {
    pub fn new(other_sample: SampleIdx, start: usize, incl_end: usize) -> Self {
        Self {
            other_sample,
            start,
            incl_end,
        }
    }

    /// Check if a marker is within this segment
    pub fn contains(&self, marker: usize) -> bool {
        marker >= self.start && marker <= self.incl_end
    }

    /// Length of segment in markers
    pub fn len(&self) -> usize {
        self.incl_end - self.start + 1
    }
}

/// Collection of IBS2 segments for all target samples
pub struct Ibs2 {
    /// Number of markers
    n_markers: usize,
    /// IBS2 segments for each sample: sample_segs[sample_idx] = Vec<Ibs2Segment>
    sample_segs: Vec<Vec<Ibs2Segment>>,
}

impl Ibs2 {
    /// Build IBS2 segments from genotype data
    ///
    /// # Arguments
    /// * `gt` - Genotype matrix (target samples)
    /// * `gen_map` - Genetic map for distance calculations
    /// * `maf` - Minor allele frequencies for each marker
    pub fn new(gt: &GenotypeMatrix, gen_map: &GeneticMap, maf: &[f32]) -> Self {
        let n_markers = gt.n_markers();
        let n_samples = gt.n_samples();

        // First pass: find initial IBS2 sets using informative markers
        let ibs2_markers = Ibs2Markers::new(gt, maf);
        let ibs2_sets = Ibs2Sets::new(gt, &ibs2_markers);

        // Build segments for each sample
        let sample_segs: Vec<Vec<Ibs2Segment>> = (0..n_samples)
            .map(|s| {
                let sample = SampleIdx::new(s as u32);
                Self::build_sample_segments(gt, gen_map, &ibs2_sets, sample)
            })
            .collect();

        Self {
            n_markers,
            sample_segs,
        }
    }

    fn build_sample_segments(
        gt: &GenotypeMatrix,
        gen_map: &GeneticMap,
        ibs2_sets: &Ibs2Sets,
        sample: SampleIdx,
    ) -> Vec<Ibs2Segment> {
        let mut segments = ibs2_sets.segments_for_sample(sample);

        // Sort by other sample
        segments.sort_by_key(|s| s.other_sample.0);

        // Merge adjacent segments
        segments = Self::merge_segments(segments, gen_map);

        // Extend segments through homozygous regions
        segments = Self::extend_segments(gt, sample, segments);

        // Merge again after extension
        segments = Self::merge_segments(segments, gen_map);

        // Filter by minimum length
        segments = Self::filter_by_length(segments, gen_map);

        segments
    }

    fn merge_segments(segments: Vec<Ibs2Segment>, gen_map: &GeneticMap) -> Vec<Ibs2Segment> {
        if segments.len() < 2 {
            return segments;
        }

        let mut merged = Vec::new();
        let mut prev = segments[0].clone();

        for next in segments.into_iter().skip(1) {
            if prev.other_sample == next.other_sample {
                let gap_cm = Self::gap_cm(&prev, &next, gen_map);
                if gap_cm <= MAX_IBS2_GAP_CM {
                    // Merge segments
                    prev = Ibs2Segment::new(
                        prev.other_sample,
                        prev.start,
                        next.incl_end,
                    );
                    continue;
                }
            }
            merged.push(prev);
            prev = next;
        }
        merged.push(prev);

        merged
    }

    fn gap_cm(prev: &Ibs2Segment, next: &Ibs2Segment, gen_map: &GeneticMap) -> f64 {
        let pos1 = prev.incl_end as u32;
        let pos2 = next.start as u32;
        gen_map.gen_dist(pos1, pos2)
    }

    fn extend_segments(
        gt: &GenotypeMatrix,
        sample: SampleIdx,
        segments: Vec<Ibs2Segment>,
    ) -> Vec<Ibs2Segment> {
        let n_markers = gt.n_markers();

        segments
            .into_iter()
            .map(|seg| {
                let other = seg.other_sample;
                let mut start = seg.start;
                let mut end = seg.incl_end;

                // Extend left
                while start > 0 && Self::is_ibs2_at(gt, start - 1, sample, other) {
                    start -= 1;
                }

                // Extend right
                while end < n_markers - 1 && Self::is_ibs2_at(gt, end + 1, sample, other) {
                    end += 1;
                }

                Ibs2Segment::new(other, start, end)
            })
            .collect()
    }

    fn filter_by_length(segments: Vec<Ibs2Segment>, gen_map: &GeneticMap) -> Vec<Ibs2Segment> {
        segments
            .into_iter()
            .filter(|seg| {
                let start_pos = seg.start as u32;
                let end_pos = seg.incl_end as u32;
                let len_cm = gen_map.gen_dist(start_pos, end_pos);
                len_cm >= MIN_IBS2_CM
            })
            .collect()
    }

    /// Check if two samples are IBS2 at a marker
    fn is_ibs2_at(gt: &GenotypeMatrix, marker: usize, s1: SampleIdx, s2: SampleIdx) -> bool {
        let m_idx = MarkerIdx::new(marker as u32);

        let a1 = gt.allele(m_idx, s1.hap1());
        let a2 = gt.allele(m_idx, s1.hap2());
        let b1 = gt.allele(m_idx, s2.hap1());
        let b2 = gt.allele(m_idx, s2.hap2());

        Self::are_phase_consistent(a1, a2, b1, b2)
            || Self::are_phase_consistent(a1, a2, b2, b1)
    }

    fn are_phase_consistent(a1: u8, a2: u8, b1: u8, b2: u8) -> bool {
        (a1 == 255 || b1 == 255 || a1 == b1) && (a2 == 255 || b2 == 255 || a2 == b2)
    }

    /// Number of markers
    pub fn n_markers(&self) -> usize {
        self.n_markers
    }

    /// Number of target samples
    pub fn n_samples(&self) -> usize {
        self.sample_segs.len()
    }

    /// Number of IBS2 segments for a sample
    pub fn n_segments(&self, sample: SampleIdx) -> usize {
        self.sample_segs
            .get(sample.0 as usize)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Check if two samples are IBS2 at a marker
    pub fn are_ibs2(&self, sample: SampleIdx, other: SampleIdx, marker: usize) -> bool {
        if sample == other {
            return true;
        }

        let segs = match self.sample_segs.get(sample.0 as usize) {
            Some(s) => s,
            None => return false,
        };

        for seg in segs {
            if seg.other_sample == other && seg.contains(marker) {
                return true;
            }
        }

        false
    }

    /// Check if two samples are IBS2 within an interval
    pub fn are_ibs2_in_interval(
        &self,
        sample: SampleIdx,
        other: SampleIdx,
        start: usize,
        incl_end: usize,
    ) -> bool {
        if sample == other {
            return true;
        }

        let segs = match self.sample_segs.get(sample.0 as usize) {
            Some(s) => s,
            None => return false,
        };

        for seg in segs {
            if seg.other_sample == other {
                // Check for overlap
                if start <= seg.incl_end && seg.start <= incl_end {
                    return true;
                }
            }
        }

        false
    }

    /// Get all IBS2 segments for a sample
    pub fn segments(&self, sample: SampleIdx) -> &[Ibs2Segment] {
        self.sample_segs
            .get(sample.0 as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get samples that are IBS2 with target at a marker
    pub fn ibs2_samples_at(&self, sample: SampleIdx, marker: usize) -> Vec<SampleIdx> {
        let mut result = Vec::new();

        let segs = match self.sample_segs.get(sample.0 as usize) {
            Some(s) => s,
            None => return result,
        };

        for seg in segs {
            if seg.contains(marker) {
                result.push(seg.other_sample);
            }
        }

        result
    }
}

/// Identifies informative markers for IBS2 detection
struct Ibs2Markers {
    /// Marker indices that are informative (not too rare, not too common)
    informative: Vec<usize>,
}

impl Ibs2Markers {
    /// Minimum MAF for informative markers
    const MIN_MAF: f32 = 0.05;

    fn new(gt: &GenotypeMatrix, maf: &[f32]) -> Self {
        let n_markers = gt.n_markers();
        let mut informative = Vec::new();

        for m in 0..n_markers {
            let marker_maf = maf.get(m).copied().unwrap_or(0.0);
            if marker_maf >= Self::MIN_MAF && marker_maf <= 1.0 - Self::MIN_MAF {
                informative.push(m);
            }
        }

        Self { informative }
    }

    fn len(&self) -> usize {
        self.informative.len()
    }
}

/// Builds initial IBS2 sets for efficient lookup
struct Ibs2Sets {
    /// For each informative marker, sets of sample pairs that are IBS2
    /// ibs2_at_marker[info_marker_idx] = HashMap<(min_sample, max_sample), ()>
    ibs2_at_marker: Vec<HashMap<(u32, u32), ()>>,
}

impl Ibs2Sets {
    fn new(gt: &GenotypeMatrix, ibs2_markers: &Ibs2Markers) -> Self {
        let n_samples = gt.n_samples();
        let n_info = ibs2_markers.len();

        let mut ibs2_at_marker: Vec<HashMap<(u32, u32), ()>> =
            vec![HashMap::new(); n_info];

        // For each informative marker, find IBS2 pairs
        for (info_idx, &marker) in ibs2_markers.informative.iter().enumerate() {
            let m_idx = MarkerIdx::new(marker as u32);

            // Group samples by genotype pattern
            let mut patterns: HashMap<(u8, u8), Vec<u32>> = HashMap::new();

            for s in 0..n_samples {
                let sample = SampleIdx::new(s as u32);
                let a1 = gt.allele(m_idx, sample.hap1());
                let a2 = gt.allele(m_idx, sample.hap2());

                // Skip missing
                if a1 == 255 || a2 == 255 {
                    continue;
                }

                // Normalize: smaller allele first
                let key = if a1 <= a2 { (a1, a2) } else { (a2, a1) };
                patterns.entry(key).or_default().push(s as u32);
            }

            // Samples with same genotype pattern are potentially IBS2
            for samples in patterns.values() {
                if samples.len() < 2 {
                    continue;
                }

                for i in 0..samples.len() {
                    for j in (i + 1)..samples.len() {
                        let s1 = samples[i].min(samples[j]);
                        let s2 = samples[i].max(samples[j]);
                        ibs2_at_marker[info_idx].insert((s1, s2), ());
                    }
                }
            }
        }

        Self { ibs2_at_marker }
    }

    /// Get initial IBS2 segments for a sample
    fn segments_for_sample(&self, sample: SampleIdx) -> Vec<Ibs2Segment> {
        let s = sample.0;
        let mut segments = Vec::new();
        let mut current_segments: HashMap<u32, usize> = HashMap::new(); // other -> start_info_idx

        for (info_idx, pairs) in self.ibs2_at_marker.iter().enumerate() {
            // Find pairs involving this sample
            let mut current_partners: Vec<u32> = Vec::new();

            for &(s1, s2) in pairs.keys() {
                if s1 == s {
                    current_partners.push(s2);
                } else if s2 == s {
                    current_partners.push(s1);
                }
            }

            // End segments for samples no longer IBS2
            let ended: Vec<u32> = current_segments
                .keys()
                .filter(|&&other| !current_partners.contains(&other))
                .copied()
                .collect();

            for other in ended {
                if let Some(start) = current_segments.remove(&other) {
                    segments.push(Ibs2Segment::new(
                        SampleIdx::new(other),
                        start,
                        info_idx.saturating_sub(1),
                    ));
                }
            }

            // Start new segments
            for other in current_partners {
                current_segments.entry(other).or_insert(info_idx);
            }
        }

        // Close remaining segments
        let n_info = self.ibs2_at_marker.len();
        for (other, start) in current_segments {
            segments.push(Ibs2Segment::new(
                SampleIdx::new(other),
                start,
                n_info.saturating_sub(1),
            ));
        }

        segments
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ibs2_segment() {
        let seg = Ibs2Segment::new(SampleIdx::new(1), 10, 20);
        assert!(seg.contains(10));
        assert!(seg.contains(15));
        assert!(seg.contains(20));
        assert!(!seg.contains(9));
        assert!(!seg.contains(21));
        assert_eq!(seg.len(), 11);
    }

    #[test]
    fn test_phase_consistent() {
        // Same genotype and same phase
        assert!(Ibs2::are_phase_consistent(0, 1, 0, 1));

        // Same genotype but swapped phase - checks EXACT phase match
        // is_ibs2_at tries both orderings for IBS2 check
        assert!(!Ibs2::are_phase_consistent(0, 1, 1, 0));

        // Different genotypes
        assert!(!Ibs2::are_phase_consistent(0, 0, 1, 1));

        // Missing data is always consistent
        assert!(Ibs2::are_phase_consistent(255, 1, 0, 1));
        assert!(Ibs2::are_phase_consistent(0, 255, 0, 1));
    }
}
