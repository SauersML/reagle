//! # IBS2 Segment Detection
//!
//! Identifies segments where samples share both haplotypes (IBS2).
//! This matches Java `phase/Ibs2.java` and `phase/Ibs2Sets.java`.
//!
//! IBS2 segments are used to:
//! - Enforce phase consistency between related samples
//! - Speed up phasing in regions of high relatedness
//! - Prune HMM search space

use std::sync::Arc;

use crate::data::ChromIdx;
use crate::data::genetic_map::GeneticMaps;
use crate::data::haplotype::{HapIdx, SampleIdx};
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
    /// IBS2 segments for each sample: sample_segs[sample_idx] = Vec<Ibs2Segment>
    sample_segs: Vec<Vec<Ibs2Segment>>,
}

impl Ibs2 {
    /// Build IBS2 segments from genotype data
    pub fn new(gt: &GenotypeMatrix, gen_maps: &GeneticMaps, chrom: ChromIdx, maf: &[f32]) -> Self {
        let n_markers = gt.n_markers();
        let n_samples = gt.n_samples();

        // Extract physical positions from markers
        let marker_positions: Vec<u32> = (0..n_markers)
            .map(|m| gt.marker(MarkerIdx::new(m as u32)).pos)
            .collect();

        // First pass: find initial IBS2 sets using recursive partitioning
        let ibs2_markers = Ibs2Markers::new(gt, gen_maps, chrom, maf);
        let ibs2_sets = Ibs2Sets::new(gt, &ibs2_markers);

        // Build segments for each sample
        let sample_segs: Vec<Vec<Ibs2Segment>> = (0..n_samples)
            .map(|s| {
                let sample = SampleIdx::new(s as u32);
                Self::build_sample_segments(
                    gt,
                    gen_maps,
                    chrom,
                    &marker_positions,
                    &ibs2_sets,
                    sample,
                )
            })
            .collect();

        Self {
            sample_segs,
        }
    }

    fn build_sample_segments(
        gt: &GenotypeMatrix,
        gen_maps: &GeneticMaps,
        chrom: ChromIdx,
        marker_positions: &[u32],
        ibs2_sets: &Ibs2Sets,
        sample: SampleIdx,
    ) -> Vec<Ibs2Segment> {
        let mut segments = ibs2_sets.segments_for_sample(sample);

        // Sort by other sample
        segments.sort_by_key(|s| s.other_sample.0);

        // Merge adjacent segments
        segments = Self::merge_segments(segments, gen_maps, chrom, marker_positions);

        // Extend segments through homozygous regions
        segments = Self::extend_segments(gt, sample, segments);

        // Merge again after extension
        segments = Self::merge_segments(segments, gen_maps, chrom, marker_positions);

        // Filter by minimum length
        segments = Self::filter_by_length(segments, gen_maps, chrom, marker_positions);

        segments
    }

    fn merge_segments(
        segments: Vec<Ibs2Segment>,
        gen_maps: &GeneticMaps,
        chrom: ChromIdx,
        marker_positions: &[u32],
    ) -> Vec<Ibs2Segment> {
        if segments.len() < 2 {
            return segments;
        }

        let mut merged = Vec::new();
        let mut prev = segments[0].clone();

        for next in segments.into_iter().skip(1) {
            if prev.other_sample == next.other_sample {
                let gap_cm = Self::gap_cm(&prev, &next, gen_maps, chrom, marker_positions);
                if gap_cm <= MAX_IBS2_GAP_CM {
                    // Merge segments
                    prev = Ibs2Segment::new(prev.other_sample, prev.start, next.incl_end);
                    continue;
                }
            }
            merged.push(prev);
            prev = next;
        }
        merged.push(prev);

        merged
    }

    fn gap_cm(
        prev: &Ibs2Segment,
        next: &Ibs2Segment,
        gen_maps: &GeneticMaps,
        chrom: ChromIdx,
        marker_positions: &[u32],
    ) -> f64 {
        let pos1 = marker_positions.get(prev.incl_end).copied().unwrap_or(0);
        let pos2 = marker_positions.get(next.start).copied().unwrap_or(0);
        gen_maps.gen_dist(chrom, pos1, pos2)
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

    fn filter_by_length(
        segments: Vec<Ibs2Segment>,
        gen_maps: &GeneticMaps,
        chrom: ChromIdx,
        marker_positions: &[u32],
    ) -> Vec<Ibs2Segment> {
        segments
            .into_iter()
            .filter(|seg| {
                let start_pos = marker_positions.get(seg.start).copied().unwrap_or(0);
                let end_pos = marker_positions.get(seg.incl_end).copied().unwrap_or(0);
                let len_cm = gen_maps.gen_dist(chrom, start_pos, end_pos);
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

        Self::are_phase_consistent(a1, a2, b1, b2) || Self::are_phase_consistent(a1, a2, b2, b1)
    }

    fn are_phase_consistent(a1: u8, a2: u8, b1: u8, b2: u8) -> bool {
        (a1 == 255 || b1 == 255 || a1 == b1) && (a2 == 255 || b2 == 255 || a2 == b2)
    }

    pub fn n_samples(&self) -> usize {
        self.sample_segs.len()
    }

    pub fn n_segments(&self, sample: SampleIdx) -> usize {
        self.sample_segs
            .get(sample.0 as usize)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    pub fn segments(&self, sample: SampleIdx) -> &[Ibs2Segment] {
        self.sample_segs
            .get(sample.0 as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

/// Identifies informative markers and partitions them into steps for IBS2 detection
struct Ibs2Markers {
    use_marker: Vec<bool>,
    step_starts: Vec<usize>,
}

impl Ibs2Markers {
    const MAX_MISS_FREQ: f32 = 0.1;
    const MIN_MINOR_FREQ: f32 = 0.1;
    const MIN_MARKER_CNT: usize = 50;
    const MIN_INTERMARKER_CM: f64 = 0.02;

    fn new(gt: &GenotypeMatrix, gen_maps: &GeneticMaps, chrom: ChromIdx, maf: &[f32]) -> Self {
        let n_markers = gt.n_markers();
        let mut use_marker = vec![false; n_markers];

        for m in 0..n_markers {
            let marker_maf = maf.get(m).copied().unwrap_or(0.0);
            if marker_maf >= Self::MIN_MINOR_FREQ && marker_maf <= 1.0 - Self::MIN_MINOR_FREQ {
                let mut miss_cnt = 0;
                let m_idx = MarkerIdx::new(m as u32);
                for h in 0..gt.n_haplotypes() {
                    if gt.allele(m_idx, HapIdx::new(h as u32)) == 255 {
                        miss_cnt += 1;
                    }
                }
                if (miss_cnt as f32 / gt.n_haplotypes() as f32) <= Self::MAX_MISS_FREQ {
                    use_marker[m] = true;
                }
            }
        }

        let mut step_starts = Vec::new();
        let mut last_start = 0;
        
        while last_start < n_markers {
            step_starts.push(last_start);
            
            let mut next_start = last_start + 1;
            let mut mkr_cnt = 0;
            let mut min_cm_pos = gen_maps.gen_pos(chrom, gt.marker(MarkerIdx::new(last_start as u32)).pos) + Self::MIN_INTERMARKER_CM;
            
            while next_start < n_markers && mkr_cnt < Self::MIN_MARKER_CNT {
                if use_marker[next_start] {
                    let cur_cm_pos = gen_maps.gen_pos(chrom, gt.marker(MarkerIdx::new(next_start as u32)).pos);
                    if cur_cm_pos < min_cm_pos {
                        use_marker[next_start] = false;
                    } else {
                        mkr_cnt += 1;
                        min_cm_pos = cur_cm_pos + Self::MIN_INTERMARKER_CM;
                    }
                }
                next_start += 1;
            }
            last_start = next_start;
        }

        Self { use_marker, step_starts }
    }

    fn markers_in_step(&self, step_idx: usize, n_markers: usize) -> Vec<usize> {
        let start = self.step_starts[step_idx];
        let end = if step_idx + 1 < self.step_starts.len() {
            self.step_starts[step_idx + 1]
        } else {
            n_markers
        };
        
        (start..end).filter(|&m| self.use_marker[m]).collect()
    }
}

/// Stores clusters of samples that are IBS2 within each step
struct Ibs2Sets {
    ibs2_sets: Vec<Vec<Option<Arc<Vec<u32>>>>>,
    step_starts: Vec<usize>,
    n_markers: usize,
}

impl Ibs2Sets {
    const MAX_MISS_STEP_FREQ: f32 = 0.1;

    fn new(gt: &GenotypeMatrix, ibs2_markers: &Ibs2Markers) -> Self {
        let n_samples = gt.n_samples();
        let n_steps = ibs2_markers.step_starts.len();
        let mut ibs2_sets = Vec::with_capacity(n_steps);

        for step in 0..n_steps {
            let step_markers = ibs2_markers.markers_in_step(step, gt.n_markers());
            if step_markers.is_empty() {
                ibs2_sets.push(vec![None; n_samples]);
                continue;
            }

            let mut init_samples = Vec::new();
            let max_miss = (Self::MAX_MISS_STEP_FREQ * step_markers.len() as f32).floor() as usize;
            for s in 0..n_samples {
                let mut miss_cnt = 0;
                for &m in &step_markers {
                    let m_idx = MarkerIdx::new(m as u32);
                    let sample = SampleIdx::new(s as u32);
                    if gt.allele(m_idx, sample.hap1()) == 255 || gt.allele(m_idx, sample.hap2()) == 255 {
                        miss_cnt += 1;
                    }
                }
                if miss_cnt <= max_miss {
                    init_samples.push(s as u32);
                }
            }

            let mut partition = vec![SampClust {
                samples: init_samples,
                is_homozygous: true,
            }];

            for &m in &step_markers {
                let mut next_partition = Vec::new();
                for parent in partition {
                    next_partition.extend(Self::partition_cluster(gt, parent, m));
                }
                partition = next_partition;
                if partition.is_empty() { break; }
            }

            let mut step_results = vec![None; n_samples];
            for clust in partition {
                if !clust.is_homozygous && clust.samples.len() > 1 {
                    let arc_samples = Arc::new(clust.samples);
                    for &s in arc_samples.iter() {
                        let s_idx = s as usize;
                        if step_results[s_idx].is_none() {
                            step_results[s_idx] = Some(Arc::clone(&arc_samples));
                        } else {
                            // Clone the inner Vec from the existing Arc
                            let mut merged: Vec<u32> = (**step_results[s_idx].as_ref().unwrap()).clone();
                            merged.extend(arc_samples.iter().copied());
                            merged.sort_unstable();
                            merged.dedup();
                            step_results[s_idx] = Some(Arc::new(merged));
                        }
                    }
                }
            }
            ibs2_sets.push(step_results);
        }

        Self {
            ibs2_sets,
            step_starts: ibs2_markers.step_starts.clone(),
            n_markers: gt.n_markers(),
        }
    }

    fn partition_cluster(gt: &GenotypeMatrix, parent: SampClust, m: usize) -> Vec<SampClust> {
        let m_idx = MarkerIdx::new(m as u32);
        let n_alleles = 1 + gt.marker(m_idx).alt_alleles.len();
        let n_gt = (n_alleles * (n_alleles + 1)) / 2;
        
        let mut gt_to_list: Vec<Option<Vec<u32>>> = vec![None; n_gt];
        let mut missing = Vec::new();
        
        let mut next_is_hom = vec![false; n_gt];
        if parent.is_homozygous {
            for a in 0..n_alleles {
                let gt_idx = (a * (a + 1)) / 2 + a;
                if gt_idx < n_gt {
                    next_is_hom[gt_idx] = true;
                }
            }
        }

        for &s in &parent.samples {
            let sample = SampleIdx::new(s);
            let a1 = gt.allele(m_idx, sample.hap1());
            let a2 = gt.allele(m_idx, sample.hap2());

            if a1 == 255 || a2 == 255 {
                missing.push(s);
                for list in gt_to_list.iter_mut().flatten() {
                    list.push(s);
                }
            } else {
                let gt_idx = if a1 <= a2 {
                    (a2 as usize * (a2 as usize + 1)) / 2 + a1 as usize
                } else {
                    (a1 as usize * (a1 as usize + 1)) / 2 + a2 as usize
                };
                
                if gt_idx < n_gt {
                    let list = gt_to_list[gt_idx].get_or_insert_with(|| missing.clone());
                    list.push(s);
                }
            }
        }

        gt_to_list.into_iter().enumerate()
            .filter_map(|(i, opt_list)| {
                let list = opt_list?;
                if list.len() > 1 {
                    Some(SampClust {
                        samples: list,
                        is_homozygous: next_is_hom[i],
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    fn segments_for_sample(&self, sample: SampleIdx) -> Vec<Ibs2Segment> {
        let s_idx = sample.0 as usize;
        let mut segments = Vec::new();

        for (step, step_results) in self.ibs2_sets.iter().enumerate() {
            if let Some(cluster) = &step_results[s_idx] {
                let start = self.step_starts[step];
                let end = if step + 1 < self.step_starts.len() {
                    self.step_starts[step + 1] - 1
                } else {
                    self.n_markers - 1
                };

                for &other in cluster.iter() {
                    if other != sample.0 {
                        segments.push(Ibs2Segment::new(SampleIdx::new(other), start, end));
                    }
                }
            }
        }

        segments
    }
}

struct SampClust {
    samples: Vec<u32>,
    is_homozygous: bool,
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