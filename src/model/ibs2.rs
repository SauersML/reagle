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
}

/// Collection of IBS2 segments for all target samples
pub struct Ibs2 {
    /// IBS2 segments for each sample: sample_segs[sample_idx] = Vec<Ibs2Segment>
    sample_segs: Vec<Vec<Ibs2Segment>>,
}

// We pack 32 markers per u64 block with 2 bits per genotype:
// - marker i occupies bits [2*i, 2*i+1]
// - values are 0, 1, or 2 (sum of the two hap alleles for biallelic sites)
// This layout lets us compare 32 markers at once using XOR + POPCNT.
const IBS2_BLOCK_MARKERS: usize = 32;
// Mask that keeps the low bit of each 2-bit lane: binary 0101... across 64 bits.
// After `diff |= diff >> 1`, any differing 2-bit genotype lane has its LSB set.
// ANDing with this mask leaves one bit per marker so POPCNT counts mismatches.
const IBS2_BLOCK_MASK: u64 = 0x5555_5555_5555_5555;

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
        let packed = PackedGenotypes::new(gt);

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
                    &packed,
                    sample,
                )
            })
            .collect();

        Self { sample_segs }
    }

    /// Create an empty IBS2 structure with no segments.
    /// Useful for testing or when IBS2 detection is not needed.
    #[cfg(test)]
    pub fn empty(n_samples: usize) -> Self {
        Self {
            sample_segs: vec![Vec::new(); n_samples],
        }
    }

    #[cfg(test)]
    pub fn from_sample_segments(sample_segs: Vec<Vec<Ibs2Segment>>) -> Self {
        Self { sample_segs }
    }

    /// Build IBS2 segments for a single sample.
    ///
    /// Pipeline:
    /// 1. Get initial segments from partition-based detection (Ibs2Sets)
    /// 2. Sort by other sample ID for efficient merging
    /// 3. Merge adjacent segments that are close in genetic distance
    /// 4. Extend segments through homozygous regions (where IBS2 is ambiguous)
    /// 5. Merge again after extension (may have created new adjacencies)
    /// 6. Filter out segments below minimum length threshold
    fn build_sample_segments(
        gt: &GenotypeMatrix,
        gen_maps: &GeneticMaps,
        chrom: ChromIdx,
        marker_positions: &[u32],
        ibs2_sets: &Ibs2Sets,
        packed: &PackedGenotypes,
        sample: SampleIdx,
    ) -> Vec<Ibs2Segment> {
        let mut segments = ibs2_sets.segments_for_sample(sample);

        // Sort by other sample
        segments.sort_by_key(|s| s.other_sample.0);

        // Merge adjacent segments
        segments = Self::merge_segments(segments, gen_maps, chrom, marker_positions);

        // Extend segments through homozygous regions
        segments = Self::extend_segments(gt, sample, segments, packed);

        // Merge again after extension
        segments = Self::merge_segments(segments, gen_maps, chrom, marker_positions);

        // Filter by minimum length
        segments = Self::filter_by_length(segments, gen_maps, chrom, marker_positions);

        segments
    }

    /// Merge adjacent IBS2 segments for the same sample pair if close in genetic distance.
    ///
    /// Short gaps between segments are common due to:
    /// - Genotyping errors or missing data
    /// - Recombination hotspots within true IBD segments
    /// - Marker filtering removing informative markers
    ///
    /// Segments with gap < MAX_IBS2_GAP_CM cM are merged into a single segment.
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

    /// Extend segments through regions where both samples are homozygous.
    ///
    /// At homozygous sites, IBS2 status is ambiguous (both orderings match).
    /// We optimistically extend segments through these regions since:
    /// 1. True IBD segments often span homozygous sites
    /// 2. Conservative extension rarely causes false positives
    /// 3. Short gaps from homozygosity don't break true IBD
    fn extend_segments(
        gt: &GenotypeMatrix,
        sample: SampleIdx,
        segments: Vec<Ibs2Segment>,
        packed: &PackedGenotypes,
    ) -> Vec<Ibs2Segment> {
        let n_markers = gt.n_markers();

        segments
            .into_iter()
            .map(|seg| {
                let other = seg.other_sample;
                let mut start = seg.start;
                let mut end = seg.incl_end;

                // Extend left through compatible markers (IBS2 or homozygous)
                start = Self::extend_left(gt, packed, sample, other, start);

                // Extend right through compatible markers
                end = Self::extend_right(gt, packed, sample, other, end, n_markers);

                Ibs2Segment::new(other, start, end)
            })
            .collect()
    }

    fn extend_left(
        gt: &GenotypeMatrix,
        packed: &PackedGenotypes,
        sample: SampleIdx,
        other: SampleIdx,
        mut start: usize,
    ) -> usize {
        if start == 0 {
            return start;
        }
        let s_idx = sample.0 as usize;
        let o_idx = other.0 as usize;

        while start > 0 && (start % IBS2_BLOCK_MARKERS) != 0 {
            let m = start - 1;
            if !Self::is_ibs2_at_fast(gt, packed, m, sample, other) {
                return start;
            }
            start -= 1;
        }

        while start >= IBS2_BLOCK_MARKERS {
            let block = start / IBS2_BLOCK_MARKERS - 1;
            if !packed.fast_block_all(block) {
                break;
            }
            if packed.block_has_missing(s_idx, o_idx, block) {
                break;
            }
            if packed.block_mismatch_count(s_idx, o_idx, block) != 0 {
                break;
            }
            start -= IBS2_BLOCK_MARKERS;
        }

        while start > 0 {
            let m = start - 1;
            if !Self::is_ibs2_at_fast(gt, packed, m, sample, other) {
                break;
            }
            start -= 1;
        }

        start
    }

    fn extend_right(
        gt: &GenotypeMatrix,
        packed: &PackedGenotypes,
        sample: SampleIdx,
        other: SampleIdx,
        mut end: usize,
        n_markers: usize,
    ) -> usize {
        if end + 1 >= n_markers {
            return end;
        }
        let s_idx = sample.0 as usize;
        let o_idx = other.0 as usize;

        while end + 1 < n_markers && ((end + 1) % IBS2_BLOCK_MARKERS) != 0 {
            let m = end + 1;
            if !Self::is_ibs2_at_fast(gt, packed, m, sample, other) {
                return end;
            }
            end += 1;
        }

        while end + IBS2_BLOCK_MARKERS < n_markers {
            let block = (end + 1) / IBS2_BLOCK_MARKERS;
            if !packed.fast_block_all(block) {
                break;
            }
            if packed.block_has_missing(s_idx, o_idx, block) {
                break;
            }
            if packed.block_mismatch_count(s_idx, o_idx, block) != 0 {
                break;
            }
            end += IBS2_BLOCK_MARKERS;
        }

        while end + 1 < n_markers {
            let m = end + 1;
            if !Self::is_ibs2_at_fast(gt, packed, m, sample, other) {
                break;
            }
            end += 1;
        }

        end
    }

    #[inline]
    fn is_ibs2_at_fast(
        gt: &GenotypeMatrix,
        packed: &PackedGenotypes,
        marker: usize,
        s1: SampleIdx,
        s2: SampleIdx,
    ) -> bool {
        if !packed.fast_marker(marker) {
            return Self::is_ibs2_at(gt, marker, s1, s2);
        }
        let s1_idx = s1.0 as usize;
        let s2_idx = s2.0 as usize;
        if packed.is_missing(s1_idx, marker) || packed.is_missing(s2_idx, marker) {
            return Self::is_ibs2_at(gt, marker, s1, s2);
        }
        packed.genotype_code(s1_idx, marker) == packed.genotype_code(s2_idx, marker)
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

    /// Check if two samples are IBS2 at a marker position.
    ///
    /// IBS2 means the samples share BOTH haplotypes (identical diploid genotype).
    /// This requires either:
    /// - Same phase: (a1, a2) matches (b1, b2)
    /// - Opposite phase: (a1, a2) matches (b2, b1)
    ///
    /// Missing data (u8::MAX) is treated as "compatible" - it doesn't break IBS2.
    fn is_ibs2_at(gt: &GenotypeMatrix, marker: usize, s1: SampleIdx, s2: SampleIdx) -> bool {
        let m_idx = MarkerIdx::new(marker as u32);

        let a1 = gt.allele(m_idx, s1.hap1());
        let a2 = gt.allele(m_idx, s1.hap2());
        let b1 = gt.allele(m_idx, s2.hap1());
        let b2 = gt.allele(m_idx, s2.hap2());

        // A marker is informative if at least one comparable allele pair is observed
        // under either phase ordering.
        let informative = (a1 != crate::data::storage::AlleleCode::MISSING.raw() && b1 != crate::data::storage::AlleleCode::MISSING.raw())
            || (a2 != crate::data::storage::AlleleCode::MISSING.raw() && b2 != crate::data::storage::AlleleCode::MISSING.raw())
            || (a1 != crate::data::storage::AlleleCode::MISSING.raw() && b2 != crate::data::storage::AlleleCode::MISSING.raw())
            || (a2 != crate::data::storage::AlleleCode::MISSING.raw() && b1 != crate::data::storage::AlleleCode::MISSING.raw());
        if !informative {
            return false;
        }

        // Check both phase orderings: (a1,a2)=(b1,b2) OR (a1,a2)=(b2,b1)
        Self::are_phase_consistent(a1, a2, b1, b2) || Self::are_phase_consistent(a1, a2, b2, b1)
    }

    /// Check if two phased genotypes are consistent (allowing missing data).
    ///
    /// Returns true if the alleles match or either is missing (u8::MAX).
    /// This is a helper for is_ibs2_at to check one phase ordering.
    fn are_phase_consistent(a1: u8, a2: u8, b1: u8, b2: u8) -> bool {
        (a1 == crate::data::storage::AlleleCode::MISSING.raw() || b1 == crate::data::storage::AlleleCode::MISSING.raw() || a1 == b1) && (a2 == crate::data::storage::AlleleCode::MISSING.raw() || b2 == crate::data::storage::AlleleCode::MISSING.raw() || a2 == b2)
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

struct PackedGenotypes {
    blocks: usize,
    genotypes: Vec<u64>,
    missing: Vec<u64>,
    fast_marker: Vec<bool>,
    fast_block_all: Vec<bool>,
}

impl PackedGenotypes {
    fn new(gt: &GenotypeMatrix) -> Self {
        let n_markers = gt.n_markers();
        let n_samples = gt.n_samples();
        let blocks = (n_markers + IBS2_BLOCK_MARKERS - 1) / IBS2_BLOCK_MARKERS;
        let mut genotypes = vec![0u64; n_samples * blocks];
        let mut missing = vec![0u64; n_samples * blocks];
        let mut fast_marker = vec![false; n_markers];

        for m in 0..n_markers {
            let m_idx = MarkerIdx::new(m as u32);
            let is_biallelic = gt.marker(m_idx).alt_alleles.len() == 1;
            if !is_biallelic {
                continue;
            }

            let block = m / IBS2_BLOCK_MARKERS;
            let bit = (m % IBS2_BLOCK_MARKERS) as u64;
            let shift = bit * 2;
            let miss_bit = 1u64 << bit;
            let mut marker_fast = true;

            for s in 0..n_samples {
                let sample = SampleIdx::new(s as u32);
                let a1 = gt.allele(m_idx, sample.hap1());
                let a2 = gt.allele(m_idx, sample.hap2());
                let idx = s * blocks + block;

                if a1 == crate::data::storage::AlleleCode::MISSING.raw() || a2 == crate::data::storage::AlleleCode::MISSING.raw() {
                    missing[idx] |= miss_bit;
                    continue;
                }
                if a1 > 1 || a2 > 1 {
                    marker_fast = false;
                    continue;
                }
                let code = (a1 + a2) as u64;
                genotypes[idx] |= code << shift;
            }

            if marker_fast {
                fast_marker[m] = true;
            }
        }

        let mut fast_block_all = vec![false; blocks];
        for block in 0..blocks {
            let start = block * IBS2_BLOCK_MARKERS;
            let end = (start + IBS2_BLOCK_MARKERS).min(n_markers);
            let mut all_fast = true;
            for m in start..end {
                if !fast_marker[m] {
                    all_fast = false;
                    break;
                }
            }
            fast_block_all[block] = all_fast;
        }

        Self {
            blocks,
            genotypes,
            missing,
            fast_marker,
            fast_block_all,
        }
    }

    #[inline]
    fn fast_marker(&self, marker: usize) -> bool {
        self.fast_marker.get(marker).copied().unwrap_or(false)
    }

    #[inline]
    fn fast_block_all(&self, block: usize) -> bool {
        self.fast_block_all.get(block).copied().unwrap_or(false)
    }

    #[inline]
    fn block_has_missing(&self, s1: usize, s2: usize, block: usize) -> bool {
        let idx1 = s1 * self.blocks + block;
        let idx2 = s2 * self.blocks + block;
        (self.missing[idx1] | self.missing[idx2]) != 0
    }

    #[inline]
    fn block_mismatch_count(&self, s1: usize, s2: usize, block: usize) -> u32 {
        let idx1 = s1 * self.blocks + block;
        let idx2 = s2 * self.blocks + block;
        // XOR exposes per-lane differences across 32 markers (2 bits per marker).
        // Example for one marker lane:
        //   00 ^ 00 -> 00 (match)
        //   01 ^ 01 -> 00 (match)
        //   10 ^ 10 -> 00 (match)
        //   01 ^ 10 -> 11 (mismatch)
        //   00 ^ 01 -> 01 (mismatch)
        //
        // `diff |= diff >> 1` collapses any difference in a 2-bit lane to its LSB:
        //   00 -> 00, 01 -> 01, 10 -> 11, 11 -> 11
        // AND with 0x5555... keeps only the LSB of each 2-bit lane, yielding
        // one bit per marker indicating mismatch. POPCNT then counts mismatches.
        let mut diff = self.genotypes[idx1] ^ self.genotypes[idx2];
        diff |= diff >> 1;
        let diff = diff & IBS2_BLOCK_MASK;
        diff.count_ones()
    }

    #[inline]
    fn is_missing(&self, sample: usize, marker: usize) -> bool {
        let block = marker / IBS2_BLOCK_MARKERS;
        let bit = (marker % IBS2_BLOCK_MARKERS) as u64;
        let idx = sample * self.blocks + block;
        (self.missing[idx] & (1u64 << bit)) != 0
    }

    #[inline]
    fn genotype_code(&self, sample: usize, marker: usize) -> u8 {
        let block = marker / IBS2_BLOCK_MARKERS;
        let shift = (marker % IBS2_BLOCK_MARKERS) as u64 * 2;
        let idx = sample * self.blocks + block;
        ((self.genotypes[idx] >> shift) & 0b11) as u8
    }
}

/// Identifies informative markers and partitions them into steps for IBS2 detection.
///
/// Not all markers are useful for IBS2 detection:
/// - Rare variants (MAF < 0.1) have low discrimination power
/// - High-missing markers reduce accuracy
/// - Very close markers are redundant
///
/// This struct filters to informative markers and groups them into "steps"
/// of approximately MIN_MARKER_CNT markers each, spaced by MIN_INTERMARKER_CM.
struct Ibs2Markers {
    /// Whether each marker is used for IBS2 detection
    use_marker: Vec<bool>,
    /// Starting marker index for each step
    step_starts: Vec<usize>,
}

impl Ibs2Markers {
    /// Maximum fraction of missing genotypes to include a marker
    const MAX_MISS_FREQ: f32 = 0.1;
    /// Minimum minor allele frequency for discrimination power
    const MIN_MINOR_FREQ: f32 = 0.1;
    /// Target number of markers per step
    const MIN_MARKER_CNT: usize = 50;
    /// Minimum genetic distance between selected markers (cM)
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
                    if gt.allele(m_idx, HapIdx::new(h as u32)) == crate::data::storage::AlleleCode::MISSING.raw() {
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
            let mut min_cm_pos = gen_maps
                .gen_pos(chrom, gt.marker(MarkerIdx::new(last_start as u32)).pos)
                + Self::MIN_INTERMARKER_CM;

            while next_start < n_markers && mkr_cnt < Self::MIN_MARKER_CNT {
                if use_marker[next_start] {
                    let cur_cm_pos =
                        gen_maps.gen_pos(chrom, gt.marker(MarkerIdx::new(next_start as u32)).pos);
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

        Self {
            use_marker,
            step_starts,
        }
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

/// Stores clusters of samples that are IBS2 within each step.
///
/// Uses recursive partitioning by genotype to identify groups of samples
/// that share both haplotypes across all markers in a step. This is the
/// core data structure for efficient IBS2 segment detection.
///
/// ## Algorithm
///
/// For each step:
/// 1. Start with all samples in one partition
/// 2. For each marker in the step, split partitions by genotype
/// 3. Samples remaining in the same partition after all markers are IBS2
/// 4. Discard homozygous-only partitions (not informative for phasing)
struct Ibs2Sets {
    /// IBS2 clusters per step: `ibs2_sets[step][sample_idx] = Some(cluster)` if sample
    /// is in an IBS2 group, where cluster is Arc<Vec<u32>> of sample indices
    ibs2_sets: Vec<Vec<Option<Arc<Vec<u32>>>>>,
    /// Starting marker index for each step (from Ibs2Markers)
    step_starts: Vec<usize>,
    /// Total number of markers
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
                    if gt.allele(m_idx, sample.hap1()) == crate::data::storage::AlleleCode::MISSING.raw()
                        || gt.allele(m_idx, sample.hap2()) == crate::data::storage::AlleleCode::MISSING.raw()
                    {
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
                if partition.is_empty() {
                    break;
                }
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
                            let mut merged: Vec<u32> =
                                (**step_results[s_idx].as_ref().unwrap()).clone();
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

    /// Partition a cluster of samples by genotype at a single marker.
    ///
    /// This is the core operation for IBS2 detection: samples with different
    /// genotypes cannot be IBS2, so we split the parent cluster by genotype.
    ///
    /// # Genotype Indexing
    /// Genotypes are indexed by unordered allele pair: gt_idx = a2*(a2+1)/2 + a1
    /// where a1 <= a2. This handles both homozygotes and heterozygotes uniformly.
    ///
    /// # Missing Data Handling
    /// Samples with missing data at this marker are added to ALL partitions,
    /// since they could potentially match any genotype.
    ///
    /// # Homozygosity Tracking
    /// Tracks whether each partition contains only homozygous genotypes,
    /// since homozygous-only partitions aren't informative for phasing.
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

            if a1 == crate::data::storage::AlleleCode::MISSING.raw() || a2 == crate::data::storage::AlleleCode::MISSING.raw() {
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

        gt_to_list
            .into_iter()
            .enumerate()
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

/// A cluster of samples during recursive partitioning.
///
/// Tracks both the sample indices and whether all genotypes seen so far are
/// homozygous. Homozygous-only clusters aren't useful for phasing since they
/// don't constrain phase relationships.
struct SampClust {
    /// Sample indices in this cluster
    samples: Vec<u32>,
    /// True if all genotypes seen in this cluster are homozygous
    is_homozygous: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ChromIdx;
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Allele, Marker, Markers, Nucleotide};
    use crate::data::storage::GenotypeColumn;
    use std::sync::Arc;

    fn single_marker_gt(a1: u8, a2: u8, b1: u8, b2: u8) -> GenotypeMatrix {
        let samples = Arc::new(Samples::from_ids(vec!["S1".to_string(), "S2".to_string()]));
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");
        markers.push(Marker::new(
            ChromIdx::new(0),
            100,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        ));
        let column = GenotypeColumn::from_alleles(&[a1, a2, b1, b2], 2);
        GenotypeMatrix::new_unphased(markers, vec![column], samples)
    }

    #[test]
    fn test_ibs2_segment() {
        let seg = Ibs2Segment::new(SampleIdx::new(1), 10, 20);
        assert!(seg.contains(10));
        assert!(seg.contains(15));
        assert!(seg.contains(20));
        assert!(!seg.contains(9));
        assert!(!seg.contains(21));
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
        assert!(Ibs2::are_phase_consistent(u8::MAX, 1, 0, 1));
        assert!(Ibs2::are_phase_consistent(0, u8::MAX, 0, 1));
    }

    #[test]
    fn test_is_ibs2_at_swapped_informative_with_missing() {
        // Observed comparison exists only under swapped ordering: (a1,b2) = (0,0).
        // Previous informative gate incorrectly rejected this case.
        let gt = single_marker_gt(0, u8::MAX, u8::MAX, 0);
        assert!(Ibs2::is_ibs2_at(
            &gt,
            0,
            SampleIdx::new(0),
            SampleIdx::new(1)
        ));
    }
}
