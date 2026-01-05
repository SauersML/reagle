//! # Sliding Window Infrastructure
//!
//! Handles chunking of genomic data into overlapping windows for processing.
//! Replaces `vcf/SlidingWindow.java`, `vcf/TargSlidingWindow.java`, etc.

use std::sync::Arc;

use crate::data::genetic_map::GeneticMap;
use crate::data::haplotype::Samples;
use crate::data::marker::{MarkerIdx, Markers};
use crate::data::storage::GenotypeMatrix;
use crate::data::ChromIdx;

/// A processing window containing a subset of markers
#[derive(Clone, Debug)]
pub struct Window {
    /// The genotype data for this window
    pub target_gt: GenotypeMatrix,

    /// Optional reference panel data
    pub ref_gt: Option<GenotypeMatrix>,

    /// Genetic map for this region
    pub gen_map: GeneticMap,

    /// Window indices
    pub indices: WindowIndices,

    /// Window number (0-indexed)
    pub window_num: usize,
}

/// Indices defining window boundaries
#[derive(Clone, Debug, Default)]
pub struct WindowIndices {
    /// Start marker index (inclusive) in the full dataset
    pub start: usize,
    /// End marker index (exclusive) in the full dataset
    pub end: usize,
    /// Previous splice point (where output starts)
    pub prev_splice: usize,
    /// Next splice point (where output ends)
    pub next_splice: usize,
    /// Overlap start (where next window's data begins)
    pub overlap_start: usize,
    /// Number of target markers
    pub n_targ_markers: usize,
    /// Number of reference markers (if imputing)
    pub n_ref_markers: usize,
}

impl WindowIndices {
    /// Number of markers in the window
    pub fn n_markers(&self) -> usize {
        self.end - self.start
    }

    /// Number of markers to output
    pub fn n_output_markers(&self) -> usize {
        self.next_splice - self.prev_splice
    }
}

impl Window {
    /// Create a new window
    pub fn new(
        target_gt: GenotypeMatrix,
        ref_gt: Option<GenotypeMatrix>,
        gen_map: GeneticMap,
        indices: WindowIndices,
        window_num: usize,
    ) -> Self {
        Self {
            target_gt,
            ref_gt,
            gen_map,
            indices,
            window_num,
        }
    }

    /// Number of target markers
    pub fn n_markers(&self) -> usize {
        self.target_gt.n_markers()
    }

    /// Number of target samples
    pub fn n_samples(&self) -> usize {
        self.target_gt.n_samples()
    }

    /// Number of target haplotypes
    pub fn n_haplotypes(&self) -> usize {
        self.target_gt.n_haplotypes()
    }

    /// Check if this window has a reference panel
    pub fn has_ref(&self) -> bool {
        self.ref_gt.is_some()
    }

    /// Get genetic distance between two marker indices (within window)
    pub fn gen_dist(&self, m1: usize, m2: usize) -> f64 {
        let pos1 = self.target_gt.marker(MarkerIdx::new(m1 as u32)).pos;
        let pos2 = self.target_gt.marker(MarkerIdx::new(m2 as u32)).pos;
        self.gen_map.gen_dist(pos1, pos2)
    }
}

/// Builder for creating windows from a genotype matrix
pub struct WindowBuilder {
    /// Window size in cM
    window_cm: f32,
    /// Overlap size in cM
    overlap_cm: f32,
    /// Maximum markers per window
    max_markers: usize,
    /// Buffer size in cM (extra overlap for HMM edge effects)
    buffer_cm: f32,
}

impl WindowBuilder {
    /// Create a new window builder with default settings
    pub fn new() -> Self {
        Self {
            window_cm: 40.0,
            overlap_cm: 2.0,
            max_markers: 4_000_000,
            buffer_cm: 1.0,
        }
    }

    /// Set window size in cM
    pub fn window_cm(mut self, cm: f32) -> Self {
        self.window_cm = cm;
        self
    }

    /// Set overlap size in cM
    pub fn overlap_cm(mut self, cm: f32) -> Self {
        self.overlap_cm = cm;
        self
    }

    /// Set maximum markers per window
    pub fn max_markers(mut self, n: usize) -> Self {
        self.max_markers = n;
        self
    }

    /// Set buffer size in cM
    pub fn buffer_cm(mut self, cm: f32) -> Self {
        self.buffer_cm = cm;
        self
    }

    /// Create a sliding window iterator
    pub fn build<'a>(
        &self,
        target: &'a GenotypeMatrix,
        reference: Option<&'a GenotypeMatrix>,
        gen_map: &'a GeneticMap,
    ) -> SlidingWindowIterator<'a> {
        SlidingWindowIterator::new(
            target,
            reference,
            gen_map,
            self.window_cm,
            self.overlap_cm,
            self.max_markers,
        )
    }
}

impl Default for WindowBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Iterator over sliding windows
pub struct SlidingWindowIterator<'a> {
    target: &'a GenotypeMatrix,
    reference: Option<&'a GenotypeMatrix>,
    gen_map: &'a GeneticMap,
    window_cm: f32,
    overlap_cm: f32,
    max_markers: usize,
    current_start: usize,
    window_num: usize,
    done: bool,
}

impl<'a> SlidingWindowIterator<'a> {
    /// Create a new sliding window iterator
    pub fn new(
        target: &'a GenotypeMatrix,
        reference: Option<&'a GenotypeMatrix>,
        gen_map: &'a GeneticMap,
        window_cm: f32,
        overlap_cm: f32,
        max_markers: usize,
    ) -> Self {
        Self {
            target,
            reference,
            gen_map,
            window_cm,
            overlap_cm,
            max_markers,
            current_start: 0,
            window_num: 0,
            done: target.n_markers() == 0,
        }
    }

    /// Find the end index for a window starting at `start`
    fn find_window_end(&self, start: usize) -> usize {
        let n_markers = self.target.n_markers();
        if start >= n_markers {
            return n_markers;
        }

        let start_pos = self.target.marker(MarkerIdx::new(start as u32)).pos;
        let start_cm = self.gen_map.gen_pos(start_pos);
        let target_cm = start_cm + self.window_cm as f64;

        // Find end by genetic distance
        let mut end = start + 1;
        while end < n_markers {
            let pos = self.target.marker(MarkerIdx::new(end as u32)).pos;
            let cm = self.gen_map.gen_pos(pos);
            if cm >= target_cm || end - start >= self.max_markers {
                break;
            }
            end += 1;
        }

        end.min(n_markers)
    }

    /// Find the overlap start for the next window
    fn find_overlap_start(&self, end: usize) -> usize {
        if end >= self.target.n_markers() {
            return end;
        }

        let end_pos = self.target.marker(MarkerIdx::new((end - 1) as u32)).pos;
        let end_cm = self.gen_map.gen_pos(end_pos);
        let target_cm = end_cm - self.overlap_cm as f64;

        // Find overlap start
        let mut overlap = end - 1;
        while overlap > 0 {
            let pos = self.target.marker(MarkerIdx::new((overlap - 1) as u32)).pos;
            let cm = self.gen_map.gen_pos(pos);
            if cm < target_cm {
                break;
            }
            overlap -= 1;
        }

        overlap
    }
}

impl<'a> Iterator for SlidingWindowIterator<'a> {
    type Item = Window;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let n_markers = self.target.n_markers();
        let start = self.current_start;
        let end = self.find_window_end(start);

        if end <= start {
            self.done = true;
            return None;
        }

        // Calculate splice points
        let prev_splice = if self.window_num == 0 { 0 } else { start };
        let is_last = end >= n_markers;
        let overlap_start = if is_last { end } else { self.find_overlap_start(end) };
        let next_splice = if is_last { end } else { overlap_start };

        let indices = WindowIndices {
            start,
            end,
            prev_splice,
            next_splice,
            overlap_start,
            n_targ_markers: end - start,
            n_ref_markers: 0, // TODO: handle reference
        };

        // Extract window data
        let target_gt = self.target.restrict(start, end);
        let ref_gt = self.reference.map(|r| r.restrict(start, end));

        // Create genetic map for window
        let chrom = self.target.marker(MarkerIdx::new(start as u32)).chrom;
        let gen_map = GeneticMap::empty(chrom); // TODO: slice from full map

        let window = Window::new(target_gt, ref_gt, gen_map, indices, self.window_num);

        // Advance to next window
        if is_last {
            self.done = true;
        } else {
            self.current_start = overlap_start;
            self.window_num += 1;
        }

        Some(window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::marker::{Allele, Marker};
    use crate::data::storage::GenotypeColumn;

    fn make_test_matrix(n_markers: usize) -> GenotypeMatrix {
        let samples = Arc::new(Samples::from_ids(vec!["S1".to_string(), "S2".to_string()]));
        let mut markers = Markers::new();
        markers.add_chrom("chr1");

        let mut columns = Vec::new();
        for i in 0..n_markers {
            let m = Marker::new(
                ChromIdx::new(0),
                (i * 1000 + 100) as u32,
                None,
                Allele::Base(0),
                vec![Allele::Base(1)],
            );
            markers.push(m);
            columns.push(GenotypeColumn::from_alleles(&[0, 1, 0, 1], 2));
        }

        GenotypeMatrix::new(markers, columns, samples, true)
    }

    #[test]
    fn test_window_builder() {
        let matrix = make_test_matrix(100);
        let gen_map = GeneticMap::empty(ChromIdx::new(0));

        let windows: Vec<_> = WindowBuilder::new()
            .window_cm(10.0)
            .overlap_cm(1.0)
            .build(&matrix, None, &gen_map)
            .collect();

        assert!(!windows.is_empty());
        assert_eq!(windows[0].window_num, 0);
    }

    #[test]
    fn test_single_window() {
        let matrix = make_test_matrix(10);
        let gen_map = GeneticMap::empty(ChromIdx::new(0));

        let windows: Vec<_> = WindowBuilder::new()
            .window_cm(100.0) // Large enough for all markers
            .build(&matrix, None, &gen_map)
            .collect();

        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].n_markers(), 10);
    }
}