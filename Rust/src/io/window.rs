//! # Sliding Window Infrastructure
//!
//! Handles chunking of genomic data into overlapping windows for processing.
//! Replaces `vcf/SlidingWindow.java`, `vcf/TargSlidingWindow.java`, etc.

use crate::data::genetic_map::GeneticMap;
use crate::data::marker::MarkerIdx;
use crate::data::storage::GenotypeMatrix;

/// A processing window containing a subset of markers
#[derive(Clone, Debug)]
pub struct Window {
    /// The genotype data for this window
    pub target_gt: GenotypeMatrix,

    /// Window number (0-indexed)
    pub window_num: usize,
}

impl Window {
    /// Create a new window
    pub fn new(
        target_gt: GenotypeMatrix,
        window_num: usize,
    ) -> Self {
        Self {
            target_gt,
            window_num,
        }
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
}

impl WindowBuilder {
    /// Create a new window builder with default settings
    pub fn new() -> Self {
        Self {
            window_cm: 40.0,
            overlap_cm: 2.0,
            max_markers: 4_000_000,
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

    /// Create a sliding window iterator
    pub fn build<'a>(
        &self,
        target: &'a GenotypeMatrix,
        gen_map: &'a GeneticMap,
    ) -> SlidingWindowIterator<'a> {
        SlidingWindowIterator::new(
            target,
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
        gen_map: &'a GeneticMap,
        window_cm: f32,
        overlap_cm: f32,
        max_markers: usize,
    ) -> Self {
        Self {
            target,
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
        let is_last = end >= n_markers;
        let overlap_start = if is_last {
            end
        } else {
            self.find_overlap_start(end)
        };

        // Extract window data
        let target_gt = self.target.restrict(start, end);

        let window = Window::new(target_gt, self.window_num);

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
    use crate::data::ChromIdx;
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Allele, Marker, Markers};
    use crate::data::storage::GenotypeColumn;
    use std::sync::Arc;

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
            .build(&matrix, &gen_map)
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
            .build(&matrix, &gen_map)
            .collect();

        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0].target_gt.n_markers(), 10);
    }
}
