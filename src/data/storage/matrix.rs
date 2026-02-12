//! # Genotype Matrix
//!
//! The main data structure: a matrix of genotypes (markers x haplotypes).
//! Replaces `vcf/RefGT.java`, `vcf/BasicGT.java`, and related classes.
//!
//! ## Type State Pattern
//!
//! The matrix uses a generic `State` parameter to track phasing status at compile time.
//! `GenotypeMatrix<Unphased>` represents unphased data, while `GenotypeMatrix<Phased>`
//! represents phased data. This enables compile-time enforcement of pipeline correctness.

use std::marker::PhantomData;
use std::sync::Arc;

use crate::data::haplotype::{HapIdx, SampleIdx, Samples};
use crate::data::marker::{AnyMarkerSpace, Marker, MarkerIdx, Markers};
use crate::data::storage::GenotypeColumn;
use crate::data::storage::phase_state::{PhaseState, Phased, Unphased};
use bitvec::prelude::*;

#[derive(Clone, Debug)]
pub struct PlMatrix {
    n_samples: usize,
    marker_offsets: Vec<usize>,
    marker_strides: Vec<usize>,
    values: Vec<u16>,
    missing: BitVec<u64, Lsb0>,
}

impl PlMatrix {
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn n_markers(&self) -> usize {
        self.marker_strides.len()
    }

    #[inline]
    pub fn sample_pl(&self, marker: usize, sample_idx: usize) -> Option<&[u16]> {
        if marker >= self.marker_strides.len() || sample_idx >= self.n_samples {
            return None;
        }
        let stride = self.marker_strides[marker];
        if stride == 0 {
            return None;
        }
        let base = *self.marker_offsets.get(marker)?;
        let sample_offset = sample_idx.checked_mul(stride)?;
        let start = base.checked_add(sample_offset)?;
        let end = start.checked_add(stride)?;
        let slice = self.values.get(start..end)?;
        let missing = self.missing.get(start..end)?;
        if missing.all() { None } else { Some(slice) }
    }

    pub fn from_marker_blocks(
        n_samples: usize,
        marker_strides: Vec<usize>,
        mut marker_blocks: Vec<Vec<u16>>,
        mut marker_missing_blocks: Vec<Vec<u8>>,
    ) -> Self {
        debug_assert_eq!(marker_strides.len(), marker_blocks.len());
        debug_assert_eq!(marker_strides.len(), marker_missing_blocks.len());

        let mut marker_offsets: Vec<usize> = Vec::with_capacity(marker_strides.len() + 1);
        marker_offsets.push(0);
        let mut values: Vec<u16> = Vec::new();
        let mut missing: BitVec<u64, Lsb0> = BitVec::new();
        let mut running: usize = 0;
        for ((stride, mut block), missing_block) in marker_strides
            .iter()
            .copied()
            .zip(marker_blocks.iter_mut())
            .zip(marker_missing_blocks.iter_mut())
        {
            if stride == 0 {
                marker_offsets.push(running);
                continue;
            }
            let block_len = stride
                .checked_mul(n_samples)
                .expect("PL block length overflow");
            assert_eq!(
                block.len(),
                block_len,
                "PL block length mismatch: expected {} values, found {}",
                block_len,
                block.len()
            );
            assert_eq!(
                missing_block.len(),
                block_len,
                "PL missing-mask length mismatch: expected {} values, found {}",
                block_len,
                missing_block.len()
            );
            values.append(&mut block);
            missing.extend(missing_block.iter().map(|&v| v != 0));
            running = running
                .checked_add(block_len)
                .expect("PL matrix offset overflow");
            marker_offsets.push(running);
        }
        let out = Self {
            n_samples,
            marker_offsets,
            marker_strides,
            values,
            missing,
        };
        debug_assert_eq!(out.n_samples(), n_samples);
        debug_assert_eq!(out.n_markers(), out.marker_strides.len());
        out
    }
}

#[derive(Clone, Debug)]
struct FlatU8Matrix {
    n_rows: usize,
    n_cols: usize,
    data: Vec<u8>,
}

impl FlatU8Matrix {
    fn from_nested(rows: Vec<Vec<u8>>, n_rows: usize, n_cols: usize, field: &str) -> Self {
        assert_eq!(
            rows.len(),
            n_rows,
            "{} row count mismatch: expected {}, found {}",
            field,
            n_rows,
            rows.len()
        );
        let total = n_rows
            .checked_mul(n_cols)
            .expect("flat u8 matrix size overflow");
        let mut data = Vec::with_capacity(total);
        for (row_idx, row) in rows.into_iter().enumerate() {
            assert_eq!(
                row.len(),
                n_cols,
                "{} row {} length mismatch: expected {}, found {}",
                field,
                row_idx,
                n_cols,
                row.len()
            );
            data.extend(row);
        }
        Self {
            n_rows,
            n_cols,
            data,
        }
    }

    #[inline]
    fn idx(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.n_rows || col >= self.n_cols {
            return None;
        }
        row.checked_mul(self.n_cols)?.checked_add(col)
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> Option<u8> {
        let idx = self.idx(row, col)?;
        self.data.get(idx).copied()
    }

    fn to_nested_clone(&self) -> Vec<Vec<u8>> {
        let mut out = Vec::with_capacity(self.n_rows);
        for row in 0..self.n_rows {
            let start = row * self.n_cols;
            out.push(self.data[start..start + self.n_cols].to_vec());
        }
        out
    }
}

#[derive(Clone, Debug)]
pub struct BitMatrix {
    n_rows: usize,
    n_cols: usize,
    bits: BitVec<u64, Lsb0>,
}

impl BitMatrix {
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn from_nested(rows: Vec<Vec<u8>>, n_rows: usize, n_cols: usize, field: &str) -> Self {
        assert_eq!(
            rows.len(),
            n_rows,
            "{} row count mismatch: expected {}, found {}",
            field,
            n_rows,
            rows.len()
        );
        let total = n_rows
            .checked_mul(n_cols)
            .expect("bit matrix size overflow");
        let mut bits = BitVec::with_capacity(total);
        for (row_idx, row) in rows.into_iter().enumerate() {
            assert_eq!(
                row.len(),
                n_cols,
                "{} row {} length mismatch: expected {}, found {}",
                field,
                row_idx,
                n_cols,
                row.len()
            );
            bits.extend(row.into_iter().map(|v| v != 0));
        }
        Self {
            n_rows,
            n_cols,
            bits,
        }
    }

    #[inline]
    fn idx(&self, row: usize, col: usize) -> Option<usize> {
        if row >= self.n_rows || col >= self.n_cols {
            return None;
        }
        row.checked_mul(self.n_cols)?.checked_add(col)
    }

    #[inline]
    pub fn get(&self, row: usize, col: usize) -> Option<u8> {
        let idx = self.idx(row, col)?;
        self.bits.get(idx).map(|bit| *bit as u8)
    }

    #[inline]
    pub fn row_has_any_set(&self, row: usize) -> bool {
        if row >= self.n_rows {
            return false;
        }
        let start = row * self.n_cols;
        let end = start + self.n_cols;
        self.bits[start..end].any()
    }

    #[inline]
    pub fn row_all_set(&self, row: usize) -> bool {
        if row >= self.n_rows {
            return false;
        }
        let start = row * self.n_cols;
        let end = start + self.n_cols;
        self.bits[start..end].all()
    }
}

fn build_missing_genotype_mask(
    columns: &[GenotypeColumn],
    samples: &Samples,
    n_markers: usize,
    n_samples: usize,
) -> BitMatrix {
    let total = n_markers
        .checked_mul(n_samples)
        .expect("missing-genotype mask size overflow");
    let mut bits = BitVec::<u64, Lsb0>::with_capacity(total);
    for m in 0..n_markers {
        for s in 0..n_samples {
            let sample = SampleIdx::new(s as u32);
            let a1 = columns[m].get(sample.hap1());
            let mut is_missing = GenotypeColumn::is_missing_allele(a1);
            if !is_missing && samples.is_diploid(sample) {
                let a2 = columns[m].get(sample.hap2());
                is_missing = GenotypeColumn::is_missing_allele(a2);
            }
            bits.push(is_missing);
        }
    }
    BitMatrix {
        n_rows: n_markers,
        n_cols: n_samples,
        bits,
    }
}

/// The main genotype matrix structure.
///
/// Type parameter `State` encodes whether data is phased at compile time,
/// enabling the compiler to enforce correct pipeline usage.
#[derive(Debug)]
#[repr(C)]
pub struct GenotypeMatrix<State: PhaseState = Unphased, Space = AnyMarkerSpace> {
    /// Marker metadata
    markers: Markers<Space>,

    /// Genotype data (one column per marker)
    columns: Vec<GenotypeColumn>,

    /// Sample metadata
    samples: Arc<Samples>,

    /// Whether markers are in reverse order
    is_reversed: bool,

    /// Optional per-sample genotype confidence scores (from GL/PL/GQ-derived posterior confidence).
    /// Stored as u8 (0-u8::MAX) representing confidence 0.0-1.0.
    /// Layout: `confidence[marker][sample]`
    /// None if no confidence information available (assume full confidence).
    confidence: Option<FlatU8Matrix>,

    /// Precomputed missing-genotype mask used when confidence scores are absent.
    /// 1 means missing genotype for this marker/sample.
    missing_genotypes: Option<BitMatrix>,

    /// Optional per-sample phase confidence scores (0-u8::MAX => 0.0-1.0).
    /// Represents confidence in the current phased orientation at heterozygotes.
    /// Layout: `phase_confidence[marker][sample]`
    phase_confidence: Option<FlatU8Matrix>,

    /// Optional per-sample phasedness mask (1 = phased, 0 = unphased or missing).
    /// Layout: `phase_mask[marker][sample]`
    phase_mask: Option<BitMatrix>,

    likelihoods_pl: Option<Arc<PlMatrix>>,

    /// Phantom data to hold the State type parameter (zero-sized)
    phantom: PhantomData<State>,
}

impl<State: PhaseState, Space> Clone for GenotypeMatrix<State, Space> {
    fn clone(&self) -> Self {
        Self {
            markers: self.markers.clone(),
            columns: self.columns.clone(),
            samples: Arc::clone(&self.samples),
            is_reversed: self.is_reversed,
            confidence: self.confidence.clone(),
            missing_genotypes: self.missing_genotypes.clone(),
            phase_confidence: self.phase_confidence.clone(),
            phase_mask: self.phase_mask.clone(),
            likelihoods_pl: self.likelihoods_pl.clone(),
            phantom: PhantomData,
        }
    }
}

// ============================================================================
// Methods available for ALL phase states
// ============================================================================

impl<S: PhaseState, Space> GenotypeMatrix<S, Space> {
    /// Number of markers
    pub fn n_markers(&self) -> usize {
        self.markers.len()
    }

    /// Number of samples
    pub fn n_samples(&self) -> usize {
        self.samples.len()
    }

    /// Number of haplotypes
    pub fn n_haplotypes(&self) -> usize {
        self.samples.n_haps()
    }

    /// Get marker by index
    pub fn marker(&self, idx: MarkerIdx<Space>) -> &Marker {
        self.markers.marker(idx)
    }

    /// Get all markers
    pub fn markers(&self) -> &Markers<Space> {
        &self.markers
    }

    /// Get samples reference

    /// Get samples Arc (cloned)
    pub fn samples_arc(&self) -> Arc<Samples> {
        Arc::clone(&self.samples)
    }

    /// Get genotype column for a marker
    pub fn column(&self, idx: MarkerIdx<Space>) -> &GenotypeColumn {
        &self.columns[idx.as_usize()]
    }

    /// Get all genotype columns

    /// Get allele at (marker, haplotype)
    #[inline]
    pub fn allele(&self, marker: MarkerIdx<Space>, hap: HapIdx) -> u8 {
        self.columns[marker.as_usize()].get(hap)
    }

    /// Fill a batch of alleles for the provided haplotypes at a marker.
    #[inline]
    pub fn fill_batch(&self, marker: MarkerIdx<Space>, haps: &[HapIdx], out: &mut [u8]) {
        let col = &self.columns[marker.as_usize()];
        col.fill_batch(haps, out);
    }

    /// Total memory usage in bytes (approximate)
    pub fn size_bytes(&self) -> usize {
        let column_bytes: usize = self.columns.iter().map(|c| c.size_bytes()).sum();
        let confidence_bytes: usize = self
            .confidence
            .as_ref()
            .map(|c| c.data.capacity() * std::mem::size_of::<u8>())
            .unwrap_or(0);
        let phase_confidence_bytes: usize = self
            .phase_confidence
            .as_ref()
            .map(|c| c.data.capacity() * std::mem::size_of::<u8>())
            .unwrap_or(0);
        let phase_mask_bytes: usize = self
            .phase_mask
            .as_ref()
            .map(|c| c.bits.capacity() / 8)
            .unwrap_or(0);
        let missing_genotypes_bytes: usize = self
            .missing_genotypes
            .as_ref()
            .map(|c| c.bits.capacity() / 8)
            .unwrap_or(0);
        let pl_values_bytes = self
            .likelihoods_pl
            .as_ref()
            .map(|pl| pl.values.capacity() * std::mem::size_of::<u16>())
            .unwrap_or(0);
        let pl_missing_bytes = self
            .likelihoods_pl
            .as_ref()
            .map(|pl| pl.missing.capacity() / 8)
            .unwrap_or(0);
        let marker_offsets_bytes = self
            .likelihoods_pl
            .as_ref()
            .map(|pl| pl.marker_offsets.capacity() * std::mem::size_of::<usize>())
            .unwrap_or(0);
        let marker_strides_bytes = self
            .likelihoods_pl
            .as_ref()
            .map(|pl| pl.marker_strides.capacity() * std::mem::size_of::<usize>())
            .unwrap_or(0);
        column_bytes
            + confidence_bytes
            + phase_confidence_bytes
            + phase_mask_bytes
            + missing_genotypes_bytes
            + pl_values_bytes
            + pl_missing_bytes
            + marker_offsets_bytes
            + marker_strides_bytes
            + std::mem::size_of::<Self>()
    }

    /// Check if confidence scores are available

    /// Get confidence score for a sample at a marker (0-u8::MAX representing 0.0-1.0).
    /// Returns `u8::MAX` (full confidence) if confidence data is not available.
    #[inline]
    pub fn sample_confidence(&self, marker: MarkerIdx<Space>, sample_idx: usize) -> u8 {
        let marker_idx = marker.as_usize();
        assert!(
            marker_idx < self.columns.len(),
            "marker index {} out of bounds for {} markers",
            marker_idx,
            self.columns.len()
        );
        assert!(
            sample_idx < self.n_samples(),
            "sample index {} out of bounds for {} samples",
            sample_idx,
            self.n_samples()
        );
        if let Some(conf) = self
            .confidence
            .as_ref()
            .and_then(|c| c.get(marker_idx, sample_idx))
        {
            return conf;
        }
        if let Some(is_missing) = self
            .missing_genotypes
            .as_ref()
            .and_then(|m| m.get(marker_idx, sample_idx))
        {
            if is_missing != 0 {
                return 0;
            }
        }
        u8::MAX
    }

    /// Get confidence score as f32 (0.0-1.0)
    #[inline]
    pub fn sample_confidence_f32(&self, marker: MarkerIdx<Space>, sample_idx: usize) -> f32 {
        self.sample_confidence(marker, sample_idx) as f32 / u8::MAX as f32
    }

    /// Clone the confidence data (for transferring to a new matrix)
    pub fn confidence_clone(&self) -> Option<Vec<Vec<u8>>> {
        self.confidence.as_ref().map(FlatU8Matrix::to_nested_clone)
    }

    /// Get phase confidence score for a sample at a marker (0-u8::MAX).
    /// Returns `u8::MAX` (full confidence) if phase confidence is not available.
    #[inline]
    pub fn sample_phase_confidence(&self, marker: MarkerIdx<Space>, sample_idx: usize) -> u8 {
        let marker_idx = marker.as_usize();
        assert!(
            marker_idx < self.columns.len(),
            "marker index {} out of bounds for {} markers",
            marker_idx,
            self.columns.len()
        );
        assert!(
            sample_idx < self.n_samples(),
            "sample index {} out of bounds for {} samples",
            sample_idx,
            self.n_samples()
        );
        self.phase_confidence
            .as_ref()
            .and_then(|c| c.get(marker_idx, sample_idx))
            .unwrap_or(u8::MAX)
    }

    /// Get phase confidence score as f32 (0.0-1.0).
    #[inline]
    pub fn sample_phase_confidence_f32(&self, marker: MarkerIdx<Space>, sample_idx: usize) -> f32 {
        self.sample_phase_confidence(marker, sample_idx) as f32 / u8::MAX as f32
    }

    /// Clone the phase confidence data (for transferring to a new matrix)
    pub fn phase_confidence_clone(&self) -> Option<Vec<Vec<u8>>> {
        self.phase_confidence
            .as_ref()
            .map(FlatU8Matrix::to_nested_clone)
    }

    pub fn phase_mask(&self) -> Option<&BitMatrix> {
        self.phase_mask.as_ref()
    }

    pub fn with_phase_mask(mut self, phase_mask: Option<Vec<Vec<u8>>>) -> Self {
        if let Some(ref mask) = phase_mask {
            debug_assert_eq!(self.markers.len(), mask.len());
        }
        self.phase_mask = phase_mask.map(|m| {
            BitMatrix::from_nested(m, self.markers.len(), self.samples.len(), "phase_mask")
        });
        self
    }

    pub fn likelihoods_pl_arc(&self) -> Option<Arc<PlMatrix>> {
        self.likelihoods_pl.as_ref().map(Arc::clone)
    }

    #[inline]
    pub fn sample_pl(&self, marker: MarkerIdx<Space>, sample_idx: usize) -> Option<&[u16]> {
        self.likelihoods_pl
            .as_ref()
            .and_then(|pl| pl.sample_pl(marker.as_usize(), sample_idx))
    }
}

// ============================================================================
// Methods ONLY for Unphased matrices
// ============================================================================

impl<Space> GenotypeMatrix<Unphased, Space> {
    /// Create a new unphased genotype matrix
    pub fn new_unphased(
        markers: Markers<Space>,
        columns: Vec<GenotypeColumn>,
        samples: Arc<Samples>,
    ) -> Self {
        debug_assert_eq!(markers.len(), columns.len());
        let missing_genotypes = Some(build_missing_genotype_mask(
            &columns,
            &samples,
            markers.len(),
            samples.len(),
        ));
        Self {
            markers,
            columns,
            samples,
            is_reversed: false,
            confidence: None,
            missing_genotypes,
            phase_confidence: None,
            phase_mask: None,
            likelihoods_pl: None,
            phantom: PhantomData,
        }
    }

    /// Create new unphased matrix with confidence scores
    pub fn new_unphased_with_confidence(
        markers: Markers<Space>,
        columns: Vec<GenotypeColumn>,
        samples: Arc<Samples>,
        confidence: Vec<Vec<u8>>,
    ) -> Self {
        debug_assert_eq!(markers.len(), columns.len());
        let n_markers = markers.len();
        let n_samples = samples.len();
        let confidence = FlatU8Matrix::from_nested(confidence, n_markers, n_samples, "confidence");
        Self {
            markers,
            columns,
            samples,
            is_reversed: false,
            confidence: Some(confidence),
            missing_genotypes: None,
            phase_confidence: None,
            phase_mask: None,
            likelihoods_pl: None,
            phantom: PhantomData,
        }
    }

    pub fn new_unphased_with_confidence_and_likelihoods(
        markers: Markers<Space>,
        columns: Vec<GenotypeColumn>,
        samples: Arc<Samples>,
        confidence: Option<Vec<Vec<u8>>>,
        likelihoods_pl: Arc<PlMatrix>,
    ) -> Self {
        debug_assert_eq!(markers.len(), columns.len());
        let n_markers = markers.len();
        let n_samples = samples.len();
        let confidence = confidence
            .map(|conf| FlatU8Matrix::from_nested(conf, n_markers, n_samples, "confidence"));
        let missing_genotypes = if confidence.is_none() {
            Some(build_missing_genotype_mask(
                &columns, &samples, n_markers, n_samples,
            ))
        } else {
            None
        };
        Self {
            markers,
            columns,
            samples,
            is_reversed: false,
            confidence,
            missing_genotypes,
            phase_confidence: None,
            phase_mask: None,
            likelihoods_pl: Some(likelihoods_pl),
            phantom: PhantomData,
        }
    }

    /// Transform into a phased matrix.
    ///
    /// This is the primary way to create a `GenotypeMatrix<Phased>`.
    /// Consumes self to prevent accidental use of unphased data.
    pub fn into_phased(self) -> GenotypeMatrix<Phased, Space> {
        GenotypeMatrix {
            markers: self.markers,
            columns: self.columns,
            samples: self.samples,
            is_reversed: self.is_reversed,
            confidence: self.confidence,
            missing_genotypes: self.missing_genotypes,
            phase_confidence: self.phase_confidence,
            phase_mask: self.phase_mask,
            likelihoods_pl: self.likelihoods_pl,
            phantom: PhantomData,
        }
    }
}

// ============================================================================
// Methods ONLY for Phased matrices
// ============================================================================

impl<Space> GenotypeMatrix<Phased, Space> {
    /// Create a new phased genotype matrix
    pub fn new_phased(
        markers: Markers<Space>,
        columns: Vec<GenotypeColumn>,
        samples: Arc<Samples>,
    ) -> Self {
        debug_assert_eq!(markers.len(), columns.len());
        let missing_genotypes = Some(build_missing_genotype_mask(
            &columns,
            &samples,
            markers.len(),
            samples.len(),
        ));
        Self {
            markers,
            columns,
            samples,
            is_reversed: false,
            confidence: None,
            missing_genotypes,
            phase_confidence: None,
            phase_mask: None,
            likelihoods_pl: None,
            phantom: PhantomData,
        }
    }

    pub fn new_phased_with_confidence_and_likelihoods(
        markers: Markers<Space>,
        columns: Vec<GenotypeColumn>,
        samples: Arc<Samples>,
        confidence: Option<Vec<Vec<u8>>>,
        likelihoods_pl: Option<Arc<PlMatrix>>,
    ) -> Self {
        debug_assert_eq!(markers.len(), columns.len());
        let n_markers = markers.len();
        let n_samples = samples.len();
        let confidence = confidence
            .map(|conf| FlatU8Matrix::from_nested(conf, n_markers, n_samples, "confidence"));
        let missing_genotypes = if confidence.is_none() {
            Some(build_missing_genotype_mask(
                &columns, &samples, n_markers, n_samples,
            ))
        } else {
            None
        };

        if likelihoods_pl.is_none() {
            if let Some(conf) = confidence {
                return Self {
                    markers,
                    columns,
                    samples,
                    is_reversed: false,
                    confidence: Some(conf),
                    missing_genotypes: None,
                    phase_confidence: None,
                    phase_mask: None,
                    likelihoods_pl: None,
                    phantom: PhantomData,
                };
            }
            return Self::new_phased(markers, columns, samples);
        }

        Self {
            markers,
            columns,
            samples,
            is_reversed: false,
            confidence,
            missing_genotypes,
            phase_confidence: None,
            phase_mask: None,
            likelihoods_pl,
            phantom: PhantomData,
        }
    }

    /// Attach phase confidence data to a phased matrix.
    pub fn with_phase_confidence(mut self, phase_confidence: Option<Vec<Vec<u8>>>) -> Self {
        self.phase_confidence = phase_confidence.map(|conf| {
            FlatU8Matrix::from_nested(
                conf,
                self.markers.len(),
                self.samples.len(),
                "phase_confidence",
            )
        });
        self
    }

    /// Get a reference as unphased (zero-cost, same memory layout)
    pub fn as_unphased_ref(&self) -> &GenotypeMatrix<Unphased, Space> {
        // SAFETY: GenotypeMatrix<Phased> and GenotypeMatrix<Unphased> have identical
        // memory layouts (PhantomData is zero-sized), differing only in the type parameter
        debug_assert_eq!(
            std::mem::size_of::<GenotypeMatrix<Phased, Space>>(),
            std::mem::size_of::<GenotypeMatrix<Unphased, Space>>()
        );
        debug_assert_eq!(
            std::mem::align_of::<GenotypeMatrix<Phased, Space>>(),
            std::mem::align_of::<GenotypeMatrix<Unphased, Space>>()
        );
        unsafe {
            &*(self as *const GenotypeMatrix<Phased, Space>
                as *const GenotypeMatrix<Unphased, Space>)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ChromIdx;
    use crate::data::marker::{Allele, Nucleotide};

    fn make_test_matrix_phased() -> GenotypeMatrix<Phased> {
        let samples = Arc::new(Samples::from_ids(vec!["S1".to_string(), "S2".to_string()]));
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");

        let m1 = Marker::new(
            ChromIdx::new(0),
            100,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        );
        let m2 = Marker::new(
            ChromIdx::new(0),
            200,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        );

        markers.push(m1);
        markers.push(m2);

        let col1 = GenotypeColumn::from_alleles(&[0, 1, 0, 1], 2);
        let col2 = GenotypeColumn::from_alleles(&[1, 1, 0, 0], 2);

        GenotypeMatrix::new_phased(markers, vec![col1, col2], samples)
    }

    fn make_test_matrix_unphased() -> GenotypeMatrix<Unphased> {
        let samples = Arc::new(Samples::from_ids(vec!["S1".to_string(), "S2".to_string()]));
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");

        let m1 = Marker::new(
            ChromIdx::new(0),
            100,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        );
        let m2 = Marker::new(
            ChromIdx::new(0),
            200,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        );

        markers.push(m1);
        markers.push(m2);

        let col1 = GenotypeColumn::from_alleles(&[0, 1, 0, 1], 2);
        let col2 = GenotypeColumn::from_alleles(&[1, 1, 0, 0], 2);

        GenotypeMatrix::new_unphased(markers, vec![col1, col2], samples)
    }

    #[test]
    fn test_matrix_access() {
        let matrix = make_test_matrix_phased();

        assert_eq!(matrix.n_markers(), 2);
        assert_eq!(matrix.n_samples(), 2);
        assert_eq!(matrix.n_haplotypes(), 4);

        assert_eq!(matrix.allele(MarkerIdx::new(0), HapIdx::new(0)), 0);
        assert_eq!(matrix.allele(MarkerIdx::new(0), HapIdx::new(1)), 1);
        assert_eq!(matrix.allele(MarkerIdx::new(1), HapIdx::new(0)), 1);
    }

    #[test]
    fn test_phase_transition() {
        let unphased = make_test_matrix_unphased();

        // Transform to phased
        let phased = unphased.into_phased();
        assert_eq!(phased.n_markers(), 2);
    }

    #[test]
    fn test_confidence_scores() {
        let samples = Arc::new(Samples::from_ids(vec!["S1".to_string(), "S2".to_string()]));
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");

        let m1 = Marker::new(
            ChromIdx::new(0),
            100,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        );
        let m2 = Marker::new(
            ChromIdx::new(0),
            200,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        );

        markers.push(m1);
        markers.push(m2);

        let col1 = GenotypeColumn::from_alleles(&[0, 1, 0, 1], 2);
        let col2 = GenotypeColumn::from_alleles(&[1, 1, 0, 0], 2);

        // Create confidence scores: marker 0 has full confidence, marker 1 has 50% for sample 0
        let confidence = vec![
            vec![u8::MAX, u8::MAX], // marker 0: full confidence for both samples
            vec![128, u8::MAX],     // marker 1: 50% for sample 0, full for sample 1
        ];

        let matrix = GenotypeMatrix::new_unphased_with_confidence(
            markers,
            vec![col1, col2],
            samples,
            confidence,
        );

        assert!(matrix.confidence_clone().is_some());
        assert_eq!(matrix.sample_confidence(MarkerIdx::new(0), 0), u8::MAX);
        assert_eq!(matrix.sample_confidence(MarkerIdx::new(1), 0), 128);
        assert_eq!(matrix.sample_confidence(MarkerIdx::new(1), 1), u8::MAX);

        // Check f32 conversion
        assert!((matrix.sample_confidence_f32(MarkerIdx::new(0), 0) - 1.0).abs() < 0.01);
        assert!((matrix.sample_confidence_f32(MarkerIdx::new(1), 0) - 0.502).abs() < 0.01);

        // Verify confidence survives phase transition
        let phased = matrix.into_phased();
        assert!(phased.confidence_clone().is_some());
        assert_eq!(phased.sample_confidence(MarkerIdx::new(1), 0), 128);
    }

    #[test]
    fn test_no_confidence_defaults_to_full() {
        let matrix = make_test_matrix_unphased();

        // Without confidence data, has_confidence returns false
        assert!(matrix.confidence_clone().is_none());

        // But sample_confidence defaults to u8::MAX (full confidence)
        assert_eq!(matrix.sample_confidence(MarkerIdx::new(0), 0), u8::MAX);
        assert!((matrix.sample_confidence_f32(MarkerIdx::new(0), 0) - 1.0).abs() < 0.01);
    }
}
