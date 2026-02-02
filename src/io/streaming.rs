//! # Streaming VCF Window Processing
//!
//! Implements memory-efficient streaming of VCF data through sliding windows.
//! This matches Java `vcf/RefTargSlidingWindow.java` and related classes.
//!
//! Instead of loading the entire VCF into memory, this module:
//! 1. Reads markers incrementally
//! 2. Maintains only the current window + overlap buffer in memory
//! 3. Processes each window and writes output before discarding

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;

use flate2::read::GzDecoder;
use noodles::bgzf::io as bgzf_io;
use noodles::vcf::Header;
use tracing::info_span;

use crate::data::ChromIdx;
use crate::data::genetic_map::GeneticMaps;
use crate::data::haplotype::Samples;
use crate::data::marker::{Allele, Marker, Markers};
use crate::data::storage::GenotypeColumn;
use crate::data::storage::matrix::GenotypeMatrix;
use crate::data::storage::matrix::PlMatrix;
use crate::error::{ReagleError, Result};

/// Configuration for streaming window processing
#[derive(Clone, Debug)]
pub struct StreamingConfig {
    /// Window size in cM
    pub window_cm: f32,
    /// Overlap size in cM
    pub overlap_cm: f32,
    /// Buffer size in cM (extra overlap for HMM edge effects)
    pub buffer_cm: f32,
    /// Maximum markers per window
    pub max_markers: usize,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            window_cm: 40.0,
            overlap_cm: 2.0,
            buffer_cm: 1.0,
            max_markers: 100_000,
        }
    }
}

/// Posterior state probabilities for soft-information handoff
#[derive(Clone, Debug)]
pub struct StateProbs {
    /// State probabilities for each haplotype at each Stage 1 marker
    /// Layout: [hap][marker_idx][state]
    pub data: Vec<Vec<Vec<f32>>>,
    /// Indices of these markers (relative to the window start)
    pub marker_indices: Vec<usize>,
    /// Number of states
    pub n_states: usize,
}

impl StateProbs {}

/// Global haplotype identifier (stable across windows).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct GlobalHapId(pub u32);

/// Haplotype-indexed state probabilities for soft-information handoff between windows.
///
/// Uses sorted dense arrays instead of HashMap for O(log K) lookup with good cache locality.
/// This is critical for HMM performance since prior lookup happens for every state at window start.
///
/// Design: Store (hap_id, prob) pairs sorted by hap_id for binary search.
/// Only significant probabilities (>0.001) are stored to save memory.
#[derive(Clone, Debug)]
pub struct HaplotypePriors {
    /// Sorted haplotype IDs (for binary search)
    hap_ids: Vec<GlobalHapId>,
    /// Corresponding probabilities (same order as hap_ids)
    probs: Vec<f32>,
}

impl HaplotypePriors {
    /// Create new priors with invariant enforcement (validation, sorting, normalization)
    pub fn new(hap_ids: Vec<GlobalHapId>, probs: Vec<f32>) -> Self {
        // Length validation
        assert_eq!(
            hap_ids.len(),
            probs.len(),
            "HaplotypePriors: hap_ids and probs must have the same length"
        );

        if hap_ids.is_empty() {
            return Self {
                hap_ids: Vec::new(),
                probs: Vec::new(),
            };
        }

        // Validate probabilities
        for &p in &probs {
            assert!(
                p.is_finite() && p >= 0.0,
                "HaplotypePriors: all probabilities must be finite and non-negative"
            );
        }

        // Sort by hap_id and check for duplicates
        let mut pairs: Vec<_> = hap_ids.into_iter().zip(probs).collect();
        pairs.sort_unstable_by_key(|(id, _)| *id);

        // Check for duplicate hap_ids
        for i in 1..pairs.len() {
            assert_ne!(
                pairs[i - 1].0,
                pairs[i].0,
                "HaplotypePriors: duplicate haplotype ID {:?}",
                pairs[i].0
            );
        }

        let (ids, mut ps): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();

        // Normalize to sum to 1.0
        let sum: f32 = ps.iter().sum();
        if sum <= 0.0 {
            return Self::empty();
        }
        for p in &mut ps {
            *p /= sum;
        }

        Self {
            hap_ids: ids,
            probs: ps,
        }
    }

    /// Create empty priors (uniform distribution)
    pub fn empty() -> Self {
        Self {
            hap_ids: Vec::new(),
            probs: Vec::new(),
        }
    }

    /// Get reference to sorted haplotype IDs
    pub fn ids(&self) -> &[GlobalHapId] {
        &self.hap_ids
    }

    /// Get reference to probabilities aligned with IDs
    pub fn probs(&self) -> &[f32] {
        &self.probs
    }

    /// Lookup the probability mass for a specific haplotype ID.
    pub fn prob_of(&self, hap_id: GlobalHapId) -> Option<f32> {
        match self.hap_ids.binary_search(&hap_id) {
            Ok(idx) => self.probs.get(idx).copied(),
            Err(_) => None,
        }
    }

    /// Check if we have any priors
    pub fn is_empty(&self) -> bool {
        self.hap_ids.is_empty()
    }
}

impl Default for HaplotypePriors {
    fn default() -> Self {
        Self::empty()
    }
}

/// Phased genotypes from overlap region to seed next window
///
/// This carries the phased alleles from the overlap region of the previous window
/// to constrain the next window's phasing for phase continuity at window boundaries.
/// Based on Java's FixedPhaseData and SplicedGT classes.
#[derive(Clone, Debug)]
pub struct PhasedOverlap {
    /// Number of markers in the overlap
    pub n_markers: usize,
    /// Phased alleles for each haplotype in the overlap region
    /// Layout: alleles[hap * n_markers + marker]
    pub alleles: Vec<u8>,
    /// Number of haplotypes
    pub n_haps: usize,
    /// Posterior state probabilities for Stage 1 markers in the overlap
    /// Used for soft-information handoff to prevent stair-step artifacts
    pub state_probs: Option<StateProbs>,
    /// Per-target-haplotype priors indexed by reference haplotype ID
    /// This enables proper soft-information handoff when HMM states differ between windows
    pub hap_priors: Option<Vec<HaplotypePriors>>,

    /// Global marker index used to export haplotype priors.
    /// This lets the next window verify it is projecting at the same physical marker.
    pub prior_stage1_global_marker: Option<usize>,

    /// Genetic position (cM) for the prior marker used to export haplotype priors.
    pub prior_stage1_gen_pos: Option<f64>,
}

impl PhasedOverlap {
    /// Create a new PhasedOverlap from phased genotype data
    ///
    /// # Arguments
    /// * `n_markers` - Number of markers in the overlap region
    /// * `n_haps` - Number of haplotypes
    /// * `alleles` - Phased alleles, layout: alleles[hap * n_markers + marker]
    pub fn new(n_markers: usize, n_haps: usize, alleles: Vec<u8>) -> Self {
        debug_assert_eq!(alleles.len(), n_markers * n_haps);
        Self {
            n_markers,
            alleles,
            n_haps,
            state_probs: None,
            hap_priors: None,
            prior_stage1_global_marker: None,
            prior_stage1_gen_pos: None,
        }
    }

    /// Set state probabilities (legacy format)
    pub fn set_state_probs(&mut self, state_probs: StateProbs) {
        self.state_probs = Some(state_probs);
    }

    /// Set haplotype-indexed priors for soft-information handoff
    pub fn set_hap_priors(&mut self, priors: Vec<HaplotypePriors>) {
        self.hap_priors = Some(priors);
    }

    /// Get haplotype priors if available
    pub fn hap_priors(&self) -> Option<&[HaplotypePriors]> {
        self.hap_priors.as_deref()
    }

    /// Set the global marker index at which haplotype priors were exported.
    pub fn set_prior_stage1_global_marker(&mut self, marker: usize) {
        self.prior_stage1_global_marker = Some(marker);
    }

    /// Get the global marker index used for haplotype prior export.
    pub fn prior_stage1_global_marker(&self) -> Option<usize> {
        self.prior_stage1_global_marker
    }

    /// Set genetic position (cM) for the prior marker used to export haplotype priors.
    pub fn set_prior_stage1_gen_pos(&mut self, gen_pos: f64) {
        self.prior_stage1_gen_pos = Some(gen_pos);
    }

    /// Get genetic position (cM) for the prior marker used to export haplotype priors.
    pub fn prior_stage1_gen_pos(&self) -> Option<f64> {
        self.prior_stage1_gen_pos
    }

    /// Get the allele for a specific haplotype at a specific marker
    #[inline]
    pub fn allele(&self, marker: usize, hap: usize) -> u8 {
        self.alleles[hap * self.n_markers + marker]
    }
}

/// A window of genotype data ready for processing
#[derive(Clone, Debug)]
pub struct StreamWindow {
    /// Genotype data for this window
    pub genotypes: GenotypeMatrix,
    /// Start marker index in full chromosome
    pub global_start: usize,
    /// End marker index in full chromosome (exclusive)
    pub global_end: usize,
    /// Index where output should start (relative to window)
    pub output_start: usize,
    /// Index where output should end (relative to window, exclusive)
    pub output_end: usize,
    /// Whether this is the first window
    pub is_first: bool,
    /// Phased genotypes from overlap region of previous window
    /// These should be used to constrain/seed the current window's phasing
    pub phased_overlap: Option<PhasedOverlap>,
}

impl StreamWindow {
    /// Returns true if this is the last window (no more data follows)
    pub fn is_last(&self) -> bool {
        self.output_end >= self.genotypes.n_markers()
    }
}

/// Buffered marker data for streaming
struct BufferedMarker {
    marker: Marker,
    column: GenotypeColumn,
    gen_pos: f64,
    confidences: Option<Vec<u8>>,
    likelihoods_pl: Option<Vec<Vec<u16>>>,
    phase_mask: Vec<u8>,
}

/// Streaming VCF reader that yields windows
pub struct StreamingVcfReader {
    /// Reader for the VCF file
    reader: Box<dyn BufRead + Send>,
    /// Sample information
    samples: Arc<Samples>,
    /// Streaming configuration
    config: StreamingConfig,
    /// Genetic maps for position conversion
    gen_maps: GeneticMaps,
    /// Buffer of markers not yet processed
    buffer: VecDeque<BufferedMarker>,
    /// Markers metadata (for chromosome tracking)
    markers_meta: Markers,
    /// Current chromosome index
    current_chrom: Option<ChromIdx>,
    /// Current window number
    window_num: usize,
    /// Global marker index
    global_marker_idx: usize,
    /// Whether we've reached EOF
    eof: bool,
    /// Current line buffer
    line_buf: Vec<u8>,
    /// Whether all genotypes seen so far were phased
    all_phased: bool,
    /// Per-sample ploidy (true=diploid, false=haploid)
    sample_ploidy: Option<Vec<bool>>,
    /// Whether any confidence scores were seen
    has_any_confidence: bool,
}

impl StreamingVcfReader {
    /// Open a VCF file for streaming
    pub fn open(path: &Path, gen_maps: GeneticMaps, config: StreamingConfig) -> Result<Self> {
        fn detect_bgzf(file: &mut File) -> Result<bool> {
            use std::io::{Read, Seek, SeekFrom};

            let mut header = [0u8; 12];
            let n = file.read(&mut header)?;
            if n < 10 {
                file.seek(SeekFrom::Start(0))?;
                return Ok(false);
            }
            if header[0] != 0x1f || header[1] != 0x8b || header[2] != 0x08 {
                file.seek(SeekFrom::Start(0))?;
                return Ok(false);
            }
            let flg = header[3];
            if flg & 0x04 == 0 {
                file.seek(SeekFrom::Start(0))?;
                return Ok(false);
            }
            if n < 12 {
                file.seek(SeekFrom::Start(0))?;
                return Ok(false);
            }
            let xlen = u16::from_le_bytes([header[10], header[11]]) as usize;
            if xlen < 4 {
                file.seek(SeekFrom::Start(0))?;
                return Ok(false);
            }
            let mut extra = vec![0u8; xlen];
            file.read_exact(&mut extra)?;
            file.seek(SeekFrom::Start(0))?;
            let mut i = 0usize;
            while i + 4 <= extra.len() {
                let si1 = extra[i];
                let si2 = extra[i + 1];
                let slen = u16::from_le_bytes([extra[i + 2], extra[i + 3]]) as usize;
                if si1 == b'B' && si2 == b'C' && slen == 2 {
                    return Ok(true);
                }
                i = i.saturating_add(4 + slen);
            }
            Ok(false)
        }

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        match ext {
            "bgz" | "bgzf" => {
                let mut file = File::open(path)?;
                if !detect_bgzf(&mut file)? {
                    return Err(anyhow::anyhow!("Expected BGZF file for extension .{}", ext).into());
                }
                let reader: Box<dyn BufRead + Send> =
                    Box::new(BufReader::with_capacity(128 * 1024, bgzf_io::Reader::new(file)));
                Self::from_reader(reader, gen_maps, config)
            }
            "gz" => {
                let mut file = File::open(path)?;
                let reader: Box<dyn BufRead + Send> = if detect_bgzf(&mut file)? {
                    Box::new(BufReader::with_capacity(128 * 1024, bgzf_io::Reader::new(file)))
                } else {
                    Box::new(BufReader::with_capacity(128 * 1024, GzDecoder::new(file)))
                };
                Self::from_reader(reader, gen_maps, config)
            }
            _ => {
                let file = File::open(path)?;
                let reader: Box<dyn BufRead + Send> =
                    Box::new(BufReader::with_capacity(128 * 1024, file));
                Self::from_reader(reader, gen_maps, config)
            }
        }
    }

    /// Create from a reader
    pub fn from_reader(
        mut reader: Box<dyn BufRead + Send>,
        gen_maps: GeneticMaps,
        config: StreamingConfig,
    ) -> Result<Self> {
        info_span!("streaming_vcf_from_reader").in_scope(|| {
            // Read header
            let mut header_str = String::new();
            let mut line = String::new();

            loop {
                line.clear();
                let bytes_read = reader.read_line(&mut line)?;
                if bytes_read == 0 {
                    break;
                }
                if line.starts_with('#') {
                    header_str.push_str(&line);
                    if line.starts_with("#CHROM") {
                        break;
                    }
                } else {
                    break;
                }
            }

            let header: Header = header_str
                .parse()
                .map_err(|e| ReagleError::vcf(format!("{}", e)))?;

            // Parse sample names from header.
            let sample_names: Vec<String> = header
                .sample_names()
                .iter()
                .map(|s| s.to_string())
                .collect();

            let samples = Arc::new(Samples::from_ids(sample_names));
            let header_samples = samples.len();
            let header_lines = header_str.lines().count();

            let mut reader = Self {
                reader,
                samples,
                config,
                gen_maps,
                buffer: VecDeque::new(),
                markers_meta: Markers::<crate::data::AnyMarkerSpace>::new(),
                current_chrom: None,
                window_num: 0,
                global_marker_idx: 0,
                eof: false,
                line_buf: Vec::new(),
                all_phased: true,
                sample_ploidy: None,
                has_any_confidence: false,
            };

            if let Err(e) = reader.prefetch_first_marker() {
                return Err(ReagleError::vcf(format!(
                    "{} (header_lines={}, header_samples={})",
                    e, header_lines, header_samples
                )));
            }

            Ok(reader)
        })
    }

    fn prefetch_first_marker(&mut self) -> Result<()> {
        if !self.buffer.is_empty() {
            return Ok(());
        }
        if let Some(bm) = self.read_next_marker()? {
            self.buffer.push_back(bm);
            return Ok(());
        }
        Err(ReagleError::vcf(
            "No variant records found while profiling; input VCF may be empty or malformed.",
        ))
    }

    /// Get samples Arc
    pub fn samples_arc(&self) -> Arc<Samples> {
        Arc::clone(&self.samples)
    }

    /// Read the next window of data
    ///
    /// Returns None when all data has been processed
    pub fn next_window(&mut self) -> Result<Option<StreamWindow>> {
        info_span!("streaming_next_window").in_scope(|| {
            if self.eof && self.buffer.is_empty() {
                return Ok(None);
            }

            // Fill buffer until we have a complete window
            self.fill_buffer_to_window()?;

            if self.buffer.is_empty() {
                return Ok(None);
            }

            // Determine window boundaries
            let window_start_gen = self.buffer.front().map(|m| m.gen_pos).unwrap_or(0.0);
            let target_end_gen = window_start_gen + self.config.window_cm as f64;
            let full_window_gen = target_end_gen + self.config.overlap_cm as f64;

            // Find end of full window (output + overlap)
            let window_end = self
                .buffer
                .iter()
                .position(|m| m.gen_pos >= full_window_gen)
                .unwrap_or(self.buffer.len())
                .min(self.config.max_markers);

            let is_last = self.eof && window_end >= self.buffer.len();

            // Determine splice points
            let output_start = 0;
            let output_end = if is_last {
                window_end
            } else {
                // Splice at the first marker past the main window
                self.buffer
                    .iter()
                    .take(window_end)
                    .position(|m| m.gen_pos >= target_end_gen)
                    .unwrap_or(window_end)
            };

            // Build GenotypeMatrix for this window
            let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
            let mut columns = Vec::with_capacity(window_end);
            let mut confidences: Vec<Vec<u8>> = Vec::new();
            let mut phase_masks: Vec<Vec<u8>> = Vec::new();
            let n_samples = self.samples.len();
            let has_any_likelihoods = self
                .buffer
                .iter()
                .take(window_end)
                .any(|bm| bm.likelihoods_pl.is_some());

            let mut marker_strides: Vec<u16> = Vec::new();
            let mut marker_blocks: Vec<Vec<u16>> = Vec::new();

            for i in 0..window_end {
                let bm = &self.buffer[i];
                let chrom_name = self
                    .markers_meta
                    .chrom_name(bm.marker.chrom)
                    .unwrap_or("UNKNOWN");
                let window_chrom_idx = markers.add_chrom(chrom_name);
                let mut marker = bm.marker.clone();
                marker.chrom = window_chrom_idx;
                markers.push(marker);
                columns.push(bm.column.clone());
                if self.has_any_confidence {
                    if let Some(conf) = &bm.confidences {
                        confidences.push(conf.clone());
                    } else {
                        confidences.push(vec![255; self.samples.len()]);
                    }
                }
                phase_masks.push(bm.phase_mask.clone());

                if has_any_likelihoods {
                    if let Some(pl_by_sample) = bm.likelihoods_pl.clone() {
                        let stride = pl_by_sample
                            .get(0)
                            .map(|v| v.len())
                            .unwrap_or(0)
                            .min(u16::MAX as usize) as u16;
                        if stride == 0 {
                            marker_strides.push(0);
                            marker_blocks.push(Vec::new());
                        } else {
                            let stride_usize = stride as usize;
                            let mut block: Vec<u16> = vec![u16::MAX; stride_usize * n_samples];
                            for (s, pls) in pl_by_sample.into_iter().enumerate().take(n_samples) {
                                if pls.len() != stride_usize {
                                    continue;
                                }
                                let start = s * stride_usize;
                                block[start..start + stride_usize].copy_from_slice(&pls);
                            }
                            marker_strides.push(stride);
                            marker_blocks.push(block);
                        }
                    } else {
                        marker_strides.push(0);
                        marker_blocks.push(Vec::new());
                    }
                }
            }

            if let Some(ref ploidy) = self.sample_ploidy {
                let sample_ids: Vec<String> = self
                    .samples
                    .ids()
                    .iter()
                    .map(|s: &std::sync::Arc<str>| s.as_ref().to_string())
                    .collect();
                self.samples = Arc::new(Samples::from_ids_with_ploidy(sample_ids, ploidy.clone()));
            }

            let confidence_opt = if self.has_any_confidence {
                Some(confidences)
            } else {
                None
            };
            let genotypes = if has_any_likelihoods {
                let pl = Arc::new(PlMatrix::from_marker_blocks(
                    n_samples,
                    marker_strides,
                    marker_blocks,
                ));
                GenotypeMatrix::new_unphased_with_confidence_and_likelihoods(
                    markers,
                    columns,
                    Arc::clone(&self.samples),
                    confidence_opt,
                    pl,
                )
            } else if let Some(conf) = confidence_opt {
                GenotypeMatrix::new_unphased_with_confidence(
                    markers,
                    columns,
                    Arc::clone(&self.samples),
                    conf,
                )
            } else {
                GenotypeMatrix::new_unphased(markers, columns, Arc::clone(&self.samples))
            };
            let genotypes = genotypes.with_phase_mask(Some(phase_masks));

            // Peek ahead to find next window start position (if available)
            // window_end is the index of the first marker NOT in the current output+overlap set.
            // But wait, window_end was calculated based on `full_window_gen`.
            // The "next window" in terms of processing logic starts at `output_end` of THIS window?
            // No, `output_end` is where we stop WRITING.
            // The next window effectively picks up where this one left off in terms of coverage.
            // To calculate the rate for the *last output marker*, we need the distance to the *next available marker*.
            // That marker is strictly at index `window_end` in the buffer (if it exists).
            // Actually, `window_end` is the end of the OVERLAP. The rate at the very end of overlap doesn't matter much.
            // But `ReferenceMap` needs rates for all markers in the Reference Window.
            // The Reference Window includes overlap.
            // So we need the rate for the last marker in the overlap region.
            // That rate connects to the marker *after* the overlap.
            // So looking at `self.buffer[window_end]` is correct.
            let window = StreamWindow {
                genotypes,
                global_start: self.global_marker_idx,
                global_end: self.global_marker_idx + window_end,
                output_start,
                output_end,
                is_first: self.window_num == 0,
                phased_overlap: None, // Caller will set this from previous window's phased output
            };

            // Remove processed markers from buffer (keep overlap)
            let keep_from = output_end;
            for _ in 0..keep_from {
                self.buffer.pop_front();
            }

            self.global_marker_idx += keep_from;
            self.window_num += 1;

            Ok(Some(window))
        })
    }

    /// Load a window for a specific genomic region (start_pos..end_pos).
    pub fn load_window_for_region(
        &mut self,
        candidates: &[String],
        start_pos: u32,
        end_pos: u32,
    ) -> Result<Option<StreamWindow>> {
        // Reset if chromosome changed
        let current_name = self
            .current_chrom
            .and_then(|idx| self.markers_meta.chrom_name(idx).map(|s| s.to_string()));
        let switched = current_name
            .as_ref()
            .map(|cur: &String| !candidates.iter().any(|c| c.as_str() == cur.as_str()))
            .unwrap_or(true);
        if switched {
            self.buffer.clear();
            self.current_chrom = None;
        }

        while !self.eof {
            let need_more = self
                .buffer
                .back()
                .map(|m| m.marker.pos < end_pos)
                .unwrap_or(true);
            if !need_more {
                break;
            }
            if let Some(bm) = self.read_next_marker()? {
                self.buffer.push_back(bm);
            } else {
                break;
            }
        }

        if self.buffer.is_empty() {
            return Ok(None);
        }

        let mut indices = Vec::new();
        for (i, bm) in self.buffer.iter().enumerate() {
            if bm.marker.pos >= start_pos && bm.marker.pos <= end_pos {
                indices.push(i);
            }
        }
        if indices.is_empty() {
            // Drop markers before start_pos to keep buffer bounded
            while self
                .buffer
                .front()
                .map(|m| m.marker.pos < start_pos)
                .unwrap_or(false)
            {
                self.buffer.pop_front();
                self.global_marker_idx += 1;
            }
            return Ok(None);
        }

        let first_idx = indices[0];
        let last_idx = *indices.last().unwrap();
        let n_markers = indices.len();

        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        let mut columns = Vec::with_capacity(n_markers);
        let mut confidences: Vec<Vec<u8>> = Vec::new();
        let mut phase_masks: Vec<Vec<u8>> = Vec::new();
        let n_samples = self.samples.len();
        let has_any_likelihoods = indices.iter().any(|&i| {
            self.buffer
                .get(i)
                .is_some_and(|bm| bm.likelihoods_pl.is_some())
        });

        let mut marker_strides: Vec<u16> = Vec::new();
        let mut marker_blocks: Vec<Vec<u16>> = Vec::new();

        for &i in &indices {
            let bm = &self.buffer[i];
            let chrom_name = self
                .markers_meta
                .chrom_name(bm.marker.chrom)
                .unwrap_or("UNKNOWN");
            let window_chrom_idx = markers.add_chrom(chrom_name);
            let mut marker = bm.marker.clone();
            marker.chrom = window_chrom_idx;
            markers.push(marker);
            columns.push(bm.column.clone());
            if self.has_any_confidence {
                if let Some(conf) = &bm.confidences {
                    confidences.push(conf.clone());
                } else {
                    confidences.push(vec![255; self.samples.len()]);
                }
            }
            phase_masks.push(bm.phase_mask.clone());

            if has_any_likelihoods {
                if let Some(pl_by_sample) = bm.likelihoods_pl.clone() {
                    let stride = pl_by_sample
                        .get(0)
                        .map(|v| v.len())
                        .unwrap_or(0)
                        .min(u16::MAX as usize) as u16;
                    if stride == 0 {
                        marker_strides.push(0);
                        marker_blocks.push(Vec::new());
                    } else {
                        let stride_usize = stride as usize;
                        let mut block: Vec<u16> = vec![u16::MAX; stride_usize * n_samples];
                        for (s, pls) in pl_by_sample.into_iter().enumerate().take(n_samples) {
                            if pls.len() != stride_usize {
                                continue;
                            }
                            let start = s * stride_usize;
                            block[start..start + stride_usize].copy_from_slice(&pls);
                        }
                        marker_strides.push(stride);
                        marker_blocks.push(block);
                    }
                } else {
                    marker_strides.push(0);
                    marker_blocks.push(Vec::new());
                }
            }
        }

        let confidence_opt = if self.has_any_confidence {
            Some(confidences)
        } else {
            None
        };
        let genotypes = if has_any_likelihoods {
            let pl = Arc::new(PlMatrix::from_marker_blocks(
                n_samples,
                marker_strides,
                marker_blocks,
            ));
            GenotypeMatrix::new_unphased_with_confidence_and_likelihoods(
                markers,
                columns,
                Arc::clone(&self.samples),
                confidence_opt,
                pl,
            )
        } else if let Some(conf) = confidence_opt {
            GenotypeMatrix::new_unphased_with_confidence(
                markers,
                columns,
                Arc::clone(&self.samples),
                conf,
            )
        } else {
            GenotypeMatrix::new_unphased(markers, columns, Arc::clone(&self.samples))
        };
        let genotypes = genotypes.with_phase_mask(Some(phase_masks));

        let window = StreamWindow {
            genotypes,
            global_start: self.global_marker_idx + first_idx,
            global_end: self.global_marker_idx + last_idx + 1,
            output_start: 0,
            output_end: n_markers,
            is_first: self.window_num == 0,
            phased_overlap: None,
        };

        while self
            .buffer
            .front()
            .map(|m| m.marker.pos < start_pos)
            .unwrap_or(false)
        {
            self.buffer.pop_front();
            self.global_marker_idx += 1;
        }
        self.window_num += 1;

        Ok(Some(window))
    }

    /// Fill buffer until we have enough data for a window
    fn fill_buffer_to_window(&mut self) -> Result<()> {
        if self.eof {
            return Ok(());
        }

        let target_cm = self.config.window_cm + self.config.overlap_cm + self.config.buffer_cm;
        let start_gen = self.buffer.front().map(|m| m.gen_pos).unwrap_or(0.0);
        let target_gen = start_gen + target_cm as f64;

        while !self.eof {
            // Check if we have enough data
            if let Some(last) = self.buffer.back() {
                if last.gen_pos >= target_gen || self.buffer.len() >= self.config.max_markers {
                    break;
                }
            }

            // Read next marker
            if let Some(bm) = self.read_next_marker()? {
                self.buffer.push_back(bm);
            }
        }

        Ok(())
    }

    /// Read the next marker from the VCF
    fn read_next_marker(&mut self) -> Result<Option<BufferedMarker>> {
        loop {
            self.line_buf.clear();
            let bytes_read = self.reader.read_until(b'\n', &mut self.line_buf)?;
            if bytes_read == 0 {
                self.eof = true;
                return Ok(None);
            }

            let line_buf = std::mem::take(&mut self.line_buf);
            let line = trim_line_bytes(&line_buf);
            if line.is_empty() || line[0] == b'#' {
                self.line_buf = line_buf;
                continue;
            }

            let parsed = self.parse_vcf_line(line).map(Some);
            self.line_buf = line_buf;
            return parsed;
        }
    }

    /// Parse a single VCF line
    fn parse_vcf_line(&mut self, line: &[u8]) -> Result<BufferedMarker> {
        let mut fields = FieldIter::new(line);
        let line_idx = self.global_marker_idx + self.buffer.len();
        let mut next_field = || {
            fields.next().ok_or_else(|| {
                ReagleError::parse(line_idx, "Expected at least 10 fields, got fewer")
            })
        };

        // Parse CHROM
        let chrom_name = std::str::from_utf8(next_field()?)
            .map_err(|_| ReagleError::parse(line_idx, "Invalid CHROM field"))?;
        let chrom_idx = self.markers_meta.add_chrom(chrom_name);

        // Update current chromosome tracking
        if self.current_chrom != Some(chrom_idx) {
            // New chromosome - could flush buffer here for multi-chrom support
            self.current_chrom = Some(chrom_idx);
        }

        // Parse POS
        let pos: u32 = parse_u32_bytes(next_field()?)
            .ok_or_else(|| ReagleError::parse(line_idx, "Invalid POS field"))?;

        // Parse ID
        let id_field = next_field()?;
        let id = if id_field == b"." {
            None
        } else {
            let id_str = std::str::from_utf8(id_field)
                .map_err(|_| ReagleError::parse(line_idx, "Invalid ID field"))?;
            Some(id_str.into())
        };

        // Parse REF
        let ref_allele_str = std::str::from_utf8(next_field()?)
            .map_err(|_| ReagleError::parse(line_idx, "Invalid REF field"))?;
        let ref_allele = Allele::from_str(ref_allele_str);

        // Parse ALT
        let alt_field = next_field()?;
        let alt_alleles: Vec<Allele> = split_bytes(alt_field, b',')
            .map(|a| {
                let s = std::str::from_utf8(a)
                    .map_err(|_| ReagleError::parse(line_idx, "Invalid ALT field"))?;
                Ok(Allele::from_str(s))
            })
            .collect::<Result<Vec<_>>>()?;

        // Skip QUAL and FILTER
        next_field()?;
        next_field()?;

        // Skip INFO
        next_field()?;

        // Parse FORMAT to find GT position
        let format = next_field()?;
        let (gt_idx, gl_idx, pl_idx) = find_format_indices(format);
        let gt_idx = gt_idx.ok_or_else(|| ReagleError::parse(line_idx, "No GT field in FORMAT"))?;

        // Parse genotypes
        let n_samples = self.samples.len();
        let mut alleles = Vec::with_capacity(n_samples * 2);
        let mut confidences: Option<Vec<u8>> = gl_idx.map(|_| Vec::with_capacity(n_samples));
        let mut likelihoods_pl: Option<Vec<Vec<u16>>> = if pl_idx.is_some() || gl_idx.is_some() {
            Some(Vec::with_capacity(n_samples))
        } else {
            None
        };
        let mut phase_mask: Vec<u8> = Vec::with_capacity(n_samples);

        if self.sample_ploidy.is_none() {
            self.sample_ploidy = Some(vec![true; n_samples]);
        }

        let first_sample = next_field()?;
        let sample_iter = std::iter::once(first_sample).chain(fields.by_ref());

        for (sample_idx, sample_field) in sample_iter.enumerate().take(n_samples) {
            let gt_field = nth_colon_field(sample_field, gt_idx).unwrap_or(b"./.");

            if gt_field.iter().any(|&b| b == b'/') {
                self.all_phased = false;
            }

            let (a1, a2) = parse_gt_bytes(gt_field);
            let is_missing = a1 == 255 || a2 == 255;
            let phased = gt_field.iter().any(|&b| b == b'|');
            phase_mask.push(if phased && !is_missing { 1 } else { 0 });

            if a1 == a2
                && !gt_field.iter().any(|&b| b == b'|')
                && !gt_field.iter().any(|&b| b == b'/')
            {
                if let Some(ref mut ploidy) = self.sample_ploidy {
                    ploidy[sample_idx] = false;
                }
            }

            alleles.push(a1);
            alleles.push(a2);

            if let Some(ref mut pl_out) = likelihoods_pl {
                let pl_vec = if let Some(pl_i) = pl_idx {
                    nth_colon_field(sample_field, pl_i)
                        .and_then(bytes_to_str)
                        .and_then(crate::io::vcf::parse_pl)
                        .unwrap_or_else(Vec::new)
                } else if let Some(gl_i) = gl_idx {
                    nth_colon_field(sample_field, gl_i)
                        .and_then(bytes_to_str)
                        .and_then(crate::io::vcf::gl_to_pl)
                        .unwrap_or_else(Vec::new)
                } else {
                    Vec::new()
                };
                pl_out.push(pl_vec);
            }

            if let Some(gl_i) = gl_idx {
                if let Some(conf_vec) = confidences.as_mut() {
                    let confidence = nth_colon_field(sample_field, gl_i)
                        .and_then(bytes_to_str)
                        .and_then(|gl_str| crate::io::vcf::compute_gl_confidence(gl_str, a1, a2))
                        .unwrap_or(255);
                    conf_vec.push(confidence);
                }
            }
        }

        if confidences.is_some() {
            self.has_any_confidence = true;
        }

        let n_alleles = 1 + alt_alleles.len();
        let marker = Marker::new(chrom_idx, pos, id, ref_allele, alt_alleles);
        let column = GenotypeColumn::from_alleles(&alleles, n_alleles);

        // Calculate genetic position
        let gen_pos = self.gen_maps.gen_pos(chrom_idx, pos);

        Ok(BufferedMarker {
            marker,
            column,
            gen_pos,
            confidences,
            likelihoods_pl,
            phase_mask,
        })
    }
}

fn trim_line_bytes(line: &[u8]) -> &[u8] {
    let mut end = line.len();
    while end > 0 && (line[end - 1] == b'\n' || line[end - 1] == b'\r') {
        end -= 1;
    }
    &line[..end]
}

struct FieldIter<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> FieldIter<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }
}

impl<'a> Iterator for FieldIter<'a> {
    type Item = &'a [u8];
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos > self.buf.len() {
            return None;
        }
        if self.pos == self.buf.len() {
            self.pos += 1;
            return Some(&[]);
        }
        let start = self.pos;
        let mut i = start;
        while i < self.buf.len() && self.buf[i] != b'\t' {
            i += 1;
        }
        self.pos = i + 1;
        Some(&self.buf[start..i])
    }
}

fn split_bytes<'a>(buf: &'a [u8], delim: u8) -> impl Iterator<Item = &'a [u8]> {
    struct SplitBytes<'a> {
        buf: &'a [u8],
        pos: usize,
        delim: u8,
    }
    impl<'a> Iterator for SplitBytes<'a> {
        type Item = &'a [u8];
        fn next(&mut self) -> Option<Self::Item> {
            if self.pos > self.buf.len() {
                return None;
            }
            if self.pos == self.buf.len() {
                self.pos += 1;
                return Some(&[]);
            }
            let start = self.pos;
            let mut i = start;
            while i < self.buf.len() && self.buf[i] != self.delim {
                i += 1;
            }
            self.pos = i + 1;
            Some(&self.buf[start..i])
        }
    }
    SplitBytes { buf, pos: 0, delim }
}

fn parse_u32_bytes(buf: &[u8]) -> Option<u32> {
    if buf.is_empty() {
        return None;
    }
    let mut val: u32 = 0;
    for &b in buf {
        if b < b'0' || b > b'9' {
            return None;
        }
        val = val.saturating_mul(10).saturating_add((b - b'0') as u32);
    }
    Some(val)
}

fn bytes_to_str(buf: &[u8]) -> Option<&str> {
    std::str::from_utf8(buf).ok()
}

fn find_format_indices(format: &[u8]) -> (Option<usize>, Option<usize>, Option<usize>) {
    let mut gt_idx = None;
    let mut gl_idx = None;
    let mut pl_idx = None;
    for (i, field) in split_bytes(format, b':').enumerate() {
        if field == b"GT" {
            gt_idx = Some(i);
        } else if field == b"GL" {
            gl_idx = Some(i);
        } else if field == b"PL" {
            pl_idx = Some(i);
        }
    }
    (gt_idx, gl_idx, pl_idx)
}

fn nth_colon_field<'a>(buf: &'a [u8], idx: usize) -> Option<&'a [u8]> {
    split_bytes(buf, b':').nth(idx)
}

/// Parse genotype field to (allele1, allele2)
fn parse_gt_bytes(gt: &[u8]) -> (u8, u8) {
    if gt == b"." || gt == b"./." || gt == b".|." {
        return (255, 255);
    }

    let mut sep: Option<u8> = None;
    for &b in gt {
        if b == b'|' {
            sep = Some(b'|');
            break;
        } else if b == b'/' {
            sep = Some(b'/');
            break;
        }
    }

    let sep = match sep {
        Some(s) => s,
        None => {
            let a1 = parse_allele_char_bytes(gt);
            return (a1, a1);
        }
    };

    let mut left = None;
    let mut right = None;
    for (i, part) in split_bytes(gt, sep).enumerate() {
        if i == 0 {
            left = Some(part);
        } else if i == 1 {
            right = Some(part);
            break;
        }
    }

    let (left, right) = match (left, right) {
        (Some(l), Some(r)) => (l, r),
        _ => return (255, 255),
    };

    let a1 = parse_allele_char_bytes(left);
    let a2 = parse_allele_char_bytes(right);

    if a1 == 255 || a2 == 255 {
        (255, 255)
    } else {
        (a1, a2)
    }
}

fn parse_allele_char_bytes(s: &[u8]) -> u8 {
    if s.is_empty() || s == b"." {
        return 255;
    }
    if s.len() == 1 {
        let c = s[0];
        if c >= b'0' && c <= b'9' {
            return c - b'0';
        }
    }
    let mut val: u16 = 0;
    for &b in s {
        if b < b'0' || b > b'9' {
            return 255;
        }
        val = val.saturating_mul(10).saturating_add((b - b'0') as u16);
    }
    if val > crate::io::vcf::MAX_ALLELE_INDEX {
        255
    } else {
        val as u8
    }
}

#[cfg(test)]
fn parse_gt(gt: &str) -> (u8, u8) {
    parse_gt_bytes(gt.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gt() {
        assert_eq!(parse_gt("0|1"), (0, 1));
        assert_eq!(parse_gt("1|0"), (1, 0));
        assert_eq!(parse_gt("0/1"), (0, 1));
        assert_eq!(parse_gt("./."), (255, 255));
        assert_eq!(parse_gt(".|."), (255, 255));
        assert_eq!(parse_gt("."), (255, 255));
    }

    #[test]
    fn test_streaming_config_default() {
        let config = StreamingConfig::default();
        assert_eq!(config.window_cm, 40.0);
        assert_eq!(config.overlap_cm, 2.0);
    }
}
