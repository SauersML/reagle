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

use noodles::bgzf::io as bgzf_io;
use noodles::vcf::Header;
use flate2::read::GzDecoder;
use tracing::info_span;

use crate::data::ChromIdx;
use crate::data::genetic_map::GeneticMaps;
use crate::data::haplotype::Samples;
use crate::data::marker::{Allele, Marker, Markers};
use crate::data::storage::{GenotypeColumn, GenotypeMatrix};
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
            max_markers: 4_000_000,
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

impl StateProbs {
    pub fn new(data: Vec<Vec<Vec<f32>>>, marker_indices: Vec<usize>, n_states: usize) -> Self {
        Self {
            data,
            marker_indices,
            n_states,
        }
    }
}

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
    hap_ids: Vec<u32>,
    /// Corresponding probabilities (same order as hap_ids)
    probs: Vec<f32>,
}

impl HaplotypePriors {
    /// Create empty priors
    pub fn new() -> Self {
        Self {
            hap_ids: Vec::new(),
            probs: Vec::new(),
        }
    }

    /// Get prior probability for a haplotype.
    /// Returns uniform prior (1/n_states) if haplotype not seen in previous window.
    /// Uses binary search for O(log K) lookup with good cache locality.
    #[inline]
    pub fn prior(&self, hap_id: u32, n_states: usize) -> f32 {
        match self.hap_ids.binary_search(&hap_id) {
            Ok(idx) => self.probs[idx],
            Err(_) => 1.0 / n_states.max(1) as f32,
        }
    }

    /// Set priors from HMM state posteriors at window boundary.
    /// Uses an adaptive threshold to avoid discarding most mass at high state counts.
    /// Sorts by hap_id for efficient binary search lookup.
    pub fn set_from_posteriors(&mut self, hap_indices: &[u32], probs: &[f32], gen_position: f64, window: usize) {
        self.hap_ids.clear();
        self.probs.clear();

        let _ = (gen_position, window);

        let adaptive_min = (1.0 / hap_indices.len().max(1) as f32) * 0.5;
        let min_prob = adaptive_min.min(0.001).max(1e-6);

        // Collect significant probabilities
        let mut pairs: Vec<(u32, f32)> = hap_indices
            .iter()
            .zip(probs.iter())
            .filter(|(_, p)| **p > min_prob)
            .map(|(&h, &p)| (h, p))
            .collect();
        
        // Sort by hap_id for binary search
        pairs.sort_unstable_by_key(|(h, _)| *h);
        
        // Split into parallel arrays
        self.hap_ids.reserve(pairs.len());
        self.probs.reserve(pairs.len());
        let total: f32 = pairs.iter().map(|(_, p)| *p).sum();
        if total > 0.0 {
            for (h, p) in pairs {
                self.hap_ids.push(h);
                self.probs.push(p / total);
            }
        }
    }

    /// Check if we have any priors
    pub fn is_empty(&self) -> bool {
        self.hap_ids.is_empty()
    }
}

impl Default for HaplotypePriors {
    fn default() -> Self {
        Self::new()
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
    line_buf: String,
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
                    return Err(anyhow::anyhow!(
                        "Expected BGZF file for extension .{}",
                        ext
                    )
                    .into());
                }
                let reader: Box<dyn BufRead + Send> =
                    Box::new(BufReader::new(bgzf_io::Reader::new(file)));
                Self::from_reader(reader, gen_maps, config)
            }
            "gz" => {
                let mut file = File::open(path)?;
                let reader: Box<dyn BufRead + Send> = if detect_bgzf(&mut file)? {
                    Box::new(BufReader::new(bgzf_io::Reader::new(file)))
                } else {
                    Box::new(BufReader::new(GzDecoder::new(file)))
                };
                Self::from_reader(reader, gen_maps, config)
            }
            _ => {
                let file = File::open(path)?;
                let reader: Box<dyn BufRead + Send> = Box::new(BufReader::new(file));
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
            markers_meta: Markers::new(),
            current_chrom: None,
            window_num: 0,
            global_marker_idx: 0,
            eof: false,
            line_buf: String::new(),
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

    /// Returns true if all genotypes seen so far were phased (used '|' separator).
    pub fn was_all_phased(&self) -> bool {
        self.all_phased
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
        let mut markers = Markers::new();
        let mut columns = Vec::with_capacity(window_end);
        let mut confidences: Vec<Vec<u8>> = Vec::new();

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
        }

        if let Some(ref ploidy) = self.sample_ploidy {
            let sample_ids: Vec<String> = self.samples.ids().iter().map(|s| s.to_string()).collect();
            self.samples = Arc::new(Samples::from_ids_with_ploidy(sample_ids, ploidy.clone()));
        }

        let genotypes = if self.has_any_confidence {
            GenotypeMatrix::new_unphased_with_confidence(markers, columns, Arc::clone(&self.samples), confidences)
        } else {
            GenotypeMatrix::new_unphased(markers, columns, Arc::clone(&self.samples))
        };

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
            .map(|cur| !candidates.iter().any(|c| c.as_str() == cur.as_str()))
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
            while self.buffer.front().map(|m| m.marker.pos < start_pos).unwrap_or(false) {
                self.buffer.pop_front();
                self.global_marker_idx += 1;
            }
            return Ok(None);
        }

        let first_idx = indices[0];
        let last_idx = *indices.last().unwrap();
        let n_markers = indices.len();

        let mut markers = Markers::new();
        let mut columns = Vec::with_capacity(n_markers);
        let mut confidences: Vec<Vec<u8>> = Vec::new();

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
        }

        let genotypes = if self.has_any_confidence {
            GenotypeMatrix::new_unphased_with_confidence(
                markers,
                columns,
                Arc::clone(&self.samples),
                confidences,
            )
        } else {
            GenotypeMatrix::new_unphased(markers, columns, Arc::clone(&self.samples))
        };

        let window = StreamWindow {
            genotypes,
            global_start: self.global_marker_idx + first_idx,
            global_end: self.global_marker_idx + last_idx + 1,
            output_start: 0,
            output_end: n_markers,
            is_first: self.window_num == 0,
            phased_overlap: None,
        };

        while self.buffer.front().map(|m| m.marker.pos < start_pos).unwrap_or(false) {
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
            let bytes_read = self.reader.read_line(&mut self.line_buf)?;
            if bytes_read == 0 {
                self.eof = true;
                return Ok(None);
            }

            let line = self.line_buf.trim().to_string();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse the VCF line
            return self.parse_vcf_line(&line).map(Some);
        }
    }

    /// Parse a single VCF line
    fn parse_vcf_line(&mut self, line: &str) -> Result<BufferedMarker> {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 10 {
            return Err(ReagleError::parse(
                self.global_marker_idx + self.buffer.len(),
                format!("Expected at least 10 fields, got {}", fields.len()),
            ));
        }

        // Parse CHROM
        let chrom_name = fields[0];
        let chrom_idx = self.markers_meta.add_chrom(chrom_name);

        // Update current chromosome tracking
        if self.current_chrom != Some(chrom_idx) {
            // New chromosome - could flush buffer here for multi-chrom support
            self.current_chrom = Some(chrom_idx);
        }

        // Parse POS
        let pos: u32 = fields[1]
            .parse()
            .map_err(|_| ReagleError::parse(self.global_marker_idx, "Invalid POS field"))?;

        // Parse ID
        let id = if fields[2] == "." {
            None
        } else {
            Some(fields[2].into())
        };

        // Parse REF
        let ref_allele = Allele::from_str(fields[3]);

        // Parse ALT
        let alt_alleles: Vec<Allele> = fields[4].split(',').map(|a| Allele::from_str(a)).collect();

        // Parse FORMAT to find GT position
        let format = fields[8];
        let gt_idx = format
            .split(':')
            .position(|f| f == "GT")
            .ok_or_else(|| ReagleError::parse(self.global_marker_idx, "No GT field in FORMAT"))?;
        let gl_idx = format.split(':').position(|f| f == "GL");

        // Parse genotypes
        let n_samples = self.samples.len();
        let mut alleles = Vec::with_capacity(n_samples * 2);
        let mut confidences: Option<Vec<u8>> = gl_idx.map(|_| Vec::with_capacity(n_samples));

        if self.sample_ploidy.is_none() {
            self.sample_ploidy = Some(vec![true; n_samples]);
        }

        for (sample_idx, sample_field) in fields[9..].iter().enumerate().take(n_samples) {
            let gt_field = sample_field.split(':').nth(gt_idx).unwrap_or("./.");

            if gt_field.contains('/') {
                self.all_phased = false;
            }

            let (a1, a2) = parse_gt(gt_field);

            if a1 == a2 && !gt_field.contains('|') && !gt_field.contains('/') {
                if let Some(ref mut ploidy) = self.sample_ploidy {
                    ploidy[sample_idx] = false;
                }
            }

            alleles.push(a1);
            alleles.push(a2);

            if let Some(gl_i) = gl_idx {
                if let Some(conf_vec) = confidences.as_mut() {
                    let confidence = sample_field
                        .split(':')
                        .nth(gl_i)
                        .and_then(|gl_str| crate::io::vcf::compute_gl_confidence(gl_str, a1, a2))
                        .unwrap_or(255);
                    conf_vec.push(confidence);
                }
            }
        }

        if confidences.is_some() {
            self.has_any_confidence = true;
        }

        let marker = Marker::new(chrom_idx, pos, id, ref_allele, alt_alleles.clone());
        let n_alleles = 1 + alt_alleles.len();
        let column = GenotypeColumn::from_alleles(&alleles, n_alleles);

        // Calculate genetic position
        let gen_pos = self.gen_maps.gen_pos(chrom_idx, pos);

        Ok(BufferedMarker {
            marker,
            column,
            gen_pos,
            confidences,
        })
    }
}

/// Parse genotype field to (allele1, allele2)
fn parse_gt(gt: &str) -> (u8, u8) {
    if gt == "." || gt == "./." || gt == ".|." {
        return (255, 255);
    }

    let sep = if gt.contains('|') { '|' } else { '/' };
    let parts: Vec<&str> = gt.split(sep).collect();

    if parts.len() == 1 {
        let a1 = parse_allele_char(parts[0]);
        return (a1, a1);
    }

    if parts.len() != 2 {
        return (255, 255);
    }

    let a1 = parse_allele_char(parts[0]);
    let a2 = parse_allele_char(parts[1]);

    if a1 == 255 || a2 == 255 {
        (255, 255)
    } else {
        (a1, a2)
    }
}

fn parse_allele_char(s: &str) -> u8 {
    if s == "." || s.is_empty() {
        return 255;
    }
    if s.len() == 1 {
        let c = s.as_bytes()[0];
        if c >= b'0' && c <= b'9' {
            return c - b'0';
        }
    }
    s.parse().unwrap_or(255)
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
