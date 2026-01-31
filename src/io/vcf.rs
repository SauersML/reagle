//! # VCF Reading and Writing
//!
//! Parse VCF/BCF files into `GenotypeMatrix`. Write phased results back to VCF.
//! Uses the `noodles` crate for VCF I/O.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use rayon::prelude::*;
use std::path::Path;
use std::sync::Arc;

use flate2::read::GzDecoder;
use noodles::bgzf::io as bgzf_io;
use noodles::vcf::Header;
use tracing::info_span;

use crate::data::haplotype::Samples;
use crate::data::marker::{Allele, Marker, MarkerIdx, Markers};
use crate::data::storage::matrix::PlMatrix;
use crate::data::storage::{GenotypeColumn, GenotypeMatrix, PhaseState, compress_block};
use crate::error::{ReagleError, Result};
use crate::utils::telemetry::TelemetryBlackboard;

pub(crate) fn parse_pl(pl_str: &str) -> Option<Vec<u16>> {
    if pl_str.is_empty() || pl_str == "." {
        return None;
    }
    let mut out = Vec::new();
    for s in pl_str.split(',') {
        if s.is_empty() || s == "." {
            return None;
        }
        let v = lexical_core::parse::<u32>(s.as_bytes()).ok()?;
        out.push(v.min(u16::MAX as u32) as u16);
    }
    if out.is_empty() { None } else { Some(out) }
}

pub(crate) fn gl_to_pl(gl_str: &str) -> Option<Vec<u16>> {
    if gl_str.is_empty() || gl_str == "." {
        return None;
    }
    let mut gls: Vec<f64> = Vec::new();
    for s in gl_str.split(',') {
        if s.is_empty() || s == "." {
            return None;
        }
        let v = lexical_core::parse::<f64>(s.as_bytes()).ok()?;
        if !v.is_finite() {
            return None;
        }
        gls.push(v);
    }
    if gls.is_empty() {
        return None;
    }
    let max_gl = gls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_gl.is_finite() {
        return None;
    }
    let mut out = Vec::with_capacity(gls.len());
    for gl in gls {
        let d = (max_gl - gl).max(0.0);
        let pl = (10.0 * d).round();
        out.push(pl.min(u16::MAX as f64) as u16);
    }
    Some(out)
}

/// Imputation quality statistics for a single marker
///
/// Calculates Dosage R-squared (DR2) using the Beagle formula:
/// R² = (Σp² - (Σp)²/N) / (Σp - (Σp)²/N)
/// where p is the posterior probability of the ALT allele for each haplotype,
/// and N is the total number of haplotypes.
#[derive(Clone, Debug, Default)]
pub struct MarkerImputationStats {
    /// Sum of probabilities (p) for each allele across all haplotypes.
    sum_p: Vec<f32>,
    /// Sum of squared probabilities (p²) for each allele across all haplotypes.
    sum_p_sq: Vec<f32>,
    /// Number of HAPLOTYPES processed.
    n_haps: usize,
    /// Whether this marker was imputed.
    pub is_imputed: bool,
}

impl MarkerImputationStats {
    /// Create new stats for a marker with the given number of alleles.
    pub fn new(n_alleles: usize) -> Self {
        Self {
            sum_p: vec![0.0; n_alleles],
            sum_p_sq: vec![0.0; n_alleles],
            n_haps: 0,
            is_imputed: false,
        }
    }

    /// Add a biallelic sample's data with compact representation (no heap allocation).
    /// p1 = P(ALT) for haplotype 1, p2 = P(ALT) for haplotype 2.
    #[inline]
    pub fn add_sample_biallelic(&mut self, p1: f32, p2: f32) {
        assert!(
            self.sum_p.len() == 2,
            "add_sample_biallelic requires biallelic marker"
        );
        self.n_haps += 2;

        let p_sum = p1 + p2;
        let p_sq_sum = p1 * p1 + p2 * p2;

        self.sum_p[1] += p_sum;
        self.sum_p_sq[1] += p_sq_sum;
    }

    /// Calculate DR2 (dosage R-squared) matching Java Beagle's implementation.
    /// Formula: (Σp² - (Σp)²/N) / (Σp - (Σp)²/N)
    pub fn dr2(&self, allele: usize) -> f32 {
        if allele == 0 || allele >= self.sum_p.len() || self.n_haps == 0 {
            return 0.0;
        }

        let sum = self.sum_p[allele];
        let n = self.n_haps as f32;
        // Monomorphic sites (sum ≈ 0 or sum ≈ n) have no variance to measure.
        // Return 1.0 because they trivially impute correctly.
        if sum <= 1e-4 || (sum - n).abs() <= 1e-4 {
            return 1.0;
        }

        let sum_sq = self.sum_p_sq[allele];

        // Java: float meanTerm = sum*sum/(nInputTargHaps);
        let mean_term = sum * sum / n;

        // Java: float num = (sum2 - meanTerm);
        let num = sum_sq - mean_term;

        // Java: float den = (sum - meanTerm);
        let den = sum - mean_term;

        // Java: return num <= 0 ? 0f : num/den;
        if num <= 0.0 {
            0.0
        } else if den == 0.0 {
            0.0
        } else {
            (num / den).clamp(0.0, 1.0)
        }
    }

    /// Calculate estimated allele frequency for the specified ALT allele
    pub fn allele_freq(&self, allele: usize) -> f32 {
        if allele == 0 || allele >= self.sum_p.len() || self.n_haps == 0 {
            return 0.0;
        }
        // AF = Total Prob Mass / Total Haplotypes
        self.sum_p[allele] / self.n_haps as f32
    }
}

/// Collection of imputation statistics for all markers
#[derive(Clone, Debug, Default)]
pub struct ImputationQuality {
    /// Per-marker statistics
    pub marker_stats: Vec<MarkerImputationStats>,
}

impl ImputationQuality {
    /// Create new quality tracker for the given number of markers
    pub fn new(n_alleles_per_marker: &[usize]) -> Self {
        let marker_stats = n_alleles_per_marker
            .iter()
            .map(|&n| MarkerImputationStats::new(n))
            .collect();
        Self { marker_stats }
    }

    /// Get mutable stats for a marker
    pub fn get_mut(&mut self, marker: usize) -> Option<&mut MarkerImputationStats> {
        self.marker_stats.get_mut(marker)
    }

    /// Get stats for a marker
    pub fn get(&self, marker: usize) -> Option<&MarkerImputationStats> {
        self.marker_stats.get(marker)
    }

    /// Mark a marker as imputed
    pub fn set_imputed(&mut self, marker: usize, imputed: bool) {
        if let Some(stats) = self.marker_stats.get_mut(marker) {
            stats.is_imputed = imputed;
        }
    }
}

/// VCF file reader
pub struct VcfReader {
    /// Sample information
    samples: Arc<Samples>,
    /// Sample indices to include (None = include all)
    include_sample_indices: Option<Vec<usize>>,
    /// Marker IDs to exclude (None = exclude none)
    exclude_marker_ids: Option<std::collections::HashSet<String>>,
    /// Per-sample ploidy detected during reading (true = diploid, false = haploid)
    /// Initialized on first variant, used to update Samples after reading
    sample_ploidy: Option<Vec<bool>>,
    /// Whether all genotypes read were phased (detected during read_all)
    all_phased: bool,
}

impl VcfReader {
    fn detect_bgzf(file: &mut File) -> Result<bool> {
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

        let mut i = 0;
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

    /// Open a VCF file and read the header
    pub fn open(path: &Path) -> Result<(Self, Box<dyn BufRead + Send>)> {
        info_span!("vcf_open", path = ?path).in_scope(|| {
            let mut file = File::open(path)?;

            // Check if gzipped
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            let reader: Box<dyn BufRead + Send> = match ext {
                "bgz" | "bgzf" => {
                    if !Self::detect_bgzf(&mut file)? {
                        return Err(
                            anyhow::anyhow!("Expected BGZF file for extension .{}", ext).into()
                        );
                    }
                    Box::new(BufReader::new(bgzf_io::Reader::new(file)))
                }
                "gz" => {
                    if Self::detect_bgzf(&mut file)? {
                        Box::new(BufReader::new(bgzf_io::Reader::new(file)))
                    } else {
                        Box::new(BufReader::new(GzDecoder::new(file)))
                    }
                }
                _ => Box::new(BufReader::new(file)),
            };

            Self::from_reader(reader)
        })
    }
    /// Create from a reader
    pub fn from_reader(
        mut reader: Box<dyn BufRead + Send>,
    ) -> Result<(Self, Box<dyn BufRead + Send>)> {
        info_span!("vcf_from_reader").in_scope(|| {
            // Read header
            let mut header_str = String::new();
            loop {
                let mut line = String::new();
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

            // Extract sample names
            let sample_names: Vec<String> = header
                .sample_names()
                .iter()
                .map(|s| s.to_string())
                .collect();

            let samples = Arc::new(Samples::from_ids(sample_names));

            Ok((
                Self {
                    samples,
                    include_sample_indices: None,
                    exclude_marker_ids: None,
                    sample_ploidy: None,
                    all_phased: true,
                },
                reader,
            ))
        })
    }

    /// Set sample exclusion filter
    ///
    /// # Arguments
    /// * `exclude_ids` - Set of sample IDs to exclude from processing
    pub fn set_exclude_samples(&mut self, exclude_ids: &std::collections::HashSet<String>) {
        if exclude_ids.is_empty() {
            self.include_sample_indices = None;
            return;
        }

        // Build list of sample indices to INCLUDE (those NOT in exclude list)
        let include_indices: Vec<usize> = self
            .samples
            .ids()
            .iter()
            .enumerate()
            .filter(|(_, id)| !exclude_ids.contains(id.as_ref()))
            .map(|(i, _)| i)
            .collect();

        // Update samples Arc to only include non-excluded samples
        let filtered_ids: Vec<String> = include_indices
            .iter()
            .map(|&i| self.samples.ids()[i].to_string())
            .collect();

        self.samples = Arc::new(Samples::from_ids(filtered_ids));
        self.include_sample_indices = Some(include_indices);
    }

    /// Set marker exclusion filter
    ///
    /// # Arguments
    /// * `exclude_ids` - Set of marker IDs to exclude from processing
    pub fn set_exclude_markers(&mut self, exclude_ids: std::collections::HashSet<String>) {
        if exclude_ids.is_empty() {
            self.exclude_marker_ids = None;
        } else {
            self.exclude_marker_ids = Some(exclude_ids);
        }
    }

    /// Get samples Arc
    pub fn samples_arc(&self) -> Arc<Samples> {
        Arc::clone(&self.samples)
    }

    /// Read all records into a GenotypeMatrix
    pub fn read_all(&mut self, mut reader: Box<dyn BufRead + Send>) -> Result<GenotypeMatrix> {
        info_span!("vcf_read_all").in_scope(|| {
            let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
            let mut columns = Vec::new();
            let n_samples = self.samples.len();
            // Accumulate per-marker confidence scores (one Vec<u8> per marker, indexed by sample)
            let mut all_confidences: Vec<Option<Vec<u8>>> = Vec::new();
            let mut has_any_confidence = false;
            let mut all_likelihoods_pl: Vec<Option<Vec<Vec<u16>>>> = Vec::new();
            let mut has_any_likelihoods = false;
            let mut all_phase_masks: Vec<Vec<u8>> = Vec::new();

            let mut line_buf: Vec<u8> = Vec::new();
            let mut line_num = 0usize;

            // Buffers for batch processing (Dictionary Compression)
            const BATCH_SIZE: usize = 64;
            let mut batch_markers: Vec<Marker> = Vec::with_capacity(BATCH_SIZE);
            let mut batch_alleles: Vec<Vec<u8>> = Vec::with_capacity(BATCH_SIZE);
            let mut batch_n_alleles: Vec<usize> = Vec::with_capacity(BATCH_SIZE);

            loop {
                line_buf.clear();
                let bytes_read = reader.read_until(b'\n', &mut line_buf)?;
                if bytes_read == 0 {
                    break;
                }
                line_num += 1;

                let line = trim_line_bytes(&line_buf);
                if line.is_empty() || line[0] == b'#' {
                    continue;
                }

                // Parse VCF record
                let (
                    marker,
                    mut alleles,
                    is_phased,
                    mut confidences,
                    mut likelihoods_pl,
                    mut phase_mask,
                ) = self.parse_record(line, &mut markers, line_num)?;

                // Track if any marker is unphased
                if !is_phased {
                    self.all_phased = false;
                }

                // Check marker exclusion filter
                if let Some(ref exclude_ids) = self.exclude_marker_ids {
                    if let Some(ref id) = marker.id {
                        if exclude_ids.contains(id.as_ref()) {
                            continue; // Skip this marker
                        }
                    }
                }

                // Apply sample filtering if set
                if let Some(ref include_indices) = self.include_sample_indices {
                    // Filter alleles to only include non-excluded samples
                    let mut filtered_alleles = Vec::with_capacity(include_indices.len() * 2);
                    for &sample_idx in include_indices {
                        let hap1_idx = sample_idx * 2;
                        let hap2_idx = sample_idx * 2 + 1;
                        if hap1_idx < alleles.len() && hap2_idx < alleles.len() {
                            filtered_alleles.push(alleles[hap1_idx]);
                            filtered_alleles.push(alleles[hap2_idx]);
                        }
                    }
                    alleles = filtered_alleles;

                    // Also filter confidence scores if present
                    if let Some(ref conf) = confidences {
                        let mut filtered_conf = Vec::with_capacity(include_indices.len());
                        for &sample_idx in include_indices {
                            if sample_idx < conf.len() {
                                filtered_conf.push(conf[sample_idx]);
                            }
                        }
                        confidences = Some(filtered_conf);
                    }

                    if let Some(ref pl) = likelihoods_pl {
                        let mut filtered_pl = Vec::with_capacity(include_indices.len());
                        for &sample_idx in include_indices {
                            if sample_idx < pl.len() {
                                filtered_pl.push(pl[sample_idx].clone());
                            }
                        }
                        likelihoods_pl = Some(filtered_pl);
                    }

                    let mut filtered_phase = Vec::with_capacity(include_indices.len());
                    for &sample_idx in include_indices {
                        if sample_idx < phase_mask.len() {
                            filtered_phase.push(phase_mask[sample_idx]);
                        }
                    }
                    phase_mask = filtered_phase;
                }

                // Store confidence scores
                if confidences.is_some() {
                    has_any_confidence = true;
                }
                all_confidences.push(confidences);

                if likelihoods_pl.is_some() {
                    has_any_likelihoods = true;
                }
                all_likelihoods_pl.push(likelihoods_pl);
                all_phase_masks.push(phase_mask);

                // Calculate actual number of alleles: 1 REF + N ALT
                let n_alleles = 1 + marker.alt_alleles.len();

                // Buffer the marker data
                batch_markers.push(marker);
                batch_alleles.push(alleles);
                batch_n_alleles.push(n_alleles);

                // Process batch if full
                if batch_markers.len() >= BATCH_SIZE {
                    Self::flush_batch(
                        &mut markers,
                        &mut columns,
                        &mut batch_markers,
                        &mut batch_alleles,
                        &mut batch_n_alleles,
                    );
                }
            }

            // Process remaining markers
            if !batch_markers.is_empty() {
                Self::flush_batch(
                    &mut markers,
                    &mut columns,
                    &mut batch_markers,
                    &mut batch_alleles,
                    &mut batch_n_alleles,
                );
            }

            // Update Samples with detected ploidy information
            self.finalize_samples();

            let confidence_opt = if has_any_confidence && all_confidences.len() == columns.len() {
                Some(
                    all_confidences
                        .into_iter()
                        .map(|c| c.unwrap_or_else(|| vec![255; n_samples]))
                        .collect(),
                )
            } else {
                None
            };
            let phase_mask_opt = if all_phase_masks.len() == columns.len() {
                Some(all_phase_masks)
            } else {
                None
            };

            let matrix = if has_any_likelihoods && all_likelihoods_pl.len() == columns.len() {
                let mut marker_strides: Vec<u16> = Vec::with_capacity(all_likelihoods_pl.len());
                let mut marker_blocks: Vec<Vec<u16>> = Vec::with_capacity(all_likelihoods_pl.len());
                for pl_opt in all_likelihoods_pl.into_iter() {
                    if let Some(pl_by_sample) = pl_opt {
                        let stride = pl_by_sample
                            .get(0)
                            .map(|v| v.len())
                            .unwrap_or(0)
                            .min(u16::MAX as usize) as u16;
                        if stride == 0 {
                            marker_strides.push(0);
                            marker_blocks.push(Vec::new());
                            continue;
                        }

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
                    } else {
                        marker_strides.push(0);
                        marker_blocks.push(Vec::new());
                    }
                }

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
            Ok(matrix.with_phase_mask(phase_mask_opt))
        })
    }

    /// Flush a batch of markers, attempting dictionary compression
    fn flush_batch(
        markers: &mut Markers,
        columns: &mut Vec<GenotypeColumn>,
        batch_markers: &mut Vec<Marker>,
        batch_alleles: &mut Vec<Vec<u8>>,
        batch_n_alleles: &mut Vec<usize>,
    ) {
        if batch_markers.is_empty() {
            return;
        }

        let n_markers = batch_markers.len();
        let n_haps = batch_alleles[0].len();

        // Check if we can compress (must have enough markers and be biallelic)
        // Beagle usually only compresses biallelic markers
        let all_biallelic = batch_n_alleles.iter().all(|&n| n == 2);

        let compressed_dict = if n_markers >= 4 && all_biallelic {
            // Create closure for allele access
            let get_allele = |m: usize, h: crate::data::haplotype::HapIdx| -> u8 {
                batch_alleles[m][h.as_usize()]
            };

            compress_block(get_allele, n_markers, n_haps, 1)
        } else {
            None
        };

        if let Some(dict) = compressed_dict {
            // Success! Share the dictionary across all columns in this batch
            let dict_arc = Arc::new(dict);

            for (i, marker) in batch_markers.drain(..).enumerate() {
                markers.push(marker);
                columns.push(GenotypeColumn::Dictionary(Arc::clone(&dict_arc), i));
            }
        } else {
            // Fallback to individual storage (Dense or Sparse)
            for ((marker, alleles), n_alleles) in batch_markers
                .drain(..)
                .zip(batch_alleles.drain(..))
                .zip(batch_n_alleles.drain(..))
            {
                markers.push(marker);
                let col = GenotypeColumn::from_alleles(&alleles, n_alleles);
                columns.push(col);
            }
        }

        // Clear buffers (drain already emptied markers/alleles/n_alleles but verify)
        // drain(..) removes elements, so they are already empty if matched.
        // But batch_alleles and batch_n_alleles were not drained in the 'if' branch above.
        batch_markers.clear();
        batch_alleles.clear();
        batch_n_alleles.clear();
    }

    /// Parse a single VCF record line
    ///
    /// Returns (marker, alleles, is_phased, confidences, likelihoods_pl).
    fn parse_record(
        &mut self,
        line: &[u8],
        markers: &mut Markers,
        line_num: usize,
    ) -> Result<(
        Marker,
        Vec<u8>,
        bool,
        Option<Vec<u8>>,
        Option<Vec<Vec<u16>>>,
        Vec<u8>,
    )> {
        let mut fields = FieldIter::new(line);
        let mut next_field = || {
            fields
                .next()
                .ok_or_else(|| ReagleError::parse(line_num, "Expected at least 10 fields, got fewer"))
        };

        // Parse CHROM
        let chrom_name = std::str::from_utf8(next_field()?)
            .map_err(|_| ReagleError::parse(line_num, "Invalid CHROM field"))?;
        let chrom_idx = markers.add_chrom(chrom_name);

        // Parse POS
        let pos: u32 = parse_u32_bytes(next_field()?)
            .ok_or_else(|| ReagleError::parse(line_num, "Invalid POS field"))?;

        // Parse ID
        let id_field = next_field()?;
        let id = if id_field == b"." {
            None
        } else {
            let id_str = std::str::from_utf8(id_field)
                .map_err(|_| ReagleError::parse(line_num, "Invalid ID field"))?;
            Some(id_str.into())
        };

        // Parse REF
        let ref_allele_str = std::str::from_utf8(next_field()?)
            .map_err(|_| ReagleError::parse(line_num, "Invalid REF field"))?;
        let ref_allele = Allele::from_str(ref_allele_str);

        // Parse ALT
        let alt_field = next_field()?;
        let alt_alleles: Vec<Allele> = split_bytes(alt_field, b',')
            .map(|a| {
                let s = std::str::from_utf8(a)
                    .map_err(|_| ReagleError::parse(line_num, "Invalid ALT field"))?;
                Ok(Allele::from_str(s))
            })
            .collect::<Result<Vec<_>>>()?;

        // Skip QUAL and FILTER
        let _ = next_field()?;
        let _ = next_field()?;

        // Parse INFO field for END tag (field[7])
        // This is important for structural variants and gVCF blocks
        let info_field = next_field()?;
        let end_pos: Option<u32> = if info_field != b"." {
            // Parse INFO field looking for END=value
            // Avoid Vec allocation by using iterator directly
            split_bytes(info_field, b';')
                .filter_map(|kv| kv.strip_prefix(b"END=").and_then(parse_u32_bytes))
                .next()
        } else {
            None
        };

        // Parse FORMAT to find GT position and optionally GL position
        // Avoid Vec allocation by using position() directly on iterator
        let format = next_field()?;
        let (gt_idx, gl_idx, pl_idx) = find_format_indices(format);
        let gt_idx = gt_idx.ok_or_else(|| ReagleError::parse(line_num, "No GT field in FORMAT"))?;

        // Parse genotypes
        let n_samples = self.samples.len();
        let mut alleles = Vec::with_capacity(n_samples * 2);
        let mut is_phased = true;
        let mut phase_mask: Vec<u8> = Vec::with_capacity(n_samples);
        // Confidence scores (only populated if GL field is present)
        let mut confidences: Option<Vec<u8>> = gl_idx.map(|_| Vec::with_capacity(n_samples));
        let mut likelihoods_pl: Option<Vec<Vec<u16>>> = if pl_idx.is_some() || gl_idx.is_some() {
            Some(Vec::with_capacity(n_samples))
        } else {
            None
        };

        // Initialize ploidy tracking on first variant if not already done
        if self.sample_ploidy.is_none() {
            self.sample_ploidy = Some(vec![true; n_samples]); // Assume all diploid initially
        }

        let first_sample = next_field()?;
        let sample_iter = std::iter::once(first_sample).chain(fields.by_ref());

        for (sample_idx, sample_field) in sample_iter.enumerate() {
            if sample_idx >= n_samples {
                break;
            }

            let gt_field = nth_colon_field(sample_field, gt_idx).unwrap_or(b"./.");

            // Parse genotype (handle both phased | and unphased /)
            let (a1, a2, phased, is_haploid) = parse_genotype_bytes(gt_field)?;

            let is_missing = a1 == 255 || a2 == 255;
            if !phased {
                is_phased = false;
            }
            phase_mask.push(if phased && !is_missing { 1 } else { 0 });

            // Track haploid samples - once detected as haploid, stays haploid
            if is_haploid {
                if let Some(ref mut ploidy) = self.sample_ploidy {
                    ploidy[sample_idx] = false; // Mark as haploid
                }
            }

            alleles.push(a1);
            alleles.push(a2);

            if let Some(ref mut pl_out) = likelihoods_pl {
                let pl_vec = if let Some(pl_i) = pl_idx {
                    nth_colon_field(sample_field, pl_i)
                        .and_then(bytes_to_str)
                        .and_then(parse_pl)
                        .unwrap_or_else(Vec::new)
                } else if let Some(gl_i) = gl_idx {
                    nth_colon_field(sample_field, gl_i)
                        .and_then(bytes_to_str)
                        .and_then(gl_to_pl)
                        .unwrap_or_else(Vec::new)
                } else {
                    Vec::new()
                };
                pl_out.push(pl_vec);
            }

            // Parse GL field if present and compute confidence
            if let Some(gl_i) = gl_idx {
                if let Some(conf_vec) = confidences.as_mut() {
                    let confidence = nth_colon_field(sample_field, gl_i)
                        .and_then(bytes_to_str)
                        .and_then(|gl_str| compute_gl_confidence(gl_str, a1, a2))
                        .unwrap_or(255); // Default to full confidence if GL missing/unparseable
                    conf_vec.push(confidence);
                }
            }
        }

        let marker = Marker::with_end(chrom_idx, pos, end_pos, id, ref_allele, alt_alleles);

        Ok((
            marker,
            alleles,
            is_phased,
            confidences,
            likelihoods_pl,
            phase_mask,
        ))
    }

    /// Rebuild Samples with detected ploidy information
    ///
    /// Call this after reading all variants to update the Samples struct
    /// with accurate ploidy information detected during parsing.
    pub fn finalize_samples(&mut self) {
        if let Some(ref ploidy) = self.sample_ploidy {
            let sample_ids: Vec<String> =
                self.samples.ids().iter().map(|s| s.to_string()).collect();
            self.samples = Arc::new(Samples::from_ids_with_ploidy(sample_ids, ploidy.clone()));
        }
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

fn parse_genotype_bytes(gt: &[u8]) -> Result<(u8, u8, bool, bool)> {
    if gt == b"." || gt == b"./." || gt == b".|." {
        return Ok((255, 255, true, false));
    }

    let mut phased = false;
    let mut sep: Option<u8> = None;
    for &b in gt {
        if b == b'|' {
            phased = true;
            sep = Some(b'|');
            break;
        } else if b == b'/' {
            phased = false;
            sep = Some(b'/');
            break;
        }
    }

    let sep = match sep {
        Some(s) => s,
        None => {
            let a1 = parse_allele_bytes(gt);
            return Ok((a1, a1, true, true));
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
        _ => return Ok((255, 255, false, false)),
    };

    let a1 = parse_allele_bytes(left);
    let a2 = parse_allele_bytes(right);

    if a1 == 255 || a2 == 255 {
        return Ok((255, 255, false, false));
    }

    Ok((a1, a2, phased, false))
}

#[inline]
fn parse_allele_bytes(s: &[u8]) -> u8 {
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
        if val > MAX_ALLELE_INDEX {
            log::warn!(
                "Allele index {} exceeds maximum supported value {}; treating as missing",
                val,
                MAX_ALLELE_INDEX
            );
            return 255;
        }
    }
    val as u8
}

/// Maximum supported allele index (u8 limitation)
/// Alleles beyond this will be treated as missing with a warning
pub const MAX_ALLELE_INDEX: u16 = 254;

/// Compute genotype confidence from GL field.
///
/// GL field contains log10 likelihoods for each possible genotype.
/// For diploid biallelic: GL = P(0/0), P(0/1), P(1/1)
///
/// Returns confidence (0-255) for the called genotype.
///
/// # Arguments
/// * `gl_str` - GL field value, e.g., "-0.48,-0.48,-0.48" or "0,-5,-10"
/// * `a1` - First called allele (0=ref, 1+=alt)
/// * `a2` - Second called allele
pub fn compute_gl_confidence(gl_str: &str, a1: u8, a2: u8) -> Option<u8> {
    // Skip missing values
    if gl_str.is_empty() || gl_str == "." {
        return None;
    }

    // Parse GL values
    let mut gls: Vec<f64> = Vec::new();
    for s in gl_str.split(',') {
        if s.is_empty() || s == "." {
            return None;
        }
        let v = lexical_core::parse::<f64>(s.as_bytes()).ok()?;
        if !v.is_finite() {
            return None;
        }
        gls.push(v);
    }

    // Need at least 3 values for diploid biallelic
    if gls.len() < 3 {
        return None;
    }

    // Map genotype to GL index:
    // For biallelic: 0/0 -> 0, 0/1 -> 1, 1/1 -> 2
    // For multiallelic: use triangular number formula
    let (min_a, max_a) = if a1 <= a2 { (a1, a2) } else { (a2, a1) };
    let gt_idx = if a1 == 255 || a2 == 255 {
        // Missing allele - can't compute confidence
        return None;
    } else {
        // Triangular number index: for (a, b) where a <= b, index = b*(b+1)/2 + a
        let max_a_usize = max_a as usize;
        let min_a_usize = min_a as usize;
        max_a_usize * (max_a_usize + 1) / 2 + min_a_usize
    };

    if gt_idx >= gls.len() {
        return None;
    }

    // Get the GL for the called genotype
    let called_gl = gls[gt_idx];
    if !called_gl.is_finite() {
        return None;
    }

    // Find max GL for numerical stability
    let max_gl = gls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if !max_gl.is_finite() {
        return None;
    }

    // --- Optimized confidence calculation ---
    // Derivation of the optimal confidence scalar (C):
    // Our goal is to ensure the HMM's emission probability (E) exactly matches
    // the true posterior probability (W) derived from the input genotype likelihoods.
    //
    // The HMM emission logic uses linear interpolation between the high-confidence
    // model and a uniform random-guess floor (0.5 for biallelic/diploid logic):
    // E = (C/255) * (1 - epsilon) + (1 - C/255) * 0.5
    //
    // Assuming epsilon (model mismatch) is negligible for the mapping derivation:
    // E = (C/255) * 1.0 + (1 - C/255) * 0.5
    // E = (C/255) * 0.5 + 0.5
    //
    // We solve for C by setting E = W (the true Bayesian posterior):
    // W = (C/255) * 0.5 + 0.5
    // W - 0.5 = (C/255) * 0.5
    // 2 * (W - 0.5) = C/255
    // 2W - 1 = C/255
    // C = 255 * (2W - 1)
    //
    // Proof of optimality:
    // This formulation is optimal because it creates a mathematically exact alignment
    // between the data likelihoods and the HMM's internal state. It replaces
    // the previous heuristic curve-fit with a principled Bayesian mapping, ensuring
    // that a genotype with probability W is weighted correctly relative to the
    // transition priors in the HMM.

    // 1. Compute true posterior W using softmax (with max subtraction for stability)
    // W = 10^(GL_call - GL_max) / sum(10^(GL_i - GL_max))

    let numerator = 10.0f64.powf(called_gl - max_gl);
    let denominator: f64 = gls.iter().map(|&gl| 10.0f64.powf(gl - max_gl)).sum();

    if denominator <= 0.0 {
        return None;
    }

    let w = numerator / denominator;

    // 2. Map W to Confidence
    // If w <= 0.5, the call is indistinguishable from (or worse than) the alternative/random guess.
    let confidence = if w <= 0.5 {
        0
    } else {
        // Map (0.5, 1.0] -> (0, 255]
        (255.0 * (2.0 * w - 1.0)).round() as u8
    };

    Some(confidence)
}

/// VCF file writer
pub struct VcfWriter {
    writer: Box<dyn Write + Send>,
    samples: Arc<Samples>,
}

impl VcfWriter {
    /// Create a new VCF writer
    pub fn create(path: &Path, samples: Arc<Samples>) -> Result<Self> {
        let file = File::create(path)?;

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        let writer: Box<dyn Write + Send> = match ext {
            "bgz" | "bgzf" => Box::new(BufWriter::new(bgzf_io::Writer::new(file))),
            // Use BGZF for .gz so downstream tools (e.g. bcftools) can index .vcf.gz.
            "gz" => Box::new(BufWriter::new(bgzf_io::Writer::new(file))),
            _ => Box::new(BufWriter::new(file)),
        };

        Ok(Self { writer, samples })
    }

    /// Write VCF header for phased output
    pub fn write_header<Space>(&mut self, markers: &Markers<Space>) -> Result<()> {
        info_span!("vcf_write_header")
            .in_scope(|| self.write_header_extended(markers, false, false, false))
    }

    /// Write VCF header with optional GP/AP fields
    ///
    /// # Arguments
    /// * `markers` - Marker metadata
    /// * `imputed` - Include imputation INFO fields (DR2, AF, IMP)
    /// * `include_gp` - Include GP (genotype probabilities) FORMAT field
    /// * `include_ap` - Include AP (allele probabilities) FORMAT field
    pub fn write_header_extended<Space>(
        &mut self,
        markers: &Markers<Space>,
        imputed: bool,
        include_gp: bool,
        include_ap: bool,
    ) -> Result<()> {
        // Write file format
        writeln!(self.writer, "##fileformat=VCFv4.2")?;

        // Write contig lines
        for chrom in markers.chrom_names() {
            writeln!(self.writer, "##contig=<ID={}>", chrom)?;
        }

        // Write INFO lines for imputation
        if imputed {
            writeln!(
                self.writer,
                "##INFO=<ID=DR2,Number=A,Type=Float,Description=\"Dosage R-squared: estimated squared correlation between estimated REF dose and true REF dose\">"
            )?;
            writeln!(
                self.writer,
                "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Estimated ALT Allele Frequencies\">"
            )?;
            writeln!(
                self.writer,
                "##INFO=<ID=IMP,Number=0,Type=Flag,Description=\"Imputed marker\">"
            )?;
        }

        // Write FORMAT lines
        writeln!(
            self.writer,
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">"
        )?;
        if imputed {
            writeln!(
                self.writer,
                "##FORMAT=<ID=DS,Number=A,Type=Float,Description=\"Estimated ALT allele dosage\">"
            )?;
        }
        if include_gp {
            writeln!(
                self.writer,
                "##FORMAT=<ID=GP,Number=G,Type=Float,Description=\"Estimated Posterior Probabilities for Genotypes 0/0, 0/1 and 1/1\">"
            )?;
        }
        if include_ap {
            writeln!(
                self.writer,
                "##FORMAT=<ID=AP1,Number=A,Type=Float,Description=\"Estimated ALT Allele Probability for Haplotype 1\">"
            )?;
            writeln!(
                self.writer,
                "##FORMAT=<ID=AP2,Number=A,Type=Float,Description=\"Estimated ALT Allele Probability for Haplotype 2\">"
            )?;
        }

        // Write header line
        write!(
            self.writer,
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
        )?;
        for sample in self.samples.ids() {
            write!(self.writer, "\t{}", sample)?;
        }
        writeln!(self.writer)?;

        Ok(())
    }

    /// Write a genotype matrix (works with any phase state)
    pub fn write_phased<S: PhaseState, Space>(
        &mut self,
        matrix: &GenotypeMatrix<S, Space>,
        start_marker: usize,
        end_marker: usize,
    ) -> Result<()> {
        info_span!("vcf_write_phased", n_markers = end_marker - start_marker).in_scope(|| {
            for m in start_marker..end_marker {
                let marker_idx = MarkerIdx::new(m as u32);
                let marker = matrix.marker(marker_idx);
                let column = matrix.column(marker_idx);

                // Write fixed fields
                write!(
                    self.writer,
                    "{}\t{}\t{}\t{}\t{}\t.\tPASS\t.\tGT",
                    matrix.markers().chrom_name(marker.chrom).unwrap_or("."),
                    marker.pos,
                    marker.id.as_ref().map(|s| s.as_ref()).unwrap_or("."),
                    marker.ref_allele,
                    marker
                        .alt_alleles
                        .iter()
                        .map(|a| a.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )?;

                // Write genotypes
                for s in 0..self.samples.len() {
                    let hap1 = crate::data::SampleIdx::new(s as u32).hap1();
                    let hap2 = crate::data::SampleIdx::new(s as u32).hap2();
                    let a1 = column.get(hap1);
                    let a2 = column.get(hap2);
                    let sep = '|';
                    write!(self.writer, "\t{}{}{}", a1, sep, a2)?;
                }
                writeln!(self.writer)?;
            }

            Ok(())
        })
    }

    /// Write imputed genotypes with STREAMING access - no pre-allocation
    ///
    /// Eliminates O(n_markers * n_samples) flat_dosages allocation by using
    /// closures to access sample-major data directly during write.
    pub fn write_imputed_streaming<Space, F, B, G, H>(
        &mut self,
        markers: &Markers<Space>,
        get_dosage: F,
        get_best_gt: B,
        get_posteriors: Option<G>,
        get_genotype_posteriors: Option<H>,
        quality: &ImputationQuality,
        start: usize,
        end: usize,
        include_gp: bool,
        include_ap: bool,
        telemetry: Option<&Arc<TelemetryBlackboard>>,
    ) -> Result<()>
    where
        Space: Sync,
        F: Fn(usize, usize) -> f32 + Sync,
        B: Fn(usize, usize) -> (u8, u8) + Sync,
        G: Fn(
            usize,
            usize,
        ) -> (
            crate::pipelines::imputation::AllelePosteriors,
            crate::pipelines::imputation::AllelePosteriors,
        ) + Sync,
        H: Fn(usize, usize) -> Option<Vec<f32>> + Sync,
    {
        let n_samples = self.samples.len();

        // Pre-compute format string (same for all markers)
        let format_str = {
            let mut parts = vec!["GT", "DS"];
            if include_gp {
                parts.push("GP");
            }
            if include_ap {
                parts.push("AP1");
                parts.push("AP2");
            }
            parts.join(":")
        };

        // Helper to format float with 4 decimal places using ryu when possible.
        // Falls back to fixed-format for scientific notation to avoid exponent truncation.
        #[inline(always)]
        fn format_f32_4dp<'a>(val: f32, ryu_buf: &'a mut ryu::Buffer) -> std::borrow::Cow<'a, str> {
            if !val.is_finite() {
                return std::borrow::Cow::Borrowed("0.0000");
            }
            let s = ryu_buf.format(val);
            if s.contains('e') || s.contains('E') {
                return std::borrow::Cow::Owned(format!("{:.4}", val));
            }
            if let Some(dot_pos) = s.find('.') {
                let end = (dot_pos + 5).min(s.len());
                std::borrow::Cow::Borrowed(&s[..end])
            } else {
                std::borrow::Cow::Borrowed(s)
            }
        }

        #[inline(always)]
        fn best_gt_from_gp(n_alleles: usize, gp: &[f32]) -> (u8, u8) {
            let mut best = (0u8, 0u8);
            let mut best_prob = -1.0f32;
            let mut idx = 0usize;
            for j in 0..n_alleles {
                for i in 0..=j {
                    let p = gp.get(idx).copied().unwrap_or(0.0);
                    if p > best_prob {
                        best_prob = p;
                        if i == j {
                            best = (i as u8, i as u8);
                        } else {
                            best = (i as u8, j as u8);
                        }
                    }
                    idx += 1;
                }
            }
            best
        }

        let get_posteriors_ref = get_posteriors.as_ref();
        let get_genotype_posteriors_ref = get_genotype_posteriors.as_ref();
        let n_markers = end.saturating_sub(start);
        let mut lines: Vec<String> = vec![String::new(); n_markers];
        lines.par_iter_mut().enumerate().for_each(|(idx, line)| {
            let m = start + idx;
            let marker_idx = MarkerIdx::new(m as u32);
            let marker = markers.marker(marker_idx);
            let n_alleles = 1 + marker.alt_alleles.len();
            let mut line_buf = String::with_capacity(n_samples * 50 + 200);
            let mut ryu_buf = ryu::Buffer::new();

            let stats = quality.get(m);
            let info_field = if let Some(stats) = stats {
                let mut info_str = String::with_capacity(64);
                if n_alleles > 1 {
                    info_str.push_str("DR2=");
                    for a in 1..n_alleles {
                        if a > 1 {
                            info_str.push(',');
                        }
                        let v = format_f32_4dp(stats.dr2(a) as f32, &mut ryu_buf);
                        info_str.push_str(&v);
                    }
                    info_str.push_str(";AF=");
                    for a in 1..n_alleles {
                        if a > 1 {
                            info_str.push(',');
                        }
                        let v = format_f32_4dp(stats.allele_freq(a) as f32, &mut ryu_buf);
                        info_str.push_str(&v);
                    }
                }
                if stats.is_imputed {
                    if !info_str.is_empty() {
                        info_str.push(';');
                    }
                    info_str.push_str("IMP");
                }
                if info_str.is_empty() {
                    ".".to_string()
                } else {
                    info_str
                }
            } else {
                ".".to_string()
            };

            use std::fmt::Write;
            write!(
                line_buf,
                "{}\t{}\t{}\t{}\t{}\t.\tPASS\t{}\t{}",
                markers.chrom_name(marker.chrom).unwrap_or("."),
                marker.pos,
                marker.id.as_ref().map(|s| s.as_ref()).unwrap_or("."),
                marker.ref_allele,
                marker
                    .alt_alleles
                    .iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                info_field,
                format_str
            )
            .unwrap();

            for s in 0..n_samples {
                let ds = get_dosage(m, s);
                let posteriors = get_posteriors_ref.map(|f| f(m, s));
                let gp_override = get_genotype_posteriors_ref
                    .and_then(|f| f(m, s))
                    .and_then(|gp| {
                        let expected = n_alleles * (n_alleles + 1) / 2;
                        if gp.len() == expected {
                            Some(gp)
                        } else {
                            None
                        }
                    });
                let (a1, a2) = if let Some(ref gp) = gp_override {
                    best_gt_from_gp(n_alleles, gp)
                } else if let Some((ref p1, ref p2)) = posteriors {
                    if n_alleles <= 2 {
                        let p1_alt = p1.prob(1);
                        let p2_alt = p2.prob(1);
                        let gp00 = (1.0 - p1_alt) * (1.0 - p2_alt);
                        let gp01 =
                            p1_alt * (1.0 - p2_alt) + (1.0 - p1_alt) * p2_alt;
                        let gp11 = p1_alt * p2_alt;
                        if gp01 >= gp00 && gp01 >= gp11 {
                            let p10 = p1_alt * (1.0 - p2_alt);
                            let p01 = (1.0 - p1_alt) * p2_alt;
                            if p10 >= p01 {
                                (1, 0)
                            } else {
                                (0, 1)
                            }
                        } else if gp11 >= gp00 {
                            (1, 1)
                        } else {
                            (0, 0)
                        }
                    } else {
                        let mut best = (0u8, 0u8);
                        let mut best_prob = -1.0f32;
                        for i in 0..n_alleles {
                            for j in i..n_alleles {
                                let p_i1 = p1.prob(i);
                                let p_i2 = p2.prob(i);
                                let p_j1 = p1.prob(j);
                                let p_j2 = p2.prob(j);
                                let prob = if i == j {
                                    p_i1 * p_i2
                                } else {
                                    p_i1 * p_j2 + p_j1 * p_i2
                                };
                                if prob > best_prob {
                                    best_prob = prob;
                                    if i == j {
                                        best = (i as u8, i as u8);
                                    } else {
                                        let p_ij = p_i1 * p_j2;
                                        let p_ji = p_j1 * p_i2;
                                        if p_ij >= p_ji {
                                            best = (i as u8, j as u8);
                                        } else {
                                            best = (j as u8, i as u8);
                                        }
                                    }
                                }
                            }
                        }
                        best
                    }
                } else {
                    get_best_gt(m, s)
                };

                line_buf.push('\t');
                if a1 == 255 {
                    line_buf.push('.');
                } else if a1 < 10 {
                    line_buf.push((b'0' + a1) as char);
                } else {
                    let mut buffer = itoa::Buffer::new();
                    line_buf.push_str(buffer.format(a1));
                }
                line_buf.push('|');
                if a2 == 255 {
                    line_buf.push('.');
                } else if a2 < 10 {
                    line_buf.push((b'0' + a2) as char);
                } else {
                    let mut buffer = itoa::Buffer::new();
                    line_buf.push_str(buffer.format(a2));
                }
                line_buf.push(':');
                let v = format_f32_4dp(ds, &mut ryu_buf);
                line_buf.push_str(&v);

                if include_gp {
                    line_buf.push(':');
                    if let Some(ref gp) = gp_override {
                        let mut first = true;
                        for p in gp.iter() {
                            if !first {
                                line_buf.push(',');
                            }
                            first = false;
                            let v = format_f32_4dp(*p, &mut ryu_buf);
                            line_buf.push_str(&v);
                        }
                    } else if let Some((ref p1, ref p2)) = posteriors {
                        let mut first = true;
                        for i2 in 0..n_alleles {
                            for i1 in 0..=i2 {
                                if !first {
                                    line_buf.push(',');
                                }
                                first = false;
                                let prob = if i1 == i2 {
                                    p1.prob(i1) * p2.prob(i2)
                                } else {
                                    p1.prob(i1) * p2.prob(i2) + p1.prob(i2) * p2.prob(i1)
                                };
                                let v = format_f32_4dp(prob, &mut ryu_buf);
                                line_buf.push_str(&v);
                            }
                        }
                    } else {
                        let n_gp = n_alleles * (n_alleles + 1) / 2;
                        for i in 0..n_gp {
                            if i > 0 {
                                line_buf.push(',');
                            }
                            line_buf.push_str("0.00");
                        }
                    }
                }

                if include_ap {
                    if let Some((ref p1, ref p2)) = posteriors {
                        line_buf.push(':');
                        for a in 1..n_alleles {
                            if a > 1 {
                                line_buf.push(',');
                            }
                            let v = format_f32_4dp(p1.prob(a), &mut ryu_buf);
                            line_buf.push_str(&v);
                        }
                        if n_alleles <= 1 {
                            line_buf.push_str("0.00");
                        }
                        line_buf.push(':');
                        for a in 1..n_alleles {
                            if a > 1 {
                                line_buf.push(',');
                            }
                            let v = format_f32_4dp(p2.prob(a), &mut ryu_buf);
                            line_buf.push_str(&v);
                        }
                        if n_alleles <= 1 {
                            line_buf.push_str("0.00");
                        }
                    } else {
                        let n_ap = n_alleles.saturating_sub(1).max(1);
                        line_buf.push(':');
                        for i in 0..n_ap {
                            if i > 0 {
                                line_buf.push(',');
                            }
                            line_buf.push_str("0.00");
                        }
                        line_buf.push(':');
                        for i in 0..n_ap {
                            if i > 0 {
                                line_buf.push(',');
                            }
                            line_buf.push_str("0.00");
                        }
                    }
                }
            }
            line_buf.push('\n');
            *line = line_buf;
        });

        for line in lines.into_iter() {
            self.writer.write_all(line.as_bytes())?;
            if let Some(bb) = telemetry {
                bb.add_markers(1);
            }
        }
        Ok(())
    }

    /// Flush the writer
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

impl Drop for VcfWriter {
    fn drop(&mut self) {
        match self.flush() {
            Ok(()) => (),
            Err(_) => (),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a genotype field (e.g., "0|1", "0/1", ".")
    ///
    /// This follows the Java VcfRecGTParser behavior:
    /// - If one allele is missing, treat both as missing
    /// - Returns (allele1, allele2, is_phased, is_haploid)
    /// - Missing alleles are represented as 255
    /// - For haploid genotypes, allele2 is set to same as allele1 (for storage compatibility)
    fn parse_genotype(gt: &str) -> Result<(u8, u8, bool, bool)> {
        // Handle completely missing genotypes
        if gt == "." || gt == "./." || gt == ".|." {
            return Ok((255, 255, true, false)); // Missing, treated as phased diploid
        }

        // Determine if phased (| separator) or unphased (/ separator)
        let phased = gt.contains('|');
        let sep = if phased { '|' } else { '/' };

        // Split genotype into alleles without allocation
        let split = gt.split_once(sep);

        // Handle haploid genotypes (single allele, e.g., "0" or "1")
        if split.is_none() {
            let a1 = parse_allele(gt);
            // Store same allele in both positions for storage compatibility,
            // but mark as haploid so phasing pipeline knows to skip
            return Ok((a1, a1, true, true)); // Haploid is always "phased"
        }

        let (left, right) = split.unwrap();
        let a1 = parse_allele(left);
        let a2 = parse_allele(right);

        // Java behavior: if one allele is missing, treat both as missing
        if a1 == 255 || a2 == 255 {
            return Ok((255, 255, false, false));
        }

        Ok((a1, a2, phased, false))
    }

    /// Parse a single allele string to a u8
    /// Returns 255 for missing (.)
    /// Returns 255 with a log warning if allele index exceeds 254 (u8 limitation)
    #[inline]
    fn parse_allele(s: &str) -> u8 {
        if s == "." || s.is_empty() {
            return 255;
        }

        // Fast path for single digit alleles (most common case)
        if s.len() == 1 {
            let c = s.as_bytes()[0];
            if c >= b'0' && c <= b'9' {
                return c - b'0';
            }
        }

        // Multi-digit alleles - check for overflow
        match s.parse::<u16>() {
            Ok(val) if val <= MAX_ALLELE_INDEX => val as u8,
            Ok(val) => {
                log::warn!(
                    "Allele index {} exceeds maximum supported value {}; treating as missing",
                    val,
                    MAX_ALLELE_INDEX
                );
                255
            }
            Err(_) => 255,
        }
    }

    #[test]
    fn test_parse_genotype() {
        // Diploid genotypes: (a1, a2, is_phased, is_haploid)
        assert_eq!(parse_genotype("0|1").unwrap(), (0, 1, true, false));
        assert_eq!(parse_genotype("1|0").unwrap(), (1, 0, true, false));
        assert_eq!(parse_genotype("0/1").unwrap(), (0, 1, false, false));
        assert_eq!(parse_genotype("./.").unwrap(), (255, 255, true, false));
        assert_eq!(parse_genotype(".|.").unwrap(), (255, 255, true, false));
    }

    #[test]
    fn test_parse_genotype_multiallelic() {
        assert_eq!(parse_genotype("0|2").unwrap(), (0, 2, true, false));
        assert_eq!(parse_genotype("1|2").unwrap(), (1, 2, true, false));
    }

    #[test]
    fn test_parse_genotype_haploid() {
        // Haploid genotypes: single allele, duplicated for storage
        assert_eq!(parse_genotype("0").unwrap(), (0, 0, true, true));
        assert_eq!(parse_genotype("1").unwrap(), (1, 1, true, true));
        assert_eq!(parse_genotype(".").unwrap(), (255, 255, true, false)); // Missing is diploid
    }

    #[test]
    fn test_marker_imputation_stats_new() {
        let stats = MarkerImputationStats::new(3);
        assert_eq!(stats.sum_p.len(), 3);
        assert_eq!(stats.sum_p_sq.len(), 3);
        assert_eq!(stats.n_haps, 0);
        assert!(!stats.is_imputed);
    }

    #[test]
    fn test_dr2_perfect_imputation() {
        // Perfect imputation with variation: mix of 0 and 1 probabilities
        let mut stats = MarkerImputationStats::new(2);
        stats.is_imputed = true;

        // 5 samples with ref/ref (p=0 for alt)
        for _ in 0..5 {
            stats.add_sample_biallelic(0.0, 0.0);
        }
        // 5 samples with alt/alt (p=1 for alt)
        for _ in 0..5 {
            stats.add_sample_biallelic(1.0, 1.0);
        }

        // DR2 should be 1.0
        let dr2 = stats.dr2(1);
        assert!(
            dr2 >= 0.99,
            "DR2 should be ~1.0 with certain variable dosages, got {}",
            dr2
        );
    }

    #[test]
    fn test_dr2_uncertain_imputation() {
        // Uncertain imputation: all samples have 50% probability
        let mut stats = MarkerImputationStats::new(2);
        stats.is_imputed = true;

        // Add 10 samples, all uncertain
        for _ in 0..10 {
            stats.add_sample_biallelic(0.5, 0.5);
        }

        // DR2 should be 0 (no variance in p)
        let dr2 = stats.dr2(1);
        assert!(
            dr2 < 0.001,
            "DR2 should be 0 for uncertain calls with no dosage variance, got {}",
            dr2
        );
    }

    #[test]
    fn test_dr2_variable_imputation() {
        // Mixed certainty
        let mut stats = MarkerImputationStats::new(2);
        stats.is_imputed = true;

        // Some certain, some uncertain
        stats.add_sample_biallelic(1.0, 1.0); // Certain alt/alt (p=1)
        stats.add_sample_biallelic(0.0, 0.0); // Certain ref/ref (p=0)
        stats.add_sample_biallelic(0.5, 0.5); // Uncertain (p=0.5)

        let dr2 = stats.dr2(1);
        assert!(
            dr2 > 0.0 && dr2 <= 1.0,
            "DR2 should be between 0 and 1, got {}",
            dr2
        );
    }

    #[test]
    fn test_allele_frequency() {
        let mut stats = MarkerImputationStats::new(2);

        // 3 samples (6 haplotypes):
        // 1. alt/alt (p=1, p=1) -> 2 alt
        // 2. ref/alt (p=0.5, p=0.5) -> 1 alt equivalent
        // 3. ref/ref (p=0, p=0) -> 0 alt
        stats.add_sample_biallelic(1.0, 1.0);
        stats.add_sample_biallelic(0.5, 0.5);
        stats.add_sample_biallelic(0.0, 0.0);

        // Total prob mass = 1 + 1 + 0.5 + 0.5 + 0 + 0 = 3.0
        // Total haplotypes = 6
        // AF = 3.0 / 6 = 0.5

        let af = stats.allele_freq(1);
        assert!((af - 0.5).abs() < 0.01, "AF should be 0.5, got {}", af);
    }

    #[test]
    fn test_imputation_quality_collection() {
        // Create quality tracker for 5 biallelic markers (2 alleles each)
        let mut quality = ImputationQuality::new(&[2, 2, 2, 2, 2]);

        assert_eq!(quality.marker_stats.len(), 5);

        // Test mutability
        if let Some(stats) = quality.get_mut(2) {
            stats.add_sample_biallelic(1.0, 1.0);
            stats.is_imputed = true;
        }

        assert!(quality.get(2).unwrap().is_imputed);
        assert_eq!(quality.get(2).unwrap().n_haps, 2);
    }

    #[test]
    fn test_dictionary_compression_integration() {
        use crate::data::marker::MarkerIdx;
        use crate::data::storage::GenotypeColumn;
        use std::io::Cursor;

        // Create VCF with 70 markers (batch size 64 + 6 remainder)
        // All identical for perfect compression
        // Use explicit \t for tabs
        let mut vcf_data = String::from(
            "##fileformat=VCFv4.2\n##FILTER=<ID=PASS,Description=\"All filters passed\">\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE1\tSAMPLE2\n",
        );

        for i in 1..=70 {
            // All samples 0|0 (pattern 00)
            vcf_data.push_str(&format!(
                "chr1\t{}\t.\tA\tG\t.\tPASS\t.\tGT\t0|0\t0|0\n",
                i * 1000
            ));
        }

        let reader = Box::new(Cursor::new(vcf_data));
        let (mut vcf_reader, reader) = VcfReader::from_reader(reader).unwrap();
        let matrix = vcf_reader.read_all(reader).unwrap();

        assert_eq!(matrix.n_markers(), 70);

        // Check first batch (0..64) - should be dictionary compressed
        if let GenotypeColumn::Dictionary(_, offset) = matrix.column(MarkerIdx::new(0)) {
            assert_eq!(*offset, 0);
        } else {
            panic!("Expected Dictionary column for marker 0");
        }

        if let GenotypeColumn::Dictionary(_, offset) = matrix.column(MarkerIdx::new(63)) {
            assert_eq!(*offset, 63);
        } else {
            panic!("Expected Dictionary column for marker 63");
        }

        // Check remainder (64..70) - 6 markers >= 4, should also be compressed!
        if let GenotypeColumn::Dictionary(_, offset) = matrix.column(MarkerIdx::new(64)) {
            assert_eq!(*offset, 0); // New dictionary, offset resets
        } else {
            panic!("Expected Dictionary column for marker 64");
        }
    }

    #[test]
    fn test_compute_gl_confidence() {
        // Test case 1: High confidence
        // GL: 0, -5, -10 (Log10)
        // Lin: 1.0, 1e-5, 1e-10
        // W approx 1.0
        // Conf approx 255
        let conf = compute_gl_confidence("0,-5,-10", 0, 0).unwrap();
        assert!(conf > 250);

        // Test case 2: Uniform distribution (no confidence)
        // GL: 0, 0, 0
        // Lin: 1, 1, 1 -> Sum = 3
        // W = 1/3 = 0.33
        // W <= 0.5 -> Conf = 0
        let conf = compute_gl_confidence("0,0,0", 0, 0).unwrap();
        assert_eq!(conf, 0);

        // Test case 3: Moderate confidence
        // GL: 0, -0.301, -10 (approx log10(0.5))
        // Lin: 1, 0.5, 0 -> Sum = 1.5
        // W = 1 / 1.5 = 0.666
        // Conf = 255 * (2 * 0.666 - 1) = 255 * 0.333 = 85
        let conf = compute_gl_confidence("0,-0.301,-10", 0, 0).unwrap();
        assert!(conf >= 80 && conf <= 90, "Expected ~85, got {}", conf);

        // Test case 4: Called genotype is NOT the best one
        // GL: -5, 0, -5 (Best is 0/1)
        // Call: 0/0 (GL -5)
        // Lin: 1e-5, 1, 1e-5
        // W_call = 1e-5 / 1.00002 approx 0
        // Conf = 0
        let conf = compute_gl_confidence("-5,0,-5", 0, 0).unwrap();
        assert_eq!(conf, 0);

        // Test case 5: Exactly 50/50 split
        // GL: 0, 0, -10
        // Lin: 1, 1, 0 -> Sum 2
        // W = 1/2 = 0.5
        // Conf = 0 (Random guess floor)
        let conf = compute_gl_confidence("0,0,-10", 0, 0).unwrap();
        assert_eq!(conf, 0);
    }
}
