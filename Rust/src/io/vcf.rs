//! # VCF Reading and Writing
//!
//! Parse VCF/BCF files into `GenotypeMatrix`. Write phased results back to VCF.
//! Uses the `noodles` crate for VCF I/O.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use noodles::bgzf::io as bgzf_io;
use noodles::vcf::Header;
use tracing::info_span;

use crate::data::haplotype::Samples;
use crate::data::marker::{Allele, Marker, MarkerIdx, Markers};
use crate::data::storage::{GenotypeColumn, GenotypeMatrix, PhaseState, compress_block};
use crate::error::{ReagleError, Result};

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
        assert!(self.sum_p.len() == 2, "add_sample_biallelic requires biallelic marker");
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
        if sum == 0.0 {
            return 0.0;
        }

        let sum_sq = self.sum_p_sq[allele];
        let n = self.n_haps as f32;

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
    /// Open a VCF file and read the header
    pub fn open(path: &Path) -> Result<(Self, Box<dyn BufRead + Send>)> {
        info_span!("vcf_open", path = ?path).in_scope(|| {
        let file = File::open(path)?;

        // Check if gzipped
        let is_gzipped = path
            .extension()
            .map(|e| e == "gz" || e == "bgz")
            .unwrap_or(false);

        let reader: Box<dyn BufRead + Send> = if is_gzipped {
            Box::new(BufReader::new(bgzf_io::Reader::new(file)))
        } else {
            Box::new(BufReader::new(file))
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

        Ok((Self {
            samples,
            include_sample_indices: None,
            exclude_marker_ids: None,
            sample_ploidy: None,
            all_phased: true,
        }, reader))
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
        let include_indices: Vec<usize> = self.samples
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
        let mut markers = Markers::new();
        let mut columns = Vec::new();
        // Accumulate per-marker confidence scores (one Vec<u8> per marker, indexed by sample)
        let mut all_confidences: Vec<Vec<u8>> = Vec::new();
        let mut has_any_confidence = false;

        let mut line = String::new();
        let mut line_num = 0usize;

        // Buffers for batch processing (Dictionary Compression)
        const BATCH_SIZE: usize = 64;
        let mut batch_markers: Vec<Marker> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_alleles: Vec<Vec<u8>> = Vec::with_capacity(BATCH_SIZE);
        let mut batch_n_alleles: Vec<usize> = Vec::with_capacity(BATCH_SIZE);

        loop {
            line.clear();
            let bytes_read = reader.read_line(&mut line)?;
            if bytes_read == 0 {
                break;
            }
            line_num += 1;

            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse VCF record (now returns confidence if GL is present)
            let (marker, mut alleles, is_phased, mut confidences) =
                self.parse_record(line, &mut markers, line_num)?;

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
            }

            // Store confidence scores
            if let Some(conf) = confidences {
                has_any_confidence = true;
                all_confidences.push(conf);
            } else if has_any_confidence {
                // If we've seen confidence before but this marker has none, fill with 255
                let n_samples = self.samples.len();
                all_confidences.push(vec![255; n_samples]);
            }

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

        // Return unphased by default - caller should phase if needed
        // The is_phased detection is informational only
        let matrix = if has_any_confidence && all_confidences.len() == columns.len() {
            GenotypeMatrix::new_unphased_with_confidence(
                markers,
                columns,
                Arc::clone(&self.samples),
                all_confidences,
            )
        } else {
            GenotypeMatrix::new_unphased(markers, columns, Arc::clone(&self.samples))
        };
        Ok(matrix)
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
    /// Returns (marker, alleles, is_phased, confidences).
    /// Confidences is Some if the GL field is present, None otherwise.
    fn parse_record(
        &mut self,
        line: &str,
        markers: &mut Markers,
        line_num: usize,
    ) -> Result<(Marker, Vec<u8>, bool, Option<Vec<u8>>)> {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 10 {
            return Err(ReagleError::parse(
                line_num,
                format!("Expected at least 10 fields, got {}", fields.len()),
            ));
        }

        // Parse CHROM
        let chrom_name = fields[0];
        let chrom_idx = markers.add_chrom(chrom_name);

        // Parse POS
        let pos: u32 = fields[1]
            .parse()
            .map_err(|_| ReagleError::parse(line_num, "Invalid POS field"))?;

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

        // Parse INFO field for END tag (field[7])
        // This is important for structural variants and gVCF blocks
        let info_field = fields[7];
        let end_pos: Option<u32> = if info_field != "." {
            // Parse INFO field looking for END=value
            // Avoid Vec allocation by using iterator directly
            info_field
                .split(';')
                .filter_map(|kv| {
                    kv.strip_prefix("END=").and_then(|v| v.parse().ok())
                })
                .next()
        } else {
            None
        };

        // Parse FORMAT to find GT position and optionally GL position
        // Avoid Vec allocation by using position() directly on iterator
        let format = fields[8];
        let gt_idx = format.split(':')
            .position(|f| f == "GT")
            .ok_or_else(|| ReagleError::parse(line_num, "No GT field in FORMAT"))?;
        let gl_idx = format.split(':').position(|f| f == "GL");

        // Parse genotypes
        let n_samples = self.samples.len();
        let mut alleles = Vec::with_capacity(n_samples * 2);
        let mut is_phased = true;
        // Confidence scores (only populated if GL field is present)
        let mut confidences: Option<Vec<u8>> = gl_idx.map(|_| Vec::with_capacity(n_samples));

        // Initialize ploidy tracking on first variant if not already done
        if self.sample_ploidy.is_none() {
            self.sample_ploidy = Some(vec![true; n_samples]); // Assume all diploid initially
        }

        for (sample_idx, sample_field) in fields[9..].iter().enumerate() {
            if sample_idx >= n_samples {
                break;
            }

            // Avoid Vec allocation: use nth() to get specific field directly
            // This is O(n) but n is small (typically ~2-4 fields) and avoids allocation
            let gt_field = sample_field.split(':').nth(gt_idx).unwrap_or("./.");

            // Parse genotype (handle both phased | and unphased /)
            let (a1, a2, phased, is_haploid) = parse_genotype(gt_field)?;

            if !phased {
                is_phased = false;
            }

            // Track haploid samples - once detected as haploid, stays haploid
            if is_haploid {
                if let Some(ref mut ploidy) = self.sample_ploidy {
                    ploidy[sample_idx] = false; // Mark as haploid
                }
            }

            alleles.push(a1);
            alleles.push(a2);

            // Parse GL field if present and compute confidence
            if let Some(gl_i) = gl_idx {
                if let Some(conf_vec) = confidences.as_mut() {
                    let confidence = sample_field.split(':')
                        .nth(gl_i)
                        .and_then(|gl_str| compute_gl_confidence(gl_str, a1, a2))
                        .unwrap_or(255); // Default to full confidence if GL missing/unparseable
                    conf_vec.push(confidence);
                }
            }
        }

        let marker = Marker::with_end(chrom_idx, pos, end_pos, id, ref_allele, alt_alleles);

        Ok((marker, alleles, is_phased, confidences))
    }

    /// Rebuild Samples with detected ploidy information
    ///
    /// Call this after reading all variants to update the Samples struct
    /// with accurate ploidy information detected during parsing.
    pub fn finalize_samples(&mut self) {
        if let Some(ref ploidy) = self.sample_ploidy {
            let sample_ids: Vec<String> = self.samples.ids().iter()
                .map(|s| s.to_string())
                .collect();
            self.samples = Arc::new(Samples::from_ids_with_ploidy(sample_ids, ploidy.clone()));
        }
    }

}

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

    // Split genotype into alleles
    let parts: Vec<&str> = gt.split(sep).collect();

    // Handle haploid genotypes (single allele, e.g., "0" or "1")
    if parts.len() == 1 {
        let a1 = parse_allele(parts[0]);
        // Store same allele in both positions for storage compatibility,
        // but mark as haploid so phasing pipeline knows to skip
        return Ok((a1, a1, true, true)); // Haploid is always "phased"
    }

    // Parse diploid genotypes
    if parts.len() != 2 {
        // Malformed, treat as missing
        return Ok((255, 255, false, false));
    }

    let a1 = parse_allele(parts[0]);
    let a2 = parse_allele(parts[1]);

    // Java behavior: if one allele is missing, treat both as missing
    if a1 == 255 || a2 == 255 {
        return Ok((255, 255, false, false));
    }

    Ok((a1, a2, phased, false))
}

/// Maximum supported allele index (u8 limitation)
/// Alleles beyond this will be treated as missing with a warning
pub const MAX_ALLELE_INDEX: u16 = 254;

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

/// Compute genotype confidence from GL field.
///
/// GL field contains log10 likelihoods for each possible genotype.
/// For diploid biallelic: GL = P(0/0), P(0/1), P(1/1)
///
/// Returns confidence (0-255) for the called genotype.
/// Low confidence (uniform GLs like -0.48,-0.48,-0.48) returns low values.
/// High confidence (one GL much higher than others) returns high values.
///
/// # Arguments
/// * `gl_str` - GL field value, e.g., "-0.48,-0.48,-0.48" or "0,-5,-10"
/// * `a1` - First called allele (0=ref, 1+=alt)
/// * `a2` - Second called allele
fn compute_gl_confidence(gl_str: &str, a1: u8, a2: u8) -> Option<u8> {
    // Skip missing values
    if gl_str.is_empty() || gl_str == "." {
        return None;
    }

    // Parse GL values
    let gls: Vec<f64> = gl_str
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();

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

    // Find max GL
    let max_gl = gls.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // If called genotype has max GL, compute confidence based on gap to second-best
    // If called genotype doesn't have max GL, confidence is low

    // Sort GLs descending
    let mut sorted_gls = gls.clone();
    sorted_gls.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Gap between best and second-best GL (in log10 units)
    let gl_gap = if sorted_gls.len() >= 2 {
        sorted_gls[0] - sorted_gls[1]
    } else {
        0.0
    };

    // Check if called genotype is the most likely
    let is_best_call = (called_gl - max_gl).abs() < 0.001;

    // Compute confidence:
    // - Uniform GLs (gap ≈ 0) -> VERY low confidence (call is random)
    // - Clear winner (gap > 1.5) -> high confidence
    // - If call doesn't match best GL -> very low confidence
    //
    // Note: Uniform GLs mean the called genotype is essentially random.
    // Using 50% confidence (the old formula) causes the HMM to incorrectly
    // penalize states that match the true allele when the call is wrong.
    // This destroys imputation accuracy at rare variants.

    let confidence = if !is_best_call {
        // Called genotype is not the most likely - very uncertain
        (10.0 * (1.0 + (called_gl - max_gl))) as u8
    } else {
        // Convert GL gap to confidence using exponential scaling
        // GL gap of 0 (uniform) -> confidence ~25 (0.1) - almost no weight
        // GL gap of 0.5 -> confidence ~90 (0.35)
        // GL gap of 1.0 -> confidence ~180 (0.7)
        // GL gap of 1.5 -> confidence ~230 (0.9)
        // GL gap of 2+ -> confidence ~255 (1.0)
        //
        // Formula: conf = 255 * (1 - exp(-1.5 * gap))
        // This ensures uniform GLs contribute almost nothing to the HMM
        let conf = 255.0 * (1.0 - (-1.5 * gl_gap).exp());
        conf.min(255.0).max(1.0) as u8
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

        let is_gzipped = path
            .extension()
            .map(|e| e == "gz" || e == "bgz")
            .unwrap_or(false);

        let writer: Box<dyn Write + Send> = if is_gzipped {
            Box::new(BufWriter::new(bgzf_io::Writer::new(file)))
        } else {
            Box::new(BufWriter::new(file))
        };

        Ok(Self { writer, samples })
    }

    /// Write VCF header for phased output
    pub fn write_header(&mut self, markers: &Markers) -> Result<()> {
        info_span!("vcf_write_header").in_scope(|| {
        self.write_header_extended(markers, false, false, false)
        })
    }

    /// Write VCF header with optional GP/AP fields
    ///
    /// # Arguments
    /// * `markers` - Marker metadata
    /// * `imputed` - Include imputation INFO fields (DR2, AF, IMP)
    /// * `include_gp` - Include GP (genotype probabilities) FORMAT field
    /// * `include_ap` - Include AP (allele probabilities) FORMAT field
    pub fn write_header_extended(
        &mut self,
        markers: &Markers,
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
    pub fn write_phased<S: PhaseState>(
        &mut self,
        matrix: &GenotypeMatrix<S>,
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
                write!(self.writer, "\t{}|{}", a1, a2)?;
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
    pub fn write_imputed_streaming<S, F, B, G>(
        &mut self,
        matrix: &GenotypeMatrix<S>,
        get_dosage: F,
        get_best_gt: B,
        get_posteriors: Option<G>,
        quality: &ImputationQuality,
        start: usize,
        end: usize,
        include_gp: bool,
        include_ap: bool,
    ) -> Result<()>
    where
        S: PhaseState,
        F: Fn(usize, usize) -> f32,
        B: Fn(usize, usize) -> (u8, u8),
        G: Fn(usize, usize) -> (crate::pipelines::imputation::AllelePosteriors, crate::pipelines::imputation::AllelePosteriors),
    {
        let n_samples = self.samples.len();

        // Pre-compute format string (same for all markers)
        let format_str = {
            let mut parts = vec!["GT", "DS"];
            if include_gp { parts.push("GP"); }
            if include_ap { parts.push("AP1"); parts.push("AP2"); }
            parts.join(":")
        };

        // Pre-allocate line buffer (estimate ~50 bytes per sample)
        let mut line_buf = String::with_capacity(n_samples * 50 + 200);
        // Buffer for ryu float formatting
        let mut ryu_buf = ryu::Buffer::new();

        // Helper to format float with 4 decimal places using ryu
        #[inline(always)]
        fn format_f32_4dp(val: f32, ryu_buf: &mut ryu::Buffer) -> &str {
            if !val.is_finite() {
                return "0.0000";
            }
            // ryu formats with full precision, we need to truncate
            let s = ryu_buf.format(val);
            // Find decimal point and truncate after 4 digits
            if let Some(dot_pos) = s.find('.') {
                let end = (dot_pos + 5).min(s.len());
                &s[..end]
            } else {
                s
            }
        }

        for m in start..end {
            line_buf.clear();
            let marker_idx = MarkerIdx::new(m as u32);
            let marker = matrix.marker(marker_idx);
            let n_alleles = 1 + marker.alt_alleles.len();

            // Build INFO field
            let stats = quality.get(m);
            let info_field = if let Some(stats) = stats {
                let mut info_str = String::with_capacity(64);
                if n_alleles > 1 {
                    info_str.push_str("DR2=");
                    for a in 1..n_alleles {
                        if a > 1 { info_str.push(','); }
                        info_str.push_str(format_f32_4dp(stats.dr2(a) as f32, &mut ryu_buf));
                    }
                    info_str.push_str(";AF=");
                    for a in 1..n_alleles {
                        if a > 1 { info_str.push(','); }
                        info_str.push_str(format_f32_4dp(stats.allele_freq(a) as f32, &mut ryu_buf));
                    }
                }
                if stats.is_imputed {
                    if !info_str.is_empty() { info_str.push(';'); }
                    info_str.push_str("IMP");
                }
                if info_str.is_empty() { ".".to_string() } else { info_str }
            } else { ".".to_string() };

            // Write fixed fields using line buffer
            use std::fmt::Write;
            write!(line_buf, "{}\t{}\t{}\t{}\t{}\t.\tPASS\t{}\t{}",
                matrix.markers().chrom_name(marker.chrom).unwrap_or("."),
                marker.pos,
                marker.id.as_ref().map(|s| s.as_ref()).unwrap_or("."),
                marker.ref_allele,
                marker.alt_alleles.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(","),
                info_field, format_str).unwrap();

            for s in 0..n_samples {
                let ds = get_dosage(m, s);
                let posteriors = get_posteriors.as_ref().map(|f| f(m, s));
                let (a1, a2) = if let Some((ref p1, ref p2)) = posteriors {
                    (p1.max_allele(), p2.max_allele())
                } else { get_best_gt(m, s) };

                // Format: \t{a1}|{a2}:{ds}
                line_buf.push('\t');
                line_buf.push((b'0' + a1) as char);
                line_buf.push('|');
                line_buf.push((b'0' + a2) as char);
                line_buf.push(':');
                line_buf.push_str(format_f32_4dp(ds, &mut ryu_buf));

                if include_gp {
                    line_buf.push(':');
                    if let Some((ref p1, ref p2)) = posteriors {
                        let mut first = true;
                        for i2 in 0..n_alleles {
                            for i1 in 0..=i2 {
                                if !first { line_buf.push(','); }
                                first = false;
                                let prob = if i1 == i2 { p1.prob(i1) * p2.prob(i2) }
                                    else { p1.prob(i1) * p2.prob(i2) + p1.prob(i2) * p2.prob(i1) };
                                line_buf.push_str(format_f32_4dp(prob, &mut ryu_buf));
                            }
                        }
                    } else {
                        // n_alleles * (n_alleles + 1) / 2 zeros
                        let n_gp = n_alleles * (n_alleles + 1) / 2;
                        for i in 0..n_gp {
                            if i > 0 { line_buf.push(','); }
                            line_buf.push_str("0.00");
                        }
                    }
                }

                if include_ap {
                    if let Some((ref p1, ref p2)) = posteriors {
                        line_buf.push(':');
                        for a in 1..n_alleles {
                            if a > 1 { line_buf.push(','); }
                            line_buf.push_str(format_f32_4dp(p1.prob(a), &mut ryu_buf));
                        }
                        if n_alleles <= 1 { line_buf.push_str("0.00"); }
                        line_buf.push(':');
                        for a in 1..n_alleles {
                            if a > 1 { line_buf.push(','); }
                            line_buf.push_str(format_f32_4dp(p2.prob(a), &mut ryu_buf));
                        }
                        if n_alleles <= 1 { line_buf.push_str("0.00"); }
                    } else {
                        let n_ap = n_alleles.saturating_sub(1).max(1);
                        line_buf.push(':');
                        for i in 0..n_ap {
                            if i > 0 { line_buf.push(','); }
                            line_buf.push_str("0.00");
                        }
                        line_buf.push(':');
                        for i in 0..n_ap {
                            if i > 0 { line_buf.push(','); }
                            line_buf.push_str("0.00");
                        }
                    }
                }
            }
            line_buf.push('\n');

            // Single write for entire line
            self.writer.write_all(line_buf.as_bytes())?;
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
        let _ = self.flush();
    }
}



#[cfg(test)]
mod tests {
    use super::*;

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
            stats.add_sample(&[1.0, 0.0], &[1.0, 0.0]);
        }
        // 5 samples with alt/alt (p=1 for alt)
        for _ in 0..5 {
            stats.add_sample(&[0.0, 1.0], &[0.0, 1.0]);
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
            stats.add_sample(&[0.5, 0.5], &[0.5, 0.5]);
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
        stats.add_sample(&[0.0, 1.0], &[0.0, 1.0]); // Certain alt/alt (p=1)
        stats.add_sample(&[1.0, 0.0], &[1.0, 0.0]); // Certain ref/ref (p=0)
        stats.add_sample(&[0.5, 0.5], &[0.5, 0.5]); // Uncertain (p=0.5)

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
        stats.add_sample(&[0.0, 1.0], &[0.0, 1.0]);
        stats.add_sample(&[0.5, 0.5], &[0.5, 0.5]);
        stats.add_sample(&[1.0, 0.0], &[1.0, 0.0]);

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
            stats.add_sample(&[0.0, 1.0], &[0.0, 1.0]);
            stats.is_imputed = true;
        }

        assert!(quality.get(2).unwrap().is_imputed);
        assert_eq!(quality.get(2).unwrap().n_haps, 2);
    }
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
