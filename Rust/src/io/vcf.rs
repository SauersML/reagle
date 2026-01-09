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

use crate::data::haplotype::Samples;
use crate::data::marker::{Allele, Marker, MarkerIdx, Markers};
use crate::data::storage::{GenotypeColumn, GenotypeMatrix, PhaseState, compress_block};
use crate::error::{ReagleError, Result};

/// Imputation quality statistics for a single marker
///
/// Used to calculate DR2 (dosage R-squared) following the Beagle formula:
/// DR2 = Var(d) / Var(X)
///
/// Where:
/// - d = estimated dosage (y1 + 2*y2) = p1 + p2
/// - X = true count (0, 1, or 2)
/// - Var(X) is estimated as Mean(m) - Mean(d)^2
/// - m = second moment = y1 + 4*y2 = p1 + p2 + 2*p1*p2
#[derive(Clone, Debug, Default)]
pub struct MarkerImputationStats {
    /// Sum of dosages (p1 + p2) for each ALT allele
    pub sum_dosages: Vec<f32>,
    /// Sum of squared dosages (p1 + p2)^2 for each ALT allele
    pub sum_dosages_sq: Vec<f32>,
    /// Sum of expected true variance second moments (p1 + p2 + 2*p1*p2)
    pub sum_expected_truth: Vec<f32>,
    /// Number of SAMPLES processed (not haplotypes)
    pub n_samples: usize,
    /// Whether this marker was imputed (not in target genotypes)
    pub is_imputed: bool,
}

impl MarkerImputationStats {
    /// Create new stats for a marker with the given number of alleles
    pub fn new(n_alleles: usize) -> Self {
        Self {
            sum_dosages: vec![0.0; n_alleles],
            sum_dosages_sq: vec![0.0; n_alleles],
            sum_expected_truth: vec![0.0; n_alleles],
            n_samples: 0,
            is_imputed: false,
        }
    }

    /// Add dosage contribution from a diploid sample where estimated = truth.
    ///
    /// For imputed markers, the estimated probabilities are the best guess for truth.
    ///
    /// # Arguments
    /// * `probs1` - Allele probabilities for haplotype 1 (length = n_alleles)
    /// * `probs2` - Allele probabilities for haplotype 2 (length = n_alleles)
    pub fn add_sample(&mut self, probs1: &[f32], probs2: &[f32]) {
        self.add_sample_with_truth(probs1, probs2, probs1, probs2);
    }

    /// Add dosage contribution from a diploid sample with separate truth values.
    ///
    /// This is used for genotyped markers where the "truth" is the known
    /// hard-call genotype, while the "estimated" is the HMM's posterior.
    ///
    /// # Arguments
    /// * `est_probs1` - Estimated allele probabilities for haplotype 1
    /// * `est_probs2` - Estimated allele probabilities for haplotype 2
    /// * `true_probs1` - True allele probabilities for haplotype 1
    /// * `true_probs2` - True allele probabilities for haplotype 2
    pub fn add_sample_with_truth(
        &mut self,
        est_probs1: &[f32],
        est_probs2: &[f32],
        true_probs1: &[f32],
        true_probs2: &[f32],
    ) {
        self.n_samples += 1;
        for a in 1..self.sum_dosages.len() {
            // Estimated dosage calculations from HMM posteriors
            let p1_est = est_probs1.get(a).copied().unwrap_or(0.0);
            let p2_est = est_probs2.get(a).copied().unwrap_or(0.0);
            let dose = p1_est + p2_est;
            let dose_sq = dose * dose;
            self.sum_dosages[a] += dose;
            self.sum_dosages_sq[a] += dose_sq;

            // True variance calculation from hard-called genotypes
            let p1_true = true_probs1.get(a).copied().unwrap_or(0.0);
            let p2_true = true_probs2.get(a).copied().unwrap_or(0.0);
            // m = second moment = E[X^2]
            let m = p1_true + p2_true + 2.0 * p1_true * p2_true;
            self.sum_expected_truth[a] += m;
        }
    }

    /// Calculate DR2 (dosage R-squared) for the specified ALT allele
    ///
    /// DR2 = Var(d) / Var(X)
    pub fn dr2(&self, allele: usize) -> f32 {
        if allele == 0 || allele >= self.sum_dosages.len() || self.n_samples == 0 {
            return 0.0;
        }

        let n = self.n_samples as f32;
        let sum_d = self.sum_dosages[allele];
        // let mean_d = sum_d / n;

        // Var(d) = Mean(d^2) - Mean(d)^2
        //        = (sum_d_sq / n) - (sum_d / n)^2
        //        = (sum_d_sq - sum_d^2/n) / n
        let sum_d_sq = self.sum_dosages_sq[allele];
        let var_d_num = sum_d_sq - (sum_d * sum_d / n);
        
        if var_d_num <= 0.0 {
            return 0.0;
        }
        let var_d = var_d_num / n;

        // Var(X) = Mean(m) - Mean(d)^2
        //        = (sum_m / n) - (sum_d / n)^2
        //        = (sum_m - sum_d^2/n) / n
        let sum_m = self.sum_expected_truth[allele];
        let var_x_num = sum_m - (sum_d * sum_d / n);

        if var_x_num <= 0.0 {
            return 0.0;
        }
        let var_x = var_x_num / n;

        if var_x <= 1e-9 {
            // No variance in truth (monomorphic site) -> undefined correlation
            0.0
        } else {
            (var_d / var_x).clamp(0.0, 1.0)
        }
    }

    /// Calculate estimated allele frequency for the specified ALT allele
    pub fn allele_freq(&self, allele: usize) -> f32 {
        if allele == 0 || allele >= self.sum_dosages.len() || self.n_samples == 0 {
            return 0.0;
        }
        // AF = Mean dosage / 2
        (self.sum_dosages[allele] / self.n_samples as f32) / 2.0
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
    }

    /// Create from a reader
    pub fn from_reader(
        mut reader: Box<dyn BufRead + Send>,
    ) -> Result<(Self, Box<dyn BufRead + Send>)> {
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
        let mut markers = Markers::new();
        let mut columns = Vec::new();

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

            // Parse VCF record
            let (marker, mut alleles, is_phased) =
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
        let matrix = GenotypeMatrix::new_unphased(markers, columns, Arc::clone(&self.samples));
        Ok(matrix)
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
    fn parse_record(
        &mut self,
        line: &str,
        markers: &mut Markers,
        line_num: usize,
    ) -> Result<(Marker, Vec<u8>, bool)> {
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
            info_field
                .split(';')
                .filter_map(|kv| {
                    let parts: Vec<&str> = kv.splitn(2, '=').collect();
                    if parts.len() == 2 && parts[0] == "END" {
                        parts[1].parse().ok()
                    } else {
                        None
                    }
                })
                .next()
        } else {
            None
        };

        // Parse FORMAT to find GT position
        let format = fields[8];
        let gt_idx = format
            .split(':')
            .position(|f| f == "GT")
            .ok_or_else(|| ReagleError::parse(line_num, "No GT field in FORMAT"))?;

        // Parse genotypes
        let n_samples = self.samples.len();
        let mut alleles = Vec::with_capacity(n_samples * 2);
        let mut is_phased = true;

        // Initialize ploidy tracking on first variant if not already done
        if self.sample_ploidy.is_none() {
            self.sample_ploidy = Some(vec![true; n_samples]); // Assume all diploid initially
        }

        for (sample_idx, sample_field) in fields[9..].iter().enumerate() {
            if sample_idx >= n_samples {
                break;
            }

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
        }

        let marker = Marker::with_end(chrom_idx, pos, end_pos, id, ref_allele, alt_alleles);

        Ok((marker, alleles, is_phased))
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

    /// Check if all genotypes read were phased
    ///
    /// Returns true if every genotype in the VCF used the "|" separator,
    /// indicating the data is already phased and doesn't need re-phasing.
    /// Must be called after read_all().
    pub fn was_all_phased(&self) -> bool {
        self.all_phased
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
        self.write_header_extended(markers, false, false, false)
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
        start: usize,
        end: usize,
    ) -> Result<()> {
        for m in start..end {
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
    }

    /// Write imputed genotypes with STREAMING access - no pre-allocation
    ///
    /// Eliminates O(n_markers * n_samples) flat_dosages allocation by using
    /// closures to access sample-major data directly during write.
    pub fn write_imputed_streaming<S, F, G>(
        &mut self,
        matrix: &GenotypeMatrix<S>,
        get_dosage: F,
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
        G: Fn(usize, usize) -> (crate::pipelines::imputation::AllelePosteriors, crate::pipelines::imputation::AllelePosteriors),
    {
        let n_samples = self.samples.len();

        for m in start..end {
            let marker_idx = MarkerIdx::new(m as u32);
            let marker = matrix.marker(marker_idx);
            let n_alleles = 1 + marker.alt_alleles.len();

            let format_str = {
                let mut parts = vec!["GT", "DS"];
                if include_gp { parts.push("GP"); }
                if include_ap { parts.push("AP1"); parts.push("AP2"); }
                parts.join(":")
            };

            let stats = quality.get(m);
            let info_field = if let Some(stats) = stats {
                let mut info_parts = Vec::new();
                if n_alleles > 1 {
                    let dr2_values: Vec<String> = (1..n_alleles).map(|a| format!("{:.2}", stats.dr2(a))).collect();
                    info_parts.push(format!("DR2={}", dr2_values.join(",")));
                    let af_values: Vec<String> = (1..n_alleles).map(|a| format!("{:.4}", stats.allele_freq(a))).collect();
                    info_parts.push(format!("AF={}", af_values.join(",")));
                }
                if stats.is_imputed { info_parts.push("IMP".to_string()); }
                if info_parts.is_empty() { ".".to_string() } else { info_parts.join(";") }
            } else { ".".to_string() };

            write!(self.writer, "{}\t{}\t{}\t{}\t{}\t.\tPASS\t{}\t{}",
                matrix.markers().chrom_name(marker.chrom).unwrap_or("."),
                marker.pos,
                marker.id.as_ref().map(|s| s.as_ref()).unwrap_or("."),
                marker.ref_allele,
                marker.alt_alleles.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(","),
                info_field, format_str)?;

            for s in 0..n_samples {
                let ds = get_dosage(m, s);
                let posteriors = get_posteriors.as_ref().map(|f| f(m, s));
                let (a1, a2) = if let Some((ref p1, ref p2)) = posteriors {
                    (p1.max_allele(), p2.max_allele())
                } else { gt_from_dosage(ds) };

                write!(self.writer, "\t{}|{}:{:.2}", a1, a2, ds)?;

                if include_gp {
                    if let Some((ref p1, ref p2)) = posteriors {
                        write!(self.writer, ":")?;
                        let mut first = true;
                        for i2 in 0..n_alleles {
                            for i1 in 0..=i2 {
                                if !first { write!(self.writer, ",")?; }
                                first = false;
                                let prob = if i1 == i2 { p1.prob(i1) * p2.prob(i2) }
                                    else { p1.prob(i1) * p2.prob(i2) + p1.prob(i2) * p2.prob(i1) };
                                write!(self.writer, "{:.2}", prob)?;
                            }
                        }
                    } else {
                        write!(self.writer, ":{}", vec!["0.00"; n_alleles * (n_alleles + 1) / 2].join(","))?;
                    }
                }

                if include_ap {
                    if let Some((ref p1, ref p2)) = posteriors {
                        write!(self.writer, ":")?;
                        let ap1: Vec<String> = (1..n_alleles).map(|a| format!("{:.2}", p1.prob(a))).collect();
                        write!(self.writer, "{}", if ap1.is_empty() { "0.00".to_string() } else { ap1.join(",") })?;
                        write!(self.writer, ":")?;
                        let ap2: Vec<String> = (1..n_alleles).map(|a| format!("{:.2}", p2.prob(a))).collect();
                        write!(self.writer, "{}", if ap2.is_empty() { "0.00".to_string() } else { ap2.join(",") })?;
                    } else {
                        let n_ap = n_alleles.saturating_sub(1).max(1);
                        write!(self.writer, ":{}", vec!["0.00"; n_ap].join(","))?;
                        write!(self.writer, ":{}", vec!["0.00"; n_ap].join(","))?;
                    }
                }
            }
            writeln!(self.writer)?;
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

/// Derive hard-call GT from ALT dosage
///
/// This matches Java Beagle's ImputedRecBuilder behavior:
/// - DS < 0.5 → 0|0 (homozygous REF)
/// - 0.5 <= DS < 1.5 → 0|1 (heterozygous)
/// - DS >= 1.5 → 1|1 (homozygous ALT)
#[inline]
fn gt_from_dosage(ds: f32) -> (u8, u8) {
    if ds < 0.5 {
        (0, 0)
    } else if ds < 1.5 {
        (0, 1)
    } else {
        (1, 1)
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
        assert_eq!(stats.sum_dosages.len(), 3);
        assert_eq!(stats.sum_dosages_sq.len(), 3);
        assert_eq!(stats.sum_expected_truth.len(), 3);
        assert_eq!(stats.n_samples, 0);
        assert!(!stats.is_imputed);
    }

    #[test]
    fn test_dr2_perfect_imputation() {
        // Perfect imputation with variation: mix of 0 and 1 dosages, all certain
        let mut stats = MarkerImputationStats::new(2);

        // Add samples with different certain values to create variance
        // 5 samples with ref/ref (dosage 0 for alt)
        for _ in 0..5 {
            stats.add_sample(&[1.0, 0.0], &[1.0, 0.0]);
        }
        // 5 samples with alt/alt (dosage 1 for alt)
        for _ in 0..5 {
            stats.add_sample(&[0.0, 1.0], &[0.0, 1.0]);
        }

        // DR2 should be high when there's variance and certainty
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

        // Add 10 samples, all uncertain
        for _ in 0..10 {
            stats.add_sample(&[0.5, 0.5], &[0.5, 0.5]);
        }

        // DR2 should be low for uncertain calls
        // Here Var(d) = 0 because everyone has same dosage, so DR2=0
        let dr2 = stats.dr2(1);
        assert!(
            dr2 < 0.1,
            "DR2 should be low (0) for uncertain calls with no dosage variance, got {}",
            dr2
        );
    }

    #[test]
    fn test_dr2_variable_imputation() {
        // Mixed certainty
        let mut stats = MarkerImputationStats::new(2);

        // Some certain, some uncertain
        stats.add_sample(&[0.0, 1.0], &[0.0, 1.0]); // Certain alt/alt (dose=2)
        stats.add_sample(&[1.0, 0.0], &[1.0, 0.0]); // Certain ref/ref (dose=0)
        stats.add_sample(&[0.5, 0.5], &[0.5, 0.5]); // Uncertain (dose=1)

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

        // 3 samples with dosages 2, 1, 0 (total 3 out of 6 alleles)
        stats.add_sample(&[0.0, 1.0], &[0.0, 1.0]); // Dosage = 2
        stats.add_sample(&[0.5, 0.5], &[0.5, 0.5]); // Dosage = 1
        stats.add_sample(&[1.0, 0.0], &[1.0, 0.0]); // Dosage = 0

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
        assert_eq!(quality.get(2).unwrap().n_samples, 1);
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
