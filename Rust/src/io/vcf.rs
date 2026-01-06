//! # VCF Reading and Writing
//!
//! Parse VCF/BCF files into `GenotypeMatrix`. Write phased results back to VCF.
//! Uses the `noodles` crate for VCF I/O.

use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::Arc;

use noodles::bgzf;
use noodles::vcf::Header;

use crate::data::haplotype::Samples;
use crate::data::marker::{Allele, Marker, MarkerIdx, Markers};
use crate::data::storage::{GenotypeColumn, GenotypeMatrix, compress_block};
use crate::error::{ReagleError, Result};

/// Imputation quality statistics for a single marker
///
/// Used to calculate DR2 (dosage R-squared) following the Beagle formula.
/// This matches Java ImputedRecBuilder's approach.
#[derive(Clone, Debug, Default)]
pub struct MarkerImputationStats {
    /// Sum of allele probabilities (dosages) for each ALT allele
    pub sum_al_probs: Vec<f32>,
    /// Sum of squared allele probabilities for each ALT allele
    pub sum_al_probs2: Vec<f32>,
    /// Number of haplotypes processed
    pub n_haps: usize,
    /// Whether this marker was imputed (not in target genotypes)
    pub is_imputed: bool,
}

impl MarkerImputationStats {
    /// Create new stats for a marker with the given number of alleles
    pub fn new(n_alleles: usize) -> Self {
        Self {
            sum_al_probs: vec![0.0; n_alleles],
            sum_al_probs2: vec![0.0; n_alleles],
            n_haps: 0,
            is_imputed: false,
        }
    }

    /// Add dosage contribution from a diploid sample
    ///
    /// # Arguments
    /// * `probs1` - Allele probabilities for haplotype 1 (length = n_alleles)
    /// * `probs2` - Allele probabilities for haplotype 2 (length = n_alleles)
    pub fn add_sample(&mut self, probs1: &[f32], probs2: &[f32]) {
        self.n_haps += 2;
        for a in 1..self.sum_al_probs.len() {
            let dose =
                probs1.get(a).copied().unwrap_or(0.0) + probs2.get(a).copied().unwrap_or(0.0);
            let dose2 = probs1.get(a).copied().unwrap_or(0.0).powi(2)
                + probs2.get(a).copied().unwrap_or(0.0).powi(2);
            self.sum_al_probs[a] += dose;
            self.sum_al_probs2[a] += dose2;
        }
    }

    /// Add dosage contribution from a haploid sample
    pub fn add_haploid(&mut self, probs: &[f32]) {
        self.n_haps += 1;
        for a in 1..self.sum_al_probs.len() {
            let dose = probs.get(a).copied().unwrap_or(0.0);
            self.sum_al_probs[a] += dose;
            self.sum_al_probs2[a] += dose.powi(2);
        }
    }

    /// Calculate DR2 (dosage R-squared) for the specified ALT allele
    ///
    /// DR2 estimates the squared correlation between estimated and true dosages.
    /// Formula follows Java ImputedRecBuilder.r2():
    /// ```text
    /// meanTerm = sum^2 / n_haps
    /// num = sum2 - meanTerm
    /// den = sum - meanTerm
    /// r2 = num / den (clamped to [0, 1])
    /// ```
    pub fn dr2(&self, allele: usize) -> f32 {
        if allele == 0 || allele >= self.sum_al_probs.len() || self.n_haps == 0 {
            return 0.0;
        }

        let sum = self.sum_al_probs[allele];
        if sum == 0.0 {
            return 0.0;
        }

        let sum2 = self.sum_al_probs2[allele];
        let mean_term = sum * sum / self.n_haps as f32;
        let num = sum2 - mean_term;
        let den = sum - mean_term;

        if num <= 0.0 || den <= 0.0 {
            0.0
        } else {
            (num / den).clamp(0.0, 1.0)
        }
    }

    /// Calculate allele frequency for the specified ALT allele
    pub fn allele_freq(&self, allele: usize) -> f32 {
        if allele == 0 || allele >= self.sum_al_probs.len() || self.n_haps == 0 {
            return 0.0;
        }
        self.sum_al_probs[allele] / self.n_haps as f32
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

    /// Create with uniform number of alleles (biallelic)
    pub fn new_biallelic(n_markers: usize) -> Self {
        Self {
            marker_stats: vec![MarkerImputationStats::new(2); n_markers],
        }
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
    /// The VCF header
    header: Header,
    /// Sample information
    samples: Arc<Samples>,
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
            Box::new(BufReader::new(bgzf::Reader::new(file)))
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

        Ok((Self { header, samples }, reader))
    }

    /// Get samples
    pub fn samples(&self) -> &Samples {
        &self.samples
    }

    /// Get samples Arc
    pub fn samples_arc(&self) -> Arc<Samples> {
        Arc::clone(&self.samples)
    }

    /// Get header
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Read all records into a GenotypeMatrix
    pub fn read_all(&mut self, mut reader: Box<dyn BufRead + Send>) -> Result<GenotypeMatrix> {
        let mut markers = Markers::new();
        let mut columns = Vec::new();
        let mut is_phased = true;

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
            let (marker, alleles, record_phased) =
                self.parse_record(line, &mut markers, line_num)?;

            if !record_phased {
                is_phased = false;
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

        let matrix = GenotypeMatrix::new(markers, columns, Arc::clone(&self.samples), is_phased);
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
        let end_pos = parse_info_end(fields[7], pos, &ref_allele);

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

        for (sample_idx, sample_field) in fields[9..].iter().enumerate() {
            if sample_idx >= n_samples {
                break;
            }

            let gt_field = sample_field.split(':').nth(gt_idx).unwrap_or("./.");

            // Parse genotype (handle both phased | and unphased /)
            let (a1, a2, phased) = parse_genotype(gt_field)?;

            if !phased {
                is_phased = false;
            }

            alleles.push(a1);
            alleles.push(a2);
        }

        let marker = Marker::with_end(chrom_idx, pos, end_pos, id, ref_allele, alt_alleles);

        Ok((marker, alleles, is_phased))
    }
}

/// Parse END tag from INFO field
///
/// Looks for "END=<number>" in the INFO field for structural variants and gVCF blocks.
/// If not found, returns pos + ref_length - 1 (standard VCF behavior).
///
/// # Arguments
/// * `info` - The INFO field string
/// * `pos` - The POS field value
/// * `ref_allele` - The reference allele (to compute default end)
fn parse_info_end(info: &str, pos: u32, ref_allele: &Allele) -> u32 {
    // Default end: pos + ref_length - 1
    let default_end = pos + ref_allele.len().saturating_sub(1) as u32;

    // Check for empty INFO
    if info.is_empty() || info == "." {
        return default_end;
    }

    // Look for END= tag
    for field in info.split(';') {
        if field.starts_with("END=") {
            if let Ok(end) = field[4..].parse::<u32>() {
                return end;
            }
        }
    }

    default_end
}

/// Parse a genotype field (e.g., "0|1", "0/1", ".")
///
/// This follows the Java VcfRecGTParser behavior:
/// - If one allele is missing, treat both as missing
/// - Returns (allele1, allele2, is_phased)
/// - Missing alleles are represented as 255
fn parse_genotype(gt: &str) -> Result<(u8, u8, bool)> {
    // Handle completely missing genotypes
    if gt == "." || gt == "./." || gt == ".|." {
        return Ok((255, 255, true)); // Missing, treated as phased
    }

    // Determine if phased (| separator) or unphased (/ separator)
    let phased = gt.contains('|');
    let sep = if phased { '|' } else { '/' };

    // Split genotype into alleles
    let parts: Vec<&str> = gt.split(sep).collect();

    // Handle haploid genotypes
    if parts.len() == 1 {
        let a1 = parse_allele(parts[0]);
        return Ok((a1, a1, true)); // Haploid is always "phased"
    }

    // Parse diploid genotypes
    if parts.len() != 2 {
        // Malformed, treat as missing
        return Ok((255, 255, false));
    }

    let a1 = parse_allele(parts[0]);
    let a2 = parse_allele(parts[1]);

    // Java behavior: if one allele is missing, treat both as missing
    if a1 == 255 || a2 == 255 {
        return Ok((255, 255, false));
    }

    Ok((a1, a2, phased))
}

/// Parse a single allele string to a u8
/// Returns 255 for missing (.)
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

    // Multi-digit alleles
    s.parse().unwrap_or(255)
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
            Box::new(BufWriter::new(bgzf::Writer::new(file)))
        } else {
            Box::new(BufWriter::new(file))
        };

        Ok(Self { writer, samples })
    }

    /// Write VCF header for phased output
    pub fn write_header(&mut self, markers: &Markers) -> Result<()> {
        self.write_header_impl(markers, false)
    }

    /// Write VCF header for imputed output (includes DR2, AF, IMP fields)
    pub fn write_header_imputed(&mut self, markers: &Markers) -> Result<()> {
        self.write_header_impl(markers, true)
    }

    fn write_header_impl(&mut self, markers: &Markers, imputed: bool) -> Result<()> {
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

    /// Write a phased genotype matrix
    pub fn write_phased(
        &mut self,
        matrix: &GenotypeMatrix,
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

    /// Write imputed genotypes with dosages and quality metrics (DR2, AF)
    ///
    /// This follows the Java ImputedRecBuilder output format.
    ///
    /// # Arguments
    /// * `matrix` - Genotype matrix with imputed alleles
    /// * `dosages` - Flattened dosage array [marker][sample]
    /// * `quality` - Per-marker imputation quality statistics
    /// * `start` - Start marker index (inclusive)
    /// * `end` - End marker index (exclusive)
    pub fn write_imputed_with_quality(
        &mut self,
        matrix: &GenotypeMatrix,
        dosages: &[f32],
        quality: &ImputationQuality,
        start: usize,
        end: usize,
    ) -> Result<()> {
        let n_samples = self.samples.len();

        for (local_m, m) in (start..end).enumerate() {
            let marker_idx = MarkerIdx::new(m as u32);
            let marker = matrix.marker(marker_idx);
            let column = matrix.column(marker_idx);
            let n_alleles = 1 + marker.alt_alleles.len();

            // Get quality stats for this marker
            let stats = quality.get(m);

            // Build INFO field
            let info_field = if let Some(stats) = stats {
                let mut info_parts = Vec::new();

                // DR2 for each ALT allele
                if n_alleles > 1 {
                    let dr2_values: Vec<String> = (1..n_alleles)
                        .map(|a| format!("{:.2}", stats.dr2(a)))
                        .collect();
                    info_parts.push(format!("DR2={}", dr2_values.join(",")));

                    // AF for each ALT allele
                    let af_values: Vec<String> = (1..n_alleles)
                        .map(|a| format!("{:.4}", stats.allele_freq(a)))
                        .collect();
                    info_parts.push(format!("AF={}", af_values.join(",")));
                }

                // IMP flag if this marker was imputed
                if stats.is_imputed {
                    info_parts.push("IMP".to_string());
                }

                if info_parts.is_empty() {
                    ".".to_string()
                } else {
                    info_parts.join(";")
                }
            } else {
                ".".to_string()
            };

            // Write fixed fields with INFO
            write!(
                self.writer,
                "{}\t{}\t{}\t{}\t{}\t.\tPASS\t{}\tGT:DS",
                matrix.markers().chrom_name(marker.chrom).unwrap_or("."),
                marker.pos,
                marker.id.as_ref().map(|s| s.as_ref()).unwrap_or("."),
                marker.ref_allele,
                marker
                    .alt_alleles
                    .iter()
                    .map(|a| a.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                info_field
            )?;

            // Write genotypes with dosages
            for s in 0..n_samples {
                let hap1 = crate::data::SampleIdx::new(s as u32).hap1();
                let hap2 = crate::data::SampleIdx::new(s as u32).hap2();
                let a1 = column.get(hap1);
                let a2 = column.get(hap2);
                let ds_idx = local_m * n_samples + s;
                let ds = if ds_idx < dosages.len() {
                    dosages[ds_idx]
                } else {
                    (a1 + a2) as f32
                };
                write!(self.writer, "\t{}|{}:{:.2}", a1, a2, ds)?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_genotype() {
        assert_eq!(parse_genotype("0|1").unwrap(), (0, 1, true));
        assert_eq!(parse_genotype("1|0").unwrap(), (1, 0, true));
        assert_eq!(parse_genotype("0/1").unwrap(), (0, 1, false));
        assert_eq!(parse_genotype("./.").unwrap(), (255, 255, true));
        assert_eq!(parse_genotype(".|.").unwrap(), (255, 255, true));
    }

    #[test]
    fn test_parse_genotype_multiallelic() {
        assert_eq!(parse_genotype("0|2").unwrap(), (0, 2, true));
        assert_eq!(parse_genotype("1|2").unwrap(), (1, 2, true));
    }

    #[test]
    fn test_marker_imputation_stats_new() {
        let stats = MarkerImputationStats::new(3);
        assert_eq!(stats.sum_al_probs.len(), 3);
        assert_eq!(stats.sum_al_probs2.len(), 3);
        assert_eq!(stats.n_haps, 0);
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
        let dr2 = stats.dr2(1);
        assert!(
            dr2 < 0.5,
            "DR2 should be low for uncertain calls, got {}",
            dr2
        );
    }

    #[test]
    fn test_dr2_variable_imputation() {
        // Mixed certainty
        let mut stats = MarkerImputationStats::new(2);

        // Some certain, some uncertain
        stats.add_sample(&[0.0, 1.0], &[0.0, 1.0]); // Certain alt/alt
        stats.add_sample(&[1.0, 0.0], &[1.0, 0.0]); // Certain ref/ref
        stats.add_sample(&[0.5, 0.5], &[0.5, 0.5]); // Uncertain

        let dr2 = stats.dr2(1);
        assert!(
            dr2 > 0.0 && dr2 < 1.0,
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
        let mut quality = ImputationQuality::new_biallelic(5);

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
#[test]
fn test_dictionary_compression_integration() {
    use crate::data::marker::MarkerIdx;
    use crate::data::storage::{GenotypeColumn, GenotypeMatrix};
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
