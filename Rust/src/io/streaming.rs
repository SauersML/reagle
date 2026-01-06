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

use noodles::bgzf;

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

/// A window of genotype data ready for processing
#[derive(Clone, Debug)]
pub struct StreamWindow {
    /// Genotype data for this window
    pub genotypes: GenotypeMatrix,
    /// Window number (0-indexed)
    pub window_num: usize,
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
}

/// Buffered marker data for streaming
struct BufferedMarker {
    marker: Marker,
    alleles: Vec<u8>,
    gen_pos: f64,
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
}

impl StreamingVcfReader {
    /// Open a VCF file for streaming
    pub fn open(path: &Path, gen_maps: GeneticMaps, config: StreamingConfig) -> Result<Self> {
        let file = File::open(path)?;

        let is_gzipped = path
            .extension()
            .map(|e| e == "gz" || e == "bgz")
            .unwrap_or(false);

        let reader: Box<dyn BufRead + Send> = if is_gzipped {
            Box::new(BufReader::new(bgzf::Reader::new(file)))
        } else {
            Box::new(BufReader::new(file))
        };

        Self::from_reader(reader, gen_maps, config)
    }

    /// Create from a reader
    pub fn from_reader(
        mut reader: Box<dyn BufRead + Send>,
        gen_maps: GeneticMaps,
        config: StreamingConfig,
    ) -> Result<Self> {
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

        // Parse sample names from header
        let sample_names: Vec<String> = if let Some(header_line) = header_str.lines().last() {
            header_line
                .split('\t')
                .skip(9)
                .map(|s| s.to_string())
                .collect()
        } else {
            Vec::new()
        };

        let samples = Arc::new(Samples::from_ids(sample_names));

        Ok(Self {
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
        })
    }

    /// Get samples
    pub fn samples(&self) -> &Samples {
        &self.samples
    }

    /// Get samples Arc
    pub fn samples_arc(&self) -> Arc<Samples> {
        Arc::clone(&self.samples)
    }

    /// Read the next window of data
    ///
    /// Returns None when all data has been processed
    pub fn next_window(&mut self) -> Result<Option<StreamWindow>> {
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

        // Find end of window
        let mut window_end = 0;
        for (i, m) in self.buffer.iter().enumerate() {
            if m.gen_pos >= target_end_gen || i >= self.config.max_markers {
                break;
            }
            window_end = i + 1;
        }

        // If we haven't found the end and we're not at EOF, need more data
        if window_end == 0 {
            window_end = self.buffer.len();
        }

        let is_last = self.eof && window_end >= self.buffer.len();

        // Determine splice points
        let output_start = 0;
        let output_end = if is_last {
            window_end
        } else {
            // Find overlap point
            let overlap_gen = self
                .buffer
                .get(window_end.saturating_sub(1))
                .map(|m| m.gen_pos - self.config.overlap_cm as f64)
                .unwrap_or(0.0);

            let mut splice = window_end;
            for i in (0..window_end).rev() {
                if self.buffer[i].gen_pos <= overlap_gen {
                    splice = i + 1;
                    break;
                }
            }
            splice
        };

        // Build GenotypeMatrix for this window
        let mut markers = Markers::new();
        let mut columns = Vec::with_capacity(window_end);

        for i in 0..window_end {
            let bm = &self.buffer[i];
            markers.push(bm.marker.clone());
            columns.push(GenotypeColumn::from_alleles(&bm.alleles, 2));
        }

        let genotypes = GenotypeMatrix::new(markers, columns, Arc::clone(&self.samples), true);

        let window = StreamWindow {
            genotypes,
            window_num: self.window_num,
            global_start: self.global_marker_idx,
            global_end: self.global_marker_idx + window_end,
            output_start,
            output_end,
            is_first: self.window_num == 0,
        };

        // Remove processed markers from buffer (keep overlap)
        let keep_from = output_end;
        for _ in 0..keep_from {
            self.buffer.pop_front();
        }

        self.global_marker_idx += keep_from;
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

        // Parse genotypes
        let n_samples = self.samples.len();
        let mut alleles = Vec::with_capacity(n_samples * 2);

        for sample_field in fields[9..].iter().take(n_samples) {
            let gt_field = sample_field.split(':').nth(gt_idx).unwrap_or("./.");

            let (a1, a2) = parse_gt(gt_field);
            alleles.push(a1);
            alleles.push(a2);
        }

        let marker = Marker::new(chrom_idx, pos, id, ref_allele, alt_alleles);

        // Calculate genetic position
        let gen_pos = self.gen_maps.gen_pos(chrom_idx, pos);

        Ok(BufferedMarker {
            marker,
            alleles,
            gen_pos,
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
