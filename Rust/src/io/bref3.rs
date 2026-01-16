//! # BREF3 Format Support
//!
//! Implements reading of BREF3 (Binary Reference Format v3) files
//! for compatibility with Beagle reference panels.
//!
//! BREF3 format structure:
//! - Magic number (4 bytes, big-endian integer: 2055763188)
//! - Program string (modified UTF-8)
//! - Sample IDs array
//! - Data blocks until END_OF_DATA (0)
//! - Index section
//!
//! Each block contains:
//! - nRecs (4 bytes): number of records in block
//! - chrom (modified UTF-8): chromosome name
//! - nSeq (2 bytes): number of unique sequences
//! - hapToSeq (2 bytes * nHaps): haplotype to sequence mapping
//! - Records (either sequence-coded or allele-coded)
//!
//! Reference: Java's bref/Bref3Reader.java, bref/AsIsBref3Writer.java

use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::sync::Arc;

use tracing::info_span;
use noodles::bgzf::io as bgzf_io;

use anyhow::{Context, Result, bail};

use crate::data::haplotype::Samples;
use crate::data::marker::{Allele, Marker, Markers};
use crate::data::storage::phase_state::Phased;
use crate::data::storage::{GenotypeColumn, GenotypeMatrix, SeqCodedBlock, SeqCodedColumn};
use crate::data::ChromIdx;

/// BREF3 magic number (big-endian integer: 2055763188)
const BREF3_MAGIC: i32 = 2055763188;

/// End of data marker
const END_OF_DATA: i32 = 0;

/// Sequence-coded record flag
const SEQ_CODED: u8 = 0;

/// Allele-coded record flag
const ALLELE_CODED: u8 = 1;

/// All 24 permutations of SNV bases (A, C, G, T) for allele code decoding
static SNV_PERMS: [[&str; 4]; 24] = [
    ["A", "C", "G", "T"],
    ["A", "C", "T", "G"],
    ["A", "G", "C", "T"],
    ["A", "G", "T", "C"],
    ["A", "T", "C", "G"],
    ["A", "T", "G", "C"],
    ["C", "A", "G", "T"],
    ["C", "A", "T", "G"],
    ["C", "G", "A", "T"],
    ["C", "G", "T", "A"],
    ["C", "T", "A", "G"],
    ["C", "T", "G", "A"],
    ["G", "A", "C", "T"],
    ["G", "A", "T", "C"],
    ["G", "C", "A", "T"],
    ["G", "C", "T", "A"],
    ["G", "T", "A", "C"],
    ["G", "T", "C", "A"],
    ["T", "A", "C", "G"],
    ["T", "A", "G", "C"],
    ["T", "C", "A", "G"],
    ["T", "C", "G", "A"],
    ["T", "G", "A", "C"],
    ["T", "G", "C", "A"],
];

/// BREF3 Reader for Beagle reference panel files
pub struct Bref3Reader {
    reader: BufReader<File>,
    samples: Samples,
    n_haps: usize,
    markers: Markers,
    chrom_map: std::collections::HashMap<String, ChromIdx>,
}

impl Bref3Reader {
    /// Open a BREF3 file
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).context("Failed to open BREF3 file")?;
        let mut reader = BufReader::new(file);

        let magic = read_be_i32(&mut reader)?;
        if magic != BREF3_MAGIC {
            bail!(
                "Invalid BREF3 magic number: expected {}, got {}",
                BREF3_MAGIC,
                magic
            );
        }

        read_utf8_string(&mut reader)?; // program string (unused)
        let sample_ids = read_string_array(&mut reader)?;
        let n_haps = sample_ids.len() * 2;
        let samples = Samples::from_ids(sample_ids);

        Ok(Self {
            reader,
            samples,
            n_haps,
            markers: Markers::new(),
            chrom_map: std::collections::HashMap::new(),
        })
    }

    /// Read all genotypes into a GenotypeMatrix (phased reference data)
    pub fn read_all(mut self) -> Result<GenotypeMatrix<Phased>> {
        info_span!("bref3_read_all").in_scope(|| {
            let mut columns: Vec<GenotypeColumn> = Vec::new();

        loop {
            let n_recs = read_be_i32(&mut self.reader)?;
            if n_recs == END_OF_DATA {
                break;
            }

            self.read_block(n_recs as usize, &mut columns)?;
        }

        let samples = Arc::new(self.samples);
        Ok(GenotypeMatrix::new_phased(self.markers, columns, samples))
        })
    }

    /// Read a single block of records
    ///
    /// Optimized to use SeqCodedBlock for sequence-coded records, avoiding
    /// expansion from the compact BREF3 format. ~6x memory savings.
    fn read_block(
        &mut self,
        n_recs: usize,
        columns: &mut Vec<GenotypeColumn>,
    ) -> Result<()> {
        let chrom_name = read_utf8_string(&mut self.reader)?;
        let chrom_idx = self.get_or_add_chrom(&chrom_name);

        let n_seq = read_be_u16(&mut self.reader)? as usize;

        let mut hap_to_seq = vec![0u16; self.n_haps];
        for i in 0..self.n_haps {
            hap_to_seq[i] = read_be_u16(&mut self.reader)?;
        }

        // Create SeqCodedBlock for sequence-coded records
        let mut seq_block = SeqCodedBlock::new(hap_to_seq.clone());
        let block_start_idx = columns.len();

        for _ in 0..n_recs {
            let marker = self.read_marker(chrom_idx)?;
            let flag = read_byte(&mut self.reader)?;

            match flag {
                SEQ_CODED => {
                    // Read directly into SeqCodedBlock without expansion
                    let mut seq_to_allele = vec![0u8; n_seq];
                    self.reader.read_exact(&mut seq_to_allele)?;
                    seq_block.push_marker(seq_to_allele);
                    self.markers.push(marker);
                    // Placeholder - will be replaced with SeqCoded below
                    columns.push(GenotypeColumn::Dense(crate::data::storage::DenseColumn::new(0, 1)));
                }
                ALLELE_CODED => {
                    let alleles = self.read_allele_coded_record(marker.n_alleles())?;
                    self.markers.push(marker.clone());
                    let col = GenotypeColumn::from_alleles(&alleles, marker.n_alleles());
                    columns.push(col);
                }
                _ => bail!("Unknown record type flag: {}", flag),
            }
        }

        // Replace placeholder columns with SeqCoded variants
        if seq_block.n_markers() > 0 {
            let block = Arc::new(seq_block);
            let mut seq_marker_idx = 0;
            for col_idx in block_start_idx..columns.len() {
                if matches!(columns[col_idx], GenotypeColumn::Dense(ref d) if d.n_haplotypes() == 0) {
                    columns[col_idx] = GenotypeColumn::SeqCoded(SeqCodedColumn::new(Arc::clone(&block), seq_marker_idx));
                    seq_marker_idx += 1;
                }
            }
        }

        Ok(())
    }

    /// Read marker info
    fn read_marker(&mut self, chrom_idx: ChromIdx) -> Result<Marker> {
        let pos = read_be_i32(&mut self.reader)? as u32;

        let n_ids = read_byte(&mut self.reader)? as usize;
        let id = if n_ids == 0 {
            None
        } else {
            let mut ids = Vec::with_capacity(n_ids);
            for _ in 0..n_ids {
                ids.push(read_utf8_string(&mut self.reader)?);
            }
            Some(Arc::from(ids.join(";")))
        };

        let allele_code = read_byte(&mut self.reader)? as i8;
        let (ref_allele, alt_alleles, end) = if allele_code == -1 {
            let allele_strs = read_string_array(&mut self.reader)?;
            let end_pos = read_be_i32(&mut self.reader)?;
            let end = if end_pos >= 0 {
                Some(end_pos as u32)
            } else {
                None
            };
            parse_alleles(&allele_strs, end)
        } else {
            let n_alleles = 1 + (allele_code & 0b11) as usize;
            let perm_index = (allele_code >> 2) as usize;
            let allele_strs: Vec<String> = SNV_PERMS[perm_index][..n_alleles]
                .iter()
                .map(|s| s.to_string())
                .collect();
            parse_alleles(&allele_strs, None)
        };

        Ok(Marker::with_end(
            chrom_idx,
            pos,
            end,
            id,
            ref_allele,
            alt_alleles,
        ))
    }

    /// Read allele-coded genotype record
    fn read_allele_coded_record(&mut self, n_alleles: usize) -> Result<Vec<u8>> {
        let mut alleles = vec![0u8; self.n_haps];

        for allele_idx in 0..n_alleles {
            let count = read_be_i32(&mut self.reader)?;
            if count == -1 {
                continue;
            }

            for _ in 0..count {
                let hap_idx = read_be_i32(&mut self.reader)? as usize;
                if hap_idx < self.n_haps {
                    alleles[hap_idx] = allele_idx as u8;
                }
            }
        }

        Ok(alleles)
    }

    /// Get or add a chromosome index
    fn get_or_add_chrom(&mut self, name: &str) -> ChromIdx {
        if let Some(&idx) = self.chrom_map.get(name) {
            idx
        } else {
            let idx = self.markers.add_chrom(name);
            self.chrom_map.insert(name.to_string(), idx);
            idx
        }
    }
}

/// Parse allele strings into Allele types
fn parse_alleles(
    allele_strs: &[String],
    end: Option<u32>,
) -> (Allele, Vec<Allele>, Option<u32>) {
    let ref_allele = if allele_strs.is_empty() {
        Allele::Missing
    } else {
        Allele::from_str(&allele_strs[0])
    };

    let alt_alleles: Vec<Allele> = allele_strs
        .iter()
        .skip(1)
        .map(|s| Allele::from_str(s))
        .collect();

    (ref_allele, alt_alleles, end)
}

/// Read a big-endian i32
fn read_be_i32<R: Read>(reader: &mut R) -> Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    Ok(i32::from_be_bytes(buf))
}

/// Read a big-endian u16
fn read_be_u16<R: Read>(reader: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf)?;
    Ok(u16::from_be_bytes(buf))
}

/// Read a single byte
fn read_byte<R: Read>(reader: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf)?;
    Ok(buf[0])
}

/// Read a Java modified UTF-8 string
///
/// Java's DataInput.readUTF() format:
/// - 2 bytes big-endian length (in bytes, not chars)
/// - UTF-8 encoded bytes (with Java's modified encoding for null and high chars)
fn read_utf8_string<R: Read>(reader: &mut R) -> Result<String> {
    let len = read_be_u16(reader)? as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf)?;

    String::from_utf8(buf)
        .context("Invalid UTF-8 in BREF3 string")
}

/// Read a string array (length-prefixed)
fn read_string_array<R: Read>(reader: &mut R) -> Result<Vec<String>> {
    let len = read_be_i32(reader)?;
    if len < 0 {
        return Ok(Vec::new());
    }

    let mut result = Vec::with_capacity(len as usize);
    for _ in 0..len {
        result.push(read_utf8_string(reader)?);
    }
    Ok(result)
}

/// A single block of reference data from streaming BREF3 reading
#[derive(Debug)]
pub struct Bref3Block {
    /// Markers in this block
    pub markers: Markers,
    /// Genotype columns for each marker
    pub columns: Vec<GenotypeColumn>,
    /// End position (bp) of this block
    pub end_pos: u32,
    /// Chromosome name
    pub chrom: String,
}

impl Bref3Block {
    /// Number of markers in this block
    pub fn n_markers(&self) -> usize {
        self.markers.len()
    }
}

/// Streaming BREF3 reader for memory-efficient windowed processing
///
/// Instead of loading the entire reference panel, this reader yields
/// blocks/windows incrementally, allowing the caller to process and
/// discard data as needed.
pub struct StreamingBref3Reader {
    reader: BufReader<File>,
    samples: Arc<Samples>,
    n_haps: usize,
    chrom_map: std::collections::HashMap<String, ChromIdx>,
    /// Whether we've reached end of data
    eof: bool,
}

impl StreamingBref3Reader {
    /// Open a BREF3 file for streaming
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).context("Failed to open BREF3 file")?;
        let mut reader = BufReader::new(file);

        let magic = read_be_i32(&mut reader)?;
        if magic != BREF3_MAGIC {
            bail!(
                "Invalid BREF3 magic number: expected {}, got {}",
                BREF3_MAGIC,
                magic
            );
        }

        read_utf8_string(&mut reader)?; // program string (unused)
        let sample_ids = read_string_array(&mut reader)?;
        let n_haps = sample_ids.len() * 2;
        let samples = Arc::new(Samples::from_ids(sample_ids));

        Ok(Self {
            reader,
            samples,
            n_haps,
            chrom_map: std::collections::HashMap::new(),
            eof: false,
        })
    }

    /// Get the samples as Arc (for sharing without cloning)
    pub fn samples_arc(&self) -> Arc<Samples> {
        Arc::clone(&self.samples)
    }

    /// Get number of haplotypes
    pub fn n_haps(&self) -> usize {
        self.n_haps
    }

    /// Check if we've reached end of data
    pub fn is_eof(&self) -> bool {
        self.eof
    }

    /// Read the next block from the file
    ///
    /// Returns None when all data has been read
    pub fn next_block(&mut self) -> Result<Option<Bref3Block>> {
        if self.eof {
            return Ok(None);
        }

        let n_recs = read_be_i32(&mut self.reader)?;
        if n_recs == END_OF_DATA {
            self.eof = true;
            return Ok(None);
        }

        let n_recs = n_recs as usize;
        let chrom_name = read_utf8_string(&mut self.reader)?;
        let chrom_idx = self.get_or_add_chrom(&chrom_name);

        let n_seq = read_be_u16(&mut self.reader)? as usize;

        let mut hap_to_seq = vec![0u16; self.n_haps];
        for i in 0..self.n_haps {
            hap_to_seq[i] = read_be_u16(&mut self.reader)?;
        }

        // Create SeqCodedBlock for sequence-coded records
        let mut seq_block = SeqCodedBlock::new(hap_to_seq);
        let mut markers = Markers::new();
        markers.add_chrom(&chrom_name);
        let mut columns: Vec<GenotypeColumn> = Vec::with_capacity(n_recs);
        let block_start_idx = 0;

        let mut end_pos = 0u32;

        for _ in 0..n_recs {
            let marker = self.read_marker(chrom_idx)?;
            end_pos = end_pos.max(marker.pos);

            let flag = read_byte(&mut self.reader)?;

            match flag {
                SEQ_CODED => {
                    let mut seq_to_allele = vec![0u8; n_seq];
                    self.reader.read_exact(&mut seq_to_allele)?;
                    seq_block.push_marker(seq_to_allele);
                    markers.push(marker);
                    // Placeholder - will be replaced with SeqCoded below
                    columns.push(GenotypeColumn::Dense(crate::data::storage::DenseColumn::new(0, 1)));
                }
                ALLELE_CODED => {
                    let alleles = self.read_allele_coded_record(marker.n_alleles())?;
                    markers.push(marker.clone());
                    let col = GenotypeColumn::from_alleles(&alleles, marker.n_alleles());
                    columns.push(col);
                }
                _ => bail!("Unknown record type flag: {}", flag),
            }
        }

        // Replace placeholder columns with SeqCoded variants
        if seq_block.n_markers() > 0 {
            let block = Arc::new(seq_block);
            let mut seq_marker_idx = 0;
            for col_idx in block_start_idx..columns.len() {
                if matches!(columns[col_idx], GenotypeColumn::Dense(ref d) if d.n_haplotypes() == 0) {
                    columns[col_idx] = GenotypeColumn::SeqCoded(SeqCodedColumn::new(Arc::clone(&block), seq_marker_idx));
                    seq_marker_idx += 1;
                }
            }
        }

        Ok(Some(Bref3Block {
            markers,
            columns,
            end_pos,
            chrom: chrom_name,
        }))
    }

    /// Read marker info (same as Bref3Reader but uses internal markers)
    fn read_marker(&mut self, chrom_idx: ChromIdx) -> Result<Marker> {
        let pos = read_be_i32(&mut self.reader)? as u32;

        let n_ids = read_byte(&mut self.reader)? as usize;
        let id = if n_ids == 0 {
            None
        } else {
            let mut ids = Vec::with_capacity(n_ids);
            for _ in 0..n_ids {
                ids.push(read_utf8_string(&mut self.reader)?);
            }
            Some(Arc::from(ids.join(";")))
        };

        let allele_code = read_byte(&mut self.reader)? as i8;
        let (ref_allele, alt_alleles, end) = if allele_code == -1 {
            let allele_strs = read_string_array(&mut self.reader)?;
            let end_pos = read_be_i32(&mut self.reader)?;
            let end = if end_pos >= 0 {
                Some(end_pos as u32)
            } else {
                None
            };
            parse_alleles(&allele_strs, end)
        } else {
            let n_alleles = 1 + (allele_code & 0b11) as usize;
            let perm_index = (allele_code >> 2) as usize;
            let allele_strs: Vec<String> = SNV_PERMS[perm_index][..n_alleles]
                .iter()
                .map(|s| s.to_string())
                .collect();
            parse_alleles(&allele_strs, None)
        };

        Ok(Marker::with_end(
            chrom_idx,
            pos,
            end,
            id,
            ref_allele,
            alt_alleles,
        ))
    }

    /// Read allele-coded genotype record
    fn read_allele_coded_record(&mut self, n_alleles: usize) -> Result<Vec<u8>> {
        let mut alleles = vec![0u8; self.n_haps];

        for allele_idx in 0..n_alleles {
            let count = read_be_i32(&mut self.reader)?;
            if count == -1 {
                continue;
            }

            for _ in 0..count {
                let hap_idx = read_be_i32(&mut self.reader)? as usize;
                if hap_idx < self.n_haps {
                    alleles[hap_idx] = allele_idx as u8;
                }
            }
        }

        Ok(alleles)
    }

    /// Get or add a chromosome index
    fn get_or_add_chrom(&mut self, name: &str) -> ChromIdx {
        if let Some(&idx) = self.chrom_map.get(name) {
            idx
        } else {
            let idx = ChromIdx::new(self.chrom_map.len() as u16);
            self.chrom_map.insert(name.to_string(), idx);
            idx
        }
    }
}

/// Configuration for windowed reference loading
#[derive(Clone, Debug)]
/// A window of reference data accumulated from multiple blocks
pub struct RefWindow {
    /// Genotype matrix for this window (phased reference data)
    pub genotypes: GenotypeMatrix<crate::data::storage::phase_state::Phased>,
    /// Global start marker index
    pub global_start: usize,
    /// Global end marker index (exclusive)
    pub global_end: usize,
    /// Output start index within window (after overlap from previous)
    pub output_start: usize,
    /// Output end index within window (before overlap for next)
    pub output_end: usize,
    /// Whether this is the first window
    pub is_first: bool,
    /// Whether this is the last window
    pub is_last: bool,
}

/// Windowed reference panel reader that accumulates blocks into windows
pub struct WindowedBref3Reader {
    inner: StreamingBref3Reader,
    /// Buffer of blocks for the next window
    block_buffer: VecDeque<Bref3Block>,
    /// Current window number
    window_num: usize,
    /// Global marker offset
    global_offset: usize,
}

impl WindowedBref3Reader {
    /// Create a windowed reader with given configuration
    pub fn new(inner: StreamingBref3Reader) -> Self {
        Self {
            inner,
            block_buffer: VecDeque::new(),
            window_num: 0,
            global_offset: 0,
        }
    }

    /// Load reference window for a specific genomic region
    ///
    /// This method loads all reference markers within [start_pos, end_pos] plus flanking markers.
    /// Used to synchronize with target windows based on genomic coordinates
    /// rather than independent window boundaries.
    ///
    /// Includes flanking markers outside the region to prevent reference bias
    /// at window boundaries (HMM needs context to stabilize).
    pub fn load_window_for_region(&mut self, start_pos: u32, end_pos: u32) -> Result<Option<RefWindow>> {
        // Add buffer zone to prevent reference bias at boundaries
        // Use 500 markers to ensure HMM has context to stabilize at boundaries
        const BUFFER_MARKERS: usize = 500;
        // First, drain blocks well before start_pos, but keep one block for pre-buffer context.
        while self.block_buffer.len() > 1 {
            let second = self.block_buffer.get(1);
            if second.map(|b| b.end_pos < start_pos).unwrap_or(false) {
                self.block_buffer.pop_front();
            } else {
                break;
            }
        }

        // Load blocks until we cover end_pos
        while !self.inner.is_eof() {
            let need_more = self.block_buffer.is_empty()
                || self.block_buffer.back().map(|b| b.end_pos < end_pos).unwrap_or(true);

            if !need_more {
                break;
            }

            if let Some(block) = self.inner.next_block()? {
                self.block_buffer.push_back(block);
            } else {
                break;
            }
        }
        // Load one extra block beyond end_pos for trailing buffer, if available.
        if !self.inner.is_eof() {
            if let Some(block) = self.inner.next_block()? {
                self.block_buffer.push_back(block);
            }
        }

        if self.block_buffer.is_empty() {
            return Ok(None);
        }

        // Merge blocks in buffer and then apply marker-count buffering
        let mut all_markers = Markers::new();
        let mut all_columns: Vec<GenotypeColumn> = Vec::new();
        let mut in_range_indices: Vec<usize> = Vec::new();
        let is_first = self.window_num == 0;
        let is_last = self.inner.is_eof();

        for block in &self.block_buffer {
            // Add chromosome if needed
            if all_markers.chrom_names().is_empty()
                || all_markers.chrom_names().last().map(|s| s.as_ref()) != Some(&block.chrom)
            {
                all_markers.add_chrom(&block.chrom);
            }

            for m in 0..block.n_markers() {
                let marker = block.markers.marker(crate::data::marker::MarkerIdx::new(m as u32));
                let idx = all_markers.len();
                all_markers.push(marker.clone());
                all_columns.push(block.columns[m].clone());
                if marker.pos >= start_pos && marker.pos <= end_pos {
                    in_range_indices.push(idx);
                }
            }
        }

        if all_markers.is_empty() || in_range_indices.is_empty() {
            return Ok(None);
        }

        let first_idx = *in_range_indices.first().unwrap_or(&0);
        let last_idx = *in_range_indices.last().unwrap_or(&0);
        let start_idx = first_idx.saturating_sub(BUFFER_MARKERS);
        let end_idx = (last_idx + 1 + BUFFER_MARKERS).min(all_markers.len());

        let mut markers = Markers::new();
        let mut columns: Vec<GenotypeColumn> = Vec::new();
        for idx in start_idx..end_idx {
            let marker = all_markers.marker(crate::data::marker::MarkerIdx::new(idx as u32));
            let chrom_name = all_markers.chrom_name(marker.chrom).unwrap_or(".");
            let chrom_idx = markers.add_chrom(chrom_name);
            let mut m = marker.clone();
            m.chrom = chrom_idx;
            markers.push(m);
            columns.push(all_columns[idx].clone());
        }

        let n_markers = markers.len();
        let output_start = first_idx - start_idx;
        let output_end = output_start + (last_idx + 1 - first_idx);
        let global_start = self.global_offset;
        let global_end = global_start + n_markers;

        self.global_offset = global_end;
        self.window_num += 1;

        // Clear blocks we've processed (keep last one for potential overlap)
        while self.block_buffer.len() > 1 && self.block_buffer.front().map(|b| b.end_pos < end_pos).unwrap_or(false) {
            self.block_buffer.pop_front();
        }

        // Create GenotypeMatrix from markers and columns
        let samples = self.inner.samples_arc();
        let genotypes = GenotypeMatrix::new_phased(markers, columns, samples);

        Ok(Some(RefWindow {
            genotypes,
            global_start,
            global_end,
            output_start,
            output_end,
            is_first,
            is_last,
        }))
    }

}

/// Scan a reference file (BREF3 or VCF) to get all marker metadata without loading genotypes.
pub fn scan_ref_markers(path: &Path, is_bref3: bool) -> Result<Markers> {
    if is_bref3 {
        let mut reader = StreamingBref3Reader::open(path)?;
        let mut all_markers = Markers::new();
        while let Some(block) = reader.next_block()? {
            let chrom_name = reader.chrom_map.iter()
                .find(|(_, &idx)| idx == block.markers.marker(crate::data::marker::MarkerIdx::new(0)).chrom)
                .map(|(s, _)| s.clone())
                .unwrap_or_else(|| ".".to_string());
            let chrom_idx = all_markers.add_chrom(&chrom_name);
            for m in 0..block.n_markers() {
                let mut marker = block.markers.marker(crate::data::marker::MarkerIdx::new(m as u32)).clone();
                marker.chrom = chrom_idx;
                all_markers.push(marker);
            }
        }
        Ok(all_markers)
    } else {
        // For VCF, we need to read the whole file, but we can be smart about it.
        let (_reader, mut file) = crate::io::vcf::VcfReader::open(path)?;
        let mut markers = Markers::new();
        let mut line = String::new();
        loop {
            line.clear();
            if file.read_line(&mut line)? == 0 { break; }
            if line.starts_with('#') { continue; }
            let fields: Vec<&str> = line.split('\t').collect();
            if fields.len() < 5 { continue; }

            let chrom_idx = markers.add_chrom(fields[0]);
            let pos: u32 = fields[1].parse()?;
            let id = if fields[2] == "." { None } else { Some(fields[2].into()) };
            let ref_allele = Allele::from_str(fields[3]);
            let alt_alleles: Vec<Allele> = fields[4].split(',').map(Allele::from_str).collect();
            let marker = Marker::new(chrom_idx, pos, id, ref_allele, alt_alleles);
            markers.push(marker);
        }
        Ok(markers)
    }
}

/// Unified reference panel reader that supports both BREF3 (streaming) and VCF (in-memory)
pub enum RefPanelReader {
    /// Streaming BREF3 reader
    Bref3(WindowedBref3Reader),
    /// In-memory VCF reader
    InMemory(InMemoryRefReader),
    /// Streaming VCF reader
    StreamingVcf(StreamingRefVcfReader),
}

impl RefPanelReader {
    /// Load reference window for a specific genomic region
    pub fn load_window_for_region(&mut self, start_pos: u32, end_pos: u32) -> Result<Option<RefWindow>> {
        match self {
            RefPanelReader::Bref3(r) => r.load_window_for_region(start_pos, end_pos),
            RefPanelReader::InMemory(r) => r.load_window_for_region(start_pos, end_pos),
            RefPanelReader::StreamingVcf(r) => r.load_window_for_region(start_pos, end_pos),
        }
    }
}

/// In-memory reference panel reader for VCF files
///
/// Provides the same interface as WindowedBref3Reader but backed by
/// an in-memory GenotypeMatrix. Used for VCF reference panels.
pub struct InMemoryRefReader {
    genotypes: Arc<GenotypeMatrix<crate::data::storage::phase_state::Phased>>,
    window_num: usize,
}

impl InMemoryRefReader {
    /// Create a new in-memory reader from a loaded reference panel
    pub fn new(genotypes: Arc<GenotypeMatrix<crate::data::storage::phase_state::Phased>>) -> Self {
        Self {
            genotypes,
            window_num: 0,
        }
    }

    /// Load reference window for a specific genomic region
    pub fn load_window_for_region(&mut self, start_pos: u32, end_pos: u32) -> Result<Option<RefWindow>> {
        use crate::data::marker::MarkerIdx;

        let n_markers = self.genotypes.n_markers();
        if n_markers == 0 {
            return Ok(None);
        }

        const BUFFER_MARKERS: usize = 500;

        // Find markers within the position range
        let mut start_idx = None;
        let mut end_idx = None;

        for m in 0..n_markers {
            let marker = self.genotypes.marker(MarkerIdx::new(m as u32));
            if marker.pos >= start_pos && marker.pos <= end_pos {
                if start_idx.is_none() {
                    start_idx = Some(m);
                }
                end_idx = Some(m + 1);
            } else if marker.pos > end_pos {
                break;
            }
        }

        let (start_idx, end_idx) = match (start_idx, end_idx) {
            (Some(s), Some(e)) => (s, e),
            _ => return Ok(None),
        };

        let buffered_start_idx = start_idx.saturating_sub(BUFFER_MARKERS);
        let buffered_end_idx = (end_idx + BUFFER_MARKERS).min(n_markers);

        // Extract markers and columns for this range
        let mut markers = crate::data::marker::Markers::new();
        let mut columns = Vec::new();

        // Add chromosome
        let first_marker = self.genotypes.marker(MarkerIdx::new(start_idx as u32));
        let chrom_name = self.genotypes.markers().chrom_name(first_marker.chrom).expect("Invalid chromosome");
        let window_chrom_idx = markers.add_chrom(chrom_name);

        for m in buffered_start_idx..buffered_end_idx {
            let mut marker = self.genotypes.marker(MarkerIdx::new(m as u32)).clone();
            marker.chrom = window_chrom_idx;
            markers.push(marker);
            columns.push(self.genotypes.column(MarkerIdx::new(m as u32)).clone());
        }

        let output_start = start_idx - buffered_start_idx;
        let output_end = output_start + (end_idx - start_idx);
        let is_first = self.window_num == 0;
        self.window_num += 1;

        let genotypes = GenotypeMatrix::new_phased(markers, columns, self.genotypes.samples_arc());

        Ok(Some(RefWindow {
            genotypes,
            global_start: buffered_start_idx,
            global_end: buffered_end_idx,
            output_start,
            output_end,
            is_first,
            is_last: false, // Can't know without looking ahead
        }))
    }
}

/// A marker buffered for streaming VCF reading
struct RefPanelMarker {
    marker: Marker,
    column: GenotypeColumn,
}

/// Streaming VCF reader for reference panels
pub struct StreamingRefVcfReader {
    reader: Box<dyn BufRead + Send>,
    samples: Arc<Samples>,
    markers: Markers,
    buffer: VecDeque<RefPanelMarker>,
    line_buf: String,
    eof: bool,
}

impl StreamingRefVcfReader {
    /// Open a VCF file for streaming
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)?;
        let is_gzipped = path.extension().map(|e| e == "gz" || e == "bgz").unwrap_or(false);
        let reader: Box<dyn BufRead + Send> = if is_gzipped {
            Box::new(BufReader::new(bgzf_io::Reader::new(file)))
        } else {
            Box::new(BufReader::new(file))
        };
        Self::from_reader(reader)
    }

    /// Create from a reader
    pub fn from_reader(mut reader: Box<dyn BufRead + Send>) -> Result<Self> {
        // Read header
        let mut header_str = String::new();
        let mut line = String::new();
        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 { break; }
            if line.starts_with('#') {
                header_str.push_str(&line);
                if line.starts_with("#CHROM") { break; }
            } else { break; }
        }

        // Parse samples
        let sample_names: Vec<String> = if let Some(header_line) = header_str.lines().last() {
             header_line.split('\t').skip(9).map(|s| s.to_string()).collect()
        } else { Vec::new() };
        let samples = Arc::new(Samples::from_ids(sample_names));

        Ok(Self {
            reader,
            samples,
            markers: Markers::new(),
            buffer: VecDeque::new(),
            line_buf: String::new(),
            eof: false,
        })
    }


    /// Load reference window for a specific genomic region
    pub fn load_window_for_region(&mut self, start_pos: u32, end_pos: u32) -> Result<Option<RefWindow>> {
        const BUFFER_MARKERS: usize = 500;
        // Ensure buffer spans the window and includes BUFFER_MARKERS after end_pos.
        loop {
            while !self.eof {
                let need_more = self.buffer.is_empty()
                    || self.buffer.back().map(|m| m.marker.pos < end_pos).unwrap_or(true);
                if !need_more {
                    break;
                }
                if let Some(marker) = self.read_next_marker()? {
                    self.buffer.push_back(marker);
                }
            }

            if self.buffer.is_empty() {
                return Ok(None);
            }

            let mut last_idx = None;
            for (i, bm) in self.buffer.iter().enumerate() {
                if bm.marker.pos >= start_pos && bm.marker.pos <= end_pos {
                    last_idx = Some(i);
                }
            }

            let last_idx = match last_idx {
                Some(e) => e,
                None => return Ok(None),
            };

            let after = self.buffer.len().saturating_sub(last_idx + 1);
            if after < BUFFER_MARKERS && !self.eof {
                if let Some(marker) = self.read_next_marker()? {
                    self.buffer.push_back(marker);
                }
                continue;
            }

            break;
        }

        let mut first_idx = None;
        let mut last_idx = None;
        for (i, bm) in self.buffer.iter().enumerate() {
            if bm.marker.pos >= start_pos && bm.marker.pos <= end_pos {
                if first_idx.is_none() {
                    first_idx = Some(i);
                }
                last_idx = Some(i);
            }
        }

        let (mut first_idx, mut last_idx) = match (first_idx, last_idx) {
            (Some(s), Some(e)) => (s, e),
            _ => return Ok(None),
        };

        let buffered_start_idx = first_idx.saturating_sub(BUFFER_MARKERS);
        let mut buffered_end_idx = (last_idx + 1 + BUFFER_MARKERS).min(self.buffer.len());

        if buffered_start_idx > 0 {
            for _ in 0..buffered_start_idx {
                self.buffer.pop_front();
            }
            first_idx -= buffered_start_idx;
            last_idx -= buffered_start_idx;
            buffered_end_idx -= buffered_start_idx;
        }

        let mut markers = Markers::new();
        let mut columns = Vec::new();
        let mut found_any = false;

        for (i, bm) in self.buffer.iter().enumerate() {
            if i >= buffered_end_idx {
                break;
            }
            let chrom_name = self.markers.chrom_name(bm.marker.chrom).expect("Invalid chromosome");
            let window_chrom_idx = markers.add_chrom(chrom_name);

            let mut m = bm.marker.clone();
            m.chrom = window_chrom_idx;
            markers.push(m);
            columns.push(bm.column.clone());
            found_any = true;
        }

        if !found_any { return Ok(None); }

        let n_markers = markers.len();
        let genotypes = GenotypeMatrix::new_phased(markers, columns, Arc::clone(&self.samples));
        let output_start = first_idx;
        let output_end = last_idx + 1;

        Ok(Some(RefWindow {
            genotypes,
            global_start: 0,
            global_end: n_markers,
            output_start,
            output_end,
            is_first: false,
            is_last: self.eof && self.buffer.is_empty(),
        }))
    }

    fn read_next_marker(&mut self) -> Result<Option<RefPanelMarker>> {
        loop {
            self.line_buf.clear();
            if self.reader.read_line(&mut self.line_buf)? == 0 {
                self.eof = true;
                return Ok(None);
            }
            let line = self.line_buf.trim().to_string();
            if line.is_empty() || line.starts_with('#') { continue; }
            return self.parse_vcf_line(&line).map(Some);
        }
    }

    fn parse_vcf_line(&mut self, line: &str) -> Result<RefPanelMarker> {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 10 {
             bail!("VCF line has too few fields");
        }

        // Parse CHROM
        let chrom_name = fields[0];
        let chrom_idx = self.markers.add_chrom(chrom_name);

        // Parse POS
        let pos: u32 = fields[1].parse().context("Invalid POS")?;

        // Parse ID
        let id = if fields[2] == "." { None } else { Some(fields[2].into()) };

        // Parse REF
        let ref_allele = Allele::from_str(fields[3]);

        // Parse ALT
        let alt_alleles: Vec<Allele> = fields[4].split(',').map(|a| Allele::from_str(a)).collect();

        let marker = Marker::new(chrom_idx, pos, id, ref_allele, alt_alleles.clone());
        let n_alleles = 1 + alt_alleles.len();

        // Parse genotypes
        let n_samples = self.samples.len();
        let mut alleles = Vec::with_capacity(n_samples * 2);

        // Find GT index
        let format = fields[8];
        let gt_idx = format.split(':').position(|f| f == "GT")
             .ok_or_else(|| anyhow::anyhow!("No GT field in FORMAT"))?;

        if fields.len() < 9 + n_samples {
            bail!("VCF line has fewer samples than header");
        }

        for sample_field in fields[9..].iter().take(n_samples) {
            let gt_field = sample_field.split(':').nth(gt_idx).unwrap_or("./.");
            let (a1, a2) = self.parse_gt_local(gt_field);
            alleles.push(a1);
            alleles.push(a2);
        }

        let column = GenotypeColumn::from_alleles(&alleles, n_alleles);

        Ok(RefPanelMarker { marker, column })
    }

    fn parse_gt_local(&self, gt: &str) -> (u8, u8) {
        if gt == "." || gt == "./." || gt == ".|." { return (255, 255); }
        let sep = if gt.contains('|') { '|' } else { '/' };
        let parts: Vec<&str> = gt.split(sep).collect();
        if parts.len() == 1 {
            let a = self.parse_allele_local(parts[0]);
            return (a, a);
        }
        if parts.len() != 2 { return (255, 255); }
        (self.parse_allele_local(parts[0]), self.parse_allele_local(parts[1]))
    }

    fn parse_allele_local(&self, s: &str) -> u8 {
         if s == "." || s.is_empty() { return 255; }
         if s.len() == 1 {
             let c = s.as_bytes()[0];
             if c >= b'0' && c <= b'9' { return c - b'0'; }
         }
         s.parse().unwrap_or(255)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_read_be_i32() {
        let data = [0x7A, 0x89, 0xAB, 0xF4u8];
        let mut cursor = Cursor::new(&data);
        let result = read_be_i32(&mut cursor).unwrap();
        assert_eq!(result, 0x7A89ABF4u32 as i32);
    }

    #[test]
    fn test_read_be_u16() {
        let data = [0x12, 0x34u8];
        let mut cursor = Cursor::new(&data);
        let result = read_be_u16(&mut cursor).unwrap();
        assert_eq!(result, 0x1234);
    }

    #[test]
    fn test_read_utf8_string() {
        let data = [0x00, 0x05, b'h', b'e', b'l', b'l', b'o'];
        let mut cursor = Cursor::new(&data);
        let result = read_utf8_string(&mut cursor).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_read_string_array() {
        let mut data = Vec::new();
        data.extend_from_slice(&2i32.to_be_bytes());
        data.extend_from_slice(&3u16.to_be_bytes());
        data.extend_from_slice(b"foo");
        data.extend_from_slice(&3u16.to_be_bytes());
        data.extend_from_slice(b"bar");

        let mut cursor = Cursor::new(&data);
        let result = read_string_array(&mut cursor).unwrap();
        assert_eq!(result, vec!["foo", "bar"]);
    }

    #[test]
    fn test_read_string_array_empty() {
        let data = (-1i32).to_be_bytes();
        let mut cursor = Cursor::new(&data);
        let result = read_string_array(&mut cursor).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_snv_perms() {
        assert_eq!(SNV_PERMS[0], ["A", "C", "G", "T"]);
        assert_eq!(SNV_PERMS[23], ["T", "G", "C", "A"]);
        assert_eq!(SNV_PERMS.len(), 24);
    }

    #[test]
    fn test_allele_code_decoding() {
        let allele_code: i8 = 0b00000001;
        let n_alleles = 1 + (allele_code & 0b11) as usize;
        let perm_index = (allele_code >> 2) as usize;

        assert_eq!(n_alleles, 2);
        assert_eq!(perm_index, 0);
        assert_eq!(&SNV_PERMS[perm_index][..n_alleles], &["A", "C"]);
    }

    #[test]
    fn test_parse_alleles_biallelic() {
        let allele_strs = vec!["A".to_string(), "G".to_string()];
        let (ref_allele, alt_alleles, end) = parse_alleles(&allele_strs, None);

        assert_eq!(ref_allele, Allele::Base(0));
        assert_eq!(alt_alleles.len(), 1);
        assert_eq!(alt_alleles[0], Allele::Base(2));
        assert!(end.is_none());
    }

    #[test]
    fn test_parse_alleles_multiallelic() {
        let allele_strs = vec![
            "A".to_string(),
            "C".to_string(),
            "G".to_string(),
            "T".to_string(),
        ];
        let (ref_allele, alt_alleles, ..) = parse_alleles(&allele_strs, None);

        assert_eq!(ref_allele, Allele::Base(0));
        assert_eq!(alt_alleles.len(), 3);
    }

    #[test]
    fn test_parse_alleles_indel() {
        let allele_strs = vec!["AT".to_string(), "A".to_string()];
        let (ref_allele, alt_alleles, _) = parse_alleles(&allele_strs, None);

        assert!(matches!(ref_allele, Allele::Seq(_)));
        assert_eq!(alt_alleles.len(), 1);
        assert_eq!(alt_alleles[0], Allele::Base(0));
    }

    #[test]
    fn test_magic_number() {
        assert_eq!(BREF3_MAGIC, 2055763188);

        let bytes = BREF3_MAGIC.to_be_bytes();
        assert_eq!(bytes[0], 0x7A);
        assert_eq!(bytes[1], 0x88);
        assert_eq!(bytes[2], 0x74);
        assert_eq!(bytes[3], 0xF4);
    }
}
