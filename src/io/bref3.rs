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

use flate2::read::GzDecoder;
use tracing::info_span;
use noodles::bgzf::io as bgzf_io;

use anyhow::{Context, Result, bail};

use crate::data::genetic_map::GeneticMaps;
use crate::data::haplotype::Samples;
use crate::data::marker::{Allele, Marker, MarkerIdx, Markers};
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
    /// Chromosome name
    pub chrom: String,
}

impl Bref3Block {}

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

        for _ in 0..n_recs {
            let marker = self.read_marker(chrom_idx)?;

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

struct Bref3BufferedMarker {
    marker: Marker,
    column: GenotypeColumn,
    gen_pos: f64,
}

/// Streaming BREF3 reader that yields reference-driven windows.
pub struct StreamingBref3WindowReader {
    inner: StreamingBref3Reader,
    buffer: VecDeque<Bref3BufferedMarker>,
    current_block: Option<(Bref3Block, usize)>,
    pending_block: Option<Bref3Block>,
    current_chrom: Option<Arc<str>>,
    window_num: usize,
    global_marker_idx: usize,
    eof: bool,
}

impl StreamingBref3WindowReader {
    pub fn new(inner: StreamingBref3Reader) -> Self {
        Self {
            inner,
            buffer: VecDeque::new(),
            current_block: None,
            pending_block: None,
            current_chrom: None,
            window_num: 0,
            global_marker_idx: 0,
            eof: false,
        }
    }

    pub fn next_window(
        &mut self,
        config: &crate::io::streaming::StreamingConfig,
        gen_maps: &GeneticMaps,
    ) -> Result<Option<RefWindow>> {
        if self.eof && self.buffer.is_empty() {
            return Ok(None);
        }

        self.fill_buffer_to_window(config, gen_maps)?;
        if self.buffer.is_empty() {
            return Ok(None);
        }

        let window_start_gen = self.buffer.front().map(|m| m.gen_pos).unwrap_or(0.0);
        let target_end_gen = window_start_gen + config.window_cm as f64;
        let full_window_gen = target_end_gen + config.overlap_cm as f64;

        let window_end = self
            .buffer
            .iter()
            .position(|m| m.gen_pos >= full_window_gen)
            .unwrap_or(self.buffer.len())
            .min(config.max_markers);

        let is_last = self.eof && window_end >= self.buffer.len();
        let output_start = 0;
        let output_end = if is_last {
            window_end
        } else {
            self.buffer
                .iter()
                .take(window_end)
                .position(|m| m.gen_pos >= target_end_gen)
                .unwrap_or(window_end)
        };

        let mut markers = Markers::new();
        let mut columns = Vec::with_capacity(window_end);
        let chrom_name = self
            .current_chrom
            .as_deref()
            .unwrap_or("UNKNOWN");
        let window_chrom_idx = markers.add_chrom(chrom_name);

        for i in 0..window_end {
            let bm = &self.buffer[i];
            let mut marker = bm.marker.clone();
            marker.chrom = window_chrom_idx;
            markers.push(marker);
            columns.push(bm.column.clone());
        }

        let genotypes = GenotypeMatrix::new_phased(markers, columns, self.inner.samples_arc());
        let window = RefWindow {
            genotypes,
            global_start: self.global_marker_idx,
            global_end: self.global_marker_idx + window_end,
            output_start,
            output_end,
            is_first: self.window_num == 0,
            is_last,
        };

        for _ in 0..output_end {
            self.buffer.pop_front();
        }
        self.global_marker_idx += output_end;
        self.window_num += 1;

        Ok(Some(window))
    }

    fn fill_buffer_to_window(
        &mut self,
        config: &crate::io::streaming::StreamingConfig,
        gen_maps: &GeneticMaps,
    ) -> Result<()> {
        if self.eof {
            return Ok(());
        }

        let target_cm = config.window_cm + config.overlap_cm + config.buffer_cm;
        let start_gen = self.buffer.front().map(|m| m.gen_pos).unwrap_or(0.0);
        let target_gen = start_gen + target_cm as f64;

        while !self.eof {
            if let Some(last) = self.buffer.back() {
                if last.gen_pos >= target_gen || self.buffer.len() >= config.max_markers {
                    break;
                }
            }

            if let Some(next_marker) = self.read_next_marker(gen_maps)? {
                self.buffer.push_back(next_marker);
            } else {
                break;
            }
        }

        Ok(())
    }

    fn read_next_marker(&mut self, gen_maps: &GeneticMaps) -> Result<Option<Bref3BufferedMarker>> {
        loop {
            if let Some((block, idx)) = self.current_block.as_mut() {
                if *idx < block.markers.len() {
                    let marker = block
                        .markers
                        .get(MarkerIdx::new(*idx as u32))
                        .cloned()
                        .expect("BREF3 marker index out of bounds");
                    let column = block.columns[*idx].clone();
                    *idx += 1;
                    let gen_pos = gen_maps.gen_pos(marker.chrom, marker.pos);
                    return Ok(Some(Bref3BufferedMarker { marker, column, gen_pos }));
                }
                self.current_block = None;
                continue;
            }

            let next_block = if let Some(pending) = self.pending_block.take() {
                Some(pending)
            } else {
                self.inner.next_block()?
            };

            let Some(block) = next_block else {
                self.eof = true;
                return Ok(None);
            };

            let block_chrom: Arc<str> = Arc::from(block.chrom.as_str());
            if let Some(cur) = self.current_chrom.as_ref() {
                if cur.as_ref() != block_chrom.as_ref() && !self.buffer.is_empty() {
                    self.pending_block = Some(block);
                    return Ok(None);
                }
                if cur.as_ref() != block_chrom.as_ref() {
                    self.current_chrom = Some(block_chrom);
                    self.window_num = 0;
                    self.global_marker_idx = 0;
                }
            } else {
                self.current_chrom = Some(block_chrom);
            }

            self.current_block = Some((block, 0));
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

/// Unified reference panel reader that supports both BREF3 (streaming) and VCF (in-memory)
pub enum RefPanelReader {
    /// Streaming BREF3 reader
    Bref3(StreamingBref3WindowReader),
    /// In-memory VCF reader
    InMemory(InMemoryRefReader),
    /// Streaming VCF reader
    StreamingVcf(StreamingRefVcfReader),
}

impl RefPanelReader {
    pub fn next_window(
        &mut self,
        config: &crate::io::streaming::StreamingConfig,
        gen_maps: &GeneticMaps,
    ) -> Result<Option<RefWindow>> {
        match self {
            RefPanelReader::Bref3(r) => r.next_window(config, gen_maps),
            RefPanelReader::InMemory(r) => r.next_window(),
            RefPanelReader::StreamingVcf(r) => r.next_window(config, gen_maps),
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

    pub fn next_window(&mut self) -> Result<Option<RefWindow>> {
        if self.window_num > 0 {
            return Ok(None);
        }
        let n_markers = self.genotypes.n_markers();
        let genotypes = (*self.genotypes).clone();
        self.window_num += 1;
        Ok(Some(RefWindow {
            genotypes,
            global_start: 0,
            global_end: n_markers,
            output_start: 0,
            output_end: n_markers,
            is_first: true,
            is_last: true,
        }))
    }
}

/// A marker buffered for streaming VCF reading
struct RefPanelMarker {
    marker: Marker,
    column: GenotypeColumn,
    gen_pos: f64,
}

/// Streaming VCF reader for reference panels
pub struct StreamingRefVcfReader {
    reader: Box<dyn BufRead + Send>,
    samples: Arc<Samples>,
    markers: Markers,
    buffer: VecDeque<RefPanelMarker>,
    pending_marker: Option<RefPanelMarker>,
    current_chrom: Option<Arc<str>>,
    line_buf: String,
    eof: bool,
    window_num: usize,
    global_marker_idx: usize,
}

impl StreamingRefVcfReader {
    /// Open a VCF file for streaming
    pub fn open(path: &Path) -> Result<Self> {
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
        let mut file = File::open(path)?;
        let reader: Box<dyn BufRead + Send> = match ext {
            "bgz" | "bgzf" => {
                if !detect_bgzf(&mut file)? {
                    anyhow::bail!("Expected BGZF file for extension .{}", ext);
                }
                Box::new(BufReader::new(bgzf_io::Reader::new(file)))
            }
            "gz" => {
                if detect_bgzf(&mut file)? {
                    Box::new(BufReader::new(bgzf_io::Reader::new(file)))
                } else {
                    Box::new(BufReader::new(GzDecoder::new(file)))
                }
            }
            _ => Box::new(BufReader::new(file)),
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
            pending_marker: None,
            current_chrom: None,
            line_buf: String::new(),
            eof: false,
            window_num: 0,
            global_marker_idx: 0,
        })
    }


    /// Read the next reference-driven window (streaming).
    pub fn next_window(
        &mut self,
        config: &crate::io::streaming::StreamingConfig,
        gen_maps: &GeneticMaps,
    ) -> Result<Option<RefWindow>> {
        if self.eof && self.buffer.is_empty() {
            return Ok(None);
        }

        self.fill_buffer_to_window(config, gen_maps)?;
        if self.buffer.is_empty() {
            return Ok(None);
        }

        let window_start_gen = self.buffer.front().map(|m| m.gen_pos).unwrap_or(0.0);
        let target_end_gen = window_start_gen + config.window_cm as f64;
        let full_window_gen = target_end_gen + config.overlap_cm as f64;

        let window_end = self
            .buffer
            .iter()
            .position(|m| m.gen_pos >= full_window_gen)
            .unwrap_or(self.buffer.len())
            .min(config.max_markers);

        let is_last = self.eof && window_end >= self.buffer.len();
        let output_start = 0;
        let output_end = if is_last {
            window_end
        } else {
            self.buffer
                .iter()
                .take(window_end)
                .position(|m| m.gen_pos >= target_end_gen)
                .unwrap_or(window_end)
        };

        let mut markers = Markers::new();
        let mut columns = Vec::with_capacity(window_end);

        for i in 0..window_end {
            let bm = &self.buffer[i];
            let chrom_name = self
                .markers
                .chrom_name(bm.marker.chrom)
                .unwrap_or("UNKNOWN");
            let window_chrom_idx = markers.add_chrom(chrom_name);
            let mut marker = bm.marker.clone();
            marker.chrom = window_chrom_idx;
            markers.push(marker);
            columns.push(bm.column.clone());
        }

        let genotypes = GenotypeMatrix::new_phased(markers, columns, Arc::clone(&self.samples));
        let window = RefWindow {
            genotypes,
            global_start: self.global_marker_idx,
            global_end: self.global_marker_idx + window_end,
            output_start,
            output_end,
            is_first: self.window_num == 0,
            is_last,
        };

        for _ in 0..output_end {
            self.buffer.pop_front();
        }
        self.global_marker_idx += output_end;
        self.window_num += 1;

        Ok(Some(window))
    }

    fn fill_buffer_to_window(
        &mut self,
        config: &crate::io::streaming::StreamingConfig,
        gen_maps: &GeneticMaps,
    ) -> Result<()> {
        if self.eof {
            return Ok(());
        }

        let target_cm = config.window_cm + config.overlap_cm + config.buffer_cm;
        let start_gen = self.buffer.front().map(|m| m.gen_pos).unwrap_or(0.0);
        let target_gen = start_gen + target_cm as f64;

        while !self.eof {
            if let Some(last) = self.buffer.back() {
                if last.gen_pos >= target_gen || self.buffer.len() >= config.max_markers {
                    break;
                }
            }

            let next_marker = if let Some(pending) = self.pending_marker.take() {
                pending
            } else if let Some(marker) = self.read_next_marker_with_gen(gen_maps)? {
                marker
            } else {
                break;
            };

            let marker_chrom = self.markers.chrom_name(next_marker.marker.chrom).unwrap_or("");
            if let Some(cur) = self.current_chrom.as_ref() {
                if marker_chrom != cur.as_ref() {
                    self.pending_marker = Some(next_marker);
                    break;
                }
            } else {
                self.current_chrom = Some(Arc::from(marker_chrom));
            }

            self.buffer.push_back(next_marker);
        }

        Ok(())
    }

    fn read_next_marker_with_gen(
        &mut self,
        gen_maps: &GeneticMaps,
    ) -> Result<Option<RefPanelMarker>> {
        loop {
            self.line_buf.clear();
            if self.reader.read_line(&mut self.line_buf)? == 0 {
                self.eof = true;
                return Ok(None);
            }
            let line = self.line_buf.trim().to_string();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let mut marker = self.parse_vcf_line(&line)?;
            marker.gen_pos = gen_maps.gen_pos(marker.marker.chrom, marker.marker.pos);
            return Ok(Some(marker));
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

        Ok(RefPanelMarker { marker, column, gen_pos: 0.0 })
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
