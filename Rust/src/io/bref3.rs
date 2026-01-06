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

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};

use crate::data::haplotype::Samples;
use crate::data::marker::{Allele, Marker, Markers};
use crate::data::storage::phase_state::Phased;
use crate::data::storage::{GenotypeColumn, GenotypeMatrix};
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
    program: String,
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

        let program = read_utf8_string(&mut reader)?;
        let sample_ids = read_string_array(&mut reader)?;
        let n_haps = sample_ids.len() * 2;
        let samples = Samples::from_ids(sample_ids);

        Ok(Self {
            reader,
            samples,
            program,
            n_haps,
            markers: Markers::new(),
            chrom_map: std::collections::HashMap::new(),
        })
    }

    /// Get the program string from the BREF3 header
    pub fn program(&self) -> &str {
        &self.program
    }

    /// Get samples
    pub fn samples(&self) -> &Samples {
        &self.samples
    }

    /// Number of haplotypes
    pub fn n_haplotypes(&self) -> usize {
        self.n_haps
    }

    /// Read all genotypes into a GenotypeMatrix (phased reference data)
    pub fn read_all(mut self) -> Result<GenotypeMatrix<Phased>> {
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
    }

    /// Read a single block of records
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

        for _ in 0..n_recs {
            let (marker, alleles) =
                self.read_record(chrom_idx, n_seq, &hap_to_seq)?;
            self.markers.push(marker.clone());
            let col = GenotypeColumn::from_alleles(&alleles, marker.n_alleles());
            columns.push(col);
        }

        Ok(())
    }

    /// Read a single record (marker + genotypes)
    fn read_record(
        &mut self,
        chrom_idx: ChromIdx,
        n_seq: usize,
        hap_to_seq: &[u16],
    ) -> Result<(Marker, Vec<u8>)> {
        let marker = self.read_marker(chrom_idx)?;
        let flag = read_byte(&mut self.reader)?;

        let alleles = match flag {
            SEQ_CODED => self.read_seq_coded_record(n_seq, hap_to_seq)?,
            ALLELE_CODED => self.read_allele_coded_record(marker.n_alleles())?,
            _ => bail!("Unknown record type flag: {}", flag),
        };

        Ok((marker, alleles))
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

    /// Read sequence-coded genotype record
    fn read_seq_coded_record(
        &mut self,
        n_seq: usize,
        hap_to_seq: &[u16],
    ) -> Result<Vec<u8>> {
        let mut seq_to_allele = vec![0u8; n_seq];
        self.reader.read_exact(&mut seq_to_allele)?;

        let mut alleles = vec![0u8; self.n_haps];
        for (hap_idx, &seq_idx) in hap_to_seq.iter().enumerate() {
            alleles[hap_idx] = seq_to_allele[seq_idx as usize];
        }

        Ok(alleles)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_bref3_header(samples: &[&str]) -> Vec<u8> {
        let mut data = Vec::new();

        data.extend_from_slice(&BREF3_MAGIC.to_be_bytes());

        let program = "test-program";
        data.extend_from_slice(&(program.len() as u16).to_be_bytes());
        data.extend_from_slice(program.as_bytes());

        data.extend_from_slice(&(samples.len() as i32).to_be_bytes());
        for &sample in samples {
            data.extend_from_slice(&(sample.len() as u16).to_be_bytes());
            data.extend_from_slice(sample.as_bytes());
        }

        data
    }

    fn append_end_of_data(data: &mut Vec<u8>) {
        data.extend_from_slice(&END_OF_DATA.to_be_bytes());
    }

    fn append_block_with_snv(
        data: &mut Vec<u8>,
        chrom: &str,
        pos: u32,
        allele_code: u8,
        hap_alleles: &[u8],
    ) {
        let n_haps = hap_alleles.len();

        data.extend_from_slice(&1i32.to_be_bytes());

        data.extend_from_slice(&(chrom.len() as u16).to_be_bytes());
        data.extend_from_slice(chrom.as_bytes());

        let n_seq = hap_alleles.iter().max().map(|&m| m as usize + 1).unwrap_or(1);
        data.extend_from_slice(&(n_seq as u16).to_be_bytes());

        for (i, &allele) in hap_alleles.iter().enumerate() {
            data.extend_from_slice(&(allele as u16).to_be_bytes());
        }

        data.extend_from_slice(&(pos as i32).to_be_bytes());
        data.push(0);
        data.push(allele_code);

        data.push(SEQ_CODED);

        for seq in 0..n_seq {
            data.push(seq as u8);
        }
    }

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
        let (ref_allele, alt_alleles, end) = parse_alleles(&allele_strs, None);

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
