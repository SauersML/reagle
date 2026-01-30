use crate::data::marker::{bits_per_allele, MarkerIdx, Markers};
use crate::data::storage::GenotypeColumn;
use crate::io::bref3::RefWindow;
use crate::utils::errors::{ReagleError, Result};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const CACHE_MAGIC: &[u8; 8] = b"RGLPRSC1";
const CACHE_VERSION: u32 = 1;

#[derive(Clone, Debug)]
pub enum PackedRefColumn {
    Bits {
        bits: u8,
        n_haps: usize,
        words: Vec<u64>,
        missing: Vec<u64>,
    },
    Bytes {
        alleles: Vec<u8>,
    },
}

impl PackedRefColumn {
    pub fn n_haplotypes(&self) -> usize {
        match self {
            PackedRefColumn::Bits { n_haps, .. } => *n_haps,
            PackedRefColumn::Bytes { alleles } => alleles.len(),
        }
    }

    pub fn allele(&self, hap: usize) -> u8 {
        match self {
            PackedRefColumn::Bytes { alleles } => alleles.get(hap).copied().unwrap_or(255),
            PackedRefColumn::Bits {
                bits,
                n_haps,
                words,
                missing,
            } => {
                if hap >= *n_haps {
                    return 255;
                }
                let word_idx = hap / 64;
                let bit_in_word = hap % 64;
                if word_idx < missing.len()
                    && ((missing[word_idx] >> bit_in_word) & 1) == 1
                {
                    return 255;
                }
                let bits = *bits as usize;
                if bits == 0 {
                    return 0;
                }
                let bit_offset = hap * bits;
                let word = bit_offset / 64;
                let shift = bit_offset % 64;
                let mut val: u64 = 0;
                if word < words.len() {
                    val = words[word] >> shift;
                    if shift + bits > 64 && word + 1 < words.len() {
                        let hi = words[word + 1] << (64 - shift);
                        val |= hi;
                    }
                }
                let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
                (val & mask) as u8
            }
        }
    }

    pub fn fill_alleles(&self, out: &mut [u8]) {
        match self {
            PackedRefColumn::Bytes { alleles } => {
                let n = out.len().min(alleles.len());
                out[..n].copy_from_slice(&alleles[..n]);
                if out.len() > n {
                    out[n..].fill(255);
                }
            }
            PackedRefColumn::Bits {
                bits,
                n_haps,
                words,
                missing,
            } => {
                let n = out.len().min(*n_haps);
                out[..n].fill(0);
                if out.len() > n {
                    out[n..].fill(255);
                }
                if *bits == 0 {
                    for i in 0..n {
                        if is_missing_bit(missing, i) {
                            out[i] = 255;
                        }
                    }
                    return;
                }
                for i in 0..n {
                    if is_missing_bit(missing, i) {
                        out[i] = 255;
                    } else {
                        out[i] = unpack_bits(words, *bits as usize, i) as u8;
                    }
                }
            }
        }
    }

    pub fn pack_from_column(marker_idx: MarkerIdx, markers: &Markers, col: &GenotypeColumn) -> Self {
        let marker = markers.marker(marker_idx);
        let n_alleles = marker.n_alleles();
        let bits = bits_per_allele(n_alleles);
        let n_haps = col.n_haplotypes();

        let mut alleles = Vec::with_capacity(n_haps);
        for h in 0..n_haps {
            alleles.push(col.get(crate::data::haplotype::HapIdx::new(h as u32)));
        }

        if bits == 0 || bits >= 8 {
            return PackedRefColumn::Bytes { alleles };
        }

        let bits_usize = bits as usize;
        let n_words = ((n_haps * bits_usize) + 63) / 64;
        let mut words = vec![0u64; n_words];
        let mut missing = vec![0u64; (n_haps + 63) / 64];
        let max_val = (1u16 << bits_usize) - 1;

        for (i, &a) in alleles.iter().enumerate() {
            let miss = a == 255 || (a as u16) > max_val;
            if miss {
                set_missing_bit(&mut missing, i);
                continue;
            }
            pack_bits(&mut words, bits_usize, i, a as u64);
        }

        PackedRefColumn::Bits {
            bits,
            n_haps,
            words,
            missing,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PackedRefWindow {
    pub markers: Markers<crate::data::marker::RefWindowSpace>,
    pub columns: Vec<PackedRefColumn>,
    pub global_start: usize,
    pub global_end: usize,
    pub output_start: usize,
    pub output_end: usize,
    pub n_ref_haps: usize,
}

pub struct PrescanCacheWriter {
    file: BufWriter<File>,
    path: PathBuf,
    n_ref_haps: usize,
    header_written: bool,
}

impl PrescanCacheWriter {
    pub fn create(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(path)?;
        Ok(Self {
            file: BufWriter::new(file),
            path: path.to_path_buf(),
            n_ref_haps: 0,
            header_written: false,
        })
    }

    pub fn set_n_ref_haps(&mut self, n_ref_haps: usize) {
        self.n_ref_haps = n_ref_haps;
    }

    pub fn write_header(&mut self) -> Result<()> {
        if self.header_written {
            return Ok(());
        }
        self.file.write_all(CACHE_MAGIC)?;
        write_u32(&mut self.file, CACHE_VERSION)?;
        write_u32(&mut self.file, self.n_ref_haps as u32)?;
        self.header_written = true;
        Ok(())
    }

    pub fn write_window(&mut self, window: &RefWindow) -> Result<()> {
        if !self.header_written {
            return Err(ReagleError::internal(
                "prescan cache header not written".to_string(),
            ));
        }
        let n_markers = window.markers.len() as u32;
        write_u32(&mut self.file, n_markers)?;
        write_u32(&mut self.file, window.output_start as u32)?;
        write_u32(&mut self.file, window.output_end as u32)?;
        write_u32(&mut self.file, window.global_start as u32)?;
        write_u32(&mut self.file, window.global_end as u32)?;

        let markers_blob = bincode::serialize(&window.markers)
            .map_err(|e| ReagleError::internal(format!("marker serialize failed: {}", e)))?;
        write_u32(&mut self.file, markers_blob.len() as u32)?;
        self.file.write_all(&markers_blob)?;

        write_u32(&mut self.file, window.ref_columns.len() as u32)?;
        for (m, col) in window.ref_columns.iter().enumerate() {
            let packed = PackedRefColumn::pack_from_column(MarkerIdx::new(m as u32), &window.markers, col);
            write_packed_column(&mut self.file, &packed)?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<PathBuf> {
        self.file.flush()?;
        Ok(self.path)
    }
}

pub struct PrescanCacheReader {
    reader: BufReader<File>,
    n_ref_haps: usize,
    eof: bool,
}

impl PrescanCacheReader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != CACHE_MAGIC {
            return Err(ReagleError::internal("invalid prescan cache magic".to_string()));
        }
        let version = read_u32(&mut reader)?;
        if version != CACHE_VERSION {
            return Err(ReagleError::internal("unsupported prescan cache version".to_string()));
        }
        let n_ref_haps = read_u32(&mut reader)? as usize;
        Ok(Self {
            reader,
            n_ref_haps,
            eof: false,
        })
    }

    pub fn n_ref_haps(&self) -> usize {
        self.n_ref_haps
    }

    pub fn next_window(&mut self) -> Result<Option<PackedRefWindow>> {
        if self.eof {
            return Ok(None);
        }
        let n_markers = match read_u32_opt(&mut self.reader)? {
            Some(v) => v as usize,
            None => {
                self.eof = true;
                return Ok(None);
            }
        };
        let output_start = read_u32(&mut self.reader)? as usize;
        let output_end = read_u32(&mut self.reader)? as usize;
        let global_start = read_u32(&mut self.reader)? as usize;
        let global_end = read_u32(&mut self.reader)? as usize;

        let markers_len = read_u32(&mut self.reader)? as usize;
        let mut markers_blob = vec![0u8; markers_len];
        self.reader.read_exact(&mut markers_blob)?;
        let markers: Markers<crate::data::marker::RefWindowSpace> =
            bincode::deserialize(&markers_blob)
                .map_err(|e| ReagleError::internal(format!("marker deserialize failed: {}", e)))?;

        let n_cols = read_u32(&mut self.reader)? as usize;
        if n_cols != n_markers {
            return Err(ReagleError::internal("cache window column count mismatch".to_string()));
        }
        let mut columns = Vec::with_capacity(n_cols);
        for _ in 0..n_cols {
            columns.push(read_packed_column(&mut self.reader)?);
        }

        Ok(Some(PackedRefWindow {
            markers,
            columns,
            global_start,
            global_end,
            output_start,
            output_end,
            n_ref_haps: self.n_ref_haps,
        }))
    }
}

pub fn create_temp_cache_path() -> PathBuf {
    let mut path = std::env::temp_dir();
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    path.push(format!("reagle_prescan_cache_{}_{}.bin", pid, nanos));
    path
}

fn write_u32<W: Write>(w: &mut W, v: u32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u32_opt<R: Read>(r: &mut R) -> Result<Option<u32>> {
    let mut buf = [0u8; 4];
    match r.read_exact(&mut buf) {
        Ok(()) => Ok(Some(u32::from_le_bytes(buf))),
        Err(e) => {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                Ok(None)
            } else {
                Err(e.into())
            }
        }
    }
}

fn write_packed_column<W: Write>(w: &mut W, col: &PackedRefColumn) -> Result<()> {
    match col {
        PackedRefColumn::Bits {
            bits,
            n_haps,
            words,
            missing,
        } => {
            w.write_all(&[0u8])?;
            w.write_all(&[*bits])?;
            write_u32(w, *n_haps as u32)?;
            write_u32(w, words.len() as u32)?;
            for &word in words {
                w.write_all(&word.to_le_bytes())?;
            }
            write_u32(w, missing.len() as u32)?;
            for &word in missing {
                w.write_all(&word.to_le_bytes())?;
            }
        }
        PackedRefColumn::Bytes { alleles } => {
            w.write_all(&[1u8])?;
            write_u32(w, alleles.len() as u32)?;
            w.write_all(alleles)?;
        }
    }
    Ok(())
}

fn read_packed_column<R: Read>(r: &mut R) -> Result<PackedRefColumn> {
    let mut tag = [0u8; 1];
    r.read_exact(&mut tag)?;
    match tag[0] {
        0 => {
            let mut bits = [0u8; 1];
            r.read_exact(&mut bits)?;
            let n_haps = read_u32(r)? as usize;
            let n_words = read_u32(r)? as usize;
            let mut words = vec![0u64; n_words];
            for w in words.iter_mut() {
                let mut buf = [0u8; 8];
                r.read_exact(&mut buf)?;
                *w = u64::from_le_bytes(buf);
            }
            let n_missing = read_u32(r)? as usize;
            let mut missing = vec![0u64; n_missing];
            for w in missing.iter_mut() {
                let mut buf = [0u8; 8];
                r.read_exact(&mut buf)?;
                *w = u64::from_le_bytes(buf);
            }
            Ok(PackedRefColumn::Bits {
                bits: bits[0],
                n_haps,
                words,
                missing,
            })
        }
        1 => {
            let n = read_u32(r)? as usize;
            let mut alleles = vec![0u8; n];
            r.read_exact(&mut alleles)?;
            Ok(PackedRefColumn::Bytes { alleles })
        }
        _ => Err(ReagleError::internal(
            "unknown prescan cache column tag".to_string(),
        )),
    }
}

fn pack_bits(words: &mut [u64], bits: usize, idx: usize, value: u64) {
    let bit_offset = idx * bits;
    let word = bit_offset / 64;
    let shift = bit_offset % 64;
    if word >= words.len() {
        return;
    }
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let v = value & mask;
    words[word] |= v << shift;
    if shift + bits > 64 && word + 1 < words.len() {
        words[word + 1] |= v >> (64 - shift);
    }
}

fn unpack_bits(words: &[u64], bits: usize, idx: usize) -> u64 {
    let bit_offset = idx * bits;
    let word = bit_offset / 64;
    let shift = bit_offset % 64;
    if word >= words.len() {
        return 0;
    }
    let mut val = words[word] >> shift;
    if shift + bits > 64 && word + 1 < words.len() {
        val |= words[word + 1] << (64 - shift);
    }
    let mask = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    val & mask
}

fn set_missing_bit(bits: &mut [u64], idx: usize) {
    let word = idx / 64;
    let bit = idx % 64;
    if word < bits.len() {
        bits[word] |= 1u64 << bit;
    }
}

fn is_missing_bit(bits: &[u64], idx: usize) -> bool {
    let word = idx / 64;
    let bit = idx % 64;
    if word >= bits.len() {
        return false;
    }
    ((bits[word] >> bit) & 1) == 1
}
