use crate::data::marker::{MarkerIdx, Markers, RefWindowSpace, bits_per_allele};
use crate::data::storage::GenotypeColumn;
use crate::error::{ReagleError, Result};
use crate::io::bref3::RefWindow;
use bincode::Options;
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
            PackedRefColumn::Bytes { alleles } => alleles.get(hap).copied().unwrap_or(crate::data::storage::AlleleCode::MISSING.raw()),
            PackedRefColumn::Bits {
                bits,
                n_haps,
                words,
                missing,
            } => {
                if hap >= *n_haps {
                    return crate::data::storage::AlleleCode::MISSING.raw();
                }
                let word_idx = hap / 64;
                let bit_in_word = hap % 64;
                if word_idx < missing.len() && ((missing[word_idx] >> bit_in_word) & 1) == 1 {
                    return crate::data::storage::AlleleCode::MISSING.raw();
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
                let mask = if bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << bits) - 1
                };
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
                    out[n..].fill(crate::data::storage::AlleleCode::MISSING.raw());
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
                    out[n..].fill(crate::data::storage::AlleleCode::MISSING.raw());
                }
                if *bits == 0 {
                    for i in 0..n {
                        if is_missing_bit(missing, i) {
                            out[i] = crate::data::storage::AlleleCode::MISSING.raw();
                        }
                    }
                    return;
                }
                if *bits == 1 {
                    fill_biallelic_bits(out, n, words, missing);
                    return;
                }
                for i in 0..n {
                    if is_missing_bit(missing, i) {
                        out[i] = crate::data::storage::AlleleCode::MISSING.raw();
                    } else {
                        out[i] = unpack_bits(words, *bits as usize, i) as u8;
                    }
                }
            }
        }
    }

    pub fn counts_biallelic(&self) -> Option<(usize, usize, usize)> {
        match self {
            PackedRefColumn::Bits {
                bits,
                n_haps,
                words,
                missing,
            } => {
                if *bits != 1 {
                    return None;
                }
                let mut ones = 0usize;
                let mut missing_count = 0usize;
                for (i, word) in words.iter().enumerate() {
                    let miss = missing.get(i).copied().unwrap_or(0);
                    ones += (word & !miss).count_ones() as usize;
                    missing_count += miss.count_ones() as usize;
                }
                let total = *n_haps;
                let zeros = total.saturating_sub(ones + missing_count);
                Some((zeros, ones, missing_count))
            }
            _ => None,
        }
    }

    pub fn pack_from_column<Space>(
        marker_idx: MarkerIdx<Space>,
        markers: &Markers<Space>,
        col: &GenotypeColumn,
    ) -> Result<Self> {
        let marker = markers.marker(marker_idx);
        let n_alleles = marker.n_alleles();
        let bits = bits_per_allele(n_alleles);
        let n_haps = col.n_haplotypes();

        if n_alleles > crate::data::storage::AlleleCode::MISSING.raw() as usize {
            return Err(ReagleError::vcf(format!(
                "marker has {} alleles; maximum supported is {} because that code is reserved for missing",
                n_alleles
                ,
                crate::data::storage::AlleleCode::MISSING.raw()
            )));
        }

        let use_bits = should_use_bit_packing(bits as usize, n_haps);
        if !use_bits {
            let mut alleles = Vec::with_capacity(n_haps);
            for h in 0..n_haps {
                let a = col.get(crate::data::haplotype::HapIdx::new(h as u32));
                if a != crate::data::storage::AlleleCode::MISSING.raw() && (a as usize) >= n_alleles {
                    return Err(ReagleError::vcf(format!(
                        "invalid allele {} at marker {} (n_alleles={})",
                        a,
                        marker_idx.as_usize(),
                        n_alleles
                    )));
                }
                alleles.push(a);
            }
            return Ok(PackedRefColumn::Bytes { alleles });
        }

        let bits_usize = bits as usize;
        let shape = PackedColumnShape::new(bits_usize, n_haps)?;
        let mut words = vec![0u64; shape.n_words];
        let mut missing = vec![0u64; shape.n_missing_words];

        for i in 0..n_haps {
            let a = col.get(crate::data::haplotype::HapIdx::new(i as u32));
            if a != crate::data::storage::AlleleCode::MISSING.raw() && (a as usize) >= n_alleles {
                return Err(ReagleError::vcf(format!(
                    "invalid allele {} at marker {} (n_alleles={})",
                    a,
                    marker_idx.as_usize(),
                    n_alleles
                )));
            }
            if a == crate::data::storage::AlleleCode::MISSING.raw() {
                set_missing_bit(&mut missing, i);
                continue;
            }
            if bits_usize > 0 {
                pack_bits(&mut words, bits_usize, i, a as u64);
            }
        }

        Ok(PackedRefColumn::Bits {
            bits,
            n_haps,
            words,
            missing,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct PackedColumnShape {
    bits: usize,
    n_haps: usize,
    n_words: usize,
    n_missing_words: usize,
}

impl PackedColumnShape {
    fn new(bits: usize, n_haps: usize) -> Result<Self> {
        if bits > 63 {
            return Err(ReagleError::vcf(format!("invalid bit width {}", bits)));
        }
        Ok(Self {
            bits,
            n_haps,
            n_words: packed_words_len(bits, n_haps),
            n_missing_words: packed_missing_words_len(n_haps),
        })
    }

    fn validate_lengths(self, words_len: usize, missing_len: usize) -> Result<()> {
        if missing_len != self.n_missing_words {
            return Err(ReagleError::vcf(format!(
                "invalid missing bitmap length: got {} expected {} for n_haps={}",
                missing_len, self.n_missing_words, self.n_haps
            )));
        }
        if words_len != self.n_words {
            return Err(ReagleError::vcf(format!(
                "invalid packed word length: got {} expected {} for bits={} n_haps={}",
                words_len, self.n_words, self.bits, self.n_haps
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct PackedRefWindow {
    pub markers: Markers<crate::data::marker::RefWindowSpace>,
    pub columns: Vec<PackedRefColumn>,
}

pub fn pack_ref_columns(
    markers: &Markers<RefWindowSpace>,
    ref_columns: &[GenotypeColumn],
) -> Result<Vec<PackedRefColumn>> {
    let mut packed = Vec::with_capacity(ref_columns.len());
    for (m, col) in ref_columns.iter().enumerate() {
        packed.push(PackedRefColumn::pack_from_column(
            MarkerIdx::new(m as u32),
            markers,
            col,
        )?);
    }
    Ok(packed)
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
            return Err(ReagleError::vcf(
                "prescan cache header not written".to_string(),
            ));
        }
        if window.markers.len() != window.ref_columns.len() {
            return Err(ReagleError::vcf(format!(
                "cache window marker/column mismatch: markers={} columns={}",
                window.markers.len(),
                window.ref_columns.len()
            )));
        }
        if let Some(first_col) = window.ref_columns.first() {
            let window_haps = first_col.n_haplotypes();
            for (i, col) in window.ref_columns.iter().enumerate().skip(1) {
                if col.n_haplotypes() != window_haps {
                    return Err(ReagleError::vcf(format!(
                        "cache window has inconsistent haplotypes across columns: col0={} col{}={}",
                        window_haps,
                        i,
                        col.n_haplotypes()
                    )));
                }
            }
            if self.n_ref_haps != window_haps {
                return Err(ReagleError::vcf(format!(
                    "cache header n_ref_haps={} does not match window haplotypes={}",
                    self.n_ref_haps, window_haps
                )));
            }
        }
        let n_markers = window.markers.len() as u32;
        write_u32(&mut self.file, n_markers)?;
        let markers_blob = marker_bincode_options()
            .serialize(&window.markers)
            .map_err(|e| ReagleError::vcf(format!("marker serialize failed: {}", e)))?;
        write_u32(&mut self.file, markers_blob.len() as u32)?;
        self.file.write_all(&markers_blob)?;

        write_u32(&mut self.file, window.ref_columns.len() as u32)?;
        let packed_cols = pack_ref_columns(&window.markers, &window.ref_columns)?;
        for packed in packed_cols {
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
    data_offset: u64,
    n_ref_haps: usize,
    eof: bool,
}

impl PrescanCacheReader {
    pub fn open(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut magic = [0u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != CACHE_MAGIC {
            return Err(ReagleError::vcf("invalid prescan cache magic".to_string()));
        }
        let version = read_u32(&mut reader)?;
        if version != CACHE_VERSION {
            return Err(ReagleError::vcf(
                "unsupported prescan cache version".to_string(),
            ));
        }
        let n_ref_haps = read_u32(&mut reader)? as usize;
        let data_offset = reader.stream_position()?;
        Ok(Self {
            reader,
            data_offset,
            n_ref_haps,
            eof: false,
        })
    }

    pub fn rewind(&mut self) -> Result<()> {
        self.reader.seek(SeekFrom::Start(self.data_offset))?;
        self.eof = false;
        Ok(())
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
        let markers_len = read_u32(&mut self.reader)? as usize;
        let mut markers_blob = vec![0u8; markers_len];
        self.reader.read_exact(&mut markers_blob)?;
        let markers: Markers<crate::data::marker::RefWindowSpace> = marker_bincode_options()
            .reject_trailing_bytes()
            .deserialize(&markers_blob)
            .map_err(|e| ReagleError::vcf(format!("marker deserialize failed: {}", e)))?;
        if markers.len() != n_markers {
            return Err(ReagleError::vcf(format!(
                "cache window marker count mismatch: header={} decoded={}",
                n_markers,
                markers.len()
            )));
        }

        let n_cols = read_u32(&mut self.reader)? as usize;
        if n_cols != n_markers {
            return Err(ReagleError::vcf(
                "cache window column count mismatch".to_string(),
            ));
        }
        let mut columns = Vec::with_capacity(n_cols);
        for _ in 0..n_cols {
            let col = read_packed_column(&mut self.reader)?;
            if col.n_haplotypes() != self.n_ref_haps {
                return Err(ReagleError::vcf(format!(
                    "cache haplotype count mismatch: header={} column={}",
                    self.n_ref_haps,
                    col.n_haplotypes()
                )));
            }
            columns.push(col);
        }

        Ok(Some(PackedRefWindow { markers, columns }))
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
            let shape = PackedColumnShape::new(bits[0] as usize, n_haps)?;
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
            shape.validate_lengths(words.len(), missing.len())?;
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
        _ => Err(ReagleError::vcf(
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
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let v = value & mask;
    words[word] |= v << shift;
    if shift + bits > 64 && word + 1 < words.len() {
        words[word + 1] |= v >> (64 - shift);
    }
}

fn packed_missing_words_len(n_haps: usize) -> usize {
    n_haps.div_ceil(64)
}

fn packed_words_len(bits: usize, n_haps: usize) -> usize {
    if bits == 0 {
        0
    } else {
        (n_haps * bits).div_ceil(64)
    }
}

fn should_use_bit_packing(bits: usize, n_haps: usize) -> bool {
    if bits == 0 {
        return true;
    }
    let byte_storage = n_haps;
    let bit_storage = packed_words_len(bits, n_haps) * std::mem::size_of::<u64>()
        + packed_missing_words_len(n_haps) * std::mem::size_of::<u64>();
    bit_storage < byte_storage
}

fn marker_bincode_options() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .allow_trailing_bytes()
}

fn fill_biallelic_bits(out: &mut [u8], n_haps: usize, words: &[u64], missing: &[u64]) {
    let n_blocks = n_haps.div_ceil(64);
    for block in 0..n_blocks {
        let start = block * 64;
        let remaining = n_haps.saturating_sub(start);
        let limit = remaining.min(64);
        if limit == 0 {
            break;
        }
        let data = words.get(block).copied().unwrap_or(0);
        let miss = missing.get(block).copied().unwrap_or(0);
        for b in 0..limit {
            let idx = start + b;
            if ((miss >> b) & 1) == 1 {
                out[idx] = crate::data::storage::AlleleCode::MISSING.raw();
            } else {
                out[idx] = ((data >> b) & 1) as u8;
            }
        }
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
    let mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::marker::{Allele, Marker, RefWindowSpace};
    use crate::data::storage::GenotypeColumn;
    use tempfile::tempdir;

    #[test]
    fn test_pack_unpack_biallelic() {
        let mut markers = Markers::<RefWindowSpace>::new();
        let chrom = markers.add_chrom("1");
        let marker = Marker::new(
            chrom,
            100,
            None,
            Allele::from_str("A"),
            vec![Allele::from_str("G")],
        );
        markers.push(marker);

        let alleles = vec![0u8, 1, 0, 1, crate::data::storage::AlleleCode::MISSING.raw()];
        let col = GenotypeColumn::from_alleles(&alleles, 2);
        let packed = PackedRefColumn::pack_from_column(MarkerIdx::new(0), &markers, &col).unwrap();

        let mut out = vec![0u8; alleles.len()];
        packed.fill_alleles(&mut out);
        assert_eq!(out, alleles);
    }

    #[test]
    fn test_cache_roundtrip() {
        let mut markers = Markers::<RefWindowSpace>::new();
        let chrom = markers.add_chrom("1");
        let marker = Marker::new(
            chrom,
            200,
            None,
            Allele::from_str("C"),
            vec![Allele::from_str("T")],
        );
        markers.push(marker);

        let alleles = vec![0u8, 1, 1, 0];
        let col = GenotypeColumn::from_alleles(&alleles, 2);
        let window = RefWindow {
            markers: markers.clone(),
            ref_columns: vec![col],
            ref_genotypes: None,
            global_start: 0,
            global_end: 1,
            output_start: 0,
            output_end: 1,
        };

        let dir = tempdir().unwrap();
        let path = dir.path().join("cache.bin");
        let mut writer = PrescanCacheWriter::create(&path).unwrap();
        writer.set_n_ref_haps(alleles.len());
        writer.write_header().unwrap();
        writer.write_window(&window).unwrap();
        writer.finish().unwrap();

        let mut reader = PrescanCacheReader::open(&path).unwrap();
        let got = reader.next_window().unwrap().unwrap();
        assert_eq!(got.markers.len(), 1);
        let mut out = vec![0u8; alleles.len()];
        got.columns[0].fill_alleles(&mut out);
        assert_eq!(out, alleles);
    }
}
