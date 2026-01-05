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
use crate::data::storage::{GenotypeColumn, GenotypeMatrix};
use crate::error::{ReagleError, Result};

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
    pub fn from_reader(mut reader: Box<dyn BufRead + Send>) -> Result<(Self, Box<dyn BufRead + Send>)> {
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

        let header: Header = header_str.parse().map_err(|e| ReagleError::vcf(format!("{}", e)))?;

        // Extract sample names
        let sample_names: Vec<String> = header
            .sample_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let samples = Arc::new(Samples::from_ids(sample_names));

        Ok((
            Self {
                header,
                samples,
            },
            reader,
        ))
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
        let _n_haps = self.samples.n_haps();
        let mut is_phased = true;

        let mut line = String::new();
        let mut line_num = 0usize;

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
            let (marker, alleles, record_phased) = self.parse_record(line, &mut markers, line_num)?;
            
            if !record_phased {
                is_phased = false;
            }

            markers.push(marker);
            let column = GenotypeColumn::from_alleles(&alleles, 2); // Assume biallelic for now
            columns.push(column);
        }

        let matrix = GenotypeMatrix::new(markers, columns, Arc::clone(&self.samples), is_phased);
        Ok(matrix)
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
        let alt_alleles: Vec<Allele> = fields[4]
            .split(',')
            .map(|a| Allele::from_str(a))
            .collect();

        let _n_alleles = 1 + alt_alleles.len();

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

            let gt_field = sample_field
                .split(':')
                .nth(gt_idx)
                .unwrap_or("./.");

            // Parse genotype (handle both phased | and unphased /)
            let (a1, a2, phased) = parse_genotype(gt_field, line_num)?;
            
            if !phased {
                is_phased = false;
            }

            alleles.push(a1);
            alleles.push(a2);
        }

        let marker = Marker::new(chrom_idx, pos, id, ref_allele, alt_alleles);

        Ok((marker, alleles, is_phased))
    }
}

/// Parse a genotype field (e.g., "0|1", "0/1", ".")
fn parse_genotype(gt: &str, line_num: usize) -> Result<(u8, u8, bool)> {
    if gt == "." || gt == "./." || gt == ".|." {
        return Ok((255, 255, true)); // Missing
    }

    let phased = gt.contains('|');
    let sep = if phased { '|' } else { '/' };

    let parts: Vec<&str> = gt.split(sep).collect();
    if parts.len() != 2 {
        // Haploid
        let a1 = if parts[0] == "." {
            255
        } else {
            parts[0]
                .parse()
                .map_err(|_| ReagleError::parse(line_num, "Invalid allele value"))?
        };
        return Ok((a1, a1, phased));
    }

    let a1 = if parts[0] == "." {
        255
    } else {
        parts[0]
            .parse()
            .map_err(|_| ReagleError::parse(line_num, "Invalid allele value"))?
    };

    let a2 = if parts[1] == "." {
        255
    } else {
        parts[1]
            .parse()
            .map_err(|_| ReagleError::parse(line_num, "Invalid allele value"))?
    };

    Ok((a1, a2, phased))
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

    /// Write VCF header
    pub fn write_header(&mut self, markers: &Markers) -> Result<()> {
        // Write file format
        writeln!(self.writer, "##fileformat=VCFv4.2")?;

        // Write contig lines
        for chrom in markers.chrom_names() {
            writeln!(self.writer, "##contig=<ID={}>", chrom)?;
        }

        // Write FORMAT lines
        writeln!(self.writer, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">")?;

        // Write header line
        write!(self.writer, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")?;
        for sample in self.samples.ids() {
            write!(self.writer, "\t{}", sample)?;
        }
        writeln!(self.writer)?;

        Ok(())
    }

    /// Write a phased genotype matrix
    pub fn write_phased(&mut self, matrix: &GenotypeMatrix, start: usize, end: usize) -> Result<()> {
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
                marker.alt_alleles.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(",")
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

    /// Write imputed genotypes with dosages
    pub fn write_imputed(
        &mut self,
        matrix: &GenotypeMatrix,
        dosages: &[f32],
        start: usize,
        end: usize,
    ) -> Result<()> {
        let n_samples = self.samples.len();

        for (local_m, m) in (start..end).enumerate() {
            let marker_idx = MarkerIdx::new(m as u32);
            let marker = matrix.marker(marker_idx);
            let column = matrix.column(marker_idx);

            // Write fixed fields
            write!(
                self.writer,
                "{}\t{}\t{}\t{}\t{}\t.\tPASS\t.\tGT:DS",
                matrix.markers().chrom_name(marker.chrom).unwrap_or("."),
                marker.pos,
                marker.id.as_ref().map(|s| s.as_ref()).unwrap_or("."),
                marker.ref_allele,
                marker.alt_alleles.iter().map(|a| a.to_string()).collect::<Vec<_>>().join(",")
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
                write!(self.writer, "\t{}|{}:{:.3}", a1, a2, ds)?;
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
        assert_eq!(parse_genotype("0|1", 0).unwrap(), (0, 1, true));
        assert_eq!(parse_genotype("1|0", 0).unwrap(), (1, 0, true));
        assert_eq!(parse_genotype("0/1", 0).unwrap(), (0, 1, false));
        assert_eq!(parse_genotype("./.", 0).unwrap(), (255, 255, true));
        assert_eq!(parse_genotype(".|.", 0).unwrap(), (255, 255, true));
    }

    #[test]
    fn test_parse_genotype_multiallelic() {
        assert_eq!(parse_genotype("0|2", 0).unwrap(), (0, 2, true));
        assert_eq!(parse_genotype("1|2", 0).unwrap(), (1, 2, true));
    }
}