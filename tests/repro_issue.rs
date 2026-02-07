
use reagle::config::Config;
use reagle::pipelines::imputation::ImputationPipeline;
// use reagle::pipelines::phasing::PhasingPipeline; // unused
use serial_test::serial;
use std::fs::File;
use std::io::Write;
use tempfile::NamedTempFile;
use noodles::vcf::variant::record::samples::Series;
use noodles::bgzf::io as bgzf_io;
use noodles::vcf as noodles_vcf;
use noodles::vcf::Record;

struct SyntheticVcfBuilder {
    n_markers: usize,
    n_samples: usize,
    positions: Vec<usize>,
    allele_generator: Box<dyn Fn(usize, usize) -> u8>,
}

impl SyntheticVcfBuilder {
    fn new(n_markers: usize, n_samples: usize) -> Self {
        Self {
            n_markers,
            n_samples,
            positions: (0..n_markers).map(|m| m * 1000 + 1).collect(),
            allele_generator: Box::new(|_, _| 0),
        }
    }

    fn positions(mut self, positions: Vec<usize>) -> Self {
        self.positions = positions;
        self
    }

    fn allele_generator(mut self, generator: impl Fn(usize, usize) -> u8 + 'static) -> Self {
        self.allele_generator = Box::new(generator);
        self
    }

    fn build(self) -> NamedTempFile {
        let mut file = tempfile::Builder::new()
            .suffix(".vcf")
            .tempfile()
            .expect("Create temp file");

        writeln!(file, "##fileformat=VCFv4.2").unwrap();
        writeln!(file, "##FILTER=<ID=PASS,Description=\"All filters passed\">").unwrap();
        writeln!(file, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">").unwrap();
        writeln!(file, "##contig=<ID=chr1,length=1000000>").unwrap();
        write!(file, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT").unwrap();
        for i in 0..self.n_samples {
            write!(file, "\tSample{}", i).unwrap();
        }
        writeln!(file).unwrap();

        for m in 0..self.n_markers {
            let pos = self.positions[m];
            write!(file, "chr1\t{}\trs{}\tA\tC\t.\tPASS\t.\tGT", pos, m).unwrap();
            for s in 0..self.n_samples {
                let h1 = s * 2;
                let h2 = s * 2 + 1;
                let a1 = (self.allele_generator)(m, h1);
                let a2 = (self.allele_generator)(m, h2);
                let s1 = if a1 == 255 { ".".to_string() } else { a1.to_string() };
                let s2 = if a2 == 255 { ".".to_string() } else { a2.to_string() };
                // Use / for unphased to trigger Phasing pipeline
                write!(file, "\t{}/{}", s1, s2).unwrap(); 
            }
            writeln!(file).unwrap();
        }
        file
    }
}

// Helper to inspect output dosages robustly using noodles
fn inspect_dosages(path: &std::path::Path, _: usize) -> Vec<Vec<f32>> {
    let file = File::open(path).expect("Open output VCF");
    let decoder = bgzf_io::Reader::new(file);
    let mut reader = noodles_vcf::io::Reader::new(decoder);

    let header = reader.read_header().expect("Read header");

    let mut all_dosages = Vec::new();

    for result in reader.records() {
        let result: std::io::Result<Record> = result;
        let record = result.expect("Read record");
        let mut site_dosages = Vec::new();

        let samples = record.samples();

        let ds_col = samples.select("DS").expect("DS column missing");

        for value in ds_col.iter(&header) {
            match value {
                Ok(Some(v)) => {
                    let s = format!("{:?}", v);
                    let parsed = if s.contains("Array") {
                        if let Some(start) = s.rfind("Some(") {
                            let after_some = &s[start + 5..];
                            if let Some(end) = after_some.find(')') {
                                after_some[..end].parse().unwrap_or(0.0)
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        }
                    } else if s.contains("Float") {
                        if let Some(start) = s.find('(') {
                            if let Some(end) = s.find(')') {
                                s[start + 1..end].parse().unwrap_or(0.0)
                            } else {
                                0.0
                            }
                        } else {
                            0.0
                        }
                    } else {
                        s.parse().unwrap_or(0.0)
                    };
                    site_dosages.push(parsed);
                }
                Ok(None) => site_dosages.push(-1.0),
                Err(e) => panic!("Error reading DS: {}", e),
            }
        }
        all_dosages.push(site_dosages);
    }
    all_dosages
}

#[test]
#[serial]
fn test_repro_synthetic_recombination() {
    let n_markers = 50;
    let positions: Vec<usize> = (0..n_markers).map(|m| m * 100000 + 1).collect();

    let ref_file = SyntheticVcfBuilder::new(n_markers, 50)
        .positions(positions.clone())
        .allele_generator(|_, h| if h < 50 { 0 } else { 1 })
        .build();

    let target_file = SyntheticVcfBuilder::new(n_markers, 1)
        .positions(positions)
        .allele_generator(|m, _| {
            if m == 15 || m == 25 {
                255
            } else if m < 20 {
                0
            } else {
                1
            }
        })
        .build();

    let temp_dir = tempfile::tempdir().unwrap();
    let out_prefix = temp_dir.path().join("output_repro");

    let mut config = Config::default();
    config.target = target_file.path().to_path_buf();
    config.r#ref = Some(ref_file.path().to_path_buf());
    config.out = out_prefix.clone();
    config.imp_states = 100;
    config.ne = 10000.0;
    config.window = 10.0;
    config.overlap = 2.0;
    config.nthreads = Some(1);

    // Let ImputationPipeline handle phasing
    let mut pipeline = ImputationPipeline::new(config, None);
    pipeline.run().expect("Pipeline run success");

    let phased_vcf = temp_dir.path().join("output_repro_phased_target.vcf.gz");
    if phased_vcf.exists() {
        let file = File::open(&phased_vcf).unwrap();
        let decoder = bgzf_io::Reader::new(file);
        let mut reader = noodles_vcf::io::Reader::new(decoder);
        let header = reader.read_header().unwrap();
        
        for result in reader.records() {
            let record = result.unwrap();
            let pos = record.variant_start().unwrap().unwrap().get();
            // Marker 15 is at 15 * 100000 + 1 = 1500001
            if pos == 1500001 {
                let samples = record.samples();
                let series = samples.select("GT").unwrap();
                for value in series.iter(&header) {
                    let gt = value.unwrap().unwrap();
                    println!("Internal Phased GT at 15: {:?}", gt);
                }
            }
        }
    } else {
        println!("Phased VCF not found at {:?}", phased_vcf);
    }

    let out_vcf = temp_dir.path().join("output_repro.vcf.gz");
    let dosages = inspect_dosages(&out_vcf, 1);

    println!("Dosage at 15: {}", dosages[15][0]);
    assert!(dosages[15][0] < 0.1, "Marker 15 should be 0, got {}", dosages[15][0]);
}
