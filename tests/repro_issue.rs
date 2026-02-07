use reagle::config::Config;
use reagle::pipelines::phasing::PhasingPipeline;
use reagle::data::AnyMarkerSpace;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use noodles::vcf;
use noodles::bgzf;
use noodles::vcf::variant::record::samples::Series;

fn create_vcf(path: &PathBuf, n_markers: usize) {
    let mut file = File::create(path).unwrap();
    writeln!(file, "##fileformat=VCFv4.2").unwrap();
    writeln!(file, "##FILTER=<ID=PASS,Description=\"All filters passed\">").unwrap();
    writeln!(file, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">").unwrap();
    writeln!(file, "##contig=<ID=chr1,length=1000000>").unwrap();
    writeln!(file, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSample1").unwrap();

    for i in 0..n_markers {
        let pos = i * 100000 + 1; // 100kb spacing
        let gt = if i == 15 || i == 25 {
            "./."
        } else if i < 20 {
            "0/0"
        } else {
            "1/1"
        };
        writeln!(file, "chr1\t{}\t.\tA\tC\t.\tPASS\t.\tGT\t{}", pos, gt).unwrap();
    }
}

fn create_ref_vcf(path: &PathBuf, n_markers: usize) {
    let mut file = File::create(path).unwrap();
    writeln!(file, "##fileformat=VCFv4.2").unwrap();
    writeln!(file, "##FILTER=<ID=PASS,Description=\"All filters passed\">").unwrap();
    writeln!(file, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">").unwrap();
    writeln!(file, "##contig=<ID=chr1,length=1000000>").unwrap();
    write!(file, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT").unwrap();
    for i in 0..50 {
        write!(file, "\tRef{}", i).unwrap();
    }
    writeln!(file).unwrap();

    for i in 0..n_markers {
        let pos = i * 100000 + 1;
        write!(file, "chr1\t{}\t.\tA\tC\t.\tPASS\t.\tGT", pos).unwrap();
        for s in 0..50 {
            // Haps 0-49 are 0. Haps 50-99 are 1.
            // s=0 -> h0,h1 (0,0). s=24 -> h48,h49 (0,0).
            // s=25 -> h50,h51 (1,1).
            if s < 25 {
                write!(file, "\t0|0").unwrap();
            } else {
                write!(file, "\t1|1").unwrap();
            }
        }
        writeln!(file).unwrap();
    }
}

#[test]
fn test_phasing_imputation_of_missing_marker_recomb() {
    let temp_dir = tempfile::tempdir().unwrap();
    let target_path = temp_dir.path().join("target.vcf");
    let ref_path = temp_dir.path().join("ref.vcf");
    let out_prefix = temp_dir.path().join("out");

    let n_markers = 50;
    create_vcf(&target_path, n_markers);
    create_ref_vcf(&ref_path, n_markers);

    let config = Config {
        target: target_path,
        r#ref: Some(ref_path),
        out: out_prefix.clone(),
        ne: 10000.0,
        ..Config::default()
    };

    let mut pipeline = PhasingPipeline::<AnyMarkerSpace>::new(config, None);
    pipeline.run().expect("Phasing failed");

    // Check output
    let out_vcf = out_prefix.with_extension("vcf.gz");
    let mut reader = vcf::io::Reader::new(bgzf::io::Reader::new(File::open(out_vcf).unwrap()));
    let header = reader.read_header().unwrap();

    for (i, result) in reader.records().enumerate() {
        let record = result.unwrap();
        if i == 15 {
            let samples = record.samples();
            let gt_series = samples.select("GT").unwrap();
            let gt_val = gt_series.iter(&header).next().unwrap().unwrap().unwrap();
            println!("Marker {} GT: {:?}", i, gt_val);
            
            let gt_str = format!("{:?}", gt_val);
            assert!(gt_str.contains("0|0") || gt_str.contains("0/0"), "Marker 15: Expected 0|0, got {}", gt_str);
        }
        if i == 25 {
            let samples = record.samples();
            let gt_series = samples.select("GT").unwrap();
            let gt_val = gt_series.iter(&header).next().unwrap().unwrap().unwrap();
            println!("Marker {} GT: {:?}", i, gt_val);
            
            let gt_str = format!("{:?}", gt_val);
            assert!(gt_str.contains("1|1") || gt_str.contains("1/1"), "Marker 25: Expected 1|1, got {}", gt_str);
        }
    }
}
