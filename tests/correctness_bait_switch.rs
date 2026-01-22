use reagle::config::Config;
use reagle::pipelines::imputation::ImputationPipeline;

use noodles::bgzf::io as bgzf_io;
use noodles::vcf as noodles_vcf;
use noodles_vcf::variant::record::samples::Sample;
use noodles_vcf::variant::record::samples::series::Value;

use serial_test::serial;

use std::fs::File;
use std::io::Write;
use std::path::Path;

fn write_biallelic_vcf(
    path: &Path,
    positions_bp: &[u32],
    sample_names: &[&str],
    gt_strings_per_sample: &[Vec<String>],
) {
    assert_eq!(gt_strings_per_sample.len(), sample_names.len());
    for gts in gt_strings_per_sample {
        assert_eq!(gts.len(), positions_bp.len());
    }

    let mut f = File::create(path).expect("create VCF");

    writeln!(f, "##fileformat=VCFv4.2").unwrap();
    writeln!(f, "##FILTER=<ID=PASS,Description=\"All filters passed\">").unwrap();
    writeln!(f, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">").unwrap();
    writeln!(f, "##contig=<ID=chr1,length=1000000>").unwrap();

    write!(
        f,
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
    )
    .unwrap();
    for s in sample_names {
        write!(f, "\t{}", s).unwrap();
    }
    writeln!(f).unwrap();

    for (m, pos) in positions_bp.iter().copied().enumerate() {
        write!(
            f,
            "chr1\t{}\trs{}\tA\tT\t.\tPASS\t.\tGT",
            pos,
            m
        )
        .unwrap();

        for sample_gts in gt_strings_per_sample {
            write!(f, "\t{}", sample_gts[m]).unwrap();
        }

        writeln!(f).unwrap();
    }
}

fn read_single_sample_ds_by_record_index(path: &Path) -> Vec<f64> {
    let file = File::open(path).expect("open output VCF");
    let decoder = bgzf_io::Reader::new(file);
    let mut reader = noodles_vcf::io::Reader::new(decoder);

    let header = reader.read_header().expect("read VCF header");
    assert!(header.formats().contains_key("DS"), "DS missing from output");

    let mut ds = Vec::new();

    for result in reader.records() {
        let record = result.expect("read VCF record");
        let samples = record.samples();
        let sample0 = samples.get_index(0).expect("sample 0");

        let val = sample0
            .get(&header, "DS")
            .transpose()
            .ok()
            .flatten()
            .flatten();

        let parsed = match val {
            Some(Value::Float(f)) => f as f64,
            Some(Value::String(s)) => s.parse::<f64>().expect("parse DS"),
            Some(other) => panic!("Unexpected DS value: {other:?}"),
            None => panic!("Missing DS"),
        };

        ds.push(parsed);
    }

    ds
}

#[test]
#[serial]
fn test_state_index_stability_bait_and_switch() {
    let n_markers = 5;
    let positions: Vec<u32> = (0..n_markers).map(|m| (m as u32) * 100_000 + 1).collect();

    let temp_dir = tempfile::tempdir().unwrap();
    let ref_path = temp_dir.path().join("ref.vcf");
    let target_path = temp_dir.path().join("target.vcf");
    let out_prefix = temp_dir.path().join("bait_switch");

    // Reference: Two phased samples.
    // - B_Good is a perfect match (all 0s)
    // - A_Decoy matches only at marker 1 but diverges after.
    write_biallelic_vcf(
        &ref_path,
        &positions,
        &["B_Good", "A_Decoy"],
        &[
            vec![
                "0|0".to_string(),
                "0|0".to_string(),
                "0|0".to_string(),
                "0|0".to_string(),
                "0|0".to_string(),
            ],
            vec![
                "1|1".to_string(),
                "0|0".to_string(),
                "1|1".to_string(),
                "1|1".to_string(),
                "1|1".to_string(),
            ],
        ],
    );

    // Target: marker 0 locks onto B_Good, marker 1 matches both, markers 2+ are missing.
    write_biallelic_vcf(
        &target_path,
        &positions,
        &["Target"],
        &[vec![
            "0/0".to_string(),
            "0/0".to_string(),
            "./.".to_string(),
            "./.".to_string(),
            "./.".to_string(),
        ]],
    );

    let mut config = Config::default();
    config.gt = target_path;
    config.r#ref = Some(ref_path);
    config.out = out_prefix.clone();

    // Encourage frequent reselection / block boundaries.
    config.cluster = 0.005;
    config.imp_step = 0.05;
    config.imp_segment = 0.05;
    config.window = 1.0;
    config.overlap = 0.2;

    config.err = Some(0.0001);
    config.ne = 10_000.0;
    config.imp_states = 10;
    config.nthreads = Some(1);

    let mut pipeline = ImputationPipeline::new(config, None);
    pipeline.run().expect("pipeline run");

    let out_vcf = temp_dir.path().join("bait_switch.vcf.gz");
    assert!(out_vcf.exists(), "expected output VCF to exist");

    let ds = read_single_sample_ds_by_record_index(&out_vcf);
    assert_eq!(ds.len(), n_markers);

    // Marker 2 should impute 0.0 (matching B_Good). If the state identity was scrambled at a
    // block boundary, mass can transfer to A_Decoy and DS becomes high.
    assert!(
        ds[2] < 0.1,
        "State transition mismatch suspected: expected DS ~0.0 at marker 2, got {}",
        ds[2]
    );
}
