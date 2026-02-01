use reagle::config::Config;
use reagle::pipelines::imputation::ImputationPipeline;

use noodles::bgzf::io as bgzf_io;
use noodles::vcf as noodles_vcf;
use noodles_vcf::variant::record::samples::Sample;
use noodles_vcf::variant::record::samples::series::Value;
use noodles_vcf::variant::record::samples::series::value::Array;

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
    writeln!(
        f,
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">"
    )
    .unwrap();
    writeln!(f, "##contig=<ID=chr1,length=1000000>").unwrap();

    write!(f, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT").unwrap();
    for s in sample_names {
        write!(f, "\t{}", s).unwrap();
    }
    writeln!(f).unwrap();

    for (m, pos) in positions_bp.iter().copied().enumerate() {
        write!(f, "chr1\t{}\trs{}\tA\tT\t.\tPASS\t.\tGT", pos, m).unwrap();

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
    assert!(
        header.formats().contains_key("DS"),
        "DS missing from output"
    );

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
            Some(Value::Array(Array::Float(values))) => {
                let v = values
                    .iter()
                    .next()
                    .expect("empty array")
                    .expect("invalid value")
                    .expect("missing value");
                v as f64
            }
            Some(other) => panic!("Unexpected DS value: {other:?}"),
            None => panic!("Missing DS"),
        };

        ds.push(parsed);
    }

    ds
}

fn read_sample_ds_by_record_index(path: &Path, sample_index: usize) -> Vec<f64> {
    let file = File::open(path).expect("open output VCF");
    let decoder = bgzf_io::Reader::new(file);
    let mut reader = noodles_vcf::io::Reader::new(decoder);

    let header = reader.read_header().expect("read VCF header");
    assert!(
        header.formats().contains_key("DS"),
        "DS missing from output"
    );

    let mut ds = Vec::new();

    for result in reader.records() {
        let record = result.expect("read VCF record");
        let samples = record.samples();
        let sample = samples
            .get_index(sample_index)
            .unwrap_or_else(|| panic!("sample {sample_index}"));

        let val = sample
            .get(&header, "DS")
            .transpose()
            .ok()
            .flatten()
            .flatten();

        let parsed = match val {
            Some(Value::Float(f)) => f as f64,
            Some(Value::String(s)) => s.parse::<f64>().expect("parse DS"),
            Some(Value::Array(Array::Float(values))) => {
                let v = values
                    .iter()
                    .next()
                    .expect("empty array")
                    .expect("invalid value")
                    .expect("missing value");
                v as f64
            }
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
    let n_markers = 6;
    let positions: Vec<u32> = (0..n_markers).map(|m| m as u32 + 1).collect();

    let temp_dir = tempfile::tempdir().unwrap();
    let ref_path = temp_dir.path().join("ref.vcf");
    let target_path = temp_dir.path().join("target.vcf");
    let out_prefix = temp_dir.path().join("bait_switch");

    // Reference: Two phased samples.
    // - B_Good is a perfect match (all 0s)
    // - A_Decoy matches early then diverges after the boundary.
    write_biallelic_vcf(
        &ref_path,
        &positions,
        &["B_Good", "A_Decoy"],
        &[
            vec!["0|0".to_string(); n_markers],
            vec![
                "0|0".to_string(),
                "0|0".to_string(),
                "0|0".to_string(),
                "1|1".to_string(),
                "1|1".to_string(),
                "1|1".to_string(),
            ],
        ],
    );

    // Target: all observed 0/0 to keep emissions informative across boundaries.
    write_biallelic_vcf(
        &target_path,
        &positions,
        &["Target"],
        &[vec!["0/0".to_string(); n_markers]],
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

    eprintln!("bait/switch DS: {:?}", ds);

    // Markers 3+ should impute 0.0 (matching B_Good). If the state identity was scrambled at a
    // block boundary, mass can transfer to A_Decoy and DS becomes high even though observed 0/0.
    assert!(
        ds[3] < 0.05,
        "State transition mismatch suspected: expected DS ~0.0 at marker 3, got {}",
        ds[3]
    );
}

#[test]
#[serial]
fn test_clean_constructed_reference_consistency() {
    let n_markers = 8;
    let positions: Vec<u32> = (0..n_markers).map(|m| m as u32 + 1).collect();

    let temp_dir = tempfile::tempdir().unwrap();
    let ref_path = temp_dir.path().join("ref.vcf");
    let target_path = temp_dir.path().join("target.vcf");
    let out_prefix = temp_dir.path().join("clean_constructed");

    // Reference: three phased samples
    // - Ref0: all 0s
    // - Ref1: all 1s
    // - Ref2: switch at marker 4 (bait/switch)
    write_biallelic_vcf(
        &ref_path,
        &positions,
        &["Ref0", "Ref1", "Ref2"],
        &[
            vec!["0|0".to_string(); n_markers],
            vec!["1|1".to_string(); n_markers],
            vec![
                "0|0".to_string(),
                "0|0".to_string(),
                "0|0".to_string(),
                "0|0".to_string(),
                "1|1".to_string(),
                "1|1".to_string(),
                "1|1".to_string(),
                "1|1".to_string(),
            ],
        ],
    );

    // Target: two phased samples with clean truth
    write_biallelic_vcf(
        &target_path,
        &positions,
        &["T0", "T1"],
        &[
            vec!["0|0".to_string(); n_markers],
            vec!["1|1".to_string(); n_markers],
        ],
    );

    let mut config = Config::default();
    config.gt = target_path;
    config.r#ref = Some(ref_path);
    config.out = out_prefix.clone();

    // Keep recombination tiny and deterministic.
    config.cluster = 0.001;
    config.imp_step = 0.001;
    config.imp_segment = 0.001;
    config.window = 1.0;
    config.overlap = 0.2;

    config.err = Some(0.0001);
    config.ne = 10_000.0;
    config.imp_states = 10;
    config.nthreads = Some(1);

    let mut pipeline = ImputationPipeline::new(config, None);
    pipeline.run().expect("pipeline run");

    let out_vcf = temp_dir.path().join("clean_constructed.vcf.gz");
    assert!(out_vcf.exists(), "expected output VCF to exist");

    let ds_t0 = read_sample_ds_by_record_index(&out_vcf, 0);
    let ds_t1 = read_sample_ds_by_record_index(&out_vcf, 1);
    assert_eq!(ds_t0.len(), n_markers);
    assert_eq!(ds_t1.len(), n_markers);

    eprintln!("clean constructed DS T0: {:?}", ds_t0);
    eprintln!("clean constructed DS T1: {:?}", ds_t1);

    // T0 should stay near 0 across all markers, including after Ref2 switches to 1s.
    for (m, ds) in ds_t0.iter().copied().enumerate() {
        assert!(
            ds < 0.05,
            "T0 expected DS ~0.0 at marker {m}, got {ds}"
        );
    }

    // T1 should stay near 2 across all markers.
    for (m, ds) in ds_t1.iter().copied().enumerate() {
        assert!(
            ds > 1.95,
            "T1 expected DS ~2.0 at marker {m}, got {ds}"
        );
    }
}
