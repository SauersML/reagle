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

fn write_trap_vcfs(
    n_markers: usize,
    n_ref_samples: usize,
    positions_bp: &[u32],
    ref_path: &Path,
    target_path: &Path,
    hero_sample_idx: usize,
    masked_marker_indices: &[usize],
) {
    assert_eq!(positions_bp.len(), n_markers);
    assert!(hero_sample_idx < n_ref_samples);

    // Reference VCF (phased)
    {
        let mut f = File::create(ref_path).expect("create ref VCF");
        writeln!(f, "##fileformat=VCFv4.2").unwrap();
        writeln!(f, "##FILTER=<ID=PASS,Description=\"All filters passed\">").unwrap();
        writeln!(
            f,
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">"
        )
        .unwrap();
        writeln!(f, "##contig=<ID=chr1,length=100000000>").unwrap();

        write!(f, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT").unwrap();
        for s in 0..n_ref_samples {
            write!(f, "\tRef{}", s).unwrap();
        }
        writeln!(f).unwrap();

        let hero_hap_a = hero_sample_idx * 2;
        let hero_hap_b = hero_hap_a + 1;

        for m in 0..n_markers {
            write!(
                f,
                "chr1\t{}\trs{}\tA\tT\t.\tPASS\t.\tGT",
                positions_bp[m], m
            )
            .unwrap();

            // Rotating waves: which small subset of distractors matches at this marker.
            let block_size = 10;
            let active_group = (m / block_size) % 20;

            for s in 0..n_ref_samples {
                let h0 = s * 2;
                let h1 = h0 + 1;

                let a0 = allele_for_hap(h0, m, hero_hap_a, hero_hap_b, active_group);
                let a1 = allele_for_hap(h1, m, hero_hap_a, hero_hap_b, active_group);

                write!(f, "\t{}|{}", a0, a1).unwrap();
            }
            writeln!(f).unwrap();
        }
    }

    // Target VCF (unphased)
    {
        let mut f = File::create(target_path).expect("create target VCF");
        writeln!(f, "##fileformat=VCFv4.2").unwrap();
        writeln!(f, "##FILTER=<ID=PASS,Description=\"All filters passed\">").unwrap();
        writeln!(
            f,
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">"
        )
        .unwrap();
        writeln!(f, "##contig=<ID=chr1,length=100000000>").unwrap();
        writeln!(
            f,
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTarget"
        )
        .unwrap();

        for m in 0..n_markers {
            let gt = if masked_marker_indices.contains(&m) {
                "./.".to_string()
            } else {
                "0/0".to_string()
            };

            writeln!(
                f,
                "chr1\t{}\trs{}\tA\tT\t.\tPASS\t.\tGT\t{}",
                positions_bp[m], m, gt
            )
            .unwrap();
        }
    }
}

fn allele_for_hap(
    hap_idx: usize,
    marker_idx: usize,
    hero_hap_a: usize,
    hero_hap_b: usize,
    active_group: usize,
) -> u8 {
    if hap_idx == hero_hap_a || hap_idx == hero_hap_b {
        return 0;
    }

    // Rotate which hap-group matches.
    // This is deterministic and creates neighbor churn.
    let hap_group = (hap_idx + 7 * marker_idx) % 20;
    if hap_group == active_group { 0 } else { 1 }
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

#[test]
#[serial]
fn test_state_index_stability_trap() {
    let n_markers = 200;
    let n_ref_samples = 100;

    // Default genetic map is 1cM/Mb => 100kb = 0.1cM spacing.
    let positions: Vec<u32> = (0..n_markers).map(|m| (m as u32) * 100_000 + 1).collect();

    let masked = [50usize, 100usize, 150usize];

    let temp_dir = tempfile::tempdir().unwrap();
    let ref_path = temp_dir.path().join("ref.vcf");
    let target_path = temp_dir.path().join("target.vcf");
    let out_prefix = temp_dir.path().join("trap_output");

    // Put the hero at a high index to maximize index shifting when the selected neighbor list churns.
    let hero_sample_idx = n_ref_samples - 1;

    write_trap_vcfs(
        n_markers,
        n_ref_samples,
        &positions,
        &ref_path,
        &target_path,
        hero_sample_idx,
        &masked,
    );

    let mut config = Config::default();
    config.gt = target_path;
    config.r#ref = Some(ref_path);
    config.out = out_prefix.clone();

    config.imp_states = 20;
    config.ne = 10_000.0;
    config.window = 40.0;
    config.overlap = 2.0;
    config.nthreads = Some(1);

    let mut pipeline = ImputationPipeline::new(config, None);
    pipeline.run().expect("pipeline run");

    let out_vcf = temp_dir.path().join("trap_output.vcf.gz");
    assert!(out_vcf.exists(), "expected output VCF to exist");

    let ds = read_single_sample_ds_by_record_index(&out_vcf);
    assert_eq!(ds.len(), n_markers);

    for &idx in &masked {
        assert!(
            ds[idx] < 0.1,
            "Trap triggered at marker {}: expected DS ~0.0, got {}",
            idx,
            ds[idx]
        );
    }
}
