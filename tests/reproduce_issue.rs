
use reagle::Config;
use reagle::pipelines::phasing::PhasingPipeline;
use std::fs::File;
use std::io::Write;
use std::path::Path;

fn write_vcf(path: &Path, content: &str) {
    let mut file = File::create(path).expect("create vcf");
    file.write_all(content.as_bytes()).expect("write vcf");
}

fn write_synthetic_vcf<F>(
    path: &Path,
    n_markers: usize,
    sample_names: &[&str],
    gt_at: F,
) where
    F: Fn(usize, usize) -> String,
{
    let mut content = String::new();
    content.push_str("##fileformat=VCFv4.2\n");
    content.push_str("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n");
    content.push_str("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT");
    for name in sample_names {
        content.push('\t');
        content.push_str(name);
    }
    content.push('\n');
    for i in 0..n_markers {
        let pos = (i as u32 + 1) * 1000;
        content.push_str(&format!(
            "chr1\t{}\t.\tA\tC\t.\tPASS\t.\tGT",
            pos
        ));
        for s in 0..sample_names.len() {
            content.push('\t');
            content.push_str(&gt_at(i, s));
        }
        content.push('\n');
    }
    write_vcf(path, &content);
}

fn run_rust_phasing_with_seed(
    gt_path: &Path,
    ref_path: &Path,
    out_prefix: &Path,
    seed: i64,
) -> reagle::Result<std::path::PathBuf> {
    let config = Config::parse_from([
        "reagle",
        "--target",
        gt_path.to_str().unwrap(),
        "--ref",
        ref_path.to_str().unwrap(),
        "--out",
        out_prefix.to_str().unwrap(),
        "--seed",
        &seed.to_string(),
    ])
    .expect("config");
    let mut pipeline = PhasingPipeline::new(config, None);
    pipeline.run()?;
    Ok(out_prefix.with_extension("vcf.gz"))
}

fn parse_vcf(path: &Path) -> Vec<String> {
    let output = std::process::Command::new("gzip")
        .args(["-dc", path.to_str().unwrap()])
        .output()
        .expect("Failed to run gzip");
    let content = String::from_utf8_lossy(&output.stdout);
    let mut gts = Vec::new();
    for line in content.lines() {
        if line.starts_with('#') { continue; }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 10 { continue; }
        let gt = fields[9].split(':').next().unwrap().to_string();
        gts.push(gt);
    }
    gts
}

#[test]
fn test_reproduce_ambiguous_phasing_determinism() {
    let work_dir = tempfile::tempdir().expect("Create temp dir");
    let ref_vcf = work_dir.path().join("ref.vcf");
    let target_vcf = work_dir.path().join("target.vcf");

    let n_markers = 60; // Reduced from 60 for speed, enough to show effect
    let ref_samples = ["R1", "R2"];
    let target_samples = ["T1"];

    write_synthetic_vcf(&ref_vcf, n_markers, &ref_samples, |_, s| {
        if s == 0 { "0|0".to_string() } else { "1|1".to_string() }
    });
    write_synthetic_vcf(&target_vcf, n_markers, &target_samples, |_, _| {
        "0/1".to_string()
    });

    let out_prefix_a = work_dir.path().join("out_a");
    let out_prefix_b = work_dir.path().join("out_b");

    let out_a = run_rust_phasing_with_seed(&target_vcf, &ref_vcf, &out_prefix_a, 12345)
        .expect("Phasing run A failed");
    let out_b = run_rust_phasing_with_seed(&target_vcf, &ref_vcf, &out_prefix_b, 67890)
        .expect("Phasing run B failed");

    let gts_a = parse_vcf(&out_a);
    let gts_b = parse_vcf(&out_b);

    assert_eq!(gts_a.len(), n_markers);
    assert_eq!(gts_b.len(), n_markers);

    let mut varied = false;
    for (ga, gb) in gts_a.iter().zip(gts_b.iter()) {
        if ga != gb {
            varied = true;
            break;
        }
    }

    assert!(varied, "Expected phasing to vary across seeds for ambiguous signal, but results were identical");
}
