
use std::path::Path;
use std::process::Command;
use std::time::Instant;
use std::{env, fs};

use reagle::{Config, ImputationPipeline};

#[path = "tests/common/mod.rs"]
mod common;
#[path = "tests/common/reference_metrics.rs"]
mod reference_metrics;
use common::{cache_dir, download_if_missing, generate_test_data};
use reference_metrics::compute_fast_metrics;

const BEAGLE_JAR_URL: &str =
    "https://faculty.washington.edu/browning/beagle/beagle.27Feb25.75f.jar";

fn count_records(vcf_gz: &Path) -> usize {
    let output = Command::new("gzip")
        .args(["-dc", vcf_gz.to_string_lossy().as_ref()])
        .output()
        .expect("Run gzip -dc");
    assert!(
        output.status.success(),
        "Failed to decompress {}: {}",
        vcf_gz.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| !line.starts_with('#'))
        .count()
}

fn run_beagle_imputation(
    beagle_jar: &Path,
    ref_vcf: &Path,
    target_vcf: &Path,
    out_prefix: &Path,
) -> f64 {
    let start = Instant::now();
    let status = Command::new("java")
        .arg("-Xmx16g")
        .arg("-jar")
        .arg(beagle_jar)
        .arg(format!("ref={}", ref_vcf.to_string_lossy()))
        .arg(format!("gt={}", target_vcf.to_string_lossy()))
        .arg(format!("out={}", out_prefix.to_string_lossy()))
        .arg("gp=true")
        .arg("nthreads=4")
        .status()
        .expect("Run BEAGLE");

    assert!(status.success(), "BEAGLE imputation failed");
    start.elapsed().as_secs_f64()
}

fn run_reagle_imputation(ref_vcf: &Path, target_vcf: &Path, out_prefix: &Path) -> f64 {
    let start = Instant::now();
    let cfg = Config::parse_from([
        "reagle",
        "--target",
        target_vcf
            .to_str()
            .expect("target path contains invalid UTF-8"),
        "--ref",
        ref_vcf.to_str().expect("ref path contains invalid UTF-8"),
        "--out",
        out_prefix
            .to_str()
            .expect("out path contains invalid UTF-8"),
    ])
    .expect("Build config from CLI + reagle.toml");

    let mut pipeline = ImputationPipeline::new(cfg, None);
    pipeline.run().expect("Run Reagle imputation");
    start.elapsed().as_secs_f64()
}

fn main() {
    let jar_dir = cache_dir();
    let beagle_jar = jar_dir.join("beagle.27Feb25.75f.jar");
    if !download_if_missing(BEAGLE_JAR_URL, &beagle_jar) {
        panic!("Failed to download BEAGLE jar");
    }

    // Use a smaller region for reproduction
    let region = "chr21:15000000-20000000";
    let data = generate_test_data(200, 10, region); // Reduced ref samples too to be faster
    let work_dir = tempfile::tempdir().expect("Create work dir");

    let beagle_out = work_dir.path().join("beagle_out");
    let reagle_out = work_dir.path().join("reagle_out");

    let beagle_runtime_sec = run_beagle_imputation(
        &beagle_jar,
        &data.ref_vcf,
        &data.target_sparse_vcf,
        &beagle_out,
    );
    let reagle_runtime_sec =
        run_reagle_imputation(&data.ref_vcf, &data.target_sparse_vcf, &reagle_out);

    let beagle_vcf = beagle_out.with_extension("vcf.gz");
    let reagle_vcf = reagle_out.with_extension("vcf.gz");

    let beagle_metrics = compute_fast_metrics(&data.target_vcf, &beagle_vcf);
    let reagle_metrics = compute_fast_metrics(&data.target_vcf, &reagle_vcf);

    println!("=== Fast Accuracy Metrics (chr21 subset, ref=200, target=10) ===");
    println!(
        "Timing: BEAGLE={:.3}s REAGLE={:.3}s",
        beagle_runtime_sec, reagle_runtime_sec
    );
    println!(
        "BEAGLE: sites={} genotypes={} r2={:?} iqs={:?} hellinger={:?} SER={:?} ({}/{}) phase_conc={:?} ({}/{})",
        beagle_metrics.sites_compared,
        beagle_metrics.genotypes_compared,
        beagle_metrics.r_squared,
        beagle_metrics.iqs,
        beagle_metrics.hellinger_score,
        beagle_metrics.switch_error_rate,
        beagle_metrics.switch_errors,
        beagle_metrics.switch_opportunities,
        beagle_metrics.phase_concordance,
        beagle_metrics.phase_concordant,
        beagle_metrics.phase_total
    );
    println!(
        "REAGLE: sites={} genotypes={} r2={:?} iqs={:?} hellinger={:?} SER={:?} ({}/{}) phase_conc={:?} ({}/{})",
        reagle_metrics.sites_compared,
        reagle_metrics.genotypes_compared,
        reagle_metrics.r_squared,
        reagle_metrics.iqs,
        reagle_metrics.hellinger_score,
        reagle_metrics.switch_error_rate,
        reagle_metrics.switch_errors,
        reagle_metrics.switch_opportunities,
        reagle_metrics.phase_concordance,
        reagle_metrics.phase_concordant,
        reagle_metrics.phase_total
    );

    let reagle_r2 = reagle_metrics.r_squared.expect("reagle r2");
    let beagle_r2 = beagle_metrics.r_squared.expect("beagle r2");
    if reagle_r2 < beagle_r2 {
        println!("FAIL: Reagle worse than Beagle on R²: reagle={:.12}, beagle={:.12}", reagle_r2, beagle_r2);
    } else {
        println!("PASS: Reagle better or equal on R²");
    }
}
