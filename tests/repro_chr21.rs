//! Full-chromosome reference comparison on chr21 with larger cohorts.
//!
//! - Reference panel: 1,000 samples
//! - Target samples: 10
//! - Region: entire chr21

use std::path::Path;
use std::process::Command;
use std::time::Instant;

use reagle::{Config, ImputationPipeline};
use serial_test::serial;

mod common;
#[path = "common/reference_metrics.rs"]
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

#[test]
#[serial]
fn test_repro_chr21_subset() {
    let jar_dir = cache_dir();
    let beagle_jar = jar_dir.join("beagle.27Feb25.75f.jar");
    assert!(
        download_if_missing(BEAGLE_JAR_URL, &beagle_jar),
        "Failed to download BEAGLE jar"
    );

    // Use a subset of chr21 to be faster
    let data = generate_test_data(100, 10, "chr21:16000000-17000000"); // 1MB region, fewer samples
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
    assert!(
        beagle_vcf.exists(),
        "Missing BEAGLE output {}",
        beagle_vcf.display()
    );
    assert!(
        reagle_vcf.exists(),
        "Missing Reagle output {}",
        reagle_vcf.display()
    );

    let beagle_n = count_records(&beagle_vcf);
    let reagle_n = count_records(&reagle_vcf);
    assert!(beagle_n > 0, "BEAGLE produced zero records");
    assert!(reagle_n > 0, "Reagle produced zero records");
    assert_eq!(
        beagle_n, reagle_n,
        "Record count mismatch on chr21 subset comparison"
    );

    // Fast quality metrics only (no heavy IQS/N50-style bookkeeping).
    let beagle_metrics = compute_fast_metrics(&data.target_vcf, &beagle_vcf);
    let reagle_metrics = compute_fast_metrics(&data.target_vcf, &reagle_vcf);

    println!("=== Fast Accuracy Metrics (chr21 subset) ===");
    println!(
        "Timing: BEAGLE={:.3}s REAGLE={:.3}s",
        beagle_runtime_sec, reagle_runtime_sec
    );
    println!(
        "BEAGLE: sites={} genotypes={} r2={:?} iqs={:?} iqs_sites={} hellinger={:?} SER={:?} ({}/{}) phase_conc={:?} ({}/{})",
        beagle_metrics.sites_compared,
        beagle_metrics.genotypes_compared,
        beagle_metrics.r_squared,
        beagle_metrics.iqs,
        beagle_metrics.iqs_sites,
        beagle_metrics.hellinger_score,
        beagle_metrics.switch_error_rate,
        beagle_metrics.switch_errors,
        beagle_metrics.switch_opportunities,
        beagle_metrics.phase_concordance,
        beagle_metrics.phase_concordant,
        beagle_metrics.phase_total
    );
    println!(
        "REAGLE: sites={} genotypes={} r2={:?} iqs={:?} iqs_sites={} hellinger={:?} SER={:?} ({}/{}) phase_conc={:?} ({}/{})",
        reagle_metrics.sites_compared,
        reagle_metrics.genotypes_compared,
        reagle_metrics.r_squared,
        reagle_metrics.iqs,
        reagle_metrics.iqs_sites,
        reagle_metrics.hellinger_score,
        reagle_metrics.switch_error_rate,
        reagle_metrics.switch_errors,
        reagle_metrics.switch_opportunities,
        reagle_metrics.phase_concordance,
        reagle_metrics.phase_concordant,
        reagle_metrics.phase_total
    );

    let reagle_r2 = reagle_metrics.r_squared.unwrap();
    let beagle_r2 = beagle_metrics.r_squared.unwrap();
    assert!(
        reagle_r2 >= beagle_r2,
        "Reagle worse than Beagle on R²: reagle={:.12}, beagle={:.12}",
        reagle_r2,
        beagle_r2
    );

    let reagle_iqs = reagle_metrics.iqs.unwrap();
    let beagle_iqs = beagle_metrics.iqs.unwrap();
    assert!(
        reagle_iqs >= beagle_iqs,
        "Reagle worse than Beagle on IQS: reagle={:.12}, beagle={:.12}",
        reagle_iqs,
        beagle_iqs
    );

    let reagle_hellinger = reagle_metrics.hellinger_score.unwrap();
    let beagle_hellinger = beagle_metrics.hellinger_score.unwrap();
    assert!(
        reagle_hellinger <= beagle_hellinger,
        "Reagle worse than Beagle on Hellinger (lower is better): reagle={:.12}, beagle={:.12}",
        reagle_hellinger,
        beagle_hellinger
    );

    let reagle_ser = reagle_metrics.switch_error_rate.unwrap();
    let beagle_ser = beagle_metrics.switch_error_rate.unwrap();
    assert!(
        reagle_ser <= beagle_ser,
        "Reagle worse than Beagle on SER/switch error rate (lower is better): reagle={:.12}, beagle={:.12}",
        reagle_ser,
        beagle_ser
    );
}
