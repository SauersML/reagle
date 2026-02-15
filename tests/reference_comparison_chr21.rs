//! Full-chromosome reference comparison on chr21 with larger cohorts.
//!
//! - Reference panel: 1,000 samples
//! - Target samples: 10
//! - Region: entire chr21

use std::path::Path;
use std::process::Command;
use std::{env, fs};
use std::time::Instant;

use reagle::{Config, ImputationPipeline};
use serial_test::serial;

mod common;
use common::{cache_dir, download_if_missing, generate_test_data};
#[path = "common/reference_metrics.rs"]
mod reference_metrics;
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

fn run_beagle_imputation(beagle_jar: &Path, ref_vcf: &Path, target_vcf: &Path, out_prefix: &Path) -> f64 {
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

fn write_metrics_artifact(
    beagle: &reference_metrics::FastMetrics,
    reagle: &reference_metrics::FastMetrics,
    beagle_runtime_sec: f64,
    reagle_runtime_sec: f64,
) {
    let fmt_opt = |x: Option<f64>| {
        x.map(|v| format!("{:.8}", v))
            .unwrap_or_else(|| "".to_string())
    };
    let exp_dir = env::var("EXP_DIR").ok().map(std::path::PathBuf::from);
    let out_path = exp_dir
        .map(|p| p.join("chr21_fast_metrics.tsv"))
        .unwrap_or_else(|| std::path::PathBuf::from("chr21_fast_metrics.tsv"));
    let parent = out_path.parent().map(|p| p.to_path_buf());
    if let Some(p) = parent {
        let _ = fs::create_dir_all(p);
    }
    let mut rows = String::new();
    rows.push_str("tool\truntime_sec\tsites_compared\tgenotypes_compared\tr_squared\tiqs\thellinger_score\tswitch_error_rate\tswitch_errors\tswitch_opportunities\tphase_concordance\tphase_concordant\tphase_total\tiqs_sites\n");
    rows.push_str(&format!(
        "beagle\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        beagle_runtime_sec,
        beagle.sites_compared,
        beagle.genotypes_compared,
        fmt_opt(beagle.r_squared),
        fmt_opt(beagle.iqs),
        fmt_opt(beagle.hellinger_score),
        fmt_opt(beagle.switch_error_rate),
        beagle.switch_errors,
        beagle.switch_opportunities,
        fmt_opt(beagle.phase_concordance),
        beagle.phase_concordant,
        beagle.phase_total,
        beagle.iqs_sites
    ));
    rows.push_str(&format!(
        "reagle\t{:.6}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
        reagle_runtime_sec,
        reagle.sites_compared,
        reagle.genotypes_compared,
        fmt_opt(reagle.r_squared),
        fmt_opt(reagle.iqs),
        fmt_opt(reagle.hellinger_score),
        fmt_opt(reagle.switch_error_rate),
        reagle.switch_errors,
        reagle.switch_opportunities,
        fmt_opt(reagle.phase_concordance),
        reagle.phase_concordant,
        reagle.phase_total,
        reagle.iqs_sites
    ));
    fs::write(&out_path, rows).expect("write chr21 fast metrics artifact");
    println!("Saved fast metrics TSV: {}", out_path.display());
}

#[test]
#[serial]
fn test_reference_comparison_full_chr21_ref1000_target10() {
    let jar_dir = cache_dir();
    let beagle_jar = jar_dir.join("beagle.27Feb25.75f.jar");
    assert!(
        download_if_missing(BEAGLE_JAR_URL, &beagle_jar),
        "Failed to download BEAGLE jar"
    );

    let data = generate_test_data(1000, 10, "chr21");
    let work_dir = tempfile::tempdir().expect("Create work dir");

    let beagle_out = work_dir.path().join("beagle_out");
    let reagle_out = work_dir.path().join("reagle_out");

    let beagle_runtime_sec =
        run_beagle_imputation(&beagle_jar, &data.ref_vcf, &data.target_sparse_vcf, &beagle_out);
    let reagle_runtime_sec =
        run_reagle_imputation(&data.ref_vcf, &data.target_sparse_vcf, &reagle_out);

    let beagle_vcf = beagle_out.with_extension("vcf.gz");
    let reagle_vcf = reagle_out.with_extension("vcf.gz");
    assert!(beagle_vcf.exists(), "Missing BEAGLE output {}", beagle_vcf.display());
    assert!(reagle_vcf.exists(), "Missing Reagle output {}", reagle_vcf.display());

    let beagle_n = count_records(&beagle_vcf);
    let reagle_n = count_records(&reagle_vcf);
    assert!(beagle_n > 0, "BEAGLE produced zero records");
    assert!(reagle_n > 0, "Reagle produced zero records");
    assert_eq!(
        beagle_n, reagle_n,
        "Record count mismatch on chr21 full comparison"
    );

    // Fast quality metrics only (no heavy IQS/N50-style bookkeeping).
    let beagle_metrics = compute_fast_metrics(&data.target_vcf, &beagle_vcf);
    let reagle_metrics = compute_fast_metrics(&data.target_vcf, &reagle_vcf);

    println!("=== Fast Accuracy Metrics (chr21, ref=1000, target=10) ===");
    println!(
        "Timing: BEAGLE={:.3}s REAGLE={:.3}s",
        beagle_runtime_sec, reagle_runtime_sec
    );
    println!(
        "BEAGLE: sites={} genotypes={} r2={:?} iqs={:?} hellinger={:?} switch={:?} ({}/{}) phase_conc={:?} ({}/{})",
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
        "REAGLE: sites={} genotypes={} r2={:?} iqs={:?} hellinger={:?} switch={:?} ({}/{}) phase_conc={:?} ({}/{})",
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
    write_metrics_artifact(
        &beagle_metrics,
        &reagle_metrics,
        beagle_runtime_sec,
        reagle_runtime_sec,
    );

    // Sanity gates only: keep this test stable while still validating metric extraction.
    assert!(
        reagle_metrics.r_squared.is_some(),
        "Missing Reagle dosage R² (DS not parsed?)"
    );
    assert!(reagle_metrics.iqs.is_some(), "Missing Reagle IQS");
    assert!(
        reagle_metrics.hellinger_score.is_some(),
        "Missing Reagle Hellinger score (GP not parsed?)"
    );
    assert!(
        reagle_metrics.phase_concordance.is_some(),
        "Missing Reagle phase concordance"
    );
    assert!(
        reagle_metrics.switch_error_rate.is_some(),
        "Missing Reagle switch error rate"
    );
}
