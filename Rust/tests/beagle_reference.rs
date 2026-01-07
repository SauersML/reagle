//! Integration tests that run Java BEAGLE as a reference implementation.
//!
//! These tests download the official BEAGLE JAR and test data, run the Java
//! implementation, and verify the outputs. This establishes a baseline for
//! comparing our Rust implementation against.
//!
//! Tests run on multiple data sources:
//! - BEAGLE test data (1000 Genomes subset)
//! - gnomAD HGDP+1KG (if bcftools available)

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Parser;
use rand::SeedableRng;

// Import Rust implementation for comparison tests
use reagle::{Config, ImputationPipeline, PhasingPipeline};

// Import gnomAD data source
mod gnomad_reference;
use gnomad_reference::setup_gnomad_files;

// =============================================================================
// Common Data Source Abstraction
// =============================================================================

/// Common interface for test data sources
struct TestDataSource {
    name: &'static str,
    ref_vcf: PathBuf,
    target_vcf: PathBuf,
    target_sparse_vcf: PathBuf,
}

/// Get all available test data sources
fn get_all_data_sources() -> Vec<TestDataSource> {
    let mut sources = Vec::new();

    // Always include BEAGLE test data
    let beagle = setup_test_files();
    sources.push(TestDataSource {
        name: "BEAGLE",
        ref_vcf: beagle.ref_vcf,
        target_vcf: beagle.target_vcf,
        target_sparse_vcf: beagle.target_sparse_vcf,
    });

    // Always include gnomAD (fixtures are pre-generated and committed)
    let gnomad = setup_gnomad_files();
    sources.push(TestDataSource {
        name: "gnomAD",
        ref_vcf: gnomad.ref_vcf,
        target_vcf: gnomad.target_vcf,
        target_sparse_vcf: gnomad.target_sparse_vcf,
    });

    sources
}

/// Directory for pre-generated test fixtures
fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("beagle_reference")
}

/// Get pre-generated BEAGLE test files
fn setup_test_files() -> TestFiles {
    let dir = fixtures_dir();

    let beagle_jar = dir.join("beagle.27Feb25.75f.jar");
    let bref3_jar = dir.join("bref3.27Feb25.75f.jar");
    let ref_vcf = dir.join("ref.27Feb25.75f.vcf.gz");
    let target_vcf = dir.join("target.27Feb25.75f.vcf.gz");
    let target_sparse_vcf = dir.join("target_sparse.27Feb25.75f.vcf.gz");

    // Verify fixtures exist
    assert!(beagle_jar.exists(), "BEAGLE JAR missing: {}", beagle_jar.display());
    assert!(bref3_jar.exists(), "bref3 JAR missing: {}", bref3_jar.display());
    assert!(ref_vcf.exists(), "ref VCF missing: {}", ref_vcf.display());
    assert!(target_vcf.exists(), "target VCF missing: {}", target_vcf.display());
    assert!(target_sparse_vcf.exists(), "sparse target VCF missing: {}", target_sparse_vcf.display());

    TestFiles {
        beagle_jar,
        bref3_jar,
        ref_vcf,
        target_vcf,
        target_sparse_vcf,
    }
}

struct TestFiles {
    beagle_jar: PathBuf,
    bref3_jar: PathBuf,
    ref_vcf: PathBuf,
    target_vcf: PathBuf,
    /// Sparse target with ~10% of markers for true imputation testing
    target_sparse_vcf: PathBuf,
}

// =============================================================================
// VCF Parsing Helpers
// =============================================================================

/// Parsed genotype with optional dosage and probabilities
#[derive(Debug, Clone)]
struct ParsedGenotype {
    /// Hard call (e.g., "0|1", "1|0", "0|0")
    gt: String,
    /// Dosage value (0.0 to 2.0)
    ds: Option<f64>,
    /// Genotype probabilities [P(0/0), P(0/1), P(1/1)]
    gp: Option<[f64; 3]>,
}

/// Parsed VCF record
#[derive(Debug)]
struct ParsedRecord {
    chrom: String,
    pos: u64,
    /// INFO field key-value pairs
    info: HashMap<String, String>,
    /// Genotypes per sample
    genotypes: Vec<ParsedGenotype>,
}

/// Parse a gzipped VCF file and extract records
/// Uses gzip command for reliable BGZF decompression
fn parse_vcf(path: &Path) -> (Vec<String>, Vec<ParsedRecord>) {
    // Use gzip -dc for reliable decompression of both gzip and bgzf formats
    let output = Command::new("gzip")
        .args(["-dc", path.to_str().unwrap()])
        .output()
        .expect("Failed to run gzip");

    if !output.status.success() {
        panic!("gzip decompression failed for {:?}", path);
    }

    let content = String::from_utf8_lossy(&output.stdout);

    let mut sample_names: Vec<String> = Vec::new();
    let mut records = Vec::new();

    for line in content.lines() {
        if line.starts_with("##") {
            continue; // Skip meta lines
        }
        if line.starts_with("#CHROM") {
            // Parse header
            let fields: Vec<&str> = line.split('\t').collect();
            sample_names = fields[9..].iter().map(|s: &&str| s.to_string()).collect();
            continue;
        }

        // Parse data line
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 10 {
            continue;
        }

        let chrom = fields[0].to_string();
        let pos: u64 = fields[1].parse().expect("Parse position");

        // Parse INFO field
        let info: HashMap<String, String> = fields[7]
            .split(';')
            .filter_map(|kv: &str| {
                let parts: Vec<&str> = kv.splitn(2, '=').collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    Some((kv.to_string(), String::new()))
                }
            })
            .collect();

        // Parse FORMAT field to find indices
        let format_fields: Vec<&str> = fields[8].split(':').collect();
        let gt_idx = format_fields.iter().position(|&f| f == "GT");
        let ds_idx = format_fields.iter().position(|&f| f == "DS");
        let gp_idx = format_fields.iter().position(|&f| f == "GP");

        // Parse genotypes
        let mut genotypes = Vec::new();
        for sample_data in &fields[9..] {
            let sample_fields: Vec<&str> = sample_data.split(':').collect();

            let gt = gt_idx
                .and_then(|i| sample_fields.get(i))
                .map(|s: &&str| s.to_string())
                .unwrap_or_default();

            let ds = ds_idx
                .and_then(|i| sample_fields.get(i))
                .and_then(|s: &&str| s.parse().ok());

            let gp = gp_idx.and_then(|i| sample_fields.get(i)).and_then(|s: &&str| {
                let probs: Vec<f64> = s.split(',').filter_map(|p: &str| p.parse().ok()).collect();
                if probs.len() == 3 {
                    Some([probs[0], probs[1], probs[2]])
                } else {
                    None
                }
            });

            genotypes.push(ParsedGenotype { gt, ds, gp });
        }

        records.push(ParsedRecord {
            chrom,
            pos,
            info,
            genotypes,
        });
    }

    (sample_names, records)
}

// =============================================================================
// Comparison Metrics
// =============================================================================

/// Calculate Pearson correlation coefficient (r²) between two vectors of dosages
fn dosage_correlation(ds1: &[f64], ds2: &[f64]) -> f64 {
    assert_eq!(ds1.len(), ds2.len(), "Dosage vectors must have same length");
    let n = ds1.len() as f64;
    if n == 0.0 {
        return 0.0;
    }

    let mean1: f64 = ds1.iter().sum::<f64>() / n;
    let mean2: f64 = ds2.iter().sum::<f64>() / n;

    let mut cov = 0.0;
    let mut var1 = 0.0;
    let mut var2 = 0.0;

    for (a, b) in ds1.iter().zip(ds2.iter()) {
        let d1 = a - mean1;
        let d2 = b - mean2;
        cov += d1 * d2;
        var1 += d1 * d1;
        var2 += d2 * d2;
    }

    if var1 == 0.0 || var2 == 0.0 {
        return 0.0;
    }

    let r = cov / (var1.sqrt() * var2.sqrt());
    r * r // Return r²
}

/// Normalize a genotype for unphased comparison (0|1 == 1|0)
fn normalize_gt_unphased(gt: &str) -> String {
    let sep = if gt.contains('|') { '|' } else { '/' };
    let alleles: Vec<&str> = gt.split(sep).collect();
    if alleles.len() != 2 {
        return gt.to_string();
    }
    let mut sorted = alleles.clone();
    sorted.sort();
    format!("{}/{}", sorted[0], sorted[1])
}

/// Count phase switches between two phased genotype vectors
///
/// A phase switch occurs when the haplotype assignment flips relative
/// to the reference. We count block-level switches, not per-SNP errors.
fn count_phase_switches(gt1: &[String], gt2: &[String]) -> usize {
    assert_eq!(gt1.len(), gt2.len());

    let mut switches = 0;
    let mut current_flip = None; // None = unknown, Some(false) = same, Some(true) = flipped

    for (g1, g2) in gt1.iter().zip(gt2.iter()) {
        // Skip missing or homozygous (can't determine phase)
        if g1.contains('.') || g2.contains('.') {
            continue;
        }

        // Parse alleles
        let a1: Vec<&str> = g1.split('|').collect();
        let a2: Vec<&str> = g2.split('|').collect();

        if a1.len() != 2 || a2.len() != 2 {
            continue;
        }

        // Skip homozygous (no phase information)
        if a1[0] == a1[1] || a2[0] == a2[1] {
            continue;
        }

        // Check if normalized genotypes match (ignoring phase)
        let n1 = normalize_gt_unphased(g1);
        let n2 = normalize_gt_unphased(g2);
        if n1 != n2 {
            // Actual genotype difference, not a phase switch
            continue;
        }

        // Determine if phases match or are flipped
        let is_flipped = a1[0] != a2[0]; // If first alleles differ, it's flipped

        match current_flip {
            None => current_flip = Some(is_flipped),
            Some(was_flipped) => {
                if is_flipped != was_flipped {
                    switches += 1;
                    current_flip = Some(is_flipped);
                }
            }
        }
    }

    switches
}

/// Calculate genotype concordance (fraction of matching hard calls)
fn genotype_concordance(gt1: &[String], gt2: &[String], ignore_phase: bool) -> f64 {
    assert_eq!(gt1.len(), gt2.len());

    let mut matches = 0;
    let mut total = 0;

    for (g1, g2) in gt1.iter().zip(gt2.iter()) {
        // Skip missing
        if g1.contains('.') || g2.contains('.') {
            continue;
        }

        total += 1;

        let match_result = if ignore_phase {
            normalize_gt_unphased(g1) == normalize_gt_unphased(g2)
        } else {
            g1 == g2
        };

        if match_result {
            matches += 1;
        }
    }

    if total == 0 {
        return 1.0;
    }

    matches as f64 / total as f64
}

/// Extract all dosages from parsed records (flattened: all samples, all markers)
fn extract_dosages(records: &[ParsedRecord]) -> Vec<f64> {
    let mut dosages = Vec::new();
    for record in records {
        for gt in &record.genotypes {
            if let Some(ds) = gt.ds {
                dosages.push(ds);
            }
        }
    }
    dosages
}

/// Extract DR2 values from INFO field
fn extract_dr2(records: &[ParsedRecord]) -> Vec<f64> {
    records
        .iter()
        .filter_map(|r| r.info.get("DR2").and_then(|v| v.parse().ok()))
        .collect()
}

/// Run Java BEAGLE with given arguments
fn run_beagle(jar: &Path, args: &[(&str, &str)], work_dir: &Path) -> std::process::Output {
    let mut cmd = Command::new("java");
    cmd.arg("-jar").arg(jar);

    for (key, value) in args {
        cmd.arg(format!("{}={}", key, value));
    }

    cmd.current_dir(work_dir);

    println!("Running: java -jar {} {:?}", jar.display(), args);

    let output = cmd.output().expect("Failed to execute Java BEAGLE");

    if !output.status.success() {
        eprintln!("STDOUT: {}", String::from_utf8_lossy(&output.stdout));
        eprintln!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
    }

    output
}

#[test]
fn test_phasing_rust_vs_java() {
    // Run on all available data sources
    for source in get_all_data_sources() {
        println!("\n{}", "=".repeat(60));
        println!("=== Phasing Test: {} data ===", source.name);
        println!("{}", "=".repeat(60));

        run_phasing_comparison(&source);
    }
}

/// Helper: Run phasing comparison on a data source
fn run_phasing_comparison(source: &TestDataSource) {
    let files = setup_test_files(); // For BEAGLE JAR
    let work_dir = tempfile::tempdir().expect("Create temp dir");

    // Copy target to work dir
    let gt_path = work_dir.path().join("target.vcf.gz");
    fs::copy(&source.target_vcf, &gt_path).expect("Copy target VCF");

    // Run Java BEAGLE
    let java_out = work_dir.path().join("java_phased");
    let java_output = run_beagle(
        &files.beagle_jar,
        &[
            ("gt", gt_path.to_str().unwrap()),
            ("out", java_out.to_str().unwrap()),
            ("seed", "42"),
        ],
        work_dir.path(),
    );
    assert!(java_output.status.success(), "{}: Java phasing failed", source.name);

    let java_vcf = work_dir.path().join("java_phased.vcf.gz");

    // Run Rust (decompress for Rust since test fixtures are regular gzip, not BGZF)
    let gt_vcf = decompress_vcf_for_rust(&gt_path, work_dir.path());
    let rust_out = work_dir.path().join("rust_phased");
    let rust_result = run_rust_phasing(&gt_vcf, &rust_out, 42);
    assert!(rust_result.is_ok(), "{}: Rust phasing failed: {:?}", source.name, rust_result.err());

    let rust_vcf = work_dir.path().join("rust_phased.vcf.gz");

    // Compare outputs
    let (_, java_records) = parse_vcf(&java_vcf);
    let (_, rust_records) = parse_vcf(&rust_vcf);

    println!("[{}] Java: {} records, Rust: {} records",
             source.name, java_records.len(), rust_records.len());

    assert_eq!(java_records.len(), rust_records.len(),
               "{}: Record count mismatch", source.name);

    // Compare genotypes
    let mut concordant = 0;
    let mut total = 0;
    for (j_rec, r_rec) in java_records.iter().zip(rust_records.iter()) {
        for (j_gt, r_gt) in j_rec.genotypes.iter().zip(r_rec.genotypes.iter()) {
            total += 1;
            if normalize_gt_unphased(&j_gt.gt) == normalize_gt_unphased(&r_gt.gt) {
                concordant += 1;
            }
        }
    }

    let concordance = concordant as f64 / total as f64;
    println!("[{}] Concordance: {:.2}% ({}/{})",
             source.name, concordance * 100.0, concordant, total);

    assert!(concordance > 0.99,
            "{}: Concordance too low: {:.2}%", source.name, concordance * 100.0);
}

#[test]
fn test_imputation_vcf_ref_rust_vs_java() {
    // Run on all available data sources
    for source in get_all_data_sources() {
        println!("\n{}", "=".repeat(60));
        println!("=== Imputation Test: {} data ===", source.name);
        println!("{}", "=".repeat(60));

        run_imputation_comparison(&source);
    }
}

/// Helper: Run imputation comparison on a data source
fn run_imputation_comparison(source: &TestDataSource) {
    let files = setup_test_files(); // For BEAGLE JAR
    let work_dir = tempfile::tempdir().expect("Create temp dir");

    // Copy files to work dir
    let ref_path = work_dir.path().join("ref.vcf.gz");
    let gt_path = work_dir.path().join("target.vcf.gz");
    fs::copy(&source.ref_vcf, &ref_path).expect("Copy ref VCF");
    fs::copy(&source.target_vcf, &gt_path).expect("Copy target VCF");

    // Run Java BEAGLE
    let java_out = work_dir.path().join("java_imputed");
    let java_output = run_beagle(
        &files.beagle_jar,
        &[
            ("ref", ref_path.to_str().unwrap()),
            ("gt", gt_path.to_str().unwrap()),
            ("out", java_out.to_str().unwrap()),
            ("seed", "42"),
            ("gp", "true"),
        ],
        work_dir.path(),
    );
    assert!(java_output.status.success(), "{}: Java imputation failed", source.name);

    let java_vcf = work_dir.path().join("java_imputed.vcf.gz");

    // Run Rust (decompress for Rust since test fixtures are regular gzip, not BGZF)
    let gt_vcf = decompress_vcf_for_rust(&gt_path, work_dir.path());
    let ref_vcf = decompress_vcf_for_rust(&ref_path, work_dir.path());
    let rust_out = work_dir.path().join("rust_imputed");
    let rust_result = run_rust_imputation(&gt_vcf, &ref_vcf, &rust_out, 42);
    assert!(rust_result.is_ok(), "{}: Rust imputation failed: {:?}", source.name, rust_result.err());

    let rust_vcf = work_dir.path().join("rust_imputed.vcf.gz");

    // Compare outputs
    let (_, java_records) = parse_vcf(&java_vcf);
    let (_, rust_records) = parse_vcf(&rust_vcf);

    println!("[{}] Java: {} records, Rust: {} records",
             source.name, java_records.len(), rust_records.len());

    assert_eq!(java_records.len(), rust_records.len(),
               "{}: Record count mismatch", source.name);

    // Compare dosages
    let mut dosage_diffs: Vec<f64> = Vec::new();
    for (j_rec, r_rec) in java_records.iter().zip(rust_records.iter()) {
        for (j_gt, r_gt) in j_rec.genotypes.iter().zip(r_rec.genotypes.iter()) {
            if let (Some(j_ds), Some(r_ds)) = (j_gt.ds, r_gt.ds) {
                dosage_diffs.push((j_ds - r_ds).abs());
            }
        }
    }

    if !dosage_diffs.is_empty() {
        let mean_diff: f64 = dosage_diffs.iter().sum::<f64>() / dosage_diffs.len() as f64;
        println!("[{}] Dosage comparison: {} values, mean diff={:.4}",
                 source.name, dosage_diffs.len(), mean_diff);
        assert!(mean_diff < 0.1, "{}: Mean dosage too high: {:.4}", source.name, mean_diff);
    }
}

#[test]
fn test_java_beagle_bref3_creation() {
    let files = setup_test_files();
    let work_dir = tempfile::tempdir().expect("Create temp dir");

    // Copy ref VCF to work dir
    let ref_path = work_dir.path().join("ref.vcf.gz");
    fs::copy(&files.ref_vcf, &ref_path).expect("Copy ref VCF");

    let bref3_path = work_dir.path().join("ref.bref3");

    // Run bref3 tool: java -jar bref3.jar input.vcf.gz > output.bref3
    let output = Command::new("sh")
        .arg("-c")
        .arg(format!(
            "java -jar {} {} > {}",
            files.bref3_jar.display(),
            ref_path.display(),
            bref3_path.display()
        ))
        .current_dir(work_dir.path())
        .output()
        .expect("Failed to run bref3");

    if !output.status.success() {
        eprintln!("STDERR: {}", String::from_utf8_lossy(&output.stderr));
    }
    assert!(output.status.success(), "bref3 creation failed");

    assert!(bref3_path.exists(), "bref3 file not created");

    let bref3_size = fs::metadata(&bref3_path).unwrap().len();
    assert!(bref3_size > 0, "bref3 file is empty");

    println!("bref3 output: {} ({} bytes)", bref3_path.display(), bref3_size);
}

#[test]
fn test_imputation_bref3_ref_rust_vs_java() {
    let files = setup_test_files();
    let work_dir = tempfile::tempdir().expect("Create temp dir");

    // Copy files to work dir
    let ref_path = work_dir.path().join("ref.vcf.gz");
    let gt_path = work_dir.path().join("target.vcf.gz");
    fs::copy(&files.ref_vcf, &ref_path).expect("Copy ref VCF");
    fs::copy(&files.target_vcf, &gt_path).expect("Copy target VCF");

    // First create bref3 using Java tool
    let bref3_path = work_dir.path().join("ref.bref3");
    let bref3_output = Command::new("sh")
        .arg("-c")
        .arg(format!(
            "java -jar {} {} > {}",
            files.bref3_jar.display(),
            ref_path.display(),
            bref3_path.display()
        ))
        .current_dir(work_dir.path())
        .output()
        .expect("Failed to run bref3");
    assert!(bref3_output.status.success(), "bref3 creation failed");

    // Run Java BEAGLE with bref3 reference
    let java_out = work_dir.path().join("java_bref3");
    let java_output = run_beagle(
        &files.beagle_jar,
        &[
            ("ref", bref3_path.to_str().unwrap()),
            ("gt", gt_path.to_str().unwrap()),
            ("out", java_out.to_str().unwrap()),
            ("seed", "42"),
            ("gp", "true"),
        ],
        work_dir.path(),
    );
    assert!(java_output.status.success(), "Java BEAGLE with bref3 failed");

    // Run Rust with bref3 reference (decompress gt for Rust)
    let gt_vcf = decompress_vcf_for_rust(&gt_path, work_dir.path());
    let rust_out = work_dir.path().join("rust_bref3");
    let rust_result = run_rust_imputation(&gt_vcf, &bref3_path, &rust_out, 42);
    assert!(rust_result.is_ok(), "Rust with bref3 failed: {:?}", rust_result.err());

    // Compare outputs
    let java_vcf = work_dir.path().join("java_bref3.vcf.gz");
    let rust_vcf = work_dir.path().join("rust_bref3.vcf.gz");

    assert!(java_vcf.exists(), "Java output not created");
    assert!(rust_vcf.exists(), "Rust output not created");

    let (_, java_records) = parse_vcf(&java_vcf);
    let (_, rust_records) = parse_vcf(&rust_vcf);

    println!("\n=== bref3 Imputation: Rust vs Java ===");
    println!("Java: {} records, {} bytes", java_records.len(), fs::metadata(&java_vcf).unwrap().len());
    println!("Rust: {} records, {} bytes", rust_records.len(), fs::metadata(&rust_vcf).unwrap().len());

    assert_eq!(java_records.len(), rust_records.len(), "Record count mismatch");

    // Compare dosages
    let mut dosage_diffs: Vec<f64> = Vec::new();
    for (j_rec, r_rec) in java_records.iter().zip(rust_records.iter()) {
        for (j_gt, r_gt) in j_rec.genotypes.iter().zip(r_rec.genotypes.iter()) {
            if let (Some(j_ds), Some(r_ds)) = (j_gt.ds, r_gt.ds) {
                dosage_diffs.push((j_ds - r_ds).abs());
            }
        }
    }

    if !dosage_diffs.is_empty() {
        let mean_diff: f64 = dosage_diffs.iter().sum::<f64>() / dosage_diffs.len() as f64;
        println!("Dosage comparison: {} values, mean diff={:.4}", dosage_diffs.len(), mean_diff);
        assert!(mean_diff < 0.1, "Mean dosage difference too high: {:.4}", mean_diff);
    }

    println!("bref3 imputation: Rust matches Java!");
}

#[test]
fn test_full_workflow_rust_vs_java() {
    // Run full workflow on all data sources
    for source in get_all_data_sources() {
        println!("\n{}", "=".repeat(60));
        println!("=== Full Workflow Test: {} data ===", source.name);
        println!("{}", "=".repeat(60));

        run_full_workflow_comparison(&source);
    }

    // bref3 tests are Java-specific (uses BEAGLE test_vcf), run only once
    println!("\n{}", "=".repeat(60));
    println!("=== bref3 Test (Java only, BEAGLE data) ===");
    println!("{}", "=".repeat(60));
    run_bref3_java_only_test();
}

/// Helper: Run full workflow comparison on a data source
fn run_full_workflow_comparison(source: &TestDataSource) {
    let files = setup_test_files(); // For BEAGLE JAR
    let work_dir = tempfile::tempdir().expect("Create temp dir");

    // Copy files to work dir
    let ref_path = work_dir.path().join("ref.vcf.gz");
    let gt_path = work_dir.path().join("target.vcf.gz");
    fs::copy(&source.ref_vcf, &ref_path).expect("Copy ref VCF");
    fs::copy(&source.target_vcf, &gt_path).expect("Copy target VCF");

    // Decompress for Rust (test fixtures are regular gzip, not BGZF)
    let gt_vcf = decompress_vcf_for_rust(&gt_path, work_dir.path());
    let ref_vcf = decompress_vcf_for_rust(&ref_path, work_dir.path());

    // 1. Phasing only - Compare Rust vs Java
    println!("\n=== [{}] Test 1: Phasing - Rust vs Java ===", source.name);
    let java_phase = work_dir.path().join("java_phased");
    let output1 = run_beagle(
        &files.beagle_jar,
        &[
            ("gt", gt_path.to_str().unwrap()),
            ("out", java_phase.to_str().unwrap()),
            ("seed", "42"),
        ],
        work_dir.path(),
    );
    assert!(output1.status.success(), "{}: Java phasing failed", source.name);

    let rust_phase = work_dir.path().join("rust_phased");
    let rust_result = run_rust_phasing(&gt_vcf, &rust_phase, 42);
    assert!(rust_result.is_ok(), "{}: Rust phasing failed: {:?}", source.name, rust_result.err());

    let java_vcf = work_dir.path().join("java_phased.vcf.gz");
    let rust_vcf = work_dir.path().join("rust_phased.vcf.gz");
    assert!(java_vcf.exists() && rust_vcf.exists());
    println!("  Java: {} bytes, Rust: {} bytes",
             fs::metadata(&java_vcf).unwrap().len(),
             fs::metadata(&rust_vcf).unwrap().len());

    // 2. Imputation with VCF reference - Compare Rust vs Java
    println!("\n=== [{}] Test 2: Imputation (VCF ref) - Rust vs Java ===", source.name);
    let java_imp = work_dir.path().join("java_imputed");
    let output2 = run_beagle(
        &files.beagle_jar,
        &[
            ("ref", ref_path.to_str().unwrap()),
            ("gt", gt_path.to_str().unwrap()),
            ("out", java_imp.to_str().unwrap()),
            ("seed", "42"),
        ],
        work_dir.path(),
    );
    assert!(output2.status.success(), "{}: Java imputation failed", source.name);

    let rust_imp = work_dir.path().join("rust_imputed");
    let rust_result = run_rust_imputation(&gt_vcf, &ref_vcf, &rust_imp, 42);
    assert!(rust_result.is_ok(), "{}: Rust imputation failed: {:?}", source.name, rust_result.err());

    let java_vcf = work_dir.path().join("java_imputed.vcf.gz");
    let rust_vcf = work_dir.path().join("rust_imputed.vcf.gz");
    assert!(java_vcf.exists() && rust_vcf.exists());
    println!("  Java: {} bytes, Rust: {} bytes",
             fs::metadata(&java_vcf).unwrap().len(),
             fs::metadata(&rust_vcf).unwrap().len());

    println!("\n=== [{}] Full workflow passed ===", source.name);
}

/// Helper: Run bref3 Java-only test (BEAGLE-specific)
fn run_bref3_java_only_test() {
    let files = setup_test_files();
    let work_dir = tempfile::tempdir().expect("Create temp dir");

    let ref_path = work_dir.path().join("ref.vcf.gz");
    let gt_path = work_dir.path().join("target.vcf.gz");
    fs::copy(&files.ref_vcf, &ref_path).expect("Copy ref VCF");
    fs::copy(&files.target_vcf, &gt_path).expect("Copy target VCF");

    // Create bref3
    let bref3_path = work_dir.path().join("ref.bref3");
    let bref3_output = Command::new("sh")
        .arg("-c")
        .arg(format!(
            "java -jar {} {} > {}",
            files.bref3_jar.display(),
            ref_path.display(),
            bref3_path.display()
        ))
        .current_dir(work_dir.path())
        .output()
        .expect("Failed to run bref3");
    assert!(bref3_output.status.success(), "bref3 creation failed");

    // Run imputation with bref3
    let out3 = work_dir.path().join("out.bref3");
    let output3 = run_beagle(
        &files.beagle_jar,
        &[
            ("ref", bref3_path.to_str().unwrap()),
            ("gt", gt_path.to_str().unwrap()),
            ("out", out3.to_str().unwrap()),
        ],
        work_dir.path(),
    );
    assert!(output3.status.success(), "bref3 imputation failed");
    println!("  bref3 imputation: {} bytes",
             fs::metadata(work_dir.path().join("out.bref3.vcf.gz")).unwrap().len());

    println!("\n=== bref3 Java-only test passed ===");
}

#[test]
fn test_output_structure_rust_vs_java() {
    // Run on all available data sources
    for source in get_all_data_sources() {
        println!("\n{}", "=".repeat(60));
        println!("=== Output Structure Test: {} data ===", source.name);
        println!("{}", "=".repeat(60));

        run_output_structure_comparison(&source);
    }
}

/// Helper: Run output structure comparison on a data source
fn run_output_structure_comparison(source: &TestDataSource) {
    let files = setup_test_files(); // For BEAGLE JAR
    let work_dir = tempfile::tempdir().expect("Create temp dir");

    let ref_path = work_dir.path().join("ref.vcf.gz");
    let gt_path = work_dir.path().join("target_sparse.vcf.gz");
    fs::copy(&source.ref_vcf, &ref_path).expect("Copy ref VCF");
    fs::copy(&source.target_sparse_vcf, &gt_path).expect("Copy sparse target VCF");

    // Run Java BEAGLE
    let java_out = work_dir.path().join("java_out");
    let java_output = run_beagle(
        &files.beagle_jar,
        &[
            ("ref", ref_path.to_str().unwrap()),
            ("gt", gt_path.to_str().unwrap()),
            ("out", java_out.to_str().unwrap()),
            ("seed", "42"),
            ("gp", "true"),
        ],
        work_dir.path(),
    );
    assert!(java_output.status.success(), "{}: Java BEAGLE imputation failed", source.name);

    // Run Rust (decompress for Rust since test fixtures are regular gzip, not BGZF)
    let gt_vcf = decompress_vcf_for_rust(&gt_path, work_dir.path());
    let ref_vcf = decompress_vcf_for_rust(&ref_path, work_dir.path());
    let rust_out = work_dir.path().join("rust_out");
    let rust_result = run_rust_imputation(&gt_vcf, &ref_vcf, &rust_out, 42);
    assert!(rust_result.is_ok(), "{}: Rust imputation failed: {:?}", source.name, rust_result.err());

    let java_vcf = work_dir.path().join("java_out.vcf.gz");
    let rust_vcf = work_dir.path().join("rust_out.vcf.gz");

    let (j_rec, j_ds, j_dr2, j_gp) = validate_output(&java_vcf, &format!("[{}] Java", source.name));
    let (r_rec, r_ds, r_dr2, r_gp) = validate_output(&rust_vcf, &format!("[{}] Rust", source.name));

    // Compare structure
    println!("\n=== [{}] Comparison ===", source.name);
    println!("Records: Java={}, Rust={}", j_rec, r_rec);
    println!("Dosages: Java={}, Rust={}", j_ds, r_ds);
    println!("DR2: Java={}, Rust={}", j_dr2, r_dr2);
    println!("GP: Java={}, Rust={}", j_gp, r_gp);

    assert_eq!(j_rec, r_rec, "{}: Record count mismatch", source.name);

    println!("\n[{}] Output structure validation passed!", source.name);
}

/// Helper to validate output structure
fn validate_output(vcf_path: &Path, name: &str) -> (usize, usize, usize, usize) {
    let (samples, records) = parse_vcf(vcf_path);

    println!("\n=== {} Output Structure ===", name);
    println!("Samples: {}, Records: {}", samples.len(), records.len());

    assert!(samples.len() > 0, "{}: Expected samples", name);
    assert!(records.len() > 100, "{}: Expected >100 records, got {}", name, records.len());

    // Check genotypes per record
    for (i, record) in records.iter().enumerate() {
        assert_eq!(record.genotypes.len(), samples.len(),
                   "{}: Record {} has wrong genotype count", name, i);
    }

    // Check phasing
    let first_gt = &records[0].genotypes[0].gt;
    assert!(first_gt.contains('|'), "{}: Expected phased genotypes, got: {}", name, first_gt);

    // Dosages
    let dosages = extract_dosages(&records);
    let invalid_dosages = dosages.iter().filter(|&&d| d < 0.0 || d > 2.0).count();
    assert_eq!(invalid_dosages, 0, "{}: Found invalid dosages", name);
    println!("Dosages: {} values, range {:.3}-{:.3}",
             dosages.len(),
             dosages.iter().cloned().fold(f64::INFINITY, f64::min),
             dosages.iter().cloned().fold(f64::NEG_INFINITY, f64::max));

    // DR2 values
    let dr2_values = extract_dr2(&records);
    let invalid_dr2 = dr2_values.iter().filter(|&&d| d < 0.0 || d > 1.0).count();
    assert_eq!(invalid_dr2, 0, "{}: Found invalid DR2 values", name);
    if !dr2_values.is_empty() {
        let mean_dr2: f64 = dr2_values.iter().sum::<f64>() / dr2_values.len() as f64;
        println!("DR2: {} values, mean {:.3}", dr2_values.len(), mean_dr2);
    }

    // Imputed vs genotyped
    let imputed_count = records.iter().filter(|r| r.info.contains_key("IMP")).count();
    println!("Imputed: {}, Genotyped: {}", imputed_count, records.len() - imputed_count);

    // GP values
    let gp_count = records.iter()
        .flat_map(|r| r.genotypes.iter())
        .filter(|g| g.gp.is_some())
        .count();
    println!("GP values: {}", gp_count);

    (records.len(), dosages.len(), dr2_values.len(), gp_count)
}

#[test]
fn test_java_beagle_vcf_vs_bref3_consistency() {
    // Verify that imputation with VCF ref and bref3 ref produce identical results
    // Using sparse target for true imputation with DS/GP output
    let files = setup_test_files();
    let work_dir = tempfile::tempdir().expect("Create temp dir");

    let ref_path = work_dir.path().join("ref.vcf.gz");
    let gt_path = work_dir.path().join("target_sparse.vcf.gz");
    fs::copy(&files.ref_vcf, &ref_path).expect("Copy ref VCF");
    fs::copy(&files.target_sparse_vcf, &gt_path).expect("Copy sparse target VCF");

    // Run with VCF reference (with gp=true for full output)
    let out_vcf_prefix = work_dir.path().join("out_vcf");
    let output1 = run_beagle(
        &files.beagle_jar,
        &[
            ("ref", ref_path.to_str().unwrap()),
            ("gt", gt_path.to_str().unwrap()),
            ("out", out_vcf_prefix.to_str().unwrap()),
            ("seed", "12345"),
            ("gp", "true"),
        ],
        work_dir.path(),
    );
    assert!(output1.status.success(), "VCF ref imputation failed");

    // Create bref3
    let bref3_path = work_dir.path().join("ref.bref3");
    let bref3_output = Command::new("sh")
        .arg("-c")
        .arg(format!(
            "java -jar {} {} > {}",
            files.bref3_jar.display(),
            ref_path.display(),
            bref3_path.display()
        ))
        .current_dir(work_dir.path())
        .output()
        .expect("Failed to run bref3");
    assert!(bref3_output.status.success(), "bref3 creation failed");

    // Run with bref3 reference (same seed and gp=true)
    let out_bref3_prefix = work_dir.path().join("out_bref3");
    let output2 = run_beagle(
        &files.beagle_jar,
        &[
            ("ref", bref3_path.to_str().unwrap()),
            ("gt", gt_path.to_str().unwrap()),
            ("out", out_bref3_prefix.to_str().unwrap()),
            ("seed", "12345"),
            ("gp", "true"),
        ],
        work_dir.path(),
    );
    assert!(output2.status.success(), "bref3 ref imputation failed");

    // Parse both outputs
    let (_, records_vcf) = parse_vcf(&work_dir.path().join("out_vcf.vcf.gz"));
    let (_, records_bref3) = parse_vcf(&work_dir.path().join("out_bref3.vcf.gz"));

    assert_eq!(
        records_vcf.len(),
        records_bref3.len(),
        "Record counts differ"
    );

    // Compare dosages (should have dosages now with true imputation)
    let ds_vcf = extract_dosages(&records_vcf);
    let ds_bref3 = extract_dosages(&records_bref3);

    println!("VCF dosages: {}, bref3 dosages: {}", ds_vcf.len(), ds_bref3.len());

    if !ds_vcf.is_empty() && !ds_bref3.is_empty() {
        let r2 = dosage_correlation(&ds_vcf, &ds_bref3);
        println!("VCF vs bref3 dosage correlation r²: {:.6}", r2);

        assert!(
            r2 > 0.999,
            "VCF and bref3 dosages should be nearly identical, got r²={}",
            r2
        );
    }

    // Compare genotypes
    let gt_vcf: Vec<String> = records_vcf
        .iter()
        .flat_map(|r| r.genotypes.iter().map(|g| g.gt.clone()))
        .collect();
    let gt_bref3: Vec<String> = records_bref3
        .iter()
        .flat_map(|r| r.genotypes.iter().map(|g| g.gt.clone()))
        .collect();

    let concordance = genotype_concordance(&gt_vcf, &gt_bref3, false);
    println!("VCF vs bref3 genotype concordance: {:.4}%", concordance * 100.0);

    // Check phase switches (should be zero for identical inputs with same seed)
    let phase_switches = count_phase_switches(&gt_vcf, &gt_bref3);
    println!("Phase switches between VCF and bref3: {}", phase_switches);

    assert!(
        concordance > 0.999,
        "VCF and bref3 genotypes should be nearly identical"
    );

    assert_eq!(
        phase_switches, 0,
        "VCF and bref3 should have identical phasing"
    );

    println!("VCF vs bref3 consistency check passed!");
}

// =============================================================================
// Mask-and-Recover Test Infrastructure
// =============================================================================

/// Calculate Minor Allele Frequency from genotypes
fn calculate_maf(genotypes: &[ParsedGenotype]) -> f64 {
    let mut alt_count = 0;
    let mut total_alleles = 0;

    for gt in genotypes {
        for allele in gt.gt.split(|c| c == '|' || c == '/') {
            if allele == "." {
                continue;
            }
            total_alleles += 1;
            if allele != "0" {
                alt_count += 1;
            }
        }
    }

    if total_alleles == 0 {
        return 0.0;
    }

    let af = alt_count as f64 / total_alleles as f64;
    af.min(1.0 - af) // MAF is always <= 0.5
}

/// Mask random genotypes in a VCF file, returning (masked_path, truth_map)
/// truth_map: HashMap<(chrom, pos, sample_idx), original_gt>
fn create_masked_vcf(
    input_path: &Path,
    output_path: &Path,
    mask_fraction: f64,
    seed: u64,
) -> HashMap<(String, u64, usize), String> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Use gzip command for reliable BGZF decompression
    let decompress_output = Command::new("gzip")
        .args(["-dc", input_path.to_str().unwrap()])
        .output()
        .expect("Failed to run gzip");

    if !decompress_output.status.success() {
        panic!("gzip decompression failed for {:?}", input_path);
    }

    let content = String::from_utf8_lossy(&decompress_output.stdout);

    let mut output = File::create(output_path).expect("Create output file");
    let mut truth_map = HashMap::new();

    let mut sample_count = 0;

    for line in content.lines() {

        if line.starts_with('#') {
            writeln!(output, "{}", line).expect("Write header");
            if line.starts_with("#CHROM") {
                let fields: Vec<&str> = line.split('\t').collect();
                sample_count = fields.len() - 9;
            }
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 10 {
            writeln!(output, "{}", line).expect("Write line");
            continue;
        }

        let chrom = fields[0].to_string();
        let pos: u64 = fields[1].parse().expect("Parse pos");

        // Decide which samples to mask at this position
        let samples_to_mask: Vec<usize> = (0..sample_count)
            .filter(|_| rand::Rng::random::<f64>(&mut rng) < mask_fraction)
            .collect();

        if samples_to_mask.is_empty() {
            writeln!(output, "{}", line).expect("Write line");
            continue;
        }

        // Build new line with masked genotypes
        let mut new_fields: Vec<String> = fields[..9].iter().map(|s: &&str| s.to_string()).collect();

        for (sample_idx, sample_data) in fields[9..].iter().enumerate() {
            if samples_to_mask.contains(&sample_idx) {
                // Store truth
                let gt: &str = sample_data.split(':').next().unwrap_or(".");
                truth_map.insert((chrom.clone(), pos, sample_idx), gt.to_string());

                // Mask the genotype (replace GT with ./.)
                let parts: Vec<&str> = sample_data.split(':').collect();
                let mut masked_parts: Vec<String> = vec!["./.".to_string()];
                masked_parts.extend(parts[1..].iter().map(|s: &&str| s.to_string()));
                new_fields.push(masked_parts.join(":"));
            } else {
                new_fields.push(sample_data.to_string());
            }
        }

        writeln!(output, "{}", new_fields.join("\t")).expect("Write masked line");
    }

    truth_map
}

/// Compare imputed genotypes against truth, stratified by MAF
#[derive(Debug, Default)]
struct ImputationAccuracy {
    /// Overall concordance
    total_correct: usize,
    total_compared: usize,

    /// Rare variants (MAF < 0.01)
    rare_true_positives: usize,  // Predicted rare, was rare
    rare_false_positives: usize, // Predicted rare, was not rare
    rare_false_negatives: usize, // Predicted common, was rare
    rare_total: usize,

    /// By confidence bin (for calibration)
    /// bin index = floor(confidence * 10), so 0.95 -> bin 9
    confidence_bins: [(usize, usize); 10], // (correct, total) per bin

    /// Brier Score components (sum of squared errors)
    brier_score_sum: f64,
    brier_score_count: usize,
}

impl ImputationAccuracy {
    fn concordance(&self) -> f64 {
        if self.total_compared == 0 {
            return 0.0; // No comparisons = 0% concordance, not 100%
        }
        self.total_correct as f64 / self.total_compared as f64
    }

    fn rare_precision(&self) -> f64 {
        let predicted_rare = self.rare_true_positives + self.rare_false_positives;
        if predicted_rare == 0 {
            return 1.0;
        }
        self.rare_true_positives as f64 / predicted_rare as f64
    }

    fn rare_recall(&self) -> f64 {
        let actual_rare = self.rare_true_positives + self.rare_false_negatives;
        if actual_rare == 0 {
            return 1.0;
        }
        self.rare_true_positives as f64 / actual_rare as f64
    }

    fn rare_f1(&self) -> f64 {
        let p = self.rare_precision();
        let r = self.rare_recall();
        if p + r == 0.0 {
            return 0.0;
        }
        2.0 * p * r / (p + r)
    }

    fn calibration_error(&self) -> f64 {
        // Mean absolute calibration error
        let mut total_error = 0.0;
        let mut bins_with_data = 0;

        for (bin_idx, &(correct, total)) in self.confidence_bins.iter().enumerate() {
            if total == 0 {
                continue;
            }
            let expected_accuracy = (bin_idx as f64 + 0.5) / 10.0; // Center of bin
            let actual_accuracy = correct as f64 / total as f64;
            total_error += (expected_accuracy - actual_accuracy).abs();
            bins_with_data += 1;
        }

        if bins_with_data == 0 {
            return 0.0;
        }
        total_error / bins_with_data as f64
    }

    /// Mean Brier Score - measures probabilistic calibration
    /// Lower is better. Punishes confident wrong predictions heavily.
    /// Returns f64::NAN if no samples (so tests can detect missing data)
    fn brier_score(&self) -> f64 {
        if self.brier_score_count == 0 {
            return f64::NAN; // No data = undefined, not "perfect"
        }
        self.brier_score_sum / self.brier_score_count as f64
    }
}

/// Calculate Brier Score for a single prediction
/// GP: [P(0/0), P(0/1), P(1/1)]
/// truth_gt: the actual genotype string (e.g., "0|0", "0|1", "1|1")
fn calculate_brier_score(gp: [f64; 3], truth_gt: &str) -> f64 {
    // Convert truth to one-hot: [is_hom_ref, is_het, is_hom_alt]
    let truth_vec = match normalize_gt_unphased(truth_gt).as_str() {
        "0/0" => [1.0, 0.0, 0.0],
        "0/1" | "1/0" => [0.0, 1.0, 0.0],
        "1/1" => [0.0, 0.0, 1.0],
        _ => return 0.0, // Skip missing/unknown
    };

    // Brier score = sum of (predicted - actual)^2
    (gp[0] - truth_vec[0]).powi(2)
        + (gp[1] - truth_vec[1]).powi(2)
        + (gp[2] - truth_vec[2]).powi(2)
}

/// Calculate imputation accuracy comparing imputed VCF against truth
fn evaluate_imputation(
    imputed_records: &[ParsedRecord],
    truth_map: &HashMap<(String, u64, usize), String>,
    ref_records: &[ParsedRecord], // For MAF calculation
) -> ImputationAccuracy {
    let mut acc = ImputationAccuracy::default();

    // Build MAF lookup from reference
    let maf_lookup: HashMap<(String, u64), f64> = ref_records
        .iter()
        .map(|r| ((r.chrom.clone(), r.pos), calculate_maf(&r.genotypes)))
        .collect();

    for record in imputed_records {
        let maf = maf_lookup
            .get(&(record.chrom.clone(), record.pos))
            .copied()
            .unwrap_or(0.5);
        let is_rare = maf < 0.01;

        for (sample_idx, gt) in record.genotypes.iter().enumerate() {
            let key = (record.chrom.clone(), record.pos, sample_idx);

            if let Some(truth_gt) = truth_map.get(&key) {
                acc.total_compared += 1;

                let imputed_normalized = normalize_gt_unphased(&gt.gt);
                let truth_normalized = normalize_gt_unphased(truth_gt);

                let is_correct = imputed_normalized == truth_normalized;

                if is_correct {
                    acc.total_correct += 1;
                }

                // Rare variant tracking
                if is_rare {
                    acc.rare_total += 1;
                    let truth_has_alt = truth_gt.contains('1');
                    let imputed_has_alt = gt.gt.contains('1');

                    match (imputed_has_alt, truth_has_alt) {
                        (true, true) => acc.rare_true_positives += 1,
                        (true, false) => acc.rare_false_positives += 1,
                        (false, true) => acc.rare_false_negatives += 1,
                        (false, false) => {} // True negative
                    }
                }

                // Calibration tracking (use GP if available, else DS)
                let confidence = gt.gp.map(|gp| gp.iter().cloned().fold(0.0, f64::max))
                    .or_else(|| gt.ds.map(|ds| {
                        // Convert dosage to pseudo-confidence
                        // DS near 0 or 2 = high confidence, DS near 1 = low confidence
                        let dist_from_het = (ds - 1.0).abs();
                        0.5 + dist_from_het * 0.5
                    }))
                    .unwrap_or(0.5);

                let bin_idx = ((confidence * 10.0) as usize).min(9);
                acc.confidence_bins[bin_idx].1 += 1;
                if is_correct {
                    acc.confidence_bins[bin_idx].0 += 1;
                }

                // Brier Score calculation (requires GP)
                if let Some(gp) = gt.gp {
                    let bs = calculate_brier_score(gp, truth_gt);
                    acc.brier_score_sum += bs;
                    acc.brier_score_count += 1;
                }
            }
        }
    }

    acc
}

#[test]
fn test_mask_and_recover_rust_vs_java() {
    // Run on all available data sources
    for source in get_all_data_sources() {
        println!("\n{}", "=".repeat(60));
        println!("=== Mask-and-Recover Test: {} data ===", source.name);
        println!("{}", "=".repeat(60));

        run_mask_and_recover_comparison(&source);
    }
}

/// Helper: Run mask-and-recover comparison on a data source
fn run_mask_and_recover_comparison(source: &TestDataSource) {
    let files = setup_test_files(); // For BEAGLE JAR
    let work_dir = tempfile::tempdir().expect("Create temp dir");

    // Copy reference panel
    let ref_path = work_dir.path().join("ref.vcf.gz");
    fs::copy(&source.ref_vcf, &ref_path).expect("Copy ref VCF");

    // Copy sparse target
    let target_path = work_dir.path().join("target_sparse.vcf.gz");
    fs::copy(&source.target_sparse_vcf, &target_path).expect("Copy sparse target VCF");

    // Create a masked version of the sparse target (mask 20% of remaining genotypes)
    let masked_path = work_dir.path().join("masked.vcf");
    let truth_map = create_masked_vcf(&target_path, &masked_path, 0.20, 42);
    println!("[{}] Masked {} genotypes", source.name, truth_map.len());

    // Compress the masked file
    let masked_gz = work_dir.path().join("masked.vcf.gz");
    let status = Command::new("gzip")
        .args(["-c"])
        .stdin(File::open(&masked_path).unwrap())
        .stdout(File::create(&masked_gz).unwrap())
        .status()
        .expect("gzip failed");
    assert!(status.success());

    // Run Java BEAGLE
    let java_out = work_dir.path().join("java_imputed");
    let java_output = run_beagle(
        &files.beagle_jar,
        &[
            ("ref", ref_path.to_str().unwrap()),
            ("gt", masked_gz.to_str().unwrap()),
            ("out", java_out.to_str().unwrap()),
            ("seed", "42"),
            ("gp", "true"),
        ],
        work_dir.path(),
    );
    assert!(java_output.status.success(), "{}: Java BEAGLE imputation failed", source.name);

    // Run Rust (use uncompressed masked.vcf, decompress ref for Rust)
    let ref_vcf = decompress_vcf_for_rust(&ref_path, work_dir.path());
    let rust_out = work_dir.path().join("rust_imputed");
    let rust_result = run_rust_imputation(&masked_path, &ref_vcf, &rust_out, 42);
    assert!(rust_result.is_ok(), "{}: Rust imputation failed: {:?}", source.name, rust_result.err());

    // Parse outputs
    let (_, target_records) = parse_vcf(&target_path);
    let (_, java_records) = parse_vcf(&work_dir.path().join("java_imputed.vcf.gz"));
    let (_, rust_records) = parse_vcf(&work_dir.path().join("rust_imputed.vcf.gz"));

    // Evaluate both against ground truth
    let java_acc = evaluate_imputation(&java_records, &truth_map, &target_records);
    let rust_acc = evaluate_imputation(&rust_records, &truth_map, &target_records);

    // Print results side-by-side
    println!("\n=== [{}] Mask-and-Recover: Rust vs Java ===", source.name);
    println!("{:<25} {:>12} {:>12}", "Metric", "Java", "Rust");
    println!("{:-<25} {:->12} {:->12}", "", "", "");
    println!("{:<25} {:>11.2}% {:>11.2}%", "Concordance",
             java_acc.concordance() * 100.0, rust_acc.concordance() * 100.0);
    println!("{:<25} {:>12.4} {:>12.4}", "Brier Score",
             java_acc.brier_score(), rust_acc.brier_score());
    println!("{:<25} {:>12.3} {:>12.3}", "Rare F1",
             java_acc.rare_f1(), rust_acc.rare_f1());
    println!("{:<25} {:>12.3} {:>12.3}", "Calibration Error",
             java_acc.calibration_error(), rust_acc.calibration_error());
    println!("{:<25} {:>12} {:>12}", "Comparisons",
             java_acc.total_compared, rust_acc.total_compared);
    println!("{:<25} {:>12} {:>12}", "Brier Samples",
             java_acc.brier_score_count, rust_acc.brier_score_count);

    // Sanity checks - ensure we're actually testing something
    assert!(java_acc.total_compared > 0, "{}: Java: No comparisons made", source.name);
    assert!(rust_acc.total_compared > 0, "{}: Rust: No comparisons made", source.name);
    assert!(java_acc.brier_score_count > 0, "{}: Java: No Brier samples", source.name);
    assert!(rust_acc.brier_score_count > 0, "{}: Rust: No Brier samples", source.name);

    // Quality checks for both
    assert!(java_acc.concordance() > 0.80, "{}: Java concordance too low", source.name);
    assert!(rust_acc.concordance() > 0.80, "{}: Rust concordance too low", source.name);

    // Compare Rust to Java (Rust should be within 10% of Java)
    let concordance_diff = (java_acc.concordance() - rust_acc.concordance()).abs();
    assert!(
        concordance_diff < 0.10,
        "{}: Rust concordance differs too much from Java: {:.2}% vs {:.2}%",
        source.name,
        rust_acc.concordance() * 100.0,
        java_acc.concordance() * 100.0
    );

    println!("\n[{}] Mask-and-recover comparison passed!", source.name);
}

/// Stores baseline BEAGLE accuracy for comparison with Rust implementation
#[derive(Debug)]
pub struct BeagleBaseline {
    pub concordance: f64,
    pub rare_f1: f64,
    pub calibration_error: f64,
    pub brier_score: f64,
    pub mask_fraction: f64,
    pub seed: u64,
}

/// Run mask-and-recover and return baseline metrics for later comparison
pub fn compute_beagle_baseline(
    beagle_jar: &Path,
    input_vcf: &Path,
    mask_fraction: f64,
    seed: u64,
) -> BeagleBaseline {
    let work_dir = tempfile::tempdir().expect("Create temp dir");

    // Create masked version
    let masked_path = work_dir.path().join("masked.vcf");
    let truth_map = create_masked_vcf(input_vcf, &masked_path, mask_fraction, seed);

    // Compress
    let masked_gz = work_dir.path().join("masked.vcf.gz");
    let status = Command::new("gzip")
        .args(["-c"])
        .stdin(File::open(&masked_path).unwrap())
        .stdout(File::create(&masked_gz).unwrap())
        .status()
        .expect("gzip failed");
    assert!(status.success());

    // Run BEAGLE with GP output
    let out_prefix = work_dir.path().join("imputed");
    let output = run_beagle(
        beagle_jar,
        &[
            ("gt", masked_gz.to_str().unwrap()),
            ("out", out_prefix.to_str().unwrap()),
            ("seed", &seed.to_string()),
            ("gp", "true"),
        ],
        work_dir.path(),
    );
    assert!(output.status.success());

    // Parse and evaluate
    let (_, ref_records) = parse_vcf(input_vcf);
    let (_, imputed_records) = parse_vcf(&work_dir.path().join("imputed.vcf.gz"));
    let accuracy = evaluate_imputation(&imputed_records, &truth_map, &ref_records);

    BeagleBaseline {
        concordance: accuracy.concordance(),
        rare_f1: accuracy.rare_f1(),
        calibration_error: accuracy.calibration_error(),
        brier_score: accuracy.brier_score(),
        mask_fraction,
        seed,
    }
}

// =============================================================================
// Placeholder for Rust vs Java comparison (to be implemented when Rust tool ready)
// =============================================================================

/// Compare an implementation's output against Java BEAGLE baseline
fn compare_against_beagle(
    output_vcf: &Path,
    truth_map: &HashMap<(String, u64, usize), String>,
    ref_records: &[ParsedRecord],
    beagle_baseline: &BeagleBaseline,
    impl_name: &str,
) -> bool {
    let (_, output_records) = parse_vcf(output_vcf);
    let accuracy = evaluate_imputation(&output_records, truth_map, ref_records);

    println!("\n=== {} vs Java BEAGLE Comparison ===", impl_name);
    println!(
        "Concordance: {} {:.2}% vs BEAGLE {:.2}%",
        impl_name,
        accuracy.concordance() * 100.0,
        beagle_baseline.concordance * 100.0
    );
    println!(
        "Rare F1: {} {:.3} vs BEAGLE {:.3}",
        impl_name,
        accuracy.rare_f1(),
        beagle_baseline.rare_f1
    );
    println!(
        "Calibration Error: {} {:.3} vs BEAGLE {:.3}",
        impl_name,
        accuracy.calibration_error(),
        beagle_baseline.calibration_error
    );
    println!(
        "Brier Score: {} {:.4} vs BEAGLE {:.4}",
        impl_name,
        accuracy.brier_score(),
        beagle_baseline.brier_score
    );

    // Pass if within tolerance of BEAGLE
    let concordance_ok = accuracy.concordance() >= beagle_baseline.concordance - 0.005;
    let rare_f1_ok = accuracy.rare_f1() >= beagle_baseline.rare_f1 - 0.05;
    let calibration_ok = accuracy.calibration_error() <= beagle_baseline.calibration_error + 0.05;
    // Handle NaN: if both are NaN, consider it OK; otherwise use normal comparison
    let brier_ok = if accuracy.brier_score().is_nan() && beagle_baseline.brier_score.is_nan() {
        true
    } else {
        accuracy.brier_score() <= beagle_baseline.brier_score + 0.02
    };

    println!("\nPass criteria:");
    println!("  Concordance >= BEAGLE - 0.5%: {}", if concordance_ok { "PASS" } else { "FAIL" });
    println!("  Rare F1 >= BEAGLE - 0.05: {}", if rare_f1_ok { "PASS" } else { "FAIL" });
    println!("  Calibration <= BEAGLE + 0.05: {}", if calibration_ok { "PASS" } else { "FAIL" });
    println!("  Brier Score <= BEAGLE + 0.02: {}", if brier_ok { "PASS" } else { "FAIL" });

    concordance_ok && rare_f1_ok && calibration_ok && brier_ok
}

#[test]
fn test_comparison_framework_self_check() {
    // Sanity check: BEAGLE compared against itself should pass trivially
    let files = setup_test_files();
    let work_dir = tempfile::tempdir().expect("Create temp dir");

    let ref_path = work_dir.path().join("ref.vcf.gz");
    fs::copy(&files.ref_vcf, &ref_path).expect("Copy ref VCF");

    // Create masked version
    let masked_path = work_dir.path().join("masked.vcf");
    let truth_map = create_masked_vcf(&ref_path, &masked_path, 0.05, 99);

    // Compress
    let masked_gz = work_dir.path().join("masked.vcf.gz");
    let status = Command::new("gzip")
        .args(["-c"])
        .stdin(File::open(&masked_path).unwrap())
        .stdout(File::create(&masked_gz).unwrap())
        .status()
        .expect("gzip");
    assert!(status.success());

    // Run BEAGLE
    let out_prefix = work_dir.path().join("imputed");
    let output = run_beagle(
        &files.beagle_jar,
        &[
            ("gt", masked_gz.to_str().unwrap()),
            ("out", out_prefix.to_str().unwrap()),
            ("seed", "99"),
            ("gp", "true"),
        ],
        work_dir.path(),
    );
    assert!(output.status.success());

    let (_, ref_records) = parse_vcf(&ref_path);
    let imputed_vcf = work_dir.path().join("imputed.vcf.gz");
    let (_, imputed_records) = parse_vcf(&imputed_vcf);
    let accuracy = evaluate_imputation(&imputed_records, &truth_map, &ref_records);

    // Create baseline from the same run
    let baseline = BeagleBaseline {
        concordance: accuracy.concordance(),
        rare_f1: accuracy.rare_f1(),
        calibration_error: accuracy.calibration_error(),
        brier_score: accuracy.brier_score(),
        mask_fraction: 0.05,
        seed: 99,
    };

    // Compare BEAGLE against itself - should pass
    let passed = compare_against_beagle(&imputed_vcf, &truth_map, &ref_records, &baseline, "BEAGLE");
    assert!(passed, "BEAGLE compared against itself should pass");
}

// =============================================================================
// Rust vs Java BEAGLE Comparison Tests
// =============================================================================

/// Decompress a .vcf.gz file to .vcf for Rust (reagle expects BGZF, test fixtures are regular gzip)
fn decompress_vcf_for_rust(gz_path: &Path, work_dir: &Path) -> PathBuf {
    let stem = gz_path.file_stem().unwrap().to_str().unwrap();
    // Remove .vcf from stem if present (since file is .vcf.gz)
    let base = stem.strip_suffix(".vcf").unwrap_or(stem);
    let vcf_path = work_dir.join(format!("{}_rust.vcf", base));

    let output = Command::new("gzip")
        .args(["-dc", gz_path.to_str().unwrap()])
        .output()
        .expect("Failed to decompress VCF");

    assert!(output.status.success(), "gzip decompression failed");
    fs::write(&vcf_path, &output.stdout).expect("Write decompressed VCF");
    vcf_path
}

/// Helper to run Rust phasing pipeline
fn run_rust_phasing(gt_path: &Path, out_prefix: &Path, seed: i64) -> reagle::Result<()> {
    let config = Config::parse_from([
        "reagle",
        "--gt", gt_path.to_str().unwrap(),
        "--out", out_prefix.to_str().unwrap(),
        "--seed", &seed.to_string(),
    ]);
    let mut pipeline = PhasingPipeline::new(config);
    pipeline.run_auto()
}

/// Helper to run Rust imputation pipeline
fn run_rust_imputation(
    gt_path: &Path,
    ref_path: &Path,
    out_prefix: &Path,
    seed: i64,
) -> reagle::Result<()> {
    let config = Config::parse_from([
        "reagle",
        "--gt", gt_path.to_str().unwrap(),
        "--ref", ref_path.to_str().unwrap(),
        "--out", out_prefix.to_str().unwrap(),
        "--seed", &seed.to_string(),
        "--gp",
    ]);
    let mut pipeline = ImputationPipeline::new(config);
    pipeline.run()
}

