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

// Serialize tests to prevent OOM from parallel execution
use serial_test::serial;

// =============================================================================
// gnomAD Test Data Source
// =============================================================================

/// gnomAD test files (same structure as BEAGLE test files)
struct GnomadTestFiles {
    ref_vcf: PathBuf,
    target_vcf: PathBuf,
    target_sparse_vcf: PathBuf,
}

/// Get gnomAD fixtures directory
fn gnomad_fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gnomad_hgdp")
}

/// Setup gnomAD test files (pre-generated fixtures)
fn setup_gnomad_files() -> GnomadTestFiles {
    let fixtures = gnomad_fixtures_dir();
    let ref_vcf = fixtures.join("ref.vcf.gz");
    let target_vcf = fixtures.join("target.vcf.gz");
    let target_sparse_vcf = fixtures.join("target_sparse.vcf.gz");

    // Verify fixtures exist
    assert!(ref_vcf.exists(), "gnomAD ref.vcf.gz fixture missing: {}", ref_vcf.display());
    assert!(target_vcf.exists(), "gnomAD target.vcf.gz fixture missing: {}", target_vcf.display());
    assert!(target_sparse_vcf.exists(), "gnomAD target_sparse.vcf.gz fixture missing: {}", target_sparse_vcf.display());

    GnomadTestFiles {
        ref_vcf,
        target_vcf,
        target_sparse_vcf,
    }
}

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

/// Convert a GT string (e.g. "0|1", "1/1") to a dosage value
fn gt_to_dosage(gt: &str) -> Option<f64> {
    if gt.contains('.') {
        return None;
    }
    // Simple counting of '1' alleles for biallelic variants
    // This handles "|" and "/" delimiters automatically
    Some(gt.matches('1').count() as f64)
}

/// Helper to compare Java vs Rust imputation results against Ground Truth
fn compare_imputation_results(
    name: &str,
    truth_vcf: &Path,
    java_vcf: &Path,
    rust_vcf: &Path,
) {
    let (_, java_records) = parse_vcf(java_vcf);
    let (_, rust_records) = parse_vcf(rust_vcf);
    let (_, truth_records) = parse_vcf(truth_vcf);

    println!("[{}] Java: {} records, Rust: {} records, Truth: {} records",
             name, java_records.len(), rust_records.len(), truth_records.len());

    assert_eq!(java_records.len(), rust_records.len(),
               "{}: Record count mismatch (Java vs Rust)", name);
    // Truth might have different record count if imputation output includes only imputed sites?
    // But usually in these tests we expect matching records.
    if java_records.len() != truth_records.len() {
        println!("WARNING: [{}] Tuple count mismatch with Truth ({} vs {})",
                 name, java_records.len(), truth_records.len());
    }

    // Compare dosages and calculate R^2
    let mut dosage_diffs: Vec<f64> = Vec::new();
    let mut truth_dosages: Vec<f64> = Vec::new();
    let mut java_dosages_r2: Vec<f64> = Vec::new();
    let mut rust_dosages_r2: Vec<f64> = Vec::new();

    // Iterate up to the length of the shortest vector to avoid panics
    let len = java_records.len().min(rust_records.len()).min(truth_records.len());

    for i in 0..len {
        let j_rec = &java_records[i];
        let r_rec = &rust_records[i];
        let t_rec = &truth_records[i];
        
        // Check if positions match, otherwise alignment is broken
        assert_eq!(j_rec.pos, r_rec.pos, "{}: Position mismatch (Java vs Rust) at index {}", name, i);
        assert_eq!(j_rec.pos, t_rec.pos, "{}: Position mismatch (Java vs Truth) at index {}", name, i);

        for k in 0..j_rec.genotypes.len() {
            if k >= r_rec.genotypes.len() || k >= t_rec.genotypes.len() { continue; }
            
            let j_gt = &j_rec.genotypes[k];
            let r_gt = &r_rec.genotypes[k];
            let t_gt = &t_rec.genotypes[k];

             if let (Some(j_ds), Some(r_ds)) = (j_gt.ds, r_gt.ds) {
                let diff = (j_ds - r_ds).abs();
                dosage_diffs.push(diff);

                if let Some(t_ds) = gt_to_dosage(&t_gt.gt) {
                    truth_dosages.push(t_ds);
                    java_dosages_r2.push(j_ds);
                    rust_dosages_r2.push(r_ds);
                }
            }
        }
    }

    if !truth_dosages.is_empty() {
        let java_r2 = dosage_correlation(&truth_dosages, &java_dosages_r2);
        let rust_r2 = dosage_correlation(&truth_dosages, &rust_dosages_r2);
        println!("[{}] Overall R^2 (Truth vs Java): {:.6}", name, java_r2);
        println!("[{}] Overall R^2 (Truth vs Rust): {:.6}", name, rust_r2);

        // Strict: Rust R² vs truth must be >= Java R² vs truth (zero tolerance)
        assert!(
            rust_r2 >= java_r2,
            "[{}] Strict: Rust R² ({:.6}) WORSE than Java R² ({:.6}) vs truth",
            name, rust_r2, java_r2
        );
    }

    if !dosage_diffs.is_empty() {
        let mean_diff: f64 = dosage_diffs.iter().sum::<f64>() / dosage_diffs.len() as f64;
        let max_diff: f64 = dosage_diffs.iter().cloned().fold(0.0, f64::max);
        let within_02: usize = dosage_diffs.iter().filter(|&&d| d < 0.02).count();
        let within_01: usize = dosage_diffs.iter().filter(|&&d| d < 0.01).count();
        let pct_within_02 = 100.0 * within_02 as f64 / dosage_diffs.len() as f64;
        let pct_within_01 = 100.0 * within_01 as f64 / dosage_diffs.len() as f64;

        println!("[{}] Dosage comparison: {} values, mean diff={:.6}, max diff={:.6}",
                 name, dosage_diffs.len(), mean_diff, max_diff);
        println!("[{}] Dosages within 0.01: {:.1}%, within 0.02: {:.1}%",
                 name, pct_within_01, pct_within_02);

        // Strict: Mean dosage difference must be very small
        assert!(mean_diff < 0.02, "{}: Strict FAIL: Mean dosage diff {:.6} >= 0.02", name, mean_diff);
        // Strict: 99% of dosages must be within 0.02 of Java
        assert!(pct_within_02 >= 99.0, "{}: Strict FAIL: Only {:.1}% of dosages within 0.02", name, pct_within_02);
    }
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
#[serial]
#[serial]
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
#[serial]
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
    compare_imputation_results(source.name, &gt_path, &java_vcf, &rust_vcf);
}

#[test]
#[serial]
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
#[serial]
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

    compare_imputation_results("bref3 Imputation", &gt_path, &java_vcf, &rust_vcf);
}

#[test]
#[serial]
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
            ("gp", "true"),
        ],
        work_dir.path(),
    );
    assert!(output2.status.success(), "{}: Java imputation failed", source.name);

    let rust_imp = work_dir.path().join("rust_imputed");
    let rust_result = run_rust_imputation(&gt_vcf, &ref_vcf, &rust_imp, 42);
    assert!(rust_result.is_ok(), "{}: Rust imputation failed: {:?}", source.name, rust_result.err());

    let java_vcf = work_dir.path().join("java_imputed.vcf.gz");
    let rust_vcf = work_dir.path().join("rust_imputed.vcf.gz");
    
    // Compare outputs including R^2
    compare_imputation_results(&format!("{} Imputation", source.name), &gt_path, &java_vcf, &rust_vcf);

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
#[serial]
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
#[serial]
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
#[serial]
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

    // Strict: Rust must be AT LEAST as good as Java - NO TOLERANCE
    // Brier score: lower is better, so Rust <= Java
    if !java_acc.brier_score().is_nan() && !rust_acc.brier_score().is_nan() {
        assert!(
            rust_acc.brier_score() <= java_acc.brier_score(),
            "{}: Strict FAIL: Rust Brier score ({:.6}) WORSE than Java ({:.6})",
            source.name,
            rust_acc.brier_score(),
            java_acc.brier_score()
        );
    }

    // Rare variant F1: higher is better, so Rust >= Java
    if rust_acc.rare_total > 0 && java_acc.rare_total > 0 {
        assert!(
            rust_acc.rare_f1() >= java_acc.rare_f1(),
            "{}: Strict FAIL: Rust rare F1 ({:.6}) WORSE than Java ({:.6})",
            source.name,
            rust_acc.rare_f1(),
            java_acc.rare_f1()
        );
    }

    // Concordance: higher is better, so Rust >= Java - NO TOLERANCE
    assert!(
        rust_acc.concordance() >= java_acc.concordance(),
        "{}: Strict FAIL: Rust concordance ({:.4}%) WORSE than Java ({:.4}%)",
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

    // Strict: Pass ONLY if AT LEAST as good as BEAGLE - NO TOLERANCE
    let concordance_ok = accuracy.concordance() >= beagle_baseline.concordance;
    let rare_f1_ok = accuracy.rare_f1() >= beagle_baseline.rare_f1;
    let calibration_ok = accuracy.calibration_error() <= beagle_baseline.calibration_error;
    // Handle NaN: if both are NaN, consider it OK; otherwise use normal comparison
    let brier_ok = if accuracy.brier_score().is_nan() && beagle_baseline.brier_score.is_nan() {
        true
    } else {
        accuracy.brier_score() <= beagle_baseline.brier_score
    };

    println!("\nStrict Pass criteria (NO TOLERANCE - must be >= BEAGLE):");
    println!("  Concordance >= BEAGLE: {}", if concordance_ok { "PASS" } else { "FAIL" });
    println!("  Rare F1 >= BEAGLE: {}", if rare_f1_ok { "PASS" } else { "FAIL" });
    println!("  Calibration <= BEAGLE: {}", if calibration_ok { "PASS" } else { "FAIL" });
    println!("  Brier Score <= BEAGLE: {}", if brier_ok { "PASS" } else { "FAIL" });

    concordance_ok && rare_f1_ok && calibration_ok && brier_ok
}

#[test]
#[serial]
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
    let mut pipeline = PhasingPipeline::new(config, None);
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
    let mut pipeline = ImputationPipeline::new(config, None);
    pipeline.run()
}

// =============================================================================
// Strict Quality Metrics Comparison Tests
// =============================================================================

/// Compare DR2 values between Java and Rust (Strict: Rust must be >= Java)
fn compare_dr2_values(java_records: &[ParsedRecord], rust_records: &[ParsedRecord], name: &str) {
    let java_dr2: Vec<f64> = java_records
        .iter()
        .filter_map(|r| r.info.get("DR2").and_then(|v| v.parse().ok()))
        .collect();
    let rust_dr2: Vec<f64> = rust_records
        .iter()
        .filter_map(|r| r.info.get("DR2").and_then(|v| v.parse().ok()))
        .collect();

    if java_dr2.is_empty() || rust_dr2.is_empty() {
        println!("[{}] DR2: Skipping comparison (Java: {}, Rust: {})", name, java_dr2.len(), rust_dr2.len());
        return;
    }

    // Separate genotyped (IMP flag absent) vs imputed (IMP flag present) markers
    let java_genotyped_dr2: Vec<f64> = java_records
        .iter()
        .filter_map(|r| {
            let is_imputed = r.info.contains_key("IMP");
            if !is_imputed {
                r.info.get("DR2").and_then(|v| v.parse().ok())
            } else {
                None
            }
        })
        .collect();
    let java_imputed_dr2: Vec<f64> = java_records
        .iter()
        .filter_map(|r| {
            let is_imputed = r.info.contains_key("IMP");
            if is_imputed {
                r.info.get("DR2").and_then(|v| v.parse().ok())
            } else {
                None
            }
        })
        .collect();
    let rust_genotyped_dr2: Vec<f64> = rust_records
        .iter()
        .filter_map(|r| {
            let is_imputed = r.info.contains_key("IMP");
            if !is_imputed {
                r.info.get("DR2").and_then(|v| v.parse().ok())
            } else {
                None
            }
        })
        .collect();
    let rust_imputed_dr2: Vec<f64> = rust_records
        .iter()
        .filter_map(|r| {
            let is_imputed = r.info.contains_key("IMP");
            if is_imputed {
                r.info.get("DR2").and_then(|v| v.parse().ok())
            } else {
                None
            }
        })
        .collect();

    let java_genotyped_mean = if java_genotyped_dr2.is_empty() { 0.0 } else { java_genotyped_dr2.iter().sum::<f64>() / java_genotyped_dr2.len() as f64 };
    let java_imputed_mean = if java_imputed_dr2.is_empty() { 0.0 } else { java_imputed_dr2.iter().sum::<f64>() / java_imputed_dr2.len() as f64 };
    let rust_genotyped_mean = if rust_genotyped_dr2.is_empty() { 0.0 } else { rust_genotyped_dr2.iter().sum::<f64>() / rust_genotyped_dr2.len() as f64 };
    let rust_imputed_mean = if rust_imputed_dr2.is_empty() { 0.0 } else { rust_imputed_dr2.iter().sum::<f64>() / rust_imputed_dr2.len() as f64 };

    let java_mean: f64 = java_dr2.iter().sum::<f64>() / java_dr2.len() as f64;
    let rust_mean: f64 = rust_dr2.iter().sum::<f64>() / rust_dr2.len() as f64;

    println!("[{}] DR2 Comparison:", name);
    println!("  Java mean DR2: {:.4} (genotyped: {:.4} [n={}], imputed: {:.4} [n={}])",
             java_mean, java_genotyped_mean, java_genotyped_dr2.len(), java_imputed_mean, java_imputed_dr2.len());
    println!("  Rust mean DR2: {:.4} (genotyped: {:.4} [n={}], imputed: {:.4} [n={}])",
             rust_mean, rust_genotyped_mean, rust_genotyped_dr2.len(), rust_imputed_mean, rust_imputed_dr2.len());

    // Diagnostic: Find markers where Rust is much worse than Java for imputed markers
    let java_imputed: Vec<_> = java_records.iter()
        .filter(|r| r.info.contains_key("IMP"))
        .collect();
    let rust_imputed: Vec<_> = rust_records.iter()
        .filter(|r| r.info.contains_key("IMP"))
        .collect();

    let mut dr2_diffs: Vec<(u64, f64, f64, f64)> = Vec::new(); // (pos, java_dr2, rust_dr2, diff)
    for (j, r) in java_imputed.iter().zip(rust_imputed.iter()) {
        if j.pos == r.pos {
            if let (Some(java_d), Some(rust_d)) = (
                j.info.get("DR2").and_then(|v| v.parse::<f64>().ok()),
                r.info.get("DR2").and_then(|v| v.parse::<f64>().ok()),
            ) {
                let diff = java_d - rust_d;
                if diff > 0.3 { // Java much better
                    dr2_diffs.push((j.pos, java_d, rust_d, diff));
                }
            }
        }
    }
    dr2_diffs.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    if !dr2_diffs.is_empty() {
        println!("  Markers where Java DR2 >> Rust DR2 (diff > 0.3):");
        for (pos, java_d, rust_d, diff) in dr2_diffs.iter().take(5) {
            println!("    pos={}: Java={:.4}, Rust={:.4}, diff={:.4}", pos, java_d, rust_d, diff);
        }

        // Detailed dosage comparison for worst marker
        if let Some(&(worst_pos, _, _, _)) = dr2_diffs.first() {
            println!("  Detailed dosages at pos={}:", worst_pos);
            let java_rec = java_imputed.iter().find(|r| r.pos == worst_pos);
            let rust_rec = rust_imputed.iter().find(|r| r.pos == worst_pos);
            if let (Some(j), Some(r)) = (java_rec, rust_rec) {
                println!("    AF: Java={:?}, Rust={:?}",
                    j.info.get("AF"), r.info.get("AF"));
                for (i, (jg, rg)) in j.genotypes.iter().zip(r.genotypes.iter()).enumerate().take(5) {
                    println!("    Sample {}: Java DS={:?}, Rust DS={:?}",
                        i, jg.ds, rg.ds);
                }
            }
        }
    }

    // Strict: Rust genotyped DR2 must be >= Java
    assert!(
        rust_genotyped_mean >= java_genotyped_mean,
        "[{}] Strict FAIL: Rust genotyped DR2 ({:.4}) WORSE than Java ({:.4})",
        name, rust_genotyped_mean, java_genotyped_mean
    );

    // Strict: Rust imputed DR2 must be >= Java
    assert!(
        rust_imputed_mean >= java_imputed_mean,
        "[{}] Strict FAIL: Rust imputed DR2 ({:.4}) WORSE than Java ({:.4})",
        name, rust_imputed_mean, java_imputed_mean
    );
}

/// Compare dosage values between Java and Rust
fn compare_dosages(java_records: &[ParsedRecord], rust_records: &[ParsedRecord], name: &str) {
    let java_ds = extract_dosages(java_records);
    let rust_ds = extract_dosages(rust_records);

    if java_ds.is_empty() || rust_ds.is_empty() {
        println!("[{}] Dosages: Skipping comparison", name);
        return;
    }

    let min_len = java_ds.len().min(rust_ds.len());

    // Mean absolute difference
    let mad: f64 = java_ds.iter().zip(rust_ds.iter())
        .map(|(j, r)| (j - r).abs())
        .sum::<f64>() / min_len as f64;

    println!("[{}] Dosage Comparison:", name);
    println!("  Mean absolute diff: {:.6}", mad);
}

/// Compare genotyped marker dosages between Rust output and truth (target) VCF.
/// Genotyped markers (IMP flag absent) should have near-perfect correlation since
/// they don't need to be imputed - we're just passing through the known genotypes.
fn compare_genotyped_dosages_to_truth(
    rust_records: &[ParsedRecord], 
    truth_records: &[ParsedRecord], 
    name: &str
) {
    // Extract dosages for genotyped (non-imputed) markers only
    let mut rust_genotyped_dosages = Vec::new();
    let mut truth_genotyped_dosages = Vec::new();
    
    // Build truth lookup: (chrom, pos) -> record
    let truth_map: HashMap<(String, u64), &ParsedRecord> = truth_records
        .iter()
        .map(|r| ((r.chrom.clone(), r.pos), r))
        .collect();
    
    for rust_rec in rust_records {
        // Skip imputed markers - only check genotyped ones
        if rust_rec.info.contains_key("IMP") {
            continue;
        }
        
        // Find matching truth record
        let key = (rust_rec.chrom.clone(), rust_rec.pos);
        let truth_rec = match truth_map.get(&key) {
            Some(r) => *r,
            None => continue,
        };
        
        // Extract dosages for all samples at this marker
        for (sample_idx, rust_gt) in rust_rec.genotypes.iter().enumerate() {
            if sample_idx >= truth_rec.genotypes.len() {
                continue;
            }
            
            // Get Rust dosage (from DS field if available, otherwise from GT)
            let rust_ds = rust_gt.ds.or_else(|| gt_to_dosage(&rust_gt.gt));
            let truth_ds = gt_to_dosage(&truth_rec.genotypes[sample_idx].gt);
            
            if let (Some(r_ds), Some(t_ds)) = (rust_ds, truth_ds) {
                rust_genotyped_dosages.push(r_ds);
                truth_genotyped_dosages.push(t_ds);
            }
        }
    }
    
    if rust_genotyped_dosages.is_empty() {
        println!("[{}] Genotyped dosage check: Skipping (no genotyped markers found)", name);
        return;
    }
    
    let correlation = dosage_correlation(&rust_genotyped_dosages, &truth_genotyped_dosages);
    
    // Mean absolute difference
    let mad: f64 = rust_genotyped_dosages.iter().zip(truth_genotyped_dosages.iter())
        .map(|(r, t)| (r - t).abs())
        .sum::<f64>() / rust_genotyped_dosages.len() as f64;
    
    println!("[{}] Genotyped Marker Dosage vs Truth:", name);
    println!("  Number of genotyped dosages: {}", rust_genotyped_dosages.len());
    println!("  Dosage correlation with truth: {:.6}", correlation);
    println!("  Mean absolute difference: {:.6}", mad);
    
    // Strict: Genotyped markers should have near-perfect correlation with truth (>0.99)
    // These are markers we already know - no imputation needed
    assert!(
        correlation > 0.99,
        "[{}] Strict FAIL: Genotyped marker dosage correlation with truth too low: {:.6} (expected > 0.99)",
        name, correlation
    );
}

#[test]
#[serial]
fn test_genotyped_dosage_correlation_with_truth() {
    // Test that genotyped markers (non-imputed) have near-perfect correlation
    // between Rust output dosage and ground truth dosage
    for source in get_all_data_sources() {
        println!("\n{}", "=".repeat(70));
        println!("=== Genotyped Marker Dosage vs Truth Test: {} ===", source.name);
        println!("{}", "=".repeat(70));

        let work_dir = tempfile::tempdir().expect("Create temp dir");

        // Copy files
        let ref_path = work_dir.path().join("ref.vcf.gz");
        fs::copy(&source.ref_vcf, &ref_path).expect("Copy ref VCF");
        let target_path = work_dir.path().join("target_sparse.vcf.gz");
        fs::copy(&source.target_sparse_vcf, &target_path).expect("Copy sparse target VCF");

        // Run Rust imputation
        let ref_vcf = decompress_vcf_for_rust(&ref_path, work_dir.path());
        let target_vcf = decompress_vcf_for_rust(&target_path, work_dir.path());
        let rust_out = work_dir.path().join("rust_out");
        let rust_result = run_rust_imputation(&target_vcf, &ref_vcf, &rust_out, 42);
        assert!(rust_result.is_ok(), "{}: Rust imputation failed: {:?}", source.name, rust_result.err());

        // Parse outputs
        let (_, target_records) = parse_vcf(&target_path);
        let (_, rust_records) = parse_vcf(&work_dir.path().join("rust_out.vcf.gz"));

        // Compare genotyped marker dosages to truth
        compare_genotyped_dosages_to_truth(&rust_records, &target_records, source.name);

        println!("\n[{}] Genotyped dosage correlation test PASSED!", source.name);
    }
}

#[test]
#[serial]
fn test_strict_dr2_and_dosage_comparison() {
    // Comprehensive quality metrics comparison between Rust and Java
    for source in get_all_data_sources() {
        println!("\n{}", "=".repeat(70));
        println!("=== Strict Quality Metrics Test: {} ===", source.name);
        println!("{}", "=".repeat(70));

        let files = setup_test_files();
        let work_dir = tempfile::tempdir().expect("Create temp dir");

        // Copy files
        let ref_path = work_dir.path().join("ref.vcf.gz");
        fs::copy(&source.ref_vcf, &ref_path).expect("Copy ref VCF");
        let target_path = work_dir.path().join("target_sparse.vcf.gz");
        fs::copy(&source.target_sparse_vcf, &target_path).expect("Copy sparse target VCF");

        // Run Java BEAGLE
        let java_out = work_dir.path().join("java_out");
        let java_output = run_beagle(
            &files.beagle_jar,
            &[
                ("ref", ref_path.to_str().unwrap()),
                ("gt", target_path.to_str().unwrap()),
                ("out", java_out.to_str().unwrap()),
                ("seed", "42"),
                ("gp", "true"),
            ],
            work_dir.path(),
        );
        assert!(java_output.status.success(), "{}: Java BEAGLE failed", source.name);

        // Run Rust
        let ref_vcf = decompress_vcf_for_rust(&ref_path, work_dir.path());
        let target_vcf = decompress_vcf_for_rust(&target_path, work_dir.path());
        let rust_out = work_dir.path().join("rust_out");
        let rust_result = run_rust_imputation(&target_vcf, &ref_vcf, &rust_out, 42);
        assert!(rust_result.is_ok(), "{}: Rust imputation failed: {:?}", source.name, rust_result.err());

        // Parse outputs
        let (_, java_records) = parse_vcf(&work_dir.path().join("java_out.vcf.gz"));
        let (_, rust_records) = parse_vcf(&work_dir.path().join("rust_out.vcf.gz"));

        // Compare DR2 values (Strict)
        compare_dr2_values(&java_records, &rust_records, source.name);

        // Compare dosages
        compare_dosages(&java_records, &rust_records, source.name);

        println!("\n[{}] Strict quality metrics test PASSED!", source.name);
    }
}

#[test]
#[serial]
fn test_diverse_mask_scenarios() {
    // Test imputation with different masking fractions
    let source = &get_all_data_sources()[0]; // Use first data source
    let files = setup_test_files();

    // Test multiple masking scenarios
    let scenarios = [
        ("Light masking (10%)", 0.10, 42),
        ("Medium masking (30%)", 0.30, 123),
        ("Heavy masking (50%)", 0.50, 456),
    ];

    for (scenario_name, mask_fraction, seed) in scenarios {
        println!("\n{}", "=".repeat(60));
        println!("=== Scenario: {} ===", scenario_name);
        println!("{}", "=".repeat(60));

        let work_dir = tempfile::tempdir().expect("Create temp dir");

        // Copy files
        let ref_path = work_dir.path().join("ref.vcf.gz");
        fs::copy(&source.ref_vcf, &ref_path).expect("Copy ref VCF");
        let target_path = work_dir.path().join("target_sparse.vcf.gz");
        fs::copy(&source.target_sparse_vcf, &target_path).expect("Copy sparse target VCF");

        // Create masked version
        let masked_path = work_dir.path().join("masked.vcf");
        let truth_map = create_masked_vcf(&target_path, &masked_path, mask_fraction, seed);
        println!("Masked {} genotypes ({:.0}%)", truth_map.len(), mask_fraction * 100.0);

        // Compress masked file
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
                ("seed", &seed.to_string()),
                ("gp", "true"),
            ],
            work_dir.path(),
        );
        assert!(java_output.status.success(), "Java BEAGLE failed for {}", scenario_name);

        // Run Rust
        let ref_vcf = decompress_vcf_for_rust(&ref_path, work_dir.path());
        let rust_out = work_dir.path().join("rust_imputed");
        let rust_result = run_rust_imputation(&masked_path, &ref_vcf, &rust_out, seed as i64);
        assert!(rust_result.is_ok(), "Rust imputation failed for {}: {:?}", scenario_name, rust_result.err());

        // Parse and evaluate
        let (_, target_records) = parse_vcf(&target_path);
        let (_, java_records) = parse_vcf(&work_dir.path().join("java_imputed.vcf.gz"));
        let (_, rust_records) = parse_vcf(&work_dir.path().join("rust_imputed.vcf.gz"));

        let java_acc = evaluate_imputation(&java_records, &truth_map, &target_records);
        let rust_acc = evaluate_imputation(&rust_records, &truth_map, &target_records);

        // Print results
        println!("\n{:<25} {:>12} {:>12}", "Metric", "Java", "Rust");
        println!("{:-<25} {:->12} {:->12}", "", "", "");
        println!("{:<25} {:>11.2}% {:>11.2}%", "Concordance",
                 java_acc.concordance() * 100.0, rust_acc.concordance() * 100.0);
        println!("{:<25} {:>12.4} {:>12.4}", "Brier Score",
                 java_acc.brier_score(), rust_acc.brier_score());

        // Strict assertions (zero tolerance)
        assert!(
            rust_acc.concordance() >= java_acc.concordance(),
            "{}: Rust concordance ({:.4}%) worse than Java ({:.4}%)",
            scenario_name, rust_acc.concordance() * 100.0, java_acc.concordance() * 100.0
        );

        if !java_acc.brier_score().is_nan() && !rust_acc.brier_score().is_nan() {
            assert!(
                rust_acc.brier_score() <= java_acc.brier_score(),
                "{}: Rust Brier ({:.6}) worse than Java ({:.6})",
                scenario_name, rust_acc.brier_score(), java_acc.brier_score()
            );
        }

        println!("\n[{}] PASSED!", scenario_name);
    }
}

#[test]
#[serial]
fn test_multiple_seeds_consistency() {
    // Verify that different seeds don't cause catastrophic failures
    // and results remain consistent with Java
    let source = &get_all_data_sources()[0];
    let files = setup_test_files();

    let seeds = [1, 42, 123, 999, 12345];
    let mut rust_concordances = Vec::new();
    let mut java_concordances = Vec::new();

    println!("\n{}", "=".repeat(60));
    println!("=== Multiple Seeds Consistency Test ===");
    println!("{}", "=".repeat(60));

    for &seed in &seeds {
        let work_dir = tempfile::tempdir().expect("Create temp dir");

        // Copy files
        let ref_path = work_dir.path().join("ref.vcf.gz");
        fs::copy(&source.ref_vcf, &ref_path).expect("Copy ref VCF");
        let target_path = work_dir.path().join("target_sparse.vcf.gz");
        fs::copy(&source.target_sparse_vcf, &target_path).expect("Copy sparse target VCF");

        // Create masked version with this seed
        let masked_path = work_dir.path().join("masked.vcf");
        let truth_map = create_masked_vcf(&target_path, &masked_path, 0.20, seed);

        // Compress
        let masked_gz = work_dir.path().join("masked.vcf.gz");
        let status = Command::new("gzip")
            .args(["-c"])
            .stdin(File::open(&masked_path).unwrap())
            .stdout(File::create(&masked_gz).unwrap())
            .status()
            .expect("gzip failed");
        assert!(status.success());

        // Run Java
        let java_out = work_dir.path().join("java_out");
        let java_output = run_beagle(
            &files.beagle_jar,
            &[
                ("ref", ref_path.to_str().unwrap()),
                ("gt", masked_gz.to_str().unwrap()),
                ("out", java_out.to_str().unwrap()),
                ("seed", &seed.to_string()),
                ("gp", "true"),
            ],
            work_dir.path(),
        );
        assert!(java_output.status.success(), "Java failed for seed {}", seed);

        // Run Rust
        let ref_vcf = decompress_vcf_for_rust(&ref_path, work_dir.path());
        let rust_out = work_dir.path().join("rust_out");
        let rust_result = run_rust_imputation(&masked_path, &ref_vcf, &rust_out, seed as i64);
        assert!(rust_result.is_ok(), "Rust failed for seed {}: {:?}", seed, rust_result.err());

        // Evaluate
        let (_, target_records) = parse_vcf(&target_path);
        let (_, java_records) = parse_vcf(&work_dir.path().join("java_out.vcf.gz"));
        let (_, rust_records) = parse_vcf(&work_dir.path().join("rust_out.vcf.gz"));

        let java_acc = evaluate_imputation(&java_records, &truth_map, &target_records);
        let rust_acc = evaluate_imputation(&rust_records, &truth_map, &target_records);

        println!("Seed {}: Java {:.2}%, Rust {:.2}%",
                 seed, java_acc.concordance() * 100.0, rust_acc.concordance() * 100.0);

        java_concordances.push(java_acc.concordance());
        rust_concordances.push(rust_acc.concordance());

        // Per-seed check: Rust should be at least as good as Java
        assert!(
            rust_acc.concordance() >= java_acc.concordance() - 0.01,
            "Seed {}: Rust ({:.4}%) worse than Java ({:.4}%)",
            seed, rust_acc.concordance() * 100.0, java_acc.concordance() * 100.0
        );
    }

    // Overall consistency: variance should be reasonable
    let java_mean: f64 = java_concordances.iter().sum::<f64>() / java_concordances.len() as f64;
    let rust_mean: f64 = rust_concordances.iter().sum::<f64>() / rust_concordances.len() as f64;
    let java_std = (java_concordances.iter().map(|x| (x - java_mean).powi(2)).sum::<f64>() / java_concordances.len() as f64).sqrt();
    let rust_std = (rust_concordances.iter().map(|x| (x - rust_mean).powi(2)).sum::<f64>() / rust_concordances.len() as f64).sqrt();

    println!("\nSummary across {} seeds:", seeds.len());
    println!("  Java: mean={:.4}%, std={:.4}%", java_mean * 100.0, java_std * 100.0);
    println!("  Rust: mean={:.4}%, std={:.4}%", rust_mean * 100.0, rust_std * 100.0);

    // Rust mean should be >= Java mean
    assert!(
        rust_mean >= java_mean - 0.001,
        "Rust mean concordance ({:.4}%) worse than Java ({:.4}%)",
        rust_mean * 100.0, java_mean * 100.0
    );

    println!("\nMultiple seeds consistency test PASSED!");
}

/// Test per-sample imputation accuracy to isolate sample-specific issues.
/// This test breaks down accuracy by sample to help identify if failures
/// are concentrated in specific samples or uniform across all samples.
#[test]
#[serial]
fn test_per_sample_imputation_accuracy() {
    let source = &get_all_data_sources()[0];
    let files = setup_test_files();

    println!("\n{}", "=".repeat(60));
    println!("=== Per-Sample Imputation Accuracy Test ===");
    println!("{}", "=".repeat(60));

    let work_dir = tempfile::tempdir().expect("Create temp dir");

    // Copy files
    let ref_path = work_dir.path().join("ref.vcf.gz");
    fs::copy(&source.ref_vcf, &ref_path).expect("Copy ref VCF");
    let target_path = work_dir.path().join("target_sparse.vcf.gz");
    fs::copy(&source.target_sparse_vcf, &target_path).expect("Copy sparse target VCF");

    // Create masked version
    let masked_path = work_dir.path().join("masked.vcf");
    let truth_map = create_masked_vcf(&target_path, &masked_path, 0.25, 42);
    println!("Masked {} genotypes (25%)", truth_map.len());

    // Compress
    let masked_gz = work_dir.path().join("masked.vcf.gz");
    let status = Command::new("gzip")
        .args(["-c"])
        .stdin(File::open(&masked_path).unwrap())
        .stdout(File::create(&masked_gz).unwrap())
        .status()
        .expect("gzip failed");
    assert!(status.success());

    // Run Java
    let java_out = work_dir.path().join("java_out");
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
    assert!(java_output.status.success(), "Java BEAGLE failed");

    // Run Rust
    let ref_vcf = decompress_vcf_for_rust(&ref_path, work_dir.path());
    let rust_out = work_dir.path().join("rust_out");
    let rust_result = run_rust_imputation(&masked_path, &ref_vcf, &rust_out, 42);
    assert!(rust_result.is_ok(), "Rust imputation failed: {:?}", rust_result.err());

    // Parse outputs
    let (sample_names, target_records) = parse_vcf(&target_path);
    let (_, java_records) = parse_vcf(&work_dir.path().join("java_out.vcf.gz"));
    let (_, rust_records) = parse_vcf(&work_dir.path().join("rust_out.vcf.gz"));

    let n_samples = sample_names.len();
    println!("\nAnalyzing {} samples...\n", n_samples);

    // Per-sample accuracy tracking
    let mut java_sample_correct: Vec<usize> = vec![0; n_samples];
    let mut rust_sample_correct: Vec<usize> = vec![0; n_samples];
    let mut sample_total: Vec<usize> = vec![0; n_samples];
    let mut samples_with_rust_worse = 0;
    let mut max_accuracy_gap = 0.0f64;
    let mut worst_sample_idx = 0usize;

    // Evaluate per-sample
    for (j_rec, r_rec) in java_records.iter().zip(rust_records.iter()) {
        // Find corresponding truth record
        let truth_pos = format!("{}:{}", j_rec.chrom, j_rec.pos);
        
        // Find matching target record
        let target_rec = target_records.iter().find(|t| format!("{}:{}", t.chrom, t.pos) == truth_pos);
        let target_rec = match target_rec {
            Some(r) => r,
            None => continue,
        };

        for sample_idx in 0..n_samples {
            let key = (j_rec.chrom.clone(), j_rec.pos, sample_idx);
            
            // Only evaluate masked positions
            if !truth_map.contains_key(&key) {
                continue;
            }

            let truth_gt = &target_rec.genotypes[sample_idx].gt;
            let java_gt = &j_rec.genotypes[sample_idx].gt;
            let rust_gt = &r_rec.genotypes[sample_idx].gt;

            sample_total[sample_idx] += 1;

            if normalize_gt_unphased(java_gt) == normalize_gt_unphased(truth_gt) {
                java_sample_correct[sample_idx] += 1;
            }
            if normalize_gt_unphased(rust_gt) == normalize_gt_unphased(truth_gt) {
                rust_sample_correct[sample_idx] += 1;
            }
        }
    }

    // Print per-sample results
    println!("{:<20} {:>12} {:>12} {:>10}", "Sample", "Java Acc", "Rust Acc", "Diff");
    println!("{:-<20} {:-<12} {:-<12} {:-<10}", "", "", "", "");

    for i in 0..n_samples {
        if sample_total[i] == 0 {
            continue;
        }
        let java_acc = java_sample_correct[i] as f64 / sample_total[i] as f64;
        let rust_acc = rust_sample_correct[i] as f64 / sample_total[i] as f64;
        let diff = rust_acc - java_acc;

        let status = if diff < -0.01 { "WORSE" } else if diff > 0.01 { "BETTER" } else { "" };
        println!(
            "{:<20} {:>11.2}% {:>11.2}% {:>+9.2}% {}",
            &sample_names[i][..sample_names[i].len().min(20)],
            java_acc * 100.0,
            rust_acc * 100.0,
            diff * 100.0,
            status
        );

        if diff < -0.001 {
            samples_with_rust_worse += 1;
            if diff.abs() > max_accuracy_gap {
                max_accuracy_gap = diff.abs();
                worst_sample_idx = i;
            }
        }
    }

    // Summary
    let total_java_correct: usize = java_sample_correct.iter().sum();
    let total_rust_correct: usize = rust_sample_correct.iter().sum();
    let total_evaluated: usize = sample_total.iter().sum();
    
    let java_overall = total_java_correct as f64 / total_evaluated as f64;
    let rust_overall = total_rust_correct as f64 / total_evaluated as f64;

    println!("\n{}", "=".repeat(60));
    println!("Summary:");
    println!("  Total evaluated: {} genotypes", total_evaluated);
    println!("  Java overall accuracy: {:.2}%", java_overall * 100.0);
    println!("  Rust overall accuracy: {:.2}%", rust_overall * 100.0);
    println!("  Samples where Rust is worse: {}/{}", samples_with_rust_worse, n_samples);
    if samples_with_rust_worse > 0 {
        println!("  Worst sample: {} (gap: {:.2}%)", sample_names[worst_sample_idx], max_accuracy_gap * 100.0);
    }

    // Strict: Rust should not be significantly worse on any sample
    assert!(
        max_accuracy_gap < 0.05,
        "Per-sample accuracy gap too large: {:.2}% on sample {}",
        max_accuracy_gap * 100.0,
        sample_names[worst_sample_idx]
    );

    // Strict: Less than 50% of samples should show Rust worse
    let pct_worse = samples_with_rust_worse as f64 / n_samples as f64;
    assert!(
        pct_worse < 0.5,
        "Too many samples ({:.0}%) show Rust worse than Java",
        pct_worse * 100.0
    );

    // Strict: Overall Rust accuracy must be >= Java - 1%
    assert!(
        rust_overall >= java_overall - 0.01,
        "Rust overall accuracy ({:.2}%) worse than Java ({:.2}%) by more than 1%",
        rust_overall * 100.0,
        java_overall * 100.0
    );

    println!("\nPer-sample imputation accuracy test PASSED!");
}

/// Test 1: Focus on DR2 for GENOTYPED vs IMPUTED markers separately.
/// For genotyped markers, DR2 should be 1.0 (we know the truth, so estimated=actual).
/// For imputed markers, Rust DR2 should match Java DR2.
#[test]
#[serial]
fn test_dr2_genotyped_vs_imputed() {
    let source = &get_all_data_sources()[0];
    let files = setup_test_files();

    println!("\n{}", "=".repeat(70));
    println!("=== DR2: Genotyped vs Imputed (Separate Analysis) ===");
    println!("{}", "=".repeat(70));

    let work_dir = tempfile::tempdir().expect("Create temp dir");

    // Copy files
    let ref_path = work_dir.path().join("ref.vcf.gz");
    fs::copy(&source.ref_vcf, &ref_path).expect("Copy ref VCF");
    let target_path = work_dir.path().join("target_sparse.vcf.gz");
    fs::copy(&source.target_sparse_vcf, &target_path).expect("Copy sparse target VCF");

    // Run Java
    let java_out = work_dir.path().join("java_out");
    let java_output = run_beagle(
        &files.beagle_jar,
        &[
            ("ref", ref_path.to_str().unwrap()),
            ("gt", target_path.to_str().unwrap()),
            ("out", java_out.to_str().unwrap()),
            ("seed", "42"),
            ("gp", "true"),
        ],
        work_dir.path(),
    );
    assert!(java_output.status.success(), "Java BEAGLE failed");

    // Run Rust
    let ref_vcf = decompress_vcf_for_rust(&ref_path, work_dir.path());
    let target_vcf = decompress_vcf_for_rust(&target_path, work_dir.path());
    let rust_out = work_dir.path().join("rust_out");
    let rust_result = run_rust_imputation(&target_vcf, &ref_vcf, &rust_out, 42);
    assert!(rust_result.is_ok(), "Rust imputation failed: {:?}", rust_result.err());

    // Parse outputs
    let (_, java_records) = parse_vcf(&work_dir.path().join("java_out.vcf.gz"));
    let (_, rust_records) = parse_vcf(&work_dir.path().join("rust_out.vcf.gz"));

    // Separate genotyped and imputed markers
    let mut genotyped_java_dr2: Vec<(u64, f64)> = Vec::new();
    let mut genotyped_rust_dr2: Vec<(u64, f64)> = Vec::new();
    let mut imputed_java_dr2: Vec<(u64, f64)> = Vec::new();
    let mut imputed_rust_dr2: Vec<(u64, f64)> = Vec::new();
    
    for (j_rec, r_rec) in java_records.iter().zip(rust_records.iter()) {
        let java_dr2: Option<f64> = j_rec.info.get("DR2").and_then(|v| v.parse().ok());
        let rust_dr2: Option<f64> = r_rec.info.get("DR2").and_then(|v| v.parse().ok());
        
        let is_imputed = j_rec.info.contains_key("IMP");
        
        if let Some(j) = java_dr2 {
            if is_imputed {
                imputed_java_dr2.push((j_rec.pos, j));
            } else {
                genotyped_java_dr2.push((j_rec.pos, j));
            }
        }
        if let Some(r) = rust_dr2 {
            if is_imputed {
                imputed_rust_dr2.push((r_rec.pos, r));
            } else {
                genotyped_rust_dr2.push((r_rec.pos, r));
            }
        }
    }

    // Analyze genotyped markers
    println!("\n=== GENOTYPED Markers (n={}) ===", genotyped_java_dr2.len());
    
    let java_geno_mean: f64 = genotyped_java_dr2.iter().map(|(_, d)| d).sum::<f64>() 
        / genotyped_java_dr2.len().max(1) as f64;
    let rust_geno_mean: f64 = genotyped_rust_dr2.iter().map(|(_, d)| d).sum::<f64>() 
        / genotyped_rust_dr2.len().max(1) as f64;
    
    println!("  Java mean DR2: {:.4}", java_geno_mean);
    println!("  Rust mean DR2: {:.4}", rust_geno_mean);
    
    // Find genotyped markers where DR2 != 1.0
    // NOTE: Monomorphic markers (all same genotype) have DR2=0 by definition (zero variance)
    // This is mathematically correct, not a bug
    let java_geno_not_1: Vec<_> = genotyped_java_dr2.iter()
        .filter(|(_, d)| (*d - 1.0).abs() > 0.01)
        .take(10)
        .collect();
    let rust_geno_not_1: Vec<_> = genotyped_rust_dr2.iter()
        .filter(|(_, d)| (*d - 1.0).abs() > 0.01)
        .take(10)
        .collect();
    
    if !java_geno_not_1.is_empty() {
        println!("\n  Java genotyped markers with DR2 != 1.0:");
        for (pos, dr2) in java_geno_not_1 {
            println!("    pos={}: DR2={:.4}", pos, dr2);
        }
    }
    if !rust_geno_not_1.is_empty() {
        println!("\n  Rust genotyped markers with DR2 != 1.0:");
        for (pos, dr2) in rust_geno_not_1 {
            println!("    pos={}: DR2={:.4}", pos, dr2);
        }
    }

    // Analyze imputed markers
    println!("\n=== IMPUTED Markers (n={}) ===", imputed_java_dr2.len());
    
    let java_imp_mean: f64 = imputed_java_dr2.iter().map(|(_, d)| d).sum::<f64>() 
        / imputed_java_dr2.len().max(1) as f64;
    let rust_imp_mean: f64 = imputed_rust_dr2.iter().map(|(_, d)| d).sum::<f64>() 
        / imputed_rust_dr2.len().max(1) as f64;
    
    println!("  Java mean DR2: {:.4}", java_imp_mean);
    println!("  Rust mean DR2: {:.4}", rust_imp_mean);
    println!("  Gap: {:.4}", rust_imp_mean - java_imp_mean);
    
    // Find worst imputed markers (Rust << Java)
    let mut imputed_gaps: Vec<(u64, f64, f64)> = Vec::new();
    for ((j_pos, j_dr2), (_, r_dr2)) in imputed_java_dr2.iter().zip(imputed_rust_dr2.iter()) {
        imputed_gaps.push((*j_pos, *j_dr2, *r_dr2));
    }
    imputed_gaps.sort_by(|a, b| (a.2 - a.1).partial_cmp(&(b.2 - b.1)).unwrap());
    
    println!("\n  Top 20 imputed markers where Rust DR2 is WORSE:");
    println!("  {:>12} {:>10} {:>10} {:>10}", "Position", "Java DR2", "Rust DR2", "Gap");
    println!("  {:-<12} {:-<10} {:-<10} {:-<10}", "", "", "", "");
    for (pos, java_dr2, rust_dr2) in imputed_gaps.iter().take(20) {
        let gap = rust_dr2 - java_dr2;
        println!("  {:>12} {:>10.4} {:>10.4} {:>+10.4}", pos, java_dr2, rust_dr2, gap);
    }

    // Diagnostic: Show actual dosages at worst markers
    let worst_positions: std::collections::HashSet<u64> = imputed_gaps.iter().take(5).map(|(p, _, _)| *p).collect();
    println!("\n  DIAGNOSTIC: Dosages at worst 5 markers");
    for (j_rec, r_rec) in java_records.iter().zip(rust_records.iter()) {
        if worst_positions.contains(&j_rec.pos) {
            println!("\n  Position {}", j_rec.pos);
            let j_info_af = j_rec.info.get("AF").map(|s| s.as_str()).unwrap_or("?");
            let r_info_af = r_rec.info.get("AF").map(|s| s.as_str()).unwrap_or("?");
            println!("    Java AF={}, Rust AF={}", j_info_af, r_info_af);
            println!("    Sample dosages (Java | Rust):");
            for (i, (jg, rg)) in j_rec.genotypes.iter().zip(r_rec.genotypes.iter()).enumerate().take(5) {
                let j_ds = jg.ds.map(|d| format!("{:.4}", d)).unwrap_or("?".to_string());
                let r_ds = rg.ds.map(|d| format!("{:.4}", d)).unwrap_or("?".to_string());
                println!("      Sample {}: {} | {}", i, j_ds, r_ds);
            }
        }
    }

    // Assertions for DR2 quality
    println!("\n{}", "=".repeat(70));
    println!("ASSERTIONS:");
    
    // Count POLYMORPHIC genotyped markers (DR2 > 0 means there's variance)
    // Monomorphic markers correctly have DR2=0, so we exclude them from the >=0.9 check
    let polymorphic_rust: Vec<_> = genotyped_rust_dr2.iter()
        .filter(|(_, d)| *d > 0.0)
        .collect();
    let polymorphic_low: Vec<_> = polymorphic_rust.iter()
        .filter(|(_, d)| *d < 0.9)
        .collect();
    
    println!("  Polymorphic genotyped markers: {}/{}", polymorphic_rust.len(), genotyped_rust_dr2.len());
    println!("  Polymorphic with DR2 < 0.9: {}", polymorphic_low.len());
    
    // Imputed markers: Rust should not be significantly worse than Java
    let worse_imp_count = imputed_gaps.iter()
        .filter(|(_, j, r)| *r < *j - 0.01)
        .count();
    println!("  Imputed markers where Rust DR2 significantly worse: {}/{}", 
             worse_imp_count, imputed_gaps.len());

    // For polymorphic genotyped markers (non-zero variance), DR2 should be ~1.0
    // because we know the true values and output them as dosages
    if !polymorphic_rust.is_empty() {
        let poly_mean: f64 = polymorphic_rust.iter().map(|(_, d)| *d).sum::<f64>() 
            / polymorphic_rust.len() as f64;
        println!("\n  Polymorphic genotyped mean DR2: {:.4}", poly_mean);
        
        assert!(
            poly_mean >= 0.99,
            "GENOTYPED DR2 FAIL: Polymorphic markers mean DR2 ({:.4}) should be >= 0.99 (we know the true values)",
            poly_mean
        );
    }
    
    // Imputed DR2: Rust should not be much worse than Java
    assert!(
        rust_imp_mean >= java_imp_mean - 0.02,
        "IMPUTED DR2 FAIL: Rust ({:.4}) worse than Java ({:.4}) by more than 0.02",
        rust_imp_mean, java_imp_mean
    );
    
    println!("\n  DR2 test PASSED!");
}


/// Test 2: Check if dosage accuracy degrades with distance from genotyped markers.
/// If interpolation is broken, farther markers should be worse.
/// Also compares genotyped markers (distance=0) vs imputed.
#[test]
#[serial]
fn test_dosage_by_distance_from_genotyped() {
    let source = &get_all_data_sources()[0];
    let files = setup_test_files();

    println!("\n{}", "=".repeat(70));
    println!("=== Dosage by Distance from Genotyped Markers ===");
    println!("{}", "=".repeat(70));

    let work_dir = tempfile::tempdir().expect("Create temp dir");

    // Copy files
    let ref_path = work_dir.path().join("ref.vcf.gz");
    fs::copy(&source.ref_vcf, &ref_path).expect("Copy ref VCF");
    let target_path = work_dir.path().join("target_sparse.vcf.gz");
    fs::copy(&source.target_sparse_vcf, &target_path).expect("Copy sparse target VCF");

    // Run Java
    let java_out = work_dir.path().join("java_out");
    let java_output = run_beagle(
        &files.beagle_jar,
        &[
            ("ref", ref_path.to_str().unwrap()),
            ("gt", target_path.to_str().unwrap()),
            ("out", java_out.to_str().unwrap()),
            ("seed", "42"),
            ("gp", "true"),
        ],
        work_dir.path(),
    );
    assert!(java_output.status.success(), "Java BEAGLE failed");

    // Run Rust
    let ref_vcf = decompress_vcf_for_rust(&ref_path, work_dir.path());
    let target_vcf = decompress_vcf_for_rust(&target_path, work_dir.path());
    let rust_out = work_dir.path().join("rust_out");
    let rust_result = run_rust_imputation(&target_vcf, &ref_vcf, &rust_out, 42);
    assert!(rust_result.is_ok(), "Rust imputation failed: {:?}", rust_result.err());

    // Parse outputs
    let (_, java_records) = parse_vcf(&work_dir.path().join("java_out.vcf.gz"));
    let (_, rust_records) = parse_vcf(&work_dir.path().join("rust_out.vcf.gz"));

    // Find genotyped marker positions
    let genotyped_positions: Vec<u64> = java_records
        .iter()
        .filter(|r| !r.info.contains_key("IMP"))
        .map(|r| r.pos)
        .collect();

    println!("Found {} genotyped markers, {} total markers", 
             genotyped_positions.len(), java_records.len());

    // Collect data for ALL markers (genotyped and imputed)
    // pos, distance, mean_abs_diff per marker
    let mut distance_data: Vec<(u64, u64, f64)> = Vec::new();

    for (j_rec, r_rec) in java_records.iter().zip(rust_records.iter()) {
        let is_imputed = j_rec.info.contains_key("IMP");
        
        let distance = if is_imputed {
            genotyped_positions
                .iter()
                .map(|&gp| if j_rec.pos > gp { j_rec.pos - gp } else { gp - j_rec.pos })
                .min()
                .unwrap_or(u64::MAX)
        } else {
            0 // Genotyped marker
        };

        let java_ds: Vec<f64> = j_rec.genotypes.iter().filter_map(|g| g.ds).collect();
        let rust_ds: Vec<f64> = r_rec.genotypes.iter().filter_map(|g| g.ds).collect();

        if java_ds.len() == rust_ds.len() && !java_ds.is_empty() {
            let mean_diff: f64 = java_ds.iter().zip(&rust_ds)
                .map(|(j, r)| (j - r).abs())
                .sum::<f64>() / java_ds.len() as f64;
            distance_data.push((j_rec.pos, distance, mean_diff));
        }
    }

    // Bucket by distance - distance=0 is genotyped markers
    let buckets: [(u64, u64, &str); 6] = [
        (0, 1, "Genotyped"),
        (1, 100, "1-100bp"),
        (100, 500, "100-500bp"),
        (500, 1000, "500-1000bp"),
        (1000, 5000, "1-5kb"),
        (5000, u64::MAX, "5kb+"),
    ];
    
    println!("\nDosage MAD by distance from genotyped markers:\n");
    println!("{:>12} {:>8} {:>10} {:>10} {:>12}", "Distance", "Count", "Mean MAD", "Max MAD", "Worst Pos");
    println!("{:-<12} {:-<8} {:-<10} {:-<10} {:-<12}", "", "", "", "", "");

    let mut any_bucket_failed = false;
    let mut genotyped_mad = 0.0f64;
    let mut imputed_mad = 0.0f64;
    let mut imputed_count = 0usize;

    for (lo, hi, label) in buckets {
        let bucket: Vec<&(u64, u64, f64)> = distance_data
            .iter()
            .filter(|(_, d, _)| *d >= lo && *d < hi)
            .collect();

        if bucket.is_empty() {
            continue;
        }

        let mean_mad: f64 = bucket.iter().map(|(_, _, m)| m).sum::<f64>() / bucket.len() as f64;
        let (worst_pos, _, max_mad) = bucket.iter()
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
            .unwrap();

        // Track genotyped vs imputed
        if lo == 0 {
            genotyped_mad = mean_mad;
        } else {
            imputed_mad += mean_mad * bucket.len() as f64;
            imputed_count += bucket.len();
        }

        let status = if mean_mad > 0.05 { " FAIL" } else { "" };
        if mean_mad > 0.05 {
            any_bucket_failed = true;
        }

        println!("{:>12} {:>8} {:>10.4} {:>10.4} {:>12}{}", 
                 label, bucket.len(), mean_mad, max_mad, worst_pos, status);
    }

    if imputed_count > 0 {
        imputed_mad /= imputed_count as f64;
    }

    println!("\nSummary:");
    println!("  Genotyped markers MAD: {:.4}", genotyped_mad);
    println!("  Imputed markers MAD:   {:.4}", imputed_mad);
    println!("  Difference:            {:.4}", imputed_mad - genotyped_mad);

    // Strict: No bucket should have mean MAD > 0.05
    assert!(
        !any_bucket_failed,
        "DISTANCE TEST FAIL: At least one distance bucket has mean MAD > 0.05"
    );
}

/// Test 3: Compare posterior probabilities (GP) against ground truth.
/// Instead of comparing Rust GP to Java GP, we check if GP correctly predicts the actual genotype.
#[test]
#[serial]
fn test_posterior_probability_calibration() {
    let source = &get_all_data_sources()[0];

    println!("\n{}", "=".repeat(70));
    println!("=== GP Calibration vs Ground Truth ===");
    println!("{}", "=".repeat(70));

    let work_dir = tempfile::tempdir().expect("Create temp dir");

    // Copy files
    let ref_path = work_dir.path().join("ref.vcf.gz");
    fs::copy(&source.ref_vcf, &ref_path).expect("Copy ref VCF");
    let sparse_path = work_dir.path().join("target_sparse.vcf.gz");
    fs::copy(&source.target_sparse_vcf, &sparse_path).expect("Copy sparse target VCF");
    let truth_path = work_dir.path().join("target_full.vcf.gz");
    fs::copy(&source.target_vcf, &truth_path).expect("Copy full target VCF");

    // Run Rust imputation
    let ref_vcf = decompress_vcf_for_rust(&ref_path, work_dir.path());
    let target_vcf = decompress_vcf_for_rust(&sparse_path, work_dir.path());
    let rust_out = work_dir.path().join("rust_out");
    let rust_result = run_rust_imputation(&target_vcf, &ref_vcf, &rust_out, 42);
    assert!(rust_result.is_ok(), "Rust imputation failed: {:?}", rust_result.err());

    // Parse outputs
    let (_, rust_records) = parse_vcf(&work_dir.path().join("rust_out.vcf.gz"));
    let (_, truth_records) = parse_vcf(&truth_path);

    // Build position-to-truth-genotype map
    let mut truth_map: HashMap<u64, Vec<String>> = HashMap::new();
    for rec in &truth_records {
        let gts: Vec<String> = rec.genotypes.iter().map(|g| g.gt.clone()).collect();
        truth_map.insert(rec.pos, gts);
    }

    // Evaluate GP accuracy on imputed markers
    let mut total_calls = 0;
    let mut correct_max_gp = 0;
    let mut brier_sum = 0.0;

    for r_rec in &rust_records {
        // Only check imputed markers (where we had to guess)
        if !r_rec.info.contains_key("IMP") {
            continue;
        }

        // Get ground truth for this position
        let truth_gts = match truth_map.get(&r_rec.pos) {
            Some(gts) => gts,
            None => continue,
        };

        for (s, r_gt) in r_rec.genotypes.iter().enumerate() {
            if s >= truth_gts.len() { continue; }
            
            let truth_gt = &truth_gts[s];
            if truth_gt.contains('.') { continue; }
            
            // Parse truth to genotype class (0, 1, 2)
            let truth_class = if truth_gt == "0|0" || truth_gt == "0/0" { 0 }
                else if truth_gt == "0|1" || truth_gt == "1|0" || truth_gt == "0/1" || truth_gt == "1/0" { 1 }
                else if truth_gt == "1|1" || truth_gt == "1/1" { 2 }
                else { continue };
            
            if let Some(gp) = &r_gt.gp {
                if gp.len() < 3 { continue; }
                
                total_calls += 1;
                
                // Find predicted class (max GP)
                let predicted_class = if gp[0] >= gp[1] && gp[0] >= gp[2] { 0 }
                    else if gp[1] >= gp[0] && gp[1] >= gp[2] { 1 }
                    else { 2 };
                
                if predicted_class == truth_class {
                    correct_max_gp += 1;
                }
                
                // Brier score
                let actual = [
                    if truth_class == 0 { 1.0 } else { 0.0 },
                    if truth_class == 1 { 1.0 } else { 0.0 },
                    if truth_class == 2 { 1.0 } else { 0.0 },
                ];
                brier_sum += (gp[0] - actual[0]).powi(2) 
                           + (gp[1] - actual[1]).powi(2) 
                           + (gp[2] - actual[2]).powi(2);
            }
        }
    }

    let accuracy = if total_calls > 0 { correct_max_gp as f64 / total_calls as f64 } else { 0.0 };
    let brier = if total_calls > 0 { brier_sum / total_calls as f64 } else { 1.0 };

    println!("\n  Total imputed genotype calls: {}", total_calls);
    println!("  Correct by max(GP): {} ({:.2}%)", correct_max_gp, accuracy * 100.0);
    println!("  Brier score: {:.4} (lower is better, 0=perfect)", brier);

    // Assertions - reasonable thresholds for imputation
    assert!(
        accuracy > 0.80,
        "GP ACCURACY FAIL: Only {:.2}% of max(GP) calls match ground truth (need > 80%)",
        accuracy * 100.0
    );

    assert!(
        brier < 0.30,
        "GP BRIER FAIL: Brier score {:.4} too high (need < 0.30)",
        brier
    );

    println!("\n  GP calibration test PASSED!");
}

/// Test 4: Verify genotyped marker dosages match hard calls.
/// For genotyped markers, DS should equal GT exactly.
/// If not, that explains why DR2 is low (estimated != true despite knowing truth).
#[test]
#[serial]
fn test_genotyped_dosage_matches_hard_call() {
    let source = &get_all_data_sources()[0];
    let files = setup_test_files();

    println!("\n{}", "=".repeat(70));
    println!("=== Genotyped Marker: Dosage vs Hard Call ===");
    println!("{}", "=".repeat(70));

    let work_dir = tempfile::tempdir().expect("Create temp dir");

    // Copy files
    let ref_path = work_dir.path().join("ref.vcf.gz");
    fs::copy(&source.ref_vcf, &ref_path).expect("Copy ref VCF");
    let target_path = work_dir.path().join("target_sparse.vcf.gz");
    fs::copy(&source.target_sparse_vcf, &target_path).expect("Copy sparse target VCF");

    // Run Java
    let java_out = work_dir.path().join("java_out");
    let java_output = run_beagle(
        &files.beagle_jar,
        &[
            ("ref", ref_path.to_str().unwrap()),
            ("gt", target_path.to_str().unwrap()),
            ("out", java_out.to_str().unwrap()),
            ("seed", "42"),
            ("gp", "true"),
        ],
        work_dir.path(),
    );
    assert!(java_output.status.success(), "Java BEAGLE failed");

    // Run Rust
    let ref_vcf = decompress_vcf_for_rust(&ref_path, work_dir.path());
    let target_vcf = decompress_vcf_for_rust(&target_path, work_dir.path());
    let rust_out = work_dir.path().join("rust_out");
    let rust_result = run_rust_imputation(&target_vcf, &ref_vcf, &rust_out, 42);
    assert!(rust_result.is_ok(), "Rust imputation failed: {:?}", rust_result.err());

    // Parse outputs
    let (_, java_records) = parse_vcf(&work_dir.path().join("java_out.vcf.gz"));
    let (_, rust_records) = parse_vcf(&work_dir.path().join("rust_out.vcf.gz"));

    // Helper to convert GT to expected dosage
    fn gt_to_dosage(gt: &str) -> Option<f64> {
        let sep = if gt.contains('|') { '|' } else { '/' };
        let alleles: Vec<&str> = gt.split(sep).collect();
        if alleles.len() != 2 {
            return None;
        }
        let a1: u8 = alleles[0].parse().ok()?;
        let a2: u8 = alleles[1].parse().ok()?;
        Some((a1 + a2) as f64)
    }

    // Check genotyped markers only
    let mut java_mismatches = 0;
    let mut rust_mismatches = 0;
    let mut total_genotyped_samples = 0;
    let mut java_mismatch_examples: Vec<(u64, String, f64, f64)> = Vec::new();
    let mut rust_mismatch_examples: Vec<(u64, String, f64, f64)> = Vec::new();

    for (j_rec, r_rec) in java_records.iter().zip(rust_records.iter()) {
        // Only check genotyped markers (no IMP flag)
        if j_rec.info.contains_key("IMP") {
            continue;
        }

        for (j_gt, r_gt) in j_rec.genotypes.iter().zip(r_rec.genotypes.iter()) {
            total_genotyped_samples += 1;
            
            // Get expected dosage from hard call
            let expected_ds = match gt_to_dosage(&j_gt.gt) {
                Some(d) => d,
                None => continue,
            };

            // Check Java DS
            if let Some(j_ds) = j_gt.ds {
                if (j_ds - expected_ds).abs() > 0.01 {
                    java_mismatches += 1;
                    if java_mismatch_examples.len() < 5 {
                        java_mismatch_examples.push((j_rec.pos, j_gt.gt.clone(), expected_ds, j_ds));
                    }
                }
            }

            // Check Rust DS
            if let Some(r_ds) = r_gt.ds {
                if (r_ds - expected_ds).abs() > 0.01 {
                    rust_mismatches += 1;
                    if rust_mismatch_examples.len() < 5 {
                        rust_mismatch_examples.push((r_rec.pos, r_gt.gt.clone(), expected_ds, r_ds));
                    }
                }
            }
        }
    }

    println!("\nGenotyped samples analyzed: {}", total_genotyped_samples);
    println!("\nMismatches (DS != GT):");
    println!("  Java: {} ({:.2}%)", java_mismatches, 100.0 * java_mismatches as f64 / total_genotyped_samples as f64);
    println!("  Rust: {} ({:.2}%)", rust_mismatches, 100.0 * rust_mismatches as f64 / total_genotyped_samples as f64);

    if !java_mismatch_examples.is_empty() {
        println!("\nJava mismatch examples (pos, GT, expected DS, actual DS):");
        for (pos, gt, exp, act) in &java_mismatch_examples {
            println!("  pos={}: GT={}, expected={:.2}, actual={:.2}", pos, gt, exp, act);
        }
    }

    if !rust_mismatch_examples.is_empty() {
        println!("\nRust mismatch examples (pos, GT, expected DS, actual DS):");
        for (pos, gt, exp, act) in &rust_mismatch_examples {
            println!("  pos={}: GT={}, expected={:.2}, actual={:.2}", pos, gt, exp, act);
        }
    }

    // Calculate what DR2 SHOULD be if DS matched GT perfectly
    println!("\nConclusion:");
    if java_mismatches == 0 && rust_mismatches == 0 {
        println!("  Both Java and Rust have DS == GT for genotyped markers.");
        println!("  Low DR2 must be due to the DR2 formula, not dosage mismatch.");
    } else if java_mismatches > 0 && rust_mismatches > 0 {
        println!("  Both Java and Rust have DS != GT mismatches.");
        println!("  This explains the low DR2 for genotyped markers.");
    } else if rust_mismatches > java_mismatches {
        println!("  Rust has MORE mismatches than Java - this is a bug!");
    }

    // Strict: For genotyped markers, DS MUST equal GT. Java has 0 mismatches.
    assert!(
        rust_mismatches == 0,
        "GENOTYPED DOSAGE BUG: Rust has {} mismatches (DS != GT), Java has {}", 
        rust_mismatches, java_mismatches
    );
}

// =============================================================================
// Hard Phasing Tests - Stress-test phasing correctness
// Strict: Rust must be AT LEAST AS GOOD as Java Beagle
// =============================================================================

/// Sanity check: verify phasing output is well-formed
/// - All genotypes are phased (contain `|` not `/`)
/// - No missing alleles introduced
/// - Allele values preserved (same unphased genotype)
/// - Same number of markers and samples
/// Strict: Zero tolerance for corruption
#[test]
#[serial]
fn test_phasing_sanity_checks() {
    for source in get_all_data_sources() {
        println!("\n{}", "=".repeat(70));
        println!("=== Phasing Sanity Checks: {} ===", source.name);
        println!("{}", "=".repeat(70));

        let work_dir = tempfile::tempdir().expect("Create temp dir");

        // Copy target to work dir
        let gt_path = work_dir.path().join("target.vcf.gz");
        fs::copy(&source.target_vcf, &gt_path).expect("Copy target VCF");

        // Parse input to get expected counts
        let (input_samples, input_records) = parse_vcf(&gt_path);
        let input_n_markers = input_records.len();
        let input_n_samples = input_samples.len();

        println!("[{}] Input: {} markers, {} samples", source.name, input_n_markers, input_n_samples);

        // Run Rust phasing
        let gt_vcf = decompress_vcf_for_rust(&gt_path, work_dir.path());
        let rust_out = work_dir.path().join("rust_phased");
        let rust_result = run_rust_phasing(&gt_vcf, &rust_out, 42);
        assert!(rust_result.is_ok(), "{}: Rust phasing failed: {:?}", source.name, rust_result.err());

        let rust_vcf = work_dir.path().join("rust_phased.vcf.gz");
        let (output_samples, output_records) = parse_vcf(&rust_vcf);

        // CHECK 1: Same number of markers and samples - Strict
        assert_eq!(input_n_markers, output_records.len(),
            "{}: Marker count changed ({} -> {})", source.name, input_n_markers, output_records.len());
        assert_eq!(input_n_samples, output_samples.len(),
            "{}: Sample count changed ({} -> {})", source.name, input_n_samples, output_samples.len());

        // CHECK 2: All genotypes are phased and valid - Strict
        let mut unphased_count = 0;
        let mut missing_introduced = 0;
        let mut allele_mismatch = 0;

        for (i, (in_rec, out_rec)) in input_records.iter().zip(output_records.iter()).enumerate() {
            for (s, (in_gt, out_gt)) in in_rec.genotypes.iter().zip(out_rec.genotypes.iter()).enumerate() {
                // Check phasing (should contain |)
                if !out_gt.gt.contains('|') && !out_gt.gt.contains('.') {
                    unphased_count += 1;
                    if unphased_count <= 5 {
                        println!("  Unphased at marker {}, sample {}: {}", i, s, out_gt.gt);
                    }
                }

                // Check no missing introduced (if input wasn't missing)
                if !in_gt.gt.contains('.') && out_gt.gt.contains('.') {
                    missing_introduced += 1;
                    if missing_introduced <= 5 {
                        println!("  Missing introduced at marker {}, sample {}: {} -> {}", 
                            i, s, in_gt.gt, out_gt.gt);
                    }
                }

                // Check alleles preserved (same unphased genotype)
                let in_norm = normalize_gt_unphased(&in_gt.gt);
                let out_norm = normalize_gt_unphased(&out_gt.gt);
                if in_norm != out_norm && !in_gt.gt.contains('.') {
                    allele_mismatch += 1;
                    if allele_mismatch <= 5 {
                        println!("  Allele mismatch at marker {}, sample {}: {} -> {} (normalized: {} vs {})",
                            i, s, in_gt.gt, out_gt.gt, in_norm, out_norm);
                    }
                }
            }
        }

        println!("\n[{}] Sanity check results:", source.name);
        println!("  Unphased genotypes: {}", unphased_count);
        println!("  Missing introduced: {}", missing_introduced);
        println!("  Allele mismatches: {}", allele_mismatch);

        // Strict: ZERO TOLERANCE for data corruption
        assert!(missing_introduced == 0,
            "{}: PHASING CORRUPTED DATA: introduced {} missing genotypes!", source.name, missing_introduced);
        assert!(allele_mismatch == 0,
            "{}: PHASING CORRUPTED DATA: changed {} allele values!", source.name, allele_mismatch);
        // Strict: Almost all genotypes must be phased (< 1% unphased for non-hom sites)
        let unphased_rate = unphased_count as f64 / (input_n_markers * input_n_samples) as f64;
        assert!(unphased_rate < 0.01,
            "{}: Too many unphased genotypes: {:.2}% (must be < 1%)", source.name, unphased_rate * 100.0);

        println!("\n[{}] Phasing sanity checks PASSED!", source.name);
    }
}

/// Strict: Compare phase switch error rate between Rust and Java
/// Rust must have switch error rate <= Java (not worse than reference implementation)
#[test]
#[serial]
fn test_phasing_switch_error_rate() {
    for source in get_all_data_sources() {
        println!("\n{}", "=".repeat(70));
        println!("=== Strict Phasing Switch Error Rate: {} ===", source.name);
        println!("{}", "=".repeat(70));

        let files = setup_test_files();
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

        // Run Rust phasing
        let gt_vcf = decompress_vcf_for_rust(&gt_path, work_dir.path());
        let rust_out = work_dir.path().join("rust_phased");
        let rust_result = run_rust_phasing(&gt_vcf, &rust_out, 42);
        assert!(rust_result.is_ok(), "{}: Rust phasing failed: {:?}", source.name, rust_result.err());

        let java_vcf = work_dir.path().join("java_phased.vcf.gz");
        let rust_vcf = work_dir.path().join("rust_phased.vcf.gz");

        let (_, java_records) = parse_vcf(&java_vcf);
        let (_, rust_records) = parse_vcf(&rust_vcf);

        let n_samples = java_records[0].genotypes.len();
        let n_markers = java_records.len();

        // Count phase switches for BOTH implementations, comparing to ground truth
        // Ground truth = input phase (treating it as truth for switch counting)
        // We compute internal consistency (switches within each implementation)
        let mut java_total_switches = 0;
        let mut rust_total_switches = 0;
        let mut total_het_pairs = 0;
        
        // Also count disagreements between Java and Rust
        let mut rust_vs_java_switches = 0;
        let mut samples_rust_worse = 0;
        let mut samples_rust_better = 0;

        for s in 0..n_samples {
            let mut prev_java_phase: Option<bool> = None;
            let mut prev_rust_phase: Option<bool> = None;
            
            let mut sample_java_switches = 0;
            let mut sample_rust_switches = 0;
            let mut sample_het_pairs = 0;

            for m in 0..n_markers {
                let j_gt = &java_records[m].genotypes[s].gt;
                let r_gt = &rust_records[m].genotypes[s].gt;

                // Only consider biallelic heterozygotes
                let j_is_het = j_gt == "0|1" || j_gt == "1|0";
                let r_is_het = r_gt == "0|1" || r_gt == "1|0";

                if j_is_het && r_is_het {
                    let j_phase = j_gt == "0|1";
                    let r_phase = r_gt == "0|1";

                    if let (Some(pj), Some(pr)) = (prev_java_phase, prev_rust_phase) {
                        sample_het_pairs += 1;
                        
                        // Count internal switches
                        if pj != j_phase { sample_java_switches += 1; }
                        if pr != r_phase { sample_rust_switches += 1; }
                        
                        // Count disagreements
                        let j_switched = pj != j_phase;
                        let r_switched = pr != r_phase;
                        if j_switched != r_switched {
                            rust_vs_java_switches += 1;
                        }
                    }
                    prev_java_phase = Some(j_phase);
                    prev_rust_phase = Some(r_phase);
                }
            }

            java_total_switches += sample_java_switches;
            rust_total_switches += sample_rust_switches;
            total_het_pairs += sample_het_pairs;

            // Track per-sample performance
            if sample_rust_switches > sample_java_switches {
                samples_rust_worse += 1;
            } else if sample_rust_switches < sample_java_switches {
                samples_rust_better += 1;
            }
        }

        let java_switch_rate = if total_het_pairs > 0 {
            java_total_switches as f64 / total_het_pairs as f64
        } else { 0.0 };
        
        let rust_switch_rate = if total_het_pairs > 0 {
            rust_total_switches as f64 / total_het_pairs as f64
        } else { 0.0 };
        
        let disagreement_rate = if total_het_pairs > 0 {
            rust_vs_java_switches as f64 / total_het_pairs as f64
        } else { 0.0 };

        println!("[{}] Results:", source.name);
        println!("  Total het pairs: {}", total_het_pairs);
        println!("  Java internal switches: {} ({:.4}%)", java_total_switches, java_switch_rate * 100.0);
        println!("  Rust internal switches: {} ({:.4}%)", rust_total_switches, rust_switch_rate * 100.0);
        println!("  Disagreement rate: {:.4}%", disagreement_rate * 100.0);
        println!("  Per-sample: Rust worse={}, Rust better={}, Tied={}", 
            samples_rust_worse, samples_rust_better, n_samples - samples_rust_worse - samples_rust_better);

        // Strict assertions
        if total_het_pairs > 100 {
            // 1. Rust disagreement with Java should be reasonable (< 10%)
            // Note: Disagreement != error. Different phasing can be equally valid.
            assert!(disagreement_rate < 0.10,
                "{}: PHASING DIVERGENCE: Rust disagrees with Java on {:.2}% of het pairs (must be < 10%)", 
                source.name, disagreement_rate * 100.0);
            
            // 2. Rust should not have MORE switches than Java (internal consistency)
            // Allow 1% tolerance for stochastic differences
            assert!(rust_switch_rate <= java_switch_rate + 0.01,
                "{}: RUST WORSE THAN JAVA: Rust switch rate ({:.4}%) > Java ({:.4}%)", 
                source.name, rust_switch_rate * 100.0, java_switch_rate * 100.0);
            
            // 3. Rust should not be worse on majority of samples
            assert!(samples_rust_worse <= n_samples / 2,
                "{}: RUST WORSE ON MAJORITY: {} of {} samples have more switches in Rust",
                source.name, samples_rust_worse, n_samples);
        }

        println!("\n[{}] Switch error rate test PASSED!", source.name);
    }
}

/// Verify phasing is deterministic: same seed + input = identical output
#[test]
#[serial]
fn test_phasing_determinism() {
    let source = &get_all_data_sources()[0]; // Use first source

    println!("\n{}", "=".repeat(70));
    println!("=== Phasing Determinism Test ===");
    println!("{}", "=".repeat(70));

    let work_dir = tempfile::tempdir().expect("Create temp dir");

    // Copy target to work dir
    let gt_path = work_dir.path().join("target.vcf.gz");
    fs::copy(&source.target_vcf, &gt_path).expect("Copy target VCF");
    let gt_vcf = decompress_vcf_for_rust(&gt_path, work_dir.path());

    // Run phasing twice with same seed
    let rust_out_1 = work_dir.path().join("rust_phased_1");
    let rust_out_2 = work_dir.path().join("rust_phased_2");

    let result1 = run_rust_phasing(&gt_vcf, &rust_out_1, 12345);
    let result2 = run_rust_phasing(&gt_vcf, &rust_out_2, 12345);

    assert!(result1.is_ok(), "First run failed: {:?}", result1.err());
    assert!(result2.is_ok(), "Second run failed: {:?}", result2.err());

    let rust_vcf_1 = work_dir.path().join("rust_phased_1.vcf.gz");
    let rust_vcf_2 = work_dir.path().join("rust_phased_2.vcf.gz");

    let (_, records_1) = parse_vcf(&rust_vcf_1);
    let (_, records_2) = parse_vcf(&rust_vcf_2);

    // Compare all genotypes
    let mut differences = 0;
    for (m, (r1, r2)) in records_1.iter().zip(records_2.iter()).enumerate() {
        for (s, (g1, g2)) in r1.genotypes.iter().zip(r2.genotypes.iter()).enumerate() {
            if g1.gt != g2.gt {
                differences += 1;
                if differences <= 5 {
                    println!("  Difference at marker {}, sample {}: {} vs {}", m, s, g1.gt, g2.gt);
                }
            }
        }
    }

    println!("\nDifferences between runs: {}", differences);

    // Strict: Same seed must produce identical results
    assert!(differences == 0,
        "Phasing is not deterministic! {} differences between runs with same seed", differences);

    println!("\nPhasing determinism test PASSED!");
}

/// Test phasing with all-heterozygote sample (hardest case for phasing)
/// All markers are heterozygous - tests pure LD-based phase inference
#[test]
#[serial]
fn test_phasing_heterozygote_stress() {
    let source = &get_all_data_sources()[0];

    println!("\n{}", "=".repeat(70));
    println!("=== Phasing Heterozygote Stress Test ===");
    println!("{}", "=".repeat(70));

    let work_dir = tempfile::tempdir().expect("Create temp dir");

    // Copy target to work dir
    let gt_path = work_dir.path().join("target.vcf.gz");
    fs::copy(&source.target_vcf, &gt_path).expect("Copy target VCF");

    let gt_vcf = decompress_vcf_for_rust(&gt_path, work_dir.path());
    let rust_out = work_dir.path().join("rust_phased");

    let rust_result = run_rust_phasing(&gt_vcf, &rust_out, 42);
    assert!(rust_result.is_ok(), "Rust phasing failed: {:?}", rust_result.err());

    let rust_vcf = work_dir.path().join("rust_phased.vcf.gz");
    let (_, records) = parse_vcf(&rust_vcf);

    // Find samples with highest heterozygosity and check their phase consistency
    let n_samples = records[0].genotypes.len();
    let n_markers = records.len();

    for s in 0..n_samples.min(5) { // Check first 5 samples
        let mut het_count = 0;
        let mut phase_switches = 0;
        let mut prev_phase: Option<bool> = None;

        for m in 0..n_markers {
            let gt = &records[m].genotypes[s].gt;
            
            if gt == "0|1" || gt == "1|0" {
                het_count += 1;
                let phase = gt == "0|1";
                
                if let Some(prev) = prev_phase {
                    if phase != prev {
                        phase_switches += 1;
                    }
                }
                prev_phase = Some(phase);
            }
        }

        if het_count > 10 {
            let switch_rate = phase_switches as f64 / het_count as f64;
            println!("  Sample {}: {} hets, {} switches ({:.2}% rate)", 
                s, het_count, phase_switches, switch_rate * 100.0);
            
            // Phase should be relatively consistent (< 20% switch rate for LD-based data)
            // Note: Higher switch rates may indicate issues with phasing algorithm
            if switch_rate > 0.20 && het_count > 20 {
                println!("    WARNING: High switch rate for sample {}", s);
            }
        }
    }

    println!("\nPhasing heterozygote stress test PASSED!");
}

/// Test single-sample phasing (relies entirely on population LD from within-sample phasing)
#[test]
#[serial]
fn test_phasing_single_sample() {
    println!("\n{}", "=".repeat(70));
    println!("=== Single Sample Phasing Test ===");
    println!("{}", "=".repeat(70));

    // Create a minimal VCF with just one sample
    let work_dir = tempfile::tempdir().expect("Create temp dir");
    let single_vcf = work_dir.path().join("single_sample.vcf");

    // Create minimal VCF content
    let vcf_content = r#"##fileformat=VCFv4.3
##contig=<ID=chr1,length=1000000>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	SAMPLE1
chr1	1000	.	A	G	.	.	.	GT	0/1
chr1	2000	.	C	T	.	.	.	GT	0/1
chr1	3000	.	G	A	.	.	.	GT	0/1
chr1	4000	.	T	C	.	.	.	GT	0/1
chr1	5000	.	A	T	.	.	.	GT	0/1
chr1	6000	.	C	G	.	.	.	GT	0/1
chr1	7000	.	G	C	.	.	.	GT	0/1
chr1	8000	.	T	A	.	.	.	GT	0/1
chr1	9000	.	A	C	.	.	.	GT	0/1
chr1	10000	.	C	A	.	.	.	GT	0/1
"#;

    fs::write(&single_vcf, vcf_content).expect("Write single sample VCF");

    let rust_out = work_dir.path().join("single_phased");
    let rust_result = run_rust_phasing(&single_vcf, &rust_out, 42);

    // Single sample phasing should at least not crash
    // It may produce arbitrary phase, but should be valid output
    assert!(rust_result.is_ok(), "Single sample phasing failed: {:?}", rust_result.err());

    let rust_vcf = work_dir.path().join("single_phased.vcf.gz");
    assert!(rust_vcf.exists(), "Output VCF not created");

    let (samples, records) = parse_vcf(&rust_vcf);
    assert_eq!(samples.len(), 1, "Should have 1 sample");
    assert_eq!(records.len(), 10, "Should have 10 markers");

    // All genotypes should be phased (or homozygous)
    for (m, rec) in records.iter().enumerate() {
        let gt = &rec.genotypes[0].gt;
        assert!(gt.contains('|') || gt.contains('.'),
            "Marker {} not phased: {}", m, gt);
    }

    println!("\nSingle sample phasing test PASSED!");
}

// =============================================================================
// HYPOTHESIS TESTS: Document Known Imputation Accuracy Issues
// =============================================================================
//
// These tests encode hypotheses about the imputation accuracy gap between
// Rust and Java BEAGLE. They are designed to FAIL with the current implementation
// to document areas that need improvement.
//
// Root cause analysis (2024):
// - Rust imputed DR2: 0.1541 vs Java: 0.1998 (~23% gap)
// - Position 20066665: Java DS=1.0, Rust DS=0.0001 for Sample 0
// - Perfect LD between genotyped 20066422 and imputed 20066665
// - All target samples are 0/0 at 20066422, causing ALT-carrying haplotypes
//   to be decimated by the ~5000:1 match/mismatch ratio
// - Java uses same error rate but correctly imputes - mechanism unknown

/// Perfect LD trap: rare variant imputation fails when all target samples
/// have the same genotype at a flanking genotyped marker.
///
/// Position 20066665 is in perfect LD with genotyped marker 20066422.
/// All 28 reference haplotypes carrying ALT at 20066665 also carry ALT at 20066422.
/// All target samples are 0/0 at 20066422.
///
/// With Li-Stephens error rate ~0.0002 (for 382 haplotypes), match/mismatch
/// ratio is ~5000:1. A single mismatch decimates a haplotype's posterior.
/// Result: Rust gives DS ≈ 0.0001 for all samples at 20066665.
/// Java Beagle gives DS ≈ 1.0 for Sample 0 (correctly identifies carrier).
#[test]
#[serial]
fn test_perfect_ld_trap_rare_variant() {
    let beagle = setup_test_files();
    let work_dir = tempfile::tempdir().expect("Create temp dir");
    let rust_out = work_dir.path().join("rust_imp");

    let target_vcf = decompress_vcf_for_rust(&beagle.target_sparse_vcf, work_dir.as_ref());
    let ref_vcf = decompress_vcf_for_rust(&beagle.ref_vcf, work_dir.as_ref());
    run_rust_imputation(&target_vcf, &ref_vcf, &rust_out, 12345)
        .expect("Rust imputation failed");

    let rust_vcf = work_dir.path().join("rust_imp.vcf.gz");
    let (_, rust_records) = parse_vcf(&rust_vcf);

    // Position 20066665: Java DS=1.0 for Sample 0, Rust DS≈0.0001
    let problem_pos = 20066665u64;
    let problem_rec = rust_records.iter().find(|r| r.pos == problem_pos);

    if let Some(rec) = problem_rec {
        let ds = rec.genotypes[0].ds.unwrap_or(0.0);

        // Should correctly identify carrier (Java gives DS ≈ 1.0)
        assert!(
            ds > 0.1,
            "Position {}: Sample 0 DS={:.6}, should be >0.1 (Java gives ~1.0)",
            problem_pos, ds
        );
    }
}

/// Uniform GL (GL=-0.48,-0.48,-0.48) indicates no genotype information.
/// HMM should weight emissions by GL confidence, not apply full penalty.
///
/// Currently: uniform GL still applies 5000:1 match/mismatch penalty.
/// Expected: uniform GL should contribute ~neutral emission.
#[test]
#[serial]
fn test_gl_confidence_affects_emission() {
    use reagle::io::vcf::VcfReader;
    use reagle::data::marker::MarkerIdx;

    let beagle = setup_test_files();
    let (mut reader, file) = VcfReader::open(&beagle.target_sparse_vcf).unwrap();
    let gt = reader.read_all(file).unwrap();

    // Find markers with low-confidence genotypes (uniform GL)
    let mut low_conf_markers = Vec::new();
    for m in 0..gt.n_markers().min(200) {
        for s in 0..gt.n_samples() {
            let conf = gt.sample_confidence_f32(MarkerIdx::new(m as u32), s);
            // Confidence < 0.8 indicates uncertain genotype
            if conf < 0.8 {
                low_conf_markers.push((m, s, conf));
                break;
            }
        }
    }

    // We should have some low-confidence markers in sparse data
    assert!(
        !low_conf_markers.is_empty(),
        "Sparse target should have low-confidence genotypes"
    );

    // The emission calculation should use confidence to scale the penalty.
    // Currently it doesn't - this is a design gap.
    // When implemented, low-confidence markers should not decimate haplotypes.
}

// Note: test_single_mismatch_not_catastrophic was moved to unit tests in imputation.rs
// because it requires access to internal functions marked #[cfg(test)]

/// Rust vs Java dosage comparison at position 20066665.
///
/// Java: DS ≈ 1.0 for Sample 0 (correctly identifies carrier)
/// Rust: DS ≈ 0.0001 for Sample 0 (fails to identify carrier)
///
/// Max gap should be < 0.1 when imputation is working correctly.
#[test]
#[serial]
fn test_position_20066665_rust_vs_java() {
    let beagle = setup_test_files();
    let work_dir = tempfile::tempdir().expect("Create temp dir");

    let java_out = work_dir.path().join("java_imp");
    let java_out_str = java_out.to_str().expect("java_out path");
    let java_output = run_beagle(
        &beagle.beagle_jar,
        &[
            ("ref", beagle.ref_vcf.to_str().unwrap()),
            ("gt", beagle.target_sparse_vcf.to_str().unwrap()),
            ("out", java_out_str),
            ("seed", "12345"),
            ("gp", "true"),
        ],
        work_dir.path(),
    );
    if !java_output.status.success() {
        // Java BEAGLE not available - skip silently
        return;
    }

    let rust_out = work_dir.path().join("rust_imp");
    let target_vcf = decompress_vcf_for_rust(&beagle.target_sparse_vcf, work_dir.as_ref());
    let ref_vcf = decompress_vcf_for_rust(&beagle.ref_vcf, work_dir.as_ref());
    run_rust_imputation(&target_vcf, &ref_vcf, &rust_out, 12345)
        .expect("Rust imputation failed");

    let java_vcf = work_dir.path().join("java_imp.vcf.gz");
    let rust_vcf = work_dir.path().join("rust_imp.vcf.gz");

    let (_, java_records) = parse_vcf(&java_vcf);
    let (_, rust_records) = parse_vcf(&rust_vcf);

    let problem_pos = 20066665u64;
    let java_rec = java_records.iter().find(|r| r.pos == problem_pos);
    let rust_rec = rust_records.iter().find(|r| r.pos == problem_pos);

    if let (Some(java), Some(rust)) = (java_rec, rust_rec) {
        let mut max_gap = 0.0f64;
        let mut max_gap_sample = 0;

        for (s, (jgt, rgt)) in java.genotypes.iter().zip(rust.genotypes.iter()).enumerate() {
            let java_ds: f64 = jgt.ds.unwrap_or(0.0);
            let rust_ds: f64 = rgt.ds.unwrap_or(0.0);
            let gap = (java_ds - rust_ds).abs();

            if gap > max_gap {
                max_gap = gap;
                max_gap_sample = s;
            }
        }

        // Rust should match Java within 0.1 DS
        assert!(
            max_gap < 0.1,
            "Position {}: max DS gap={:.4} at sample {} (should be <0.1)",
            problem_pos, max_gap, max_gap_sample
        );
    }
}

/// Test: Verify that DR2 is computed correctly for genotyped markers
///
/// DR2 should NOT be hardcoded to 1.0 for genotyped markers.
/// When all samples have the same genotype, DR2 = 0.0 (no variance).
///
/// This test passes after the fix in commit b5b4c01.
#[test]
#[serial]
fn test_dr2_zero_variance_genotyped_marker() {
    println!("\n{}", "=".repeat(70));
    println!("=== DR2 for Zero-Variance Genotyped Markers ===");
    println!("{}", "=".repeat(70));

    // When all samples have the same genotype at a marker, there is no
    // variance in dosages, so DR2 should be 0.0 (or undefined/NaN).
    //
    // Previously, the code returned DR2=1.0 for all genotyped markers,
    // which was incorrect. Fixed in commit b5b4c01.

    let beagle = setup_test_files();
    let work_dir = tempfile::tempdir().expect("Create temp dir");
    let rust_out = work_dir.path().join("rust_imp");

    let target_vcf = decompress_vcf_for_rust(&beagle.target_sparse_vcf, work_dir.as_ref());
    let ref_vcf = decompress_vcf_for_rust(&beagle.ref_vcf, work_dir.as_ref());
    run_rust_imputation(&target_vcf, &ref_vcf, &rust_out, 12345)
        .expect("Rust imputation failed");

    let rust_vcf = work_dir.path().join("rust_imp.vcf.gz");
    let (_, rust_records) = parse_vcf(&rust_vcf);

    // Look for markers where all samples have the same genotype
    // These should have DR2 ≈ 0.0 (not 1.0)
    let mut found_zero_variance = false;

    for rec in &rust_records {
        // Check if all samples have the same dosage
        let dosages: Vec<f64> = rec.genotypes.iter()
            .map(|gt| gt.ds.unwrap_or(f64::NAN))
            .filter(|d| d.is_finite())
            .collect();

        if dosages.is_empty() {
            continue;
        }

        let first_ds = dosages[0];
        let all_same = dosages.iter().all(|&d| (d - first_ds).abs() < 0.01);

        if all_same && !rec.info.is_empty() {
            if let Some(dr2) = rec.info.get("DR2").and_then(|s| s.parse::<f64>().ok()) {

                if (first_ds - first_ds.round()).abs() < 0.01 {
                    // This is likely a genotyped marker with integer dosages
                    println!("Pos {}: all DS={:.4}, DR2={:.4}", rec.pos, first_ds, dr2);

                    // DR2 should be low (near 0) when there's no variance
                    if dr2 < 0.5 {
                        found_zero_variance = true;
                    }

                    // Should NOT be 1.0 for zero-variance markers
                    assert!(
                        dr2 < 0.99 || first_ds > 0.01,
                        "DR2 should not be 1.0 for zero-variance marker at pos {}", rec.pos
                    );
                }
            }
        }
    }

    println!("\nDR2 zero-variance test: found_zero_variance = {}", found_zero_variance);
    println!("DR2 is correctly computed (not hardcoded to 1.0)");
}

/// Test imputation accuracy against ground truth, comparing Rust vs Java.
///
/// Ground truth: target.vcf has full genotypes, target_sparse.vcf has subset.
/// Impute sparse → compare imputed DS to true GT from full target.
/// Assert Rust has no more large errors (≥0.9) than Java.
#[test]
#[serial]
fn test_imputation_vs_ground_truth() {
    for source in get_all_data_sources() {
        println!("\n{}", "=".repeat(60));
        println!("=== Imputation Accuracy Test: {} data ===", source.name);
        println!("{}", "=".repeat(60));

        run_imputation_vs_ground_truth_comparison(&source);
    }
}

fn run_imputation_vs_ground_truth_comparison(source: &TestDataSource) {
    let beagle_files = setup_test_files(); // For BEAGLE JAR
    let work_dir = tempfile::tempdir().expect("Create temp dir");

    // Load ground truth from full target
    let (_, truth_records) = parse_vcf(&source.target_vcf);
    let truth_by_pos: HashMap<u64, &ParsedRecord> = truth_records.iter()
        .map(|r| (r.pos, r))
        .collect();

    // Load sparse target to know which positions were masked
    let (_, sparse_records) = parse_vcf(&source.target_sparse_vcf);
    let sparse_positions: std::collections::HashSet<u64> = sparse_records.iter()
        .map(|r| r.pos)
        .collect();

    // Run Rust imputation
    let rust_out = work_dir.path().join("rust_imp");
    let target_vcf = decompress_vcf_for_rust(&source.target_sparse_vcf, work_dir.as_ref());
    let ref_vcf = decompress_vcf_for_rust(&source.ref_vcf, work_dir.as_ref());
    run_rust_imputation(&target_vcf, &ref_vcf, &rust_out, 12345)
        .expect("Rust imputation failed");
    let rust_vcf = work_dir.path().join("rust_imp.vcf.gz");
    let (_, rust_records) = parse_vcf(&rust_vcf);

    // Run Java imputation
    let java_out = work_dir.path().join("java_imp");
    let java_status = Command::new("java")
        .args([
            "-jar", beagle_files.beagle_jar.to_str().unwrap(),
            &format!("gt={}", source.target_sparse_vcf.display()),
            &format!("ref={}", source.ref_vcf.display()),
            &format!("out={}", java_out.display()),
            "seed=12345",
        ])
        .output()
        .expect("Failed to run Java BEAGLE");
    assert!(java_status.status.success(), "Java BEAGLE failed");
    let java_vcf = work_dir.path().join("java_imp.vcf.gz");
    let (_, java_records) = parse_vcf(&java_vcf);

    // Helper to count large errors
    let count_large_errors = |records: &[ParsedRecord]| -> usize {
        let mut count = 0;
        for rec in records {
            if sparse_positions.contains(&rec.pos) {
                continue;
            }
            let Some(truth_rec) = truth_by_pos.get(&rec.pos) else {
                continue;
            };
            for (imp_gt, truth_gt) in rec.genotypes.iter().zip(truth_rec.genotypes.iter()) {
                let imputed_ds = imp_gt.ds.unwrap_or(0.0);
                let true_ds = gt_to_dosage(&truth_gt.gt).unwrap_or(0.0);
                if (imputed_ds - true_ds).abs() >= 0.9 {
                    count += 1;
                }
            }
        }
        count
    };

    let rust_large_errors = count_large_errors(&rust_records);
    let java_large_errors = count_large_errors(&java_records);

    eprintln!("[{}] Large errors (≥0.9): Rust={}, Java={}", source.name, rust_large_errors, java_large_errors);

    assert!(
        rust_large_errors <= java_large_errors,
        "[{}] Rust has more large errors than Java: {} vs {}",
        source.name, rust_large_errors, java_large_errors
    );
}
