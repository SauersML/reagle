//! gnomAD/HGDP+1KG data source for integration tests.
//!
//! This module provides access to gnomAD HGDP+1KG phased haplotypes as an
//! additional test data source. Uses bcftools to stream a small region
//! without downloading the full ~4GB file.
//!
//! The extracted data is cached in tests/fixtures/gnomad_hgdp/

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// gnomAD HGDP+1KG URLs
const GNOMAD_CHR22_BCF: &str = "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf";
const GNOMAD_CHR22_CSI: &str = "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf.csi";

/// Region to extract (~200 variants, smaller = faster tests)
const EXTRACT_REGION: &str = "chr22:16000000-16200000";

/// Number of samples to subset (full file has ~4000)
const SAMPLE_SUBSET: usize = 50;

/// Directory for cached gnomAD fixtures
pub fn gnomad_fixtures_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gnomad_hgdp");
    fs::create_dir_all(&dir).expect("Create gnomAD fixtures directory");
    dir
}

/// Check if bcftools is available
pub fn bcftools_available() -> bool {
    Command::new("bcftools")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// gnomAD test files (same structure as BEAGLE test files)
pub struct GnomadTestFiles {
    pub ref_vcf: PathBuf,
    pub target_vcf: PathBuf,
    pub target_sparse_vcf: PathBuf,
}

/// Extract a small region from gnomAD BCF using bcftools streaming
fn extract_gnomad_region(output_vcf: &Path) -> bool {
    if output_vcf.exists() {
        return true;
    }

    if !bcftools_available() {
        eprintln!("bcftools not available - cannot extract gnomAD data");
        return false;
    }

    let fixtures = gnomad_fixtures_dir();

    // Download just the index (small, ~1MB)
    let csi_path = fixtures.join("chr22.bcf.csi");
    if !csi_path.exists() {
        println!("Downloading BCF index...");
        let status = Command::new("curl")
            .args(["-L", "-s", "-o", csi_path.to_str().unwrap(), GNOMAD_CHR22_CSI])
            .status();
        if status.map(|s| !s.success()).unwrap_or(true) {
            return false;
        }
    }

    // Get sample list
    println!("Getting sample list from gnomAD...");
    let sample_output = Command::new("bcftools")
        .args(["query", "-l", GNOMAD_CHR22_BCF])
        .output();

    let samples: Vec<String> = match sample_output {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout)
                .lines()
                .take(SAMPLE_SUBSET)
                .map(|s| s.to_string())
                .collect()
        }
        _ => return false,
    };

    if samples.len() < SAMPLE_SUBSET {
        return false;
    }

    // Write sample subset file
    let sample_file = fixtures.join("sample_subset.txt");
    if fs::write(&sample_file, samples.join("\n")).is_err() {
        return false;
    }

    // Extract region with sample subset
    println!("Streaming region {} from gnomAD ({} samples)...", EXTRACT_REGION, SAMPLE_SUBSET);
    let temp_vcf = fixtures.join("temp_extract.vcf");

    let status = Command::new("bcftools")
        .args([
            "view",
            "-r", EXTRACT_REGION,
            "-S", sample_file.to_str().unwrap(),
            "-O", "v",
            "-o", temp_vcf.to_str().unwrap(),
            GNOMAD_CHR22_BCF,
        ])
        .status();

    if status.map(|s| !s.success()).unwrap_or(true) {
        return false;
    }

    // Compress
    let _ = Command::new("gzip").args(["-f", temp_vcf.to_str().unwrap()]).status();
    if fs::rename(fixtures.join("temp_extract.vcf.gz"), output_vcf).is_err() {
        return false;
    }

    true
}

/// Setup gnomAD test files (cached across runs)
pub fn setup_gnomad_files() -> Option<GnomadTestFiles> {
    let fixtures = gnomad_fixtures_dir();
    let full_vcf = fixtures.join("chr22_region.vcf.gz");
    let ref_vcf = fixtures.join("ref.vcf.gz");
    let target_vcf = fixtures.join("target.vcf.gz");
    let target_sparse_vcf = fixtures.join("target_sparse.vcf.gz");

    // Extract from gnomAD if not cached
    if !extract_gnomad_region(&full_vcf) {
        return None;
    }

    // Create ref/target split if not cached
    // With 50 samples: 40 ref + 10 target
    if !ref_vcf.exists() || !target_vcf.exists() {
        println!("Creating gnomAD ref/target split...");

        // ref: first 40 samples (columns 1-49 = 9 metadata + 40 samples, already phased)
        let status = Command::new("sh")
            .arg("-c")
            .arg(format!(
                "gzip -dc {} | cut -f1-49 | gzip > {}",
                full_vcf.display(), ref_vcf.display()
            ))
            .status().ok()?;
        if !status.success() { return None; }

        // target: last 10 samples, unphased (columns 1-9 metadata + 50-59 samples)
        let status = Command::new("sh")
            .arg("-c")
            .arg(format!(
                "gzip -dc {} | cut -f1-9,50-59 | sed 's/|/\\//g' | gzip > {}",
                full_vcf.display(), target_vcf.display()
            ))
            .status().ok()?;
        if !status.success() { return None; }
    }

    // Create sparse target if not cached
    if !target_sparse_vcf.exists() {
        let status = Command::new("sh")
            .arg("-c")
            .arg(format!(
                "gzip -dc {} | awk 'NR==1 || /^#/ || NR%10==0' | gzip > {}",
                target_vcf.display(), target_sparse_vcf.display()
            ))
            .status().ok()?;
        if !status.success() { return None; }
    }

    Some(GnomadTestFiles {
        ref_vcf,
        target_vcf,
        target_sparse_vcf,
    })
}
