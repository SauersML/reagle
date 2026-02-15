//! Shared test utilities for downloading and generating test data.
//!
//! Provides:
//! - Streaming extraction from HGDP+1KG reference panel (no full download needed)
//! - GSA microarray site list for sparse target generation
//!
//! And generators for:
//! - Reference panel (N samples from HGDP+1KG)
//! - Target samples (M samples held out)
//! - Sparse target (target filtered to GSA sites)
pub mod reference_metrics;

use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

/// URLs for test data
const HGDP_1KG_BCF_URL_TEMPLATE: &str = "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr{chrom}.filtered.SNV_INDEL.phased.shapeit5.bcf";
const GSA_SITES_URL: &str =
    "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/GSAv2_hg38.tsv";

/// Get the test data cache directory
pub fn cache_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("cache");
    fs::create_dir_all(&dir).ok();
    dir
}

/// Download a file if it doesn't exist
pub fn download_if_missing(url: &str, dest: &Path) -> bool {
    if dest.exists() {
        return true;
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).ok();
    }
    eprintln!("Downloading {} to {:?}...", url, dest);
    let status = Command::new("curl")
        .args(["-fsSL", url, "-o"])
        .arg(dest)
        .status();
    match status {
        Ok(s) if s.success() => true,
        _ => {
            eprintln!("Failed to download {}", url);
            false
        }
    }
}

fn hgdp_1kg_chrom_url(chrom: &str) -> String {
    let chr = chrom.trim_start_matches("chr");
    HGDP_1KG_BCF_URL_TEMPLATE.replace("{chrom}", chr)
}

/// Get sample list from remote HGDP+1KG BCF (streams header only)
fn get_remote_sample_list(source_url: &str) -> Vec<String> {
    // bcftools can read directly from URL - only fetches header for sample list
    let output = Command::new("bcftools")
        .args(["query", "-l", source_url])
        .output()
        .expect("bcftools query -l from remote URL");

    if !output.status.success() {
        panic!(
            "Failed to get sample list from remote BCF: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    std::str::from_utf8(&output.stdout)
        .expect("UTF-8")
        .lines()
        .map(|s| s.to_string())
        .collect()
}

/// Stream a region from remote HGDP+1KG BCF with sample subset
fn stream_region_from_remote(samples_file: &Path, region: &str, source_url: &str, output_vcf: &Path) {
    eprintln!(
        "Streaming region {} from remote HGDP+1KG to {:?}...",
        region, output_vcf
    );

    // bcftools can stream directly from URL with region query
    // This only downloads the index + requested region, not the whole file
    let status = Command::new("bcftools")
        .args([
            "view",
            "-S",
            samples_file.to_str().unwrap(),
            "-r",
            region,
            "-O",
            "z",
            "-o",
            output_vcf.to_str().unwrap(),
            source_url,
        ])
        .status()
        .expect("bcftools view from remote");

    assert!(
        status.success(),
        "Failed to stream region from remote HGDP+1KG"
    );

    // Index the output
    Command::new("bcftools")
        .args(["index", "-f"])
        .arg(output_vcf)
        .status()
        .expect("Index VCF");
}

/// Get path to cached GSA sites file (downloads if needed)
pub fn gsa_sites_file() -> PathBuf {
    let path = cache_dir().join("GSAv2_hg38.tsv");
    if !path.exists() {
        if !download_if_missing(GSA_SITES_URL, &path) {
            panic!("Failed to download GSA sites");
        }
    }
    path
}

/// Load GSA sites for a chromosome as a set of positions
pub fn load_gsa_positions(chrom: &str) -> Vec<u64> {
    let file = gsa_sites_file();
    let reader = BufReader::new(File::open(&file).expect("Open GSA file"));

    let mut positions = Vec::new();
    for line in reader.lines() {
        let line = line.expect("Read line");
        if line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 {
            let chr = parts[0].trim_start_matches("chr");
            if chr == chrom.trim_start_matches("chr") {
                if let Ok(pos) = parts[1].parse::<u64>() {
                    positions.push(pos);
                }
            }
        }
    }
    positions.sort();
    positions
}

/// Test data paths for a test run
pub struct TestData {
    pub ref_vcf: PathBuf,
    pub target_vcf: PathBuf,
    pub target_sparse_vcf: PathBuf,
}

fn has_index(vcf: &Path) -> bool {
    PathBuf::from(format!("{}.csi", vcf.display())).exists()
        || PathBuf::from(format!("{}.tbi", vcf.display())).exists()
}

fn ensure_index(vcf: &Path) {
    if has_index(vcf) {
        return;
    }
    let status = Command::new("bcftools")
        .args(["index", "-f"])
        .arg(vcf)
        .status()
        .expect("Index VCF");
    assert!(status.success(), "Failed to index {}", vcf.display());
}

fn sanitize_for_path(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

fn dataset_cache_dir(ref_samples: usize, target_samples: usize, region: &str) -> PathBuf {
    let key = format!(
        "ref{}_target{}_{}",
        ref_samples,
        target_samples,
        sanitize_for_path(region)
    );
    let dir = cache_dir().join("generated").join(key);
    fs::create_dir_all(&dir).expect("Create generated cache dir");
    dir
}

/// Generate test data from HGDP+1KG panel (streams region, doesn't download full file)
///
/// - ref_samples: number of samples for reference panel
/// - target_samples: number of samples to hold out as target
/// - region: genomic region to extract (e.g., "chr22:16000000-17000000")
pub fn generate_test_data(ref_samples: usize, target_samples: usize, region: &str) -> TestData {
    let work_dir = dataset_cache_dir(ref_samples, target_samples, region);
    let chrom = region.split(':').next().unwrap_or("chr22");
    let source_url = hgdp_1kg_chrom_url(chrom);

    let ref_vcf = work_dir.join("ref.vcf.gz");
    let target_vcf = work_dir.join("target.vcf.gz");
    let target_sparse_vcf = work_dir.join("target_sparse.vcf.gz");

    if ref_vcf.exists() && target_vcf.exists() && target_sparse_vcf.exists() {
        ensure_index(&ref_vcf);
        ensure_index(&target_vcf);
        ensure_index(&target_sparse_vcf);
        eprintln!(
            "Using cached generated dataset: {}",
            work_dir.to_string_lossy()
        );
        return TestData {
            ref_vcf,
            target_vcf,
            target_sparse_vcf,
        };
    }

    // Get sample list from remote (only fetches header)
    let all_samples = get_remote_sample_list(&source_url);

    let total_needed = ref_samples + target_samples;
    if all_samples.len() < total_needed {
        panic!(
            "Not enough samples in panel: need {}, have {}",
            total_needed,
            all_samples.len()
        );
    }

    // Split samples
    let ref_sample_list: Vec<&str> = all_samples[..ref_samples]
        .iter()
        .map(|s| s.as_str())
        .collect();
    let target_sample_list: Vec<&str> = all_samples[ref_samples..total_needed]
        .iter()
        .map(|s| s.as_str())
        .collect();

    // Write sample lists
    let ref_samples_file = work_dir.join("ref_samples.txt");
    let target_samples_file = work_dir.join("target_samples.txt");

    fs::write(&ref_samples_file, ref_sample_list.join("\n")).expect("Write ref samples");
    fs::write(&target_samples_file, target_sample_list.join("\n")).expect("Write target samples");

    // Stream reference panel from remote (only downloads the region we need)
    stream_region_from_remote(&ref_samples_file, region, &source_url, &ref_vcf);

    // Stream target from remote
    stream_region_from_remote(&target_samples_file, region, &source_url, &target_vcf);

    // Create sparse target (filter to GSA sites)
    let gsa_positions = load_gsa_positions(chrom);

    // Create regions file for GSA sites in this region
    let gsa_regions_file = work_dir.join("gsa_regions.txt");
    {
        let mut f = File::create(&gsa_regions_file).expect("Create GSA regions");
        let region_parts: Vec<&str> = region.split(':').collect();
        let chrom = region_parts[0];
        let (start, end) = if region_parts.len() > 1 {
            let range: Vec<&str> = region_parts[1].split('-').collect();
            (
                range[0].parse::<u64>().unwrap_or(0),
                range[1].parse::<u64>().unwrap_or(u64::MAX),
            )
        } else {
            (0, u64::MAX)
        };

        for pos in &gsa_positions {
            if *pos >= start && *pos <= end {
                // bcftools -R expects tab-separated: CHROM\tFROM\tTO
                writeln!(f, "{}\t{}\t{}", chrom, pos, pos).expect("Write region");
            }
        }
    }

    let status = Command::new("bcftools")
        .args([
            "view",
            "-R",
            gsa_regions_file.to_str().unwrap(),
            "-O",
            "z",
            "-o",
            target_sparse_vcf.to_str().unwrap(),
        ])
        .arg(&target_vcf)
        .status()
        .expect("bcftools view sparse");
    assert!(status.success(), "Failed to create sparse target");

    ensure_index(&target_sparse_vcf);
    ensure_index(&ref_vcf);
    ensure_index(&target_vcf);

    TestData {
        ref_vcf,
        target_vcf,
        target_sparse_vcf,
    }
}
