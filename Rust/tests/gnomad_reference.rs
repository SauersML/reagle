//! gnomAD/HGDP+1KG data source for integration tests.
//!
//! This module provides access to gnomAD HGDP+1KG phased haplotypes as an
//! additional test data source. The fixtures are pre-generated and committed
//! to the repository in tests/fixtures/gnomad_hgdp/.

use std::path::PathBuf;

/// gnomAD test files (same structure as BEAGLE test files)
pub struct GnomadTestFiles {
    pub ref_vcf: PathBuf,
    pub target_vcf: PathBuf,
    pub target_sparse_vcf: PathBuf,
}

/// Get gnomAD fixtures directory
fn gnomad_fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gnomad_hgdp")
}

/// Setup gnomAD test files (pre-generated fixtures)
pub fn setup_gnomad_files() -> GnomadTestFiles {
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
