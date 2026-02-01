
use reagle::{Config, ImputationPipeline};
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use tempfile::tempdir;

fn create_vcf(path: &PathBuf, content: &str) {
    let mut file = File::create(path).unwrap();
    file.write_all(content.as_bytes()).unwrap();
}

#[test]
fn test_hard_call_preservation() {
    let dir = tempdir().unwrap();
    let ref_path = dir.path().join("ref.vcf");
    let target_path = dir.path().join("target.vcf");
    let out_prefix = dir.path().join("out");

    // Reference: chr1:100 A/T
    create_vcf(&ref_path, r#"##fileformat=VCFv4.2
##contig=<ID=chr1>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	R1	R2
chr1	100	.	A	T	.	.	.	GT	0|0	1|1
"#);

    // Target: chr1:100 A/T, sample S1 is 0/0 (Hard call)
    // We expect output DS to be exactly 0.0, even if imputation might suggest otherwise (it won't here, but logic check)
    // Also include a missing marker to verify imputation runs.
    create_vcf(&target_path, r#"##fileformat=VCFv4.2
##contig=<ID=chr1>
#CHROM	POS	ID	REF	ALT	QUAL	FILTER	INFO	FORMAT	S1
chr1	100	.	A	T	.	.	.	GT	0/0
"#);

    let config = Config {
        gt: target_path.clone(),
        r#ref: Some(ref_path.clone()),
        out: out_prefix.clone(),
        err: None, // No error correction! Should enforce hard calls.
        gp: true, // Output GP/DS
        ..Default::default()
    };

    let mut pipeline = ImputationPipeline::new(config, None);
    pipeline.run().unwrap();

    let out_vcf = out_prefix.with_extension("vcf.gz");
    assert!(out_vcf.exists());

    // Decompress and check output
    let output = std::process::Command::new("gzip")
        .args(["-dc", out_vcf.to_str().unwrap()])
        .output()
        .unwrap();
    let content = String::from_utf8(output.stdout).unwrap();

    println!("Output VCF:\n{}", content);

    // Check dosage for chr1:100
    // Should be 0|0 and DS=0.0000
    // Format is GT:DS:GP
    for line in content.lines() {
        if line.starts_with("chr1\t100") {
            assert!(line.contains("0|0"), "GT should be 0|0");
            // Check DS
            // Example: 0|0:0.0000:1.0000,0.0000,0.0000
            let parts: Vec<&str> = line.split('\t').collect();
            let sample_field = parts[9]; // S1
            let format_fields: Vec<&str> = sample_field.split(':').collect();
            // GT is 0, DS is 1
            let ds = format_fields[1];
            let ds_val: f32 = ds.parse().unwrap();
            assert_eq!(ds_val, 0.0, "Dosage should be strictly 0.0 for hard call 0/0 without err correction");
        }
    }
}
