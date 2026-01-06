use reagle::config::Config;
use reagle::pipelines::imputation::ImputationPipeline;
use std::io::Write;
use tempfile::NamedTempFile;
use std::path::PathBuf;
use std::fs::File;
use std::io::BufReader;

/// Create a synthetic VCF file with specific pattern
fn create_synthetic_vcf(
    n_markers: usize,
    n_samples: usize,
    pattern: &dyn Fn(usize, usize) -> u8,
    is_phased: bool
) -> NamedTempFile {
    let mut file = tempfile::Builder::new()
        .suffix(".vcf")
        .tempfile()
        .expect("Create temp file");
    
    // Write Header
    writeln!(file, "##fileformat=VCFv4.2").unwrap();
    writeln!(file, "##FILTER=<ID=PASS,Description=\"All filters passed\">").unwrap();
    writeln!(file, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">").unwrap();
    writeln!(file, "##contig=<ID=chr1,length=1000000>").unwrap();
    
    write!(file, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT").unwrap();
    for i in 0..n_samples {
        write!(file, "\tSample{}", i).unwrap();
    }
    writeln!(file).unwrap();
    
    // Write Data
    for m in 0..n_markers {
        let pos = m * 1000 + 1;
        write!(file, "chr1\t{}\trs{}\tA\tT\t.\tPASS\t.\tGT", pos, m).unwrap();
        
        for s in 0..n_samples {
            // Haplotype indices
            let h1 = s * 2;
            let h2 = s * 2 + 1;
            
            let a1 = pattern(m, h1);
            let a2 = pattern(m, h2);
            
            let s1 = if a1 == 255 { "." } else { if a1 == 1 { "1" } else { "0" } };
            let s2 = if a2 == 255 { "." } else { if a2 == 1 { "1" } else { "0" } };
            
            let sep = if is_phased { "|" } else { "/" };
            write!(file, "\t{}{}{}", s1, sep, s2).unwrap();
        }
        writeln!(file).unwrap();
    }
    
    file
}

/// Parse dosages from output VCF
fn parse_dosages(path: &std::path::Path, n_markers: usize, n_samples: usize) -> Vec<f32> {
    // Pipeline writes bgzipped VCF
    use noodles::bgzf;
    let file = File::open(path).expect("Open output VCF");
    let reader = BufReader::new(bgzf::Reader::new(file));
    let mut dosages = Vec::new();
    
    use std::io::BufRead;
    
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with('#') { continue; }
        
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 9 + n_samples { continue; }
        
        // Ensure format includes DS
        let format = parts[8];
        let ds_idx = format.split(':').position(|f| f == "DS").unwrap_or(0); // Default to first if not found? No, DS must exist.
        
        if !format.contains("DS") {
            // Fallback to GT if DS missing (shouldn't happen for imputed)
            for s in 0..n_samples {
                let sample_str = parts[9+s];
                let gt_str = sample_str.split(':').next().unwrap();
                // simple 0|0 parsing
                if gt_str.contains('.') {
                    dosages.push(0.0); // or NaN
                } else {
                    let a1 = gt_str.chars().nth(0).unwrap().to_digit(10).unwrap() as f32;
                    let a2 = gt_str.chars().nth(2).unwrap().to_digit(10).unwrap() as f32;
                    dosages.push(a1+a2);
                }
            }
        } else {
            for s in 0..n_samples {
                let sample_str = parts[9+s];
                let ds_str = sample_str.split(':').nth(ds_idx).unwrap();
                let ds = ds_str.parse::<f32>().unwrap_or(0.0);
                dosages.push(ds);
            }
        }
    }
    
    dosages
}

// Helper to create default config since Config doesn't impl Default
fn default_test_config() -> Config {
    Config {
        gt: PathBuf::from(""),
        r#ref: None,
        out: PathBuf::from(""),
        map: None,
        chrom: None,
        excludesamples: None,
        excludemarkers: None,
        burnin: 3,
        iterations: 12,
        phase_states: 280,
        rare: 0.002,
        impute: true,
        imp_states: 1600,
        imp_segment: 6.0,
        imp_step: 0.1,
        imp_nsteps: 7,
        cluster: 0.005,
        ap: false,
        gp: false,
        ne: 100000.0,
        err: None,
        em: true,
        window: 40.0,
        window_markers: 4000000,
        overlap: 2.0,
        streaming: None,
        seed: -99999,
        nthreads: None,
    }
}

#[test]
fn test_synthetic_slam_dunk() {
    let n_markers = 50;
    
    // 1. Reference: 50 samples (100 haps). 50 '0's, 50 '1's.
    let ref_file = create_synthetic_vcf(n_markers, 50, &|_, h| {
        if h < 50 { 0 } else { 1 }
    }, true); // Phased reference
    
    // 2. Target: 1 sample. All 0s. Mask odd.
    let target_file = create_synthetic_vcf(n_markers, 1, &|m, _| {
        if m % 2 != 0 { 255 } else { 0 }
    }, false); // Unphased target
    
    let temp_dir = tempfile::tempdir().unwrap();
    let out_prefix = temp_dir.path().join("output");
    
    let mut config = default_test_config();
    config.gt = target_file.path().to_path_buf();
    config.r#ref = Some(ref_file.path().to_path_buf());
    config.out = out_prefix.clone();
    config.imp_states = 50;
    config.imp_nsteps = 10;
    config.ne = 10000.0;
    config.err = Some(0.0001);
    // Force multiple windows to test stitching (Total map ~0.05cM)
    config.window = 0.02;
    config.overlap = 0.005;
    // Single threaded for testing
    config.nthreads = Some(1);
    
    let mut pipeline = ImputationPipeline::new(config);
    pipeline.run().expect("Pipeline run success");
    
    // Output is at out_prefix.vcf.gz
    let out_vcf = temp_dir.path().join("output.vcf.gz");
    assert!(out_vcf.exists());
    
    // Verify
    use noodles::bgzf;
    let file = File::open(&out_vcf).unwrap();
    let reader = BufReader::new(bgzf::Reader::new(file));
    
    let mut dosages = Vec::new();
    use std::io::BufRead;
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with('#') { continue; }
        
        let parts: Vec<&str> = line.split('\t').collect();
        // Format is typically GT:DS:GP
        let format = parts[8];
        let ds_idx = format.split(':').position(|f| f == "DS");
        
        if let Some(idx) = ds_idx {
             let sample_str = parts[9]; // Sample 0
             let ds_str = sample_str.split(':').nth(idx).unwrap();
             dosages.push(ds_str.parse::<f32>().unwrap());
        } else {
             dosages.push(0.0);
        }
    }
    
    // Odd markers (1, 3...) masked -> should be 0
    for m in (1..n_markers).step_by(2) {
        let ds = dosages[m];
        assert!(ds < 0.1, "Marker {} should be 0, got {}", m, ds);
    }
}

#[test]
fn test_synthetic_recombination() {
    let n_markers = 50;
    let ref_file = create_synthetic_vcf(n_markers, 50, &|_, h| {
        if h < 50 { 0 } else { 1 }
    }, true);
    
    // Target: Switch at 20. 0->1. Mask 15 and 25.
    let target_file = create_synthetic_vcf(n_markers, 1, &|m, _| {
        if m == 15 || m == 25 { 
            255 
        } else {
            if m < 20 { 0 } else { 1 }
        }
    }, false);

    let temp_dir = tempfile::tempdir().unwrap();
    let out_prefix = temp_dir.path().join("output_rec");
    
    let mut config = default_test_config();
    config.gt = target_file.path().to_path_buf();
    config.r#ref = Some(ref_file.path().to_path_buf());
    config.out = out_prefix.clone();
    config.imp_states = 50;
    config.ne = 10000.0;
    config.window = 0.02;
    config.overlap = 0.005;
    config.nthreads = Some(1);
    
    let mut pipeline = ImputationPipeline::new(config);
    pipeline.run().expect("Pipeline run success");
    
    let out_vcf = temp_dir.path().join("output_rec.vcf.gz");
    assert!(out_vcf.exists());
    
    use noodles::bgzf;
    let file = File::open(&out_vcf).unwrap();
    let reader = BufReader::new(bgzf::Reader::new(file));
    
    let mut dosages = Vec::new();
    use std::io::BufRead;
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with('#') { continue; }
        let parts: Vec<&str> = line.split('\t').collect();
        let format = parts[8];
        let ds_idx = format.split(':').position(|f| f == "DS").unwrap();
        let sample_str = parts[9];
        let ds_str = sample_str.split(':').nth(ds_idx).unwrap();
        dosages.push(ds_str.parse::<f32>().unwrap());
    }
    
    let ds_15 = dosages[15];
    let ds_25 = dosages[25];
    
    assert!(ds_15 < 0.1, "Marker 15 should be 0, got {}", ds_15);
    assert!(ds_25 > 1.9, "Marker 25 should be 2, got {}", ds_25);
}
