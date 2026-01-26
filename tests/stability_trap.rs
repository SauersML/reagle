use reagle::config::Config;
use reagle::pipelines::phasing::PhasingPipeline;
use std::fs::File;
use std::io::Write;
use tempfile::NamedTempFile;

// Helper to write a simplified VCF
fn write_vcf(path: &std::path::Path, samples: &[String], haplotypes: &[Vec<u8>], marker_pos: &[u32]) {
    let mut file = File::create(path).unwrap();
    writeln!(file, "##fileformat=VCFv4.2").unwrap();
    writeln!(file, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">").unwrap();
    write!(file, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT").unwrap();
    for s in samples {
        write!(file, "\t{}", s).unwrap();
    }
    writeln!(file).unwrap();

    for (m_idx, &pos) in marker_pos.iter().enumerate() {
        write!(file, "chr1\t{}\t.\tA\tC\t.\t.\t.\tGT", pos).unwrap();
        for (s_idx, _) in samples.iter().enumerate() {
            let h1 = haplotypes[s_idx * 2][m_idx];
            let h2 = haplotypes[s_idx * 2 + 1][m_idx];
            write!(file, "\t{}|{}", h1, h2).unwrap();
        }
        writeln!(file).unwrap();
    }
}

// Write target VCF (unphased 0/1)
fn write_target_vcf(path: &std::path::Path, marker_pos: &[u32]) {
    let mut file = File::create(path).unwrap();
    writeln!(file, "##fileformat=VCFv4.2").unwrap();
    writeln!(file, "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">").unwrap();
    write!(file, "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT").unwrap();
    writeln!(file, "\tTarget").unwrap();

    for &pos in marker_pos.iter() {
        // Hero allele is (m % 2).
        // Target is homozygous for Hero allele.
        // Hero: 0, 1, 0, 1...
        // Target: 0/0, 1/1, 0/0, 1/1...
        // This is effectively PHASED but we write it as unphased 0/0 or 1/1.
        // Wait, if it is homozygous, phasing is trivial. 
        // Let's make it Heterozygous to force phasing.
        // IF we make it heterozygous (0/1), it needs to decide phase.
        // H1 should be Hero (0, 1, 0, 1).
        // H2 should be Anti-Hero (1, 0, 1, 0).
        // If we provide Hero in ref, H1 should latch to Hero.
        
        // Let's stick to the User's plan "Target: Unphased, matches Hero Hap".
        // If target is unphased, we usually write 0/1. 
        // But if matches Hero perfectly, does it mean H1 matches Hero?
        // Let's try homozygous target as initially planned to test CONFIDENCE drop.
        
         // Target must be Heterozygous (0/1) to be processed as high-freq marker.
         // If it's homozygous, MAF=0 and it gets skipped.
         write!(file, "chr1\t{}\t.\tA\tC\t.\t.\t.\tGT\t0/1", pos).unwrap();
         writeln!(file).unwrap();
    }
}

#[test]
fn test_stability_trap_file_based() {
    let n_markers = 50;
    let n_ref_haps = 100;
    let hero_idx = 99;

    // 1. Generate Reference VCF
    let ref_file = NamedTempFile::new().unwrap();
    let ref_path = ref_file.path().to_path_buf();
    
    let mut ref_samples = Vec::new();
    let mut ref_haps = Vec::new();
    
    // Init haps
    for i in 0..n_ref_haps {
        ref_samples.push(format!("R{}", i));
        ref_haps.push(vec![0u8; n_markers]); // Hap 1
        ref_haps.push(vec![0u8; n_markers]); // Hap 2
    }
    
    // Fill random distractors
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for h in 0..ref_haps.len() {
        for m in 0..n_markers {
            ref_haps[h][m] = if rng.random_bool(0.5) { 1 } else { 0 };
        }
    }
    
    // Set Hero Hap (Index 99 * 2 = 198)
    let hero_hap_idx = hero_idx * 2;
    for m in 0..n_markers {
        ref_haps[hero_hap_idx][m] = (m % 2) as u8;
    }
    
    let marker_pos: Vec<u32> = (0..n_markers).map(|i| (i * 1000 + 100) as u32).collect();
    write_vcf(&ref_path, &ref_samples, &ref_haps, &marker_pos);

    // 2. Generate Target VCF
    let target_file = NamedTempFile::new().unwrap();
    let target_path = target_file.path().to_path_buf();
    write_target_vcf(&target_path, &marker_pos);

    // 3. Configure
    let out_file = NamedTempFile::new().unwrap();
    let out_path = out_file.path().to_path_buf();
    
    let mut config = Config::default();
    config.gt = target_path.clone();
    config.r#ref = Some(ref_path.clone());
    config.out = out_path.clone(); // Reagle will likely append .vcf.gz
    config.phase_states = 20; // TRAP: Capacity < 100
    config.burnin = 0;
    config.iterations = 2; 
    config.nthreads = Some(1); 
    config.ne = 10000.0;
    // Set low error rate to force strong preference for matches
    config.err = Some(0.0001); 
    
    // 4. Run
    let mut pipeline = PhasingPipeline::new(config, None);
    pipeline.run().expect("Pipeline run failed");
    
    // 5. Verify
    // Parse the output VCF to check phasing quality
    let expected_out_path = out_path.with_extension("vcf.gz");
    assert!(expected_out_path.exists(), "Output VCF does not exist");

    // Helper to read VCF (simple parser since it's just a test)
    use flate2::read::MultiGzDecoder;
    use std::io::BufReader;
    use std::io::BufRead;
    
    let file = File::open(&expected_out_path).unwrap();
    let decoder = MultiGzDecoder::new(file);
    let reader = BufReader::new(decoder);
    
    let mut phased_haps: Vec<(u8, u8)> = Vec::new();
    
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with('#') { continue; }
        
        let parts: Vec<&str> = line.split('\t').collect();
        // GT is usually the first field in the sample column (index 9)
        let sample_field = parts[9]; 
        // Format: GT:DS... e.g. "0|1:..."
        let gt_str = sample_field.split(':').next().unwrap();
        
        // Parse "0|1" or "1|0"
        let alleles: Vec<u8> = gt_str.split(['|', '/'])
            .map(|s| s.parse().unwrap_or(0)) // fail safe
            .collect();
        
        if alleles.len() >= 2 {
            phased_haps.push((alleles[0], alleles[1]));
        }
    }
    
    assert_eq!(phased_haps.len(), n_markers, "Parsed wrong number of markers");

    // Calculate Switch Error Rate relative to Hero Hap
    // Hero Alleles: (m % 2)
    // At m=0 (Hero=0): Target is 0/1. If Hap1=Hero, GT=0|1.
    // At m=1 (Hero=1): Target is 0/1. If Hap1=Hero, GT=1|0.
    // At m=2 (Hero=0): Target is 0/1. If Hap1=Hero, GT=0|1.
    
    // Pattern A (Hap1 tracks Hero): 0|1, 1|0, 0|1, 1|0...
    // Pattern B (Hap2 tracks Hero): 1|0, 0|1, 1|0, 0|1...
    
    // Standard Switch Error Calculation
    // Iterate and compare phase of (m-1, m)
    let mut switch_errors = 0;
    for m in 1..n_markers {
        let hero_prev = ((m-1) % 2) as u8;
        let hero_curr = (m % 2) as u8;
        
        // Use m in a dummy way if not used otherwise, but here we use it for hero_curr calculation
        // and accessing phased_haps[m]!
        // The compiler complaint about 'm' unused at line 37 was for the WRITING loop. 
        // I need to fix THAT loop too.
        
        let (h1_prev, _) = phased_haps[m-1];
        let (h1_curr, _) = phased_haps[m];
        
        // Did Hap1 continue to match Hero?
        let prev_match = h1_prev == hero_prev;
        let curr_match = h1_curr == hero_curr;
        
        if prev_match != curr_match {
            switch_errors += 1;
        }
    }
    
    println!("Marker Count: {}", n_markers);
    println!("Switch Errors: {}", switch_errors);
    let ser = switch_errors as f32 / (n_markers - 1) as f32;
    println!("Switch Error Rate: {:.4}", ser);
    
    // If the trap works (bug exists), SER should be high (e.g. > 10-20%)
    // If fixed/stable, SER should be very low (ideally 0 for perfect match)
    // We ASSERT that it is LOW (< 5%). If this fails, the bug is reproduced.
    assert!(ser < 0.05, "Stability Trap Triggered! SER is too high: {:.4}", ser);
}
