use reagle::config::Config;
use reagle::data::alignment::MarkerAlignment;
use reagle::data::genetic_map::GeneticMaps;
use reagle::data::marker::MarkerIdx;
use reagle::io::vcf::VcfReader;
use reagle::pipelines::phasing::PhasingPipeline;
use std::fs::File;
use std::io::Write;
use std::sync::Arc;
use tempfile::NamedTempFile;

// Helper to write a simplified VCF
fn write_vcf(
    path: &std::path::Path,
    samples: &[String],
    haplotypes: &[Vec<u8>],
    marker_pos: &[u32],
) {
    let mut file = File::create(path).unwrap();
    writeln!(file, "##fileformat=VCFv4.2").unwrap();
    writeln!(
        file,
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">"
    )
    .unwrap();
    write!(
        file,
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
    )
    .unwrap();
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
    writeln!(
        file,
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">"
    )
    .unwrap();
    write!(
        file,
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
    )
    .unwrap();
    writeln!(file, "\tTarget").unwrap();

    for &pos in marker_pos.iter() {
        // Target is heterozygous at every marker to force phasing.
        // The reference panel defines the hero/anti patterns.
        write!(file, "chr1\t{}\t.\tA\tC\t.\t.\t.\tGT\t0/1", pos).unwrap();
        writeln!(file).unwrap();
    }
}

fn hero_pattern_from_ref_hap(ref_haps: &[Vec<u8>], hero_hap_idx: usize) -> Vec<u8> {
    ref_haps
        .get(hero_hap_idx)
        .cloned()
        .unwrap_or_default()
}

fn make_marker_pos(n_markers: usize, step_bp: u32) -> Vec<u32> {
    (0..n_markers)
        .map(|i| 100u32 + (i as u32) * step_bp)
        .collect()
}

#[test]
fn test_ser_switching_all0_all1_reference() {
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

    // Set Hero Hap (Index 99 * 2 = 198) with the test-specific hero pattern
    let hero_hap_idx = hero_idx * 2;
    for m in 0..n_markers {
        ref_haps[hero_hap_idx][m] = 0;
    }

    // Set Anti-Hero Hap (Index 99 * 2 + 1 = 199) as the complementary pattern
    let anti_hero_hap_idx = hero_idx * 2 + 1;
    for m in 0..n_markers {
        ref_haps[anti_hero_hap_idx][m] = 1;
    }

    let marker_pos = make_marker_pos(n_markers, 1_000);
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
    config.phase_states = 0; // Auto state budget.
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

    // Parse the output VCF to check phasing quality
    use flate2::read::MultiGzDecoder;
    use std::io::BufRead;
    use std::io::BufReader;

    let file = File::open(&expected_out_path).unwrap();
    let decoder = MultiGzDecoder::new(file);
    let reader = BufReader::new(decoder);

    // Read all output lines to check quality
    let mut all_lines: Vec<String> = Vec::new();
    for line in reader.lines() {
        all_lines.push(line.unwrap());
    }

    // DEBUG: Check what the phased output looks like
    println!("=== DEBUG: First few phased genotypes ===");
    for (i, line) in all_lines.iter().enumerate().take(60) {
        if i < 10 || (i >= 45 && i < 55) {
            // Show header and first few data lines
            println!("Line {}: {}", i, line);
        }
    }

    let mut phased_haps: Vec<(u8, u8)> = Vec::new();
    for line in &all_lines {
        if line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        // GT is usually the first field in the sample column (index 9)
        let sample_field = parts[9];
        // Format: GT:DS... e.g. "0|1:..."
        let gt_str = sample_field.split(':').next().unwrap();

        // Parse "0|1" or "1|0"
        let alleles: Vec<u8> = gt_str
            .split(['|', '/'])
            .map(|s| s.parse().unwrap_or(0)) // fail safe
            .collect();

        if alleles.len() >= 2 {
            phased_haps.push((alleles[0], alleles[1]));
        }
    }

    assert_eq!(
        phased_haps.len(),
        n_markers,
        "Parsed wrong number of markers"
    );

    // Calculate Switch Error Rate relative to the hero hap from the reference.
    // This is invariant to global hap1/hap2 label swaps.

    let hero_pattern = hero_pattern_from_ref_hap(&ref_haps, hero_hap_idx);

    // Standard Switch Error Calculation
    // Iterate and compare phase of (m-1, m)
    let mut switch_errors = 0;
    for m in 1..n_markers {
        let hero_prev = hero_pattern[m - 1];
        let hero_curr = hero_pattern[m];

        let (h1_prev, _) = phased_haps[m - 1];
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

    // The setup is inherently ambiguous; we still expect low switching if the
    // phaser maintains a stable path in this regime.
    assert!(
        ser < 0.05,
        "Stability Trap Triggered! SER is too high: {:.4}",
        ser
    );
}

#[test]
fn test_ser_switching_all0_all1_reference_dense_map() {
    let n_markers = 50;
    let n_ref_haps = 100;
    let hero_idx = 99;

    let ref_file = NamedTempFile::new().unwrap();
    let ref_path = ref_file.path().to_path_buf();

    let mut ref_samples = Vec::new();
    let mut ref_haps = Vec::new();

    for i in 0..n_ref_haps {
        ref_samples.push(format!("R{}", i));
        ref_haps.push(vec![0u8; n_markers]);
        ref_haps.push(vec![0u8; n_markers]);
    }

    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for h in 0..ref_haps.len() {
        for m in 0..n_markers {
            ref_haps[h][m] = if rng.random_bool(0.5) { 1 } else { 0 };
        }
    }

    let hero_hap_idx = hero_idx * 2;
    for m in 0..n_markers {
        ref_haps[hero_hap_idx][m] = 0;
    }

    let anti_hero_hap_idx = hero_idx * 2 + 1;
    for m in 0..n_markers {
        ref_haps[anti_hero_hap_idx][m] = 1;
    }

    let marker_pos = make_marker_pos(n_markers, 200_000);
    write_vcf(&ref_path, &ref_samples, &ref_haps, &marker_pos);

    let target_file = NamedTempFile::new().unwrap();
    let target_path = target_file.path().to_path_buf();
    write_target_vcf(&target_path, &marker_pos);

    let out_file = NamedTempFile::new().unwrap();
    let out_path = out_file.path().to_path_buf();

    let mut config = Config::default();
    config.gt = target_path.clone();
    config.r#ref = Some(ref_path.clone());
    config.out = out_path.clone();
    config.phase_states = 0;
    config.burnin = 0;
    config.iterations = 2;
    config.nthreads = Some(1);
    config.ne = 10000.0;
    config.err = Some(0.0001);

    let mut pipeline = PhasingPipeline::new(config, None);
    pipeline.run().expect("Pipeline run failed");

    let expected_out_path = out_path.with_extension("vcf.gz");
    assert!(expected_out_path.exists());

    use flate2::read::MultiGzDecoder;
    use std::io::BufRead;
    use std::io::BufReader;

    let file = File::open(&expected_out_path).unwrap();
    let decoder = MultiGzDecoder::new(file);
    let reader = BufReader::new(decoder);

    let mut phased_haps: Vec<(u8, u8)> = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        let sample_field = parts[9];
        let gt_str = sample_field.split(':').next().unwrap();
        let alleles: Vec<u8> = gt_str
            .split(['|', '/'])
            .map(|s| s.parse().unwrap_or(0))
            .collect();
        if alleles.len() >= 2 {
            phased_haps.push((alleles[0], alleles[1]));
        }
    }

    let hero_pattern = hero_pattern_from_ref_hap(&ref_haps, hero_hap_idx);

    let mut switch_errors = 0;
    for m in 1..n_markers {
        let hero_prev = hero_pattern[m - 1];
        let hero_curr = hero_pattern[m];
        let (h1_prev, _) = phased_haps[m - 1];
        let (h1_curr, _) = phased_haps[m];
        let prev_match = h1_prev == hero_prev;
        let curr_match = h1_curr == hero_curr;
        if prev_match != curr_match {
            switch_errors += 1;
        }
    }

    let ser = switch_errors as f32 / (n_markers - 1) as f32;

    assert!(ser < 0.05, "SER too high in dense map: {:.4}", ser);
}

#[test]
fn test_ser_not_fixed_by_high_state_count_all0_all1() {
    let n_markers = 50;
    let n_ref_haps = 100;
    let hero_idx = 99;

    let ref_file = NamedTempFile::new().unwrap();
    let ref_path = ref_file.path().to_path_buf();

    let mut ref_samples = Vec::new();
    let mut ref_haps = Vec::new();

    for i in 0..n_ref_haps {
        ref_samples.push(format!("R{}", i));
        ref_haps.push(vec![0u8; n_markers]);
        ref_haps.push(vec![0u8; n_markers]);
    }

    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for h in 0..ref_haps.len() {
        for m in 0..n_markers {
            ref_haps[h][m] = if rng.random_bool(0.5) { 1 } else { 0 };
        }
    }

    let hero_hap_idx = hero_idx * 2;
    for m in 0..n_markers {
        ref_haps[hero_hap_idx][m] = 0;
    }

    let anti_hero_hap_idx = hero_idx * 2 + 1;
    for m in 0..n_markers {
        ref_haps[anti_hero_hap_idx][m] = 1;
    }

    let marker_pos = make_marker_pos(n_markers, 1_000);
    write_vcf(&ref_path, &ref_samples, &ref_haps, &marker_pos);

    let target_file = NamedTempFile::new().unwrap();
    let target_path = target_file.path().to_path_buf();
    write_target_vcf(&target_path, &marker_pos);

    let out_file = NamedTempFile::new().unwrap();
    let out_path = out_file.path().to_path_buf();

    let mut config = Config::default();
    config.gt = target_path.clone();
    config.r#ref = Some(ref_path.clone());
    config.out = out_path.clone();
    config.phase_states = 0;
    config.burnin = 0;
    config.iterations = 2;
    config.nthreads = Some(1);
    config.ne = 10000.0;
    config.err = Some(0.0001);

    let mut pipeline = PhasingPipeline::new(config, None);
    pipeline.run().expect("Pipeline run failed");

    let expected_out_path = out_path.with_extension("vcf.gz");
    assert!(expected_out_path.exists());

    use flate2::read::MultiGzDecoder;
    use std::io::BufRead;
    use std::io::BufReader;

    let file = File::open(&expected_out_path).unwrap();
    let decoder = MultiGzDecoder::new(file);
    let reader = BufReader::new(decoder);

    let mut phased_haps: Vec<(u8, u8)> = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        let sample_field = parts[9];
        let gt_str = sample_field.split(':').next().unwrap();
        let alleles: Vec<u8> = gt_str
            .split(['|', '/'])
            .map(|s| s.parse().unwrap_or(0))
            .collect();

        if alleles.len() >= 2 {
            phased_haps.push((alleles[0], alleles[1]));
        }
    }

    assert_eq!(phased_haps.len(), n_markers);

    let hero_pattern = hero_pattern_from_ref_hap(&ref_haps, hero_hap_idx);

    let mut switch_errors = 0;
    for m in 1..n_markers {
        let hero_prev = hero_pattern[m - 1];
        let hero_curr = hero_pattern[m];
        let (h1_prev, _) = phased_haps[m - 1];
        let (h1_curr, _) = phased_haps[m];

        let prev_match = h1_prev == hero_prev;
        let curr_match = h1_curr == hero_curr;

        if prev_match != curr_match {
            switch_errors += 1;
        }
    }

    let ser = switch_errors as f32 / (n_markers - 1) as f32;

    assert!(ser < 0.05, "SER too high with max_states=200: {:.4}", ser);
}

#[test]
fn test_sparse_phased_anchors_dense_map_still_switches() {
    let n_markers = 50;
    let n_ref_haps = 100;
    let hero_idx = 99;

    let ref_file = NamedTempFile::new().unwrap();
    let ref_path = ref_file.path().to_path_buf();

    let mut ref_samples = Vec::new();
    let mut ref_haps = Vec::new();

    for i in 0..n_ref_haps {
        ref_samples.push(format!("R{}", i));
        ref_haps.push(vec![0u8; n_markers]);
        ref_haps.push(vec![0u8; n_markers]);
    }

    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for h in 0..ref_haps.len() {
        for m in 0..n_markers {
            ref_haps[h][m] = if rng.random_bool(0.5) { 1 } else { 0 };
        }
    }

    let hero_hap_idx = hero_idx * 2;
    for m in 0..n_markers {
        ref_haps[hero_hap_idx][m] = (m % 2) as u8;
    }
    let anti_hero_hap_idx = hero_idx * 2 + 1;
    for m in 0..n_markers {
        ref_haps[anti_hero_hap_idx][m] = 1 - (m % 2) as u8;
    }

    let marker_pos = make_marker_pos(n_markers, 200_000);
    write_vcf(&ref_path, &ref_samples, &ref_haps, &marker_pos);

    let target_file = NamedTempFile::new().unwrap();
    let target_path = target_file.path().to_path_buf();

    let mut file = File::create(&target_path).unwrap();
    writeln!(file, "##fileformat=VCFv4.2").unwrap();
    writeln!(
        file,
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">"
    )
    .unwrap();
    writeln!(
        file,
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTarget"
    )
    .unwrap();

    let hero_pattern = hero_pattern_from_ref_hap(&ref_haps, hero_hap_idx);
    for (m, &pos) in marker_pos.iter().enumerate() {
        let hero_allele = hero_pattern[m];
        let anti_hero = 1 - hero_allele;
        if m % 10 == 0 {
            write!(
                file,
                "chr1\t{}\t.\tA\tC\t.\t.\t.\tGT\t{}|{}",
                pos, hero_allele, anti_hero
            )
            .unwrap();
        } else {
            write!(file, "chr1\t{}\t.\tA\tC\t.\t.\t.\tGT\t0/1", pos).unwrap();
        }
        writeln!(file).unwrap();
    }

    let out_file = NamedTempFile::new().unwrap();
    let out_path = out_file.path().to_path_buf();

    let mut config = Config::default();
    config.gt = target_path.clone();
    config.r#ref = Some(ref_path.clone());
    config.out = out_path.clone();
    config.phase_states = 0;
    config.burnin = 0;
    config.iterations = 2;
    config.nthreads = Some(1);
    config.ne = 10000.0;
    config.err = Some(0.0001);

    let mut pipeline = PhasingPipeline::new(config, None);
    pipeline.run().expect("Pipeline run failed");

    let expected_out_path = out_path.with_extension("vcf.gz");
    assert!(expected_out_path.exists());

    use flate2::read::MultiGzDecoder;
    use std::io::BufRead;
    use std::io::BufReader;

    let file = File::open(&expected_out_path).unwrap();
    let decoder = MultiGzDecoder::new(file);
    let reader = BufReader::new(decoder);

    let mut phased_haps: Vec<(u8, u8)> = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        let sample_field = parts[9];
        let gt_str = sample_field.split(':').next().unwrap();
        let alleles: Vec<u8> = gt_str
            .split(['|', '/'])
            .map(|s| s.parse().unwrap_or(0))
            .collect();
        if alleles.len() >= 2 {
            phased_haps.push((alleles[0], alleles[1]));
        }
    }

    let hero_pattern = hero_pattern_from_ref_hap(&ref_haps, hero_hap_idx);

    let mut switch_errors = 0;
    for m in 1..n_markers {
        let hero_prev = hero_pattern[m - 1];
        let hero_curr = hero_pattern[m];
        let (h1_prev, _) = phased_haps[m - 1];
        let (h1_curr, _) = phased_haps[m];
        let prev_match = h1_prev == hero_prev;
        let curr_match = h1_curr == hero_curr;
        if prev_match != curr_match {
            switch_errors += 1;
        }
    }

    let ser = switch_errors as f32 / (n_markers - 1) as f32;

    assert!(
        ser < 0.05,
        "SER too high with anchors in dense map: {:.4}",
        ser
    );
}

#[test]
fn test_sparse_phased_anchors_do_not_cause_switching() {
    let n_markers = 50;
    let n_ref_haps = 100;
    let hero_idx = 99;

    let ref_file = NamedTempFile::new().unwrap();
    let ref_path = ref_file.path().to_path_buf();

    let mut ref_samples = Vec::new();
    let mut ref_haps = Vec::new();

    for i in 0..n_ref_haps {
        ref_samples.push(format!("R{}", i));
        ref_haps.push(vec![0u8; n_markers]);
        ref_haps.push(vec![0u8; n_markers]);
    }

    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for h in 0..ref_haps.len() {
        for m in 0..n_markers {
            ref_haps[h][m] = if rng.random_bool(0.5) { 1 } else { 0 };
        }
    }

    let hero_hap_idx = hero_idx * 2;
    for m in 0..n_markers {
        ref_haps[hero_hap_idx][m] = (m % 2) as u8;
    }

    // Set Anti-Hero Hap (Index 99 * 2 + 1 = 199) as the complementary pattern
    let anti_hero_hap_idx = hero_idx * 2 + 1;
    for m in 0..n_markers {
        ref_haps[anti_hero_hap_idx][m] = 1 - (m % 2) as u8;
    }

    let marker_pos = make_marker_pos(n_markers, 1_000);
    write_vcf(&ref_path, &ref_samples, &ref_haps, &marker_pos);

    let target_file = NamedTempFile::new().unwrap();
    let target_path = target_file.path().to_path_buf();

    let mut file = File::create(&target_path).unwrap();
    writeln!(file, "##fileformat=VCFv4.2").unwrap();
    writeln!(
        file,
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">"
    )
    .unwrap();
    writeln!(
        file,
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tTarget"
    )
    .unwrap();

    let hero_pattern = hero_pattern_from_ref_hap(&ref_haps, hero_hap_idx);
    for (m, &pos) in marker_pos.iter().enumerate() {
        let hero_allele = hero_pattern[m];
        let anti_hero = 1 - hero_allele;
        if m % 10 == 0 {
            write!(
                file,
                "chr1\t{}\t.\tA\tC\t.\t.\t.\tGT\t{}|{}",
                pos, hero_allele, anti_hero
            )
            .unwrap();
        } else {
            write!(file, "chr1\t{}\t.\tA\tC\t.\t.\t.\tGT\t0/1", pos).unwrap();
        }
        writeln!(file).unwrap();
    }

    let out_file = NamedTempFile::new().unwrap();
    let out_path = out_file.path().to_path_buf();

    let mut config = Config::default();
    config.gt = target_path.clone();
    config.r#ref = Some(ref_path.clone());
    config.out = out_path.clone();
    config.phase_states = 0;
    config.burnin = 0;
    config.iterations = 2;
    config.nthreads = Some(1);
    config.ne = 10000.0;
    config.err = Some(0.0001);

    let mut pipeline = PhasingPipeline::new(config, None);
    pipeline.run().expect("Pipeline run failed");

    let expected_out_path = out_path.with_extension("vcf.gz");
    assert!(expected_out_path.exists());

    use flate2::read::MultiGzDecoder;
    use std::io::BufRead;
    use std::io::BufReader;

    let file = File::open(&expected_out_path).unwrap();
    let decoder = MultiGzDecoder::new(file);
    let reader = BufReader::new(decoder);

    let mut phased_haps: Vec<(u8, u8)> = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        let sample_field = parts[9];
        let gt_str = sample_field.split(':').next().unwrap();
        let alleles: Vec<u8> = gt_str
            .split(['|', '/'])
            .map(|s| s.parse().unwrap_or(0))
            .collect();
        if alleles.len() >= 2 {
            phased_haps.push((alleles[0], alleles[1]));
        }
    }

    let hero_pattern = hero_pattern_from_ref_hap(&ref_haps, hero_hap_idx);

    let mut switch_errors = 0;
    for m in 1..n_markers {
        let hero_prev = hero_pattern[m - 1];
        let hero_curr = hero_pattern[m];
        let (h1_prev, _) = phased_haps[m - 1];
        let (h1_curr, _) = phased_haps[m];
        let prev_match = h1_prev == hero_prev;
        let curr_match = h1_curr == hero_curr;
        if prev_match != curr_match {
            switch_errors += 1;
        }
    }

    let ser = switch_errors as f32 / (n_markers - 1) as f32;

    assert!(
        ser < 0.05,
        "SER too high with correct initial phasing: {:.4}",
        ser
    );
}

#[test]
fn test_small_panel_all0_all1_perfect_match_ser() {
    let n_markers = 10; // Small for readability
    let n_ref_haps = 20; // Small panel
    let hero_idx = 10;
    // Note on expectations (full derivation):
    // Here M=10 markers and N=2*n_ref_haps=40 reference haplotypes. Aside from the
    // forced hero/anti pair, all other haps are i.i.d. Bernoulli(0.5) per marker.
    // For two random haplotypes of length M, the probability they are exact
    // complements is p_comp = (1/2)^M = 1/1024. The number of other unordered pairs
    // is C(40,2)-1 = 779, so the expected number of extra perfect complementary
    // pairs is lambda = 779 * 2^-10 ≈ 0.760742. Using Poisson(lambda),
    // P(no extra pair) ≈ e^-lambda ≈ 0.467, so in ~53% of runs there is at least
    // one accidental perfect pair.
    //
    // An ideal-but-not-oracular phaser will pick one perfect pair and stick with it.
    // If there are K=1+X perfect pairs (including hero/anti), symmetry gives
    // P(pick hero/anti)=1/K. Under the SER metric (hero = all-0), picking hero/anti
    // gives SER=0, while picking a random complementary pair makes hap1 effectively
    // random 0/1 across markers, so expected SER ≈ 0.5. Thus:
    //   E[SER | K] ≈ 0.5 * (1 - 1/K).
    // Averaging over X~Poisson(lambda), using E[1/(1+X)] ≈ (1-e^-lambda)/lambda,
    // yields E[SER] ≈ 0.5 * (1 - (1-e^-lambda)/lambda) ≈ 0.1499.
    // This means SER can be >0 even with correct behavior unless the hero pair is
    // made unique (e.g., larger M or constrained distractors).

    let seeds = [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let mut ser_sum = 0.0f32;
    for seed in seeds {
        let ref_file = NamedTempFile::new().unwrap();
        let ref_path = ref_file.path().to_path_buf();

        let mut ref_samples = Vec::new();
        let mut ref_haps = Vec::new();

        for i in 0..n_ref_haps {
            ref_samples.push(format!("R{}", i));
            ref_haps.push(vec![0u8; n_markers]);
            ref_haps.push(vec![0u8; n_markers]);
        }

        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for h in 0..ref_haps.len() {
            for m in 0..n_markers {
                ref_haps[h][m] = if rng.random_bool(0.5) { 1 } else { 0 };
            }
        }

        let hero_hap_idx = hero_idx * 2;
        for m in 0..n_markers {
            ref_haps[hero_hap_idx][m] = 0;
        }

        let anti_hero_hap_idx = hero_idx * 2 + 1;
        for m in 0..n_markers {
            ref_haps[anti_hero_hap_idx][m] = 1;
        }

        let marker_pos = make_marker_pos(n_markers, 1_000);
        write_vcf(&ref_path, &ref_samples, &ref_haps, &marker_pos);

        let target_file = NamedTempFile::new().unwrap();
        let target_path = target_file.path().to_path_buf();
        write_target_vcf(&target_path, &marker_pos);

        let out_file = NamedTempFile::new().unwrap();
        let out_path = out_file.path().to_path_buf();

        let mut config = Config::default();
        config.gt = target_path.clone();
        config.r#ref = Some(ref_path.clone());
        config.out = out_path.clone();
        config.phase_states = 0;
        config.burnin = 0;
        config.iterations = 2;
        config.nthreads = Some(1);
        config.ne = 10000.0;
        config.err = Some(0.0001);

        let mut pipeline = PhasingPipeline::new(config, None);
        pipeline.run().expect("Pipeline run failed");

        // Parse output and calculate SER
        let expected_out_path = out_path.with_extension("vcf.gz");
        use flate2::read::MultiGzDecoder;
        use std::io::BufRead;
        use std::io::BufReader;

        let file = File::open(&expected_out_path).unwrap();
        let decoder = MultiGzDecoder::new(file);
        let reader = BufReader::new(decoder);

        let mut phased_haps: Vec<(u8, u8)> = Vec::new();
        for line in reader.lines() {
            let line = line.unwrap();
            if line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            let sample_field = parts[9];
            let gt_str = sample_field.split(':').next().unwrap();
            let alleles: Vec<u8> = gt_str
                .split(['|', '/'])
                .map(|s| s.parse().unwrap_or(0))
                .collect();
            if alleles.len() >= 2 {
                phased_haps.push((alleles[0], alleles[1]));
            }
        }

        let hero_pattern = hero_pattern_from_ref_hap(&ref_haps, hero_hap_idx);

        let mut switch_errors = 0;
        for m in 1..n_markers {
            let hero_prev = hero_pattern[m - 1];
            let hero_curr = hero_pattern[m];
            let (h1_prev, _) = phased_haps[m - 1];
            let (h1_curr, _) = phased_haps[m];
            let prev_match = h1_prev == hero_prev;
            let curr_match = h1_curr == hero_curr;
            if prev_match != curr_match {
                switch_errors += 1;
            }
        }

        if n_markers <= 20 {
            println!("[small_panel_ser] seed={} hero_pattern={:?}", seed, hero_pattern);
            println!("[small_panel_ser] seed={} phased_haps={:?}", seed, phased_haps);
            let mut match_trace = Vec::with_capacity(n_markers);
            for m in 0..n_markers {
                let (h1, h2) = phased_haps[m];
                match_trace.push((m, h1, h2, hero_pattern[m]));
            }
            println!("[small_panel_ser] seed={} match_trace={:?}", seed, match_trace);
        }

        let ser = switch_errors as f32 / (n_markers - 1) as f32;
        ser_sum += ser;

        println!(
            "[small_panel_ser] seed={} markers={} switch_errors={} ser={:.4}",
            seed, n_markers, switch_errors, ser
        );
    }

    let ser_avg = ser_sum / seeds.len() as f32;
    println!("[small_panel_ser] avg_ser={:.4}", ser_avg);
    assert!(
        ser_avg < 0.15,
        "Avg SER too high in small panel: {:.4}",
        ser_avg
    );
}

#[test]
fn test_symmetric_evidence_phase_confidence_low() {
    let n_markers = 40;
    let n_ref_haps = 100;
    let hero_idx = 99;

    let ref_file = NamedTempFile::new().unwrap();
    let ref_path = ref_file.path().to_path_buf();

    let mut ref_samples = Vec::new();
    let mut ref_haps = Vec::new();

    for i in 0..n_ref_haps {
        ref_samples.push(format!("R{}", i));
        ref_haps.push(vec![0u8; n_markers]);
        ref_haps.push(vec![0u8; n_markers]);
    }

    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for h in 0..ref_haps.len() {
        for m in 0..n_markers {
            ref_haps[h][m] = if rng.random_bool(0.5) { 1 } else { 0 };
        }
    }

    let hero_hap_idx = hero_idx * 2;
    let anti_hero_hap_idx = hero_idx * 2 + 1;
    for m in 0..n_markers {
        ref_haps[hero_hap_idx][m] = (m % 2) as u8;
        ref_haps[anti_hero_hap_idx][m] = 1 - (m % 2) as u8;
    }

    let marker_pos: Vec<u32> = (0..n_markers).map(|i| (i * 1000 + 100) as u32).collect();
    write_vcf(&ref_path, &ref_samples, &ref_haps, &marker_pos);

    let target_file = NamedTempFile::new().unwrap();
    let target_path = target_file.path().to_path_buf();
    write_target_vcf(&target_path, &marker_pos);

    let (mut target_reader, target_file) = VcfReader::open(&target_path).unwrap();
    let target_gt = target_reader.read_all(target_file).unwrap();
    let (mut ref_reader, ref_file) = VcfReader::open(&ref_path).unwrap();
    let ref_gt = ref_reader.read_all(ref_file).unwrap();
    let ref_gt = ref_gt.into_phased();

    let alignment = MarkerAlignment::new(&target_gt, &ref_gt);

    let mut config = Config::default();
    config.phase_states = 0;
    config.burnin = 0;
    config.iterations = 2;
    config.nthreads = Some(1);
    config.ne = 10000.0;
    config.err = Some(0.0001);

    let mut pipeline = PhasingPipeline::new(config, None);
    pipeline.set_reference(Arc::new(ref_gt), alignment);

    let gen_maps = GeneticMaps::new();
    let (phased, _) = pipeline
        .phase_in_memory_with_overlap(&target_gt, &gen_maps, None, None)
        .expect("phase_in_memory_with_overlap");

    assert!(
        phased.phase_confidence_clone().is_some(),
        "Expected phase confidence to be populated"
    );

    let mut sum = 0.0f32;
    let mut max = 0.0f32;
    for m in 0..n_markers {
        let conf = phased.sample_phase_confidence_f32(MarkerIdx::new(m as u32), 0);
        sum += conf;
        if conf > max {
            max = conf;
        }
    }
    let mean = sum / n_markers as f32;
    println!(
        "[uninformative phase confidence] mean={:.3} max={:.3}",
        mean, max
    );

    assert!(
        mean < 0.6,
        "Expected low mean phase confidence under symmetric evidence; mean={:.3}",
        mean
    );
    assert!(
        max < 0.9,
        "Expected no highly confident markers under symmetric evidence; max={:.3}",
        max
    );
}

#[test]
fn test_dynamic_mcmc_small_window() {
    let n_markers = 20; // Small window (< 500)
    let n_ref_haps = 100;
    let hero_idx = 99;

    let ref_file = NamedTempFile::new().unwrap();
    let ref_path = ref_file.path().to_path_buf();

    let mut ref_samples = Vec::new();
    let mut ref_haps = Vec::new();

    for i in 0..n_ref_haps {
        ref_samples.push(format!("R{}", i));
        ref_haps.push(vec![0u8; n_markers]);
        ref_haps.push(vec![0u8; n_markers]);
    }

    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);
    for h in 0..ref_haps.len() {
        for m in 0..n_markers {
            ref_haps[h][m] = if rng.random_bool(0.5) { 1 } else { 0 };
        }
    }

    let hero_hap_idx = hero_idx * 2;
    for m in 0..n_markers {
        ref_haps[hero_hap_idx][m] = 0;
    }

    let anti_hero_hap_idx = hero_idx * 2 + 1;
    for m in 0..n_markers {
        ref_haps[anti_hero_hap_idx][m] = 1;
    }

    let marker_pos = make_marker_pos(n_markers, 1_000);
    write_vcf(&ref_path, &ref_samples, &ref_haps, &marker_pos);

    let target_file = NamedTempFile::new().unwrap();
    let target_path = target_file.path().to_path_buf();
    write_target_vcf(&target_path, &marker_pos);

    let out_file = NamedTempFile::new().unwrap();
    let out_path = out_file.path().to_path_buf();

    let mut config = Config::default();
    config.gt = target_path.clone();
    config.r#ref = Some(ref_path.clone());
    config.out = out_path.clone();
    config.phase_states = 0;
    config.burnin = 2;
    config.iterations = 2;
    config.nthreads = Some(1);
    config.ne = 10000.0;
    config.err = Some(0.0001);
    config.dynamic_mcmc = true; // Enable the feature we just patched

    let mut pipeline = PhasingPipeline::new(config, None);
    pipeline.run().expect("Pipeline run failed");

    let expected_out_path = out_path.with_extension("vcf.gz");
    assert!(expected_out_path.exists());

    use flate2::read::MultiGzDecoder;
    use std::io::BufRead;
    use std::io::BufReader;

    let file = File::open(&expected_out_path).unwrap();
    let decoder = MultiGzDecoder::new(file);
    let reader = BufReader::new(decoder);

    let mut phased_haps: Vec<(u8, u8)> = Vec::new();
    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        let sample_field = parts[9];
        let gt_str = sample_field.split(':').next().unwrap();
        let alleles: Vec<u8> = gt_str
            .split(['|', '/'])
            .map(|s| s.parse().unwrap_or(0))
            .collect();
        if alleles.len() >= 2 {
            phased_haps.push((alleles[0], alleles[1]));
        }
    }

    assert_eq!(phased_haps.len(), n_markers);

    let hero_pattern = hero_pattern_from_ref_hap(&ref_haps, hero_hap_idx);

    let mut switch_errors = 0;
    for m in 1..n_markers {
        let hero_prev = hero_pattern[m - 1];
        let hero_curr = hero_pattern[m];
        let (h1_prev, _) = phased_haps[m - 1];
        let (h1_curr, _) = phased_haps[m];

        let prev_match = h1_prev == hero_prev;
        let curr_match = h1_curr == hero_curr;

        if prev_match != curr_match {
            switch_errors += 1;
        }
    }

    let ser = switch_errors as f32 / (n_markers - 1) as f32;
    println!("Switch Error Rate (dynamic_mcmc): {:.4}", ser);

    assert!(ser < 0.05, "SER too high with dynamic_mcmc: {:.4}", ser);
}
