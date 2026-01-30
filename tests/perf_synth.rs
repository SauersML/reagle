use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Command, Stdio};
use std::thread::sleep;
use std::time::{Duration, Instant};

use tempfile::TempDir;

fn write_synth_vcf(
    path: &std::path::Path,
    n_samples: usize,
    n_markers: usize,
    phased: bool,
    downsample_every: usize,
) -> std::io::Result<()> {
    let chrom = "chr22";
    let start_pos = 1_000_000u32;
    let step = 20u32;

    let mut out = BufWriter::new(std::fs::File::create(path)?);
    writeln!(out, "##fileformat=VCFv4.2")?;
    writeln!(out, "##contig=<ID={}>", chrom)?;
    writeln!(out, "##FORMAT=<ID=GT,Number=1,Type=String,Description=Genotype>")?;

    // Header samples
    write!(
        out,
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT"
    )?;
    for i in 0..n_samples {
        write!(out, "\tS{}", i)?;
    }
    writeln!(out)?;

    // Prebuild a deterministic genotype string for speed.
    let mut geno = String::new();
    for i in 0..n_samples {
        let gt = match i % 20 {
            0 => if phased { "0|1" } else { "0/1" },
            1 => if phased { "1|0" } else { "1/0" },
            2 => "1|1",
            _ => "0|0",
        };
        geno.push('\t');
        geno.push_str(gt);
    }

    let mut pos = start_pos;
    for m in 0..n_markers {
        if downsample_every > 1 && (m % downsample_every) != 0 {
            pos += step;
            continue;
        }
        write!(
            out,
            "{chrom}\t{pos}\t.\tA\tC\t.\tPASS\t.\tGT{geno}\n",
            chrom = chrom,
            pos = pos,
            geno = geno
        )?;
        pos += step;
    }
    out.flush()?;
    Ok(())
}

fn assert_vcf_fully_phased(path: &std::path::Path) {
    let file = std::fs::File::open(path).expect("open vcf");
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let line = line.expect("read vcf line");
        if line.starts_with('#') {
            continue;
        }
        if line.contains('/') {
            panic!("unphased genotype found in {}", path.display());
        }
    }
}

#[test]
fn synth_impute_runtime_under_2_min() {
    // Match CI-ish hap count, but only ~5% of chr22 markers to keep runtime bounded.
    let ref_samples = 3273; // 6546 haplotypes
    let target_samples = 818; // 1636 haplotypes
    let n_markers = 2762; // ~5% of 55,237
    let downsample_every = 20; // target ~5% markers

    let bin = std::env::var("REAGLE_BIN").ok().or_else(|| {
        let debug = std::path::PathBuf::from("target/debug/reagle");
        if debug.exists() {
            Some(debug.to_string_lossy().to_string())
        } else {
            let release = std::path::PathBuf::from("target/release/reagle");
            if release.exists() {
                Some(release.to_string_lossy().to_string())
            } else {
                None
            }
        }
    }).expect("reagle binary not found; set REAGLE_BIN or build target/debug/reagle");
    eprintln!("Using reagle binary: {}", bin);

    let tmp = TempDir::new().expect("tempdir");
    let ref_vcf = tmp.path().join("ref.vcf");
    let input_vcf = tmp.path().join("input.vcf");
    let out_prefix = tmp.path().join("reagle_out");

    write_synth_vcf(&ref_vcf, ref_samples, n_markers, true, 1).expect("write ref");
    // Use phased input to skip phasing; focus on imputation runtime.
    write_synth_vcf(&input_vcf, target_samples, n_markers, true, downsample_every)
        .expect("write input");
    assert_vcf_fully_phased(&input_vcf);

    let mut child = Command::new(bin)
        .arg("--ref")
        .arg(&ref_vcf)
        .arg("--gt")
        .arg(&input_vcf)
        .arg("--out")
        .arg(&out_prefix)
        .arg("--gp")
        .arg("--nthreads")
        .arg("4")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn reagle");

    let start = Instant::now();
    let timeout_secs = std::env::var("REAGLE_PERF_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120);
    let timeout = Duration::from_secs(timeout_secs);
    loop {
        if let Some(status) = child.try_wait().expect("try_wait") {
            assert!(status.success(), "reagle exited with {}", status);
            break;
        }
        if start.elapsed() > timeout {
            let _ = child.kill();
            panic!(
                "reagle exceeded {}s on synthetic CI-like workload",
                timeout_secs
            );
        }
        sleep(Duration::from_millis(200));
    }
}
