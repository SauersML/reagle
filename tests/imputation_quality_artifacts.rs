use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::io::{BufRead, BufReader};
use std::process::{Child, Stdio};

use serde_json::Value;

const BEAGLE_URL: &str = "https://faculty.washington.edu/browning/beagle/beagle.27Feb25.75f.jar";
const WORKFLOW_NAME: &str = "imputation_quality_report.yml";
const RUN_SCAN_LIMIT: &str = "200";

#[test]
fn kat_23andme_artifact_imputation_quality_rust_beats_beagle_on_dosage_corr() {
    ensure_cmd_exists("gh");
    ensure_cmd_exists("bcftools");
    ensure_cmd_exists("java");
    ensure_cmd_exists("curl");

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let test_tmp_root = manifest_dir.join(".tmp").join("imputation_quality_artifacts");
    fs::create_dir_all(&test_tmp_root).expect("create test tmp root");
    let work_dir = tempfile::Builder::new()
        .prefix("run-")
        .tempdir_in(&test_tmp_root)
        .expect("create temp work dir");
    let dataset_dir = work_dir.path().join("dataset");
    fs::create_dir_all(&dataset_dir).expect("create dataset dir");

    let repo = detect_repo_slug(&manifest_dir).expect("repo slug resolution failed");
    let (reference_dir, outputs_dir) = download_latest_quality_artifacts(&repo, work_dir.path());
    stage_required_inputs(&reference_dir, &outputs_dir, work_dir.path());

    let ref_vcf = dataset_dir.join("ref.vcf.gz");
    let input_vcf = dataset_dir.join("input.vcf.gz");
    let truth_vcf = dataset_dir.join("truth.vcf.gz");
    assert!(ref_vcf.exists(), "missing ref.vcf.gz after staging");
    assert!(input_vcf.exists(), "missing input.vcf.gz after staging");
    assert!(truth_vcf.exists(), "missing truth.vcf.gz after staging");

    let reagle_bin = manifest_dir.join("target").join("release").join("reagle");
    assert!(
        reagle_bin.exists(),
        "reagle binary missing at {} (build with `cargo test --release --test imputation_quality_artifacts` or `cargo build --release`)",
        reagle_bin.display()
    );

    let beagle_jar = work_dir.path().join("beagle.jar");
    run(
        "curl",
        [
            "-L",
            "-o",
            beagle_jar.to_str().expect("beagle path utf8"),
            BEAGLE_URL,
        ],
    );
    assert!(beagle_jar.exists(), "failed to download beagle.jar");

    let reagle_out_prefix = work_dir.path().join("reagle_out");
    let beagle_out_prefix = work_dir.path().join("beagle_out");

    run(
        reagle_bin
            .to_str()
            .expect("reagle binary path should be valid utf8"),
        [
            "--ref",
            ref_vcf.to_str().expect("ref path utf8"),
            "--target",
            input_vcf.to_str().expect("input path utf8"),
            "--out",
            reagle_out_prefix
                .to_str()
                .expect("reagle out prefix path utf8"),
        ],
    );

    run(
        "java",
        [
            "-Xmx6g",
            "-jar",
            beagle_jar.to_str().expect("beagle path utf8"),
            &format!("ref={}", ref_vcf.display()),
            &format!("gt={}", input_vcf.display()),
            &format!("out={}", beagle_out_prefix.display()),
            "chrom=chr22",
            "nthreads=4",
            "gp=true",
        ],
    );

    let reagle_vcf = reagle_out_prefix.with_extension("vcf.gz");
    let beagle_vcf = beagle_out_prefix.with_extension("vcf.gz");
    assert!(reagle_vcf.exists(), "missing reagle output {}", reagle_vcf.display());
    assert!(beagle_vcf.exists(), "missing beagle output {}", beagle_vcf.display());

    harmonize_sample_names(work_dir.path(), &[&truth_vcf, &reagle_vcf, &beagle_vcf]);

    let (reagle_r2, reagle_n) = compute_dosage_r2_streaming(&truth_vcf, &reagle_vcf);
    let (beagle_r2, beagle_n) = compute_dosage_r2_streaming(&truth_vcf, &beagle_vcf);

    println!(
        "Native dosage r_squared | Reagle: {:.6} (n={}) | Beagle: {:.6} (n={})",
        reagle_r2, reagle_n, beagle_r2, beagle_n
    );

    assert!(
        reagle_r2 > beagle_r2,
        "Expected Reagle r_squared > Beagle r_squared; reagle={:.6}, beagle={:.6}",
        reagle_r2, beagle_r2
    );
}

fn ensure_cmd_exists(cmd: &str) {
    let status = Command::new("which")
        .arg(cmd)
        .status()
        .expect("which command failed");
    assert!(status.success(), "required command not found in PATH: {cmd}");
}

fn run(cmd: &str, args: impl IntoIterator<Item = impl AsRef<OsStr>>) {
    let output = Command::new(cmd)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to execute {cmd}: {e}"));
    if !output.status.success() {
        panic!(
            "command failed: {}\nstdout:\n{}\nstderr:\n{}",
            cmd,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn run_capture(cmd: &str, args: impl IntoIterator<Item = impl AsRef<OsStr>>) -> String {
    let output = Command::new(cmd)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to execute {cmd}: {e}"));
    if !output.status.success() {
        panic!(
            "command failed: {}\nstdout:\n{}\nstderr:\n{}",
            cmd,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8(output.stdout).expect("stdout is not utf8")
}

fn detect_repo_slug(repo_root: &Path) -> Option<String> {
    let output = Command::new("git")
        .current_dir(repo_root)
        .args(["remote", "get-url", "origin"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let url = String::from_utf8(output.stdout).ok()?;
    parse_repo_slug(url.trim())
}

fn parse_repo_slug(remote: &str) -> Option<String> {
    let trimmed = remote.strip_suffix(".git").unwrap_or(remote);
    if let Some(idx) = trimmed.find("github.com/") {
        let slug = &trimmed[idx + "github.com/".len()..];
        if slug.split('/').count() == 2 {
            return Some(slug.to_string());
        }
    }
    if let Some(idx) = trimmed.find(':') {
        let slug = &trimmed[idx + 1..];
        if slug.split('/').count() == 2 {
            return Some(slug.to_string());
        }
    }
    None
}

fn download_latest_quality_artifacts(repo: &str, out_dir: &Path) -> (PathBuf, PathBuf) {
    let runs_json = run_capture(
        "gh",
        [
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            WORKFLOW_NAME,
            "--limit",
            RUN_SCAN_LIMIT,
            "--json",
            "databaseId,status,conclusion",
        ],
    );
    let runs: Value = serde_json::from_str(&runs_json).expect("parse run list json");
    let runs_arr = runs
        .as_array()
        .expect("expected array from gh run list --json");

    let mut reference_run_id: Option<i64> = None;
    for run_entry in runs_arr {
        let status = run_entry
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let conclusion = run_entry
            .get("conclusion")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if status != "completed" || (conclusion != "success" && conclusion != "failure") {
            continue;
        }
        let run_id = match run_entry.get("databaseId").and_then(Value::as_i64) {
            Some(v) => v,
            None => continue,
        };
        let artifacts_json = run_capture(
            "gh",
            [
                "api",
                &format!("/repos/{repo}/actions/runs/{run_id}/artifacts"),
            ],
        );
        let artifacts: Value = serde_json::from_str(&artifacts_json).expect("parse artifacts json");
        let artifact_list = artifacts
            .get("artifacts")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut has_reference = false;
        for artifact in artifact_list {
            let Some(name) = artifact.get("name").and_then(Value::as_str) else {
                continue;
            };
            if name == "reference-panel" {
                has_reference = true;
            }
        }
        if has_reference {
            reference_run_id = Some(run_id);
            break;
        }
    }
    let ref_run_id = reference_run_id.expect("no recent run contains reference-panel artifact");
    let reference_dir = out_dir.join(format!("reference-run-{ref_run_id}"));
    fs::create_dir_all(&reference_dir).expect("create reference artifact dir");
    run(
        "gh",
        [
            "run",
            "download",
            "--repo",
            repo,
            &ref_run_id.to_string(),
            "-n",
            "reference-panel",
            "-D",
            reference_dir.to_str().expect("reference_dir utf8"),
        ],
    );

    for run_entry in runs_arr {
        let status = run_entry
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let conclusion = run_entry
            .get("conclusion")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if status != "completed" || (conclusion != "success" && conclusion != "failure") {
            continue;
        }
        let run_id = match run_entry.get("databaseId").and_then(Value::as_i64) {
            Some(v) => v,
            None => continue,
        };
        let artifacts_json = run_capture(
            "gh",
            [
                "api",
                &format!("/repos/{repo}/actions/runs/{run_id}/artifacts"),
            ],
        );
        let artifacts: Value = serde_json::from_str(&artifacts_json).expect("parse artifacts json");
        let artifact_list = artifacts
            .get("artifacts")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let mut outputs_kat_name: Option<String> = None;
        for artifact in artifact_list {
            let Some(name) = artifact.get("name").and_then(Value::as_str) else {
                continue;
            };
            if name.starts_with("outputs-Kat-") {
                outputs_kat_name = Some(name.to_string());
                break;
            }
        }
        let Some(outputs_name) = outputs_kat_name else {
            continue;
        };
        let candidate_dir = out_dir.join(format!("outputs-run-{run_id}"));
        fs::create_dir_all(&candidate_dir).expect("create outputs artifact dir");
        run(
            "gh",
            [
                "run",
                "download",
                "--repo",
                repo,
                &run_id.to_string(),
                "-n",
                outputs_name.as_str(),
                "-D",
                candidate_dir.to_str().expect("candidate_dir utf8"),
            ],
        );

        let has_input = find_file_recursive(&candidate_dir, "target.vcf.gz").is_some()
            || find_file_recursive(&candidate_dir, "input.vcf.gz").is_some();
        let has_truth = find_file_recursive(&candidate_dir, "truth.vcf.gz").is_some();

        if has_input && has_truth {
            return (reference_dir, candidate_dir);
        }

        let _ = fs::remove_dir_all(&candidate_dir);
    }
    panic!(
        "no recent {} run had outputs-Kat-* containing target/input.vcf.gz + truth.vcf.gz",
        WORKFLOW_NAME
    );
}

fn stage_required_inputs(reference_dir: &Path, outputs_dir: &Path, work_dir: &Path) {
    let dataset_dir = work_dir.join("dataset");
    fs::create_dir_all(&dataset_dir).expect("create dataset dir");

    let ref_src = find_file_recursive(reference_dir, "ref.vcf.gz")
        .expect("ref.vcf.gz not found in downloaded artifacts");
    copy_with_index(&ref_src, &dataset_dir.join("ref.vcf.gz"));

    let input_src = find_file_recursive(outputs_dir, "target.vcf.gz")
        .or_else(|| find_file_recursive(outputs_dir, "input.vcf.gz"))
        .expect("target.vcf.gz/input.vcf.gz not found in downloaded artifacts");
    copy_with_index(&input_src, &dataset_dir.join("input.vcf.gz"));

    let truth_src = find_file_recursive(outputs_dir, "truth.vcf.gz")
        .expect("truth.vcf.gz not found in downloaded artifacts");
    copy_with_index(&truth_src, &dataset_dir.join("truth.vcf.gz"));
}

fn copy_with_index(src_vcf: &Path, dst_vcf: &Path) {
    fs::copy(src_vcf, dst_vcf).unwrap_or_else(|e| {
        panic!(
            "failed copying {} -> {}: {e}",
            src_vcf.display(),
            dst_vcf.display()
        )
    });

    let index_candidates = [
        format!("{}.csi", src_vcf.display()),
        format!("{}.tbi", src_vcf.display()),
    ];
    for candidate in &index_candidates {
        let path = Path::new(candidate);
        if path.exists() {
            let ext = path
                .extension()
                .and_then(OsStr::to_str)
                .expect("index extension");
            let dst = dst_vcf.with_extension(format!("vcf.gz.{ext}"));
            fs::copy(path, &dst).unwrap_or_else(|e| {
                panic!(
                    "failed copying index {} -> {}: {e}",
                    path.display(),
                    dst.display()
                )
            });
            return;
        }
    }
}

fn find_file_recursive(root: &Path, filename: &str) -> Option<PathBuf> {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = match fs::read_dir(&dir) {
            Ok(v) => v,
            Err(_) => continue,
        };
        for entry in entries {
            let Ok(entry) = entry else { continue };
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path
                .file_name()
                .and_then(OsStr::to_str)
                .map(|n| n == filename)
                .unwrap_or(false)
            {
                return Some(path);
            }
        }
    }
    None
}

#[derive(Clone)]
struct TruthSite {
    chrom: String,
    pos: i64,
    ref_allele: String,
    alt_allele: String,
    gt: String,
}

#[derive(Clone)]
struct ImputedSite {
    chrom: String,
    pos: i64,
    ref_allele: String,
    alt_allele: String,
    gt: String,
    ds: Option<f64>,
    gp: Option<Vec<f64>>,
}

fn next_truth_site(reader: &mut BufReader<std::process::ChildStdout>) -> Option<TruthSite> {
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).expect("read truth query output");
        if n == 0 {
            return None;
        }
        let fields: Vec<&str> = line.trim_end().split('\t').collect();
        if fields.len() < 5 {
            continue;
        }
        let pos = match fields[1].parse::<i64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        return Some(TruthSite {
            chrom: fields[0].to_string(),
            pos,
            ref_allele: fields[2].to_string(),
            alt_allele: fields[3].to_string(),
            gt: fields[4].to_string(),
        });
    }
}

fn next_imputed_site(reader: &mut BufReader<std::process::ChildStdout>) -> Option<ImputedSite> {
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader.read_line(&mut line).expect("read imputed query output");
        if n == 0 {
            return None;
        }
        let fields: Vec<&str> = line.trim_end().split('\t').collect();
        if fields.len() < 5 {
            continue;
        }
        let pos = match fields[1].parse::<i64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let (gt, ds, gp) = parse_imputed_sample(fields[4]);
        return Some(ImputedSite {
            chrom: fields[0].to_string(),
            pos,
            ref_allele: fields[2].to_string(),
            alt_allele: fields[3].to_string(),
            gt,
            ds,
            gp,
        });
    }
}

fn parse_imputed_sample(sample: &str) -> (String, Option<f64>, Option<Vec<f64>>) {
    let parts: Vec<&str> = sample.split(':').collect();
    let gt = parts.first().unwrap_or(&".").to_string();
    let ds = parts.get(1).and_then(|v| parse_ds(v));
    let gp = parts.get(2).and_then(|v| parse_gp(v));
    (gt, ds, gp)
}

fn parse_ds(v: &str) -> Option<f64> {
    if v.is_empty() || v == "." {
        return None;
    }
    if v.contains(',') {
        let mut sum = 0.0;
        for tok in v.split(',') {
            if tok.is_empty() || tok == "." {
                continue;
            }
            let value = tok.parse::<f64>().ok()?;
            sum += value;
        }
        return Some(sum);
    }
    v.parse::<f64>().ok()
}

fn parse_gp(v: &str) -> Option<Vec<f64>> {
    if v.is_empty() || v == "." {
        return None;
    }
    let mut out = Vec::new();
    for tok in v.split(',') {
        if tok.is_empty() || tok == "." {
            return None;
        }
        out.push(tok.parse::<f64>().ok()?);
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn gt_to_nonref_dosage(gt: &str) -> Option<f64> {
    if gt.is_empty() || gt == "." || gt == "./." || gt == ".|." {
        return None;
    }
    let sep = if gt.contains('|') { '|' } else { '/' };
    let parts: Vec<&str> = gt.split(sep).collect();
    if parts.len() != 2 {
        return None;
    }
    let a = parts[0].parse::<i32>().ok()?;
    let b = parts[1].parse::<i32>().ok()?;
    Some((if a != 0 { 1.0 } else { 0.0 }) + (if b != 0 { 1.0 } else { 0.0 }))
}

fn ds_from_gp_biallelic(gp: &[f64]) -> Option<f64> {
    if gp.len() < 3 {
        return None;
    }
    let p0 = gp[0];
    let p1 = gp[1];
    let p2 = gp[2];
    let sum = p0 + p1 + p2;
    if sum <= 0.0 {
        return None;
    }
    Some((p1 + 2.0 * p2) / sum)
}

fn compare_site_key(t: &TruthSite, i: &ImputedSite) -> std::cmp::Ordering {
    match t.chrom.cmp(&i.chrom) {
        std::cmp::Ordering::Equal => t.pos.cmp(&i.pos),
        other => other,
    }
}

fn spawn_bcftools_query(vcf: &Path, format: &str) -> (Child, BufReader<std::process::ChildStdout>) {
    let mut child = Command::new("bcftools")
        .args(["query", "-f", format, vcf.to_str().expect("vcf path utf8")])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn bcftools query");
    let stdout = child.stdout.take().expect("bcftools stdout missing");
    (child, BufReader::new(stdout))
}

fn wait_child_ok(child: Child, label: &str) {
    let output = child.wait_with_output().expect("wait on child process");
    assert!(
        output.status.success(),
        "{} failed\nstdout:\n{}\nstderr:\n{}",
        label,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn compute_dosage_r2_streaming(truth_vcf: &Path, imputed_vcf: &Path) -> (f64, usize) {
    let (truth_child, mut truth_reader) =
        spawn_bcftools_query(truth_vcf, "%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n");
    let (imputed_child, mut imputed_reader) =
        spawn_bcftools_query(imputed_vcf, "%CHROM\t%POS\t%REF\t%ALT[\t%GT:%DS:%GP]\n");

    let mut sum_t = 0.0_f64;
    let mut sum_i = 0.0_f64;
    let mut sum_ti = 0.0_f64;
    let mut sum_tt = 0.0_f64;
    let mut sum_ii = 0.0_f64;
    let mut n: usize = 0;

    let mut t_site = next_truth_site(&mut truth_reader);
    let mut i_site = next_imputed_site(&mut imputed_reader);

    while let (Some(t), Some(i)) = (&t_site, &i_site) {
        match compare_site_key(t, i) {
            std::cmp::Ordering::Less => t_site = next_truth_site(&mut truth_reader),
            std::cmp::Ordering::Greater => i_site = next_imputed_site(&mut imputed_reader),
            std::cmp::Ordering::Equal => {
                let t_cur = t.clone();
                let i_cur = i.clone();

                let mut i_dos = i_cur
                    .ds
                    .or_else(|| i_cur.gp.as_ref().and_then(|g| ds_from_gp_biallelic(g)))
                    .or_else(|| gt_to_nonref_dosage(&i_cur.gt));
                if let (Some(t_dos), Some(mut i_dos_val)) = (gt_to_nonref_dosage(&t_cur.gt), i_dos.take()) {
                    if t_cur.ref_allele != i_cur.ref_allele || t_cur.alt_allele != i_cur.alt_allele {
                        let biallelic_truth = !t_cur.alt_allele.contains(',');
                        let biallelic_imp = !i_cur.alt_allele.contains(',');
                        let swapped = biallelic_truth
                            && biallelic_imp
                            && t_cur.ref_allele == i_cur.alt_allele
                            && t_cur.alt_allele == i_cur.ref_allele;
                        if swapped {
                            i_dos_val = 2.0 - i_dos_val;
                        } else {
                            t_site = next_truth_site(&mut truth_reader);
                            i_site = next_imputed_site(&mut imputed_reader);
                            continue;
                        }
                    }
                    sum_t += t_dos;
                    sum_i += i_dos_val;
                    sum_ti += t_dos * i_dos_val;
                    sum_tt += t_dos * t_dos;
                    sum_ii += i_dos_val * i_dos_val;
                    n += 1;
                }
                t_site = next_truth_site(&mut truth_reader);
                i_site = next_imputed_site(&mut imputed_reader);
            }
        }
    }

    wait_child_ok(truth_child, "bcftools truth query");
    wait_child_ok(imputed_child, "bcftools imputed query");

    assert!(n > 1, "insufficient comparable points for r_squared (n={n})");
    let n_f = n as f64;
    let mean_t = sum_t / n_f;
    let mean_i = sum_i / n_f;
    let cov = sum_ti / n_f - mean_t * mean_i;
    let var_t = sum_tt / n_f - mean_t * mean_t;
    let var_i = sum_ii / n_f - mean_i * mean_i;
    assert!(
        var_t > 0.0 && var_i > 0.0,
        "non-positive variance for r_squared (var_t={var_t}, var_i={var_i}, n={n})"
    );
    let r = cov / (var_t * var_i).sqrt();
    (r * r, n)
}

fn harmonize_sample_names(work_dir: &Path, vcfs: &[&Path]) {
    let sample_file = work_dir.join("sample_name.txt");
    fs::write(&sample_file, "SAMPLE\n").expect("write sample name mapping");
    for vcf in vcfs {
        let tmp = PathBuf::from(format!("{}.tmp", vcf.display()));
        run(
            "bcftools",
            [
                "reheader",
                "-s",
                sample_file.to_str().expect("sample file path utf8"),
                vcf.to_str().expect("vcf path utf8"),
                "-o",
                tmp.to_str().expect("tmp path utf8"),
            ],
        );
        fs::rename(&tmp, vcf).unwrap_or_else(|e| {
            panic!(
                "failed replacing reheadered file {} -> {}: {e}",
                tmp.display(),
                vcf.display()
            )
        });
        run(
            "bcftools",
            ["index", "-f", vcf.to_str().expect("vcf path utf8")],
        );
    }
}
