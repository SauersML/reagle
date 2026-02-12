use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

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
    ensure_cmd_exists("cargo");

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let work_dir = tempfile::tempdir().expect("create temp work dir");
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

    run_in(&manifest_dir, "cargo", ["build", "--release"]);
    let reagle_bin = manifest_dir.join("target").join("release").join("reagle");
    assert!(
        reagle_bin.exists(),
        "reagle binary missing after cargo build --release: {}",
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

    let input_sites = load_input_sites(&input_vcf);
    let truth_dosage = load_truth_dosage(&truth_vcf, &input_sites);
    let reagle_ds = load_ds_map(&reagle_vcf, &truth_dosage);
    let beagle_ds = load_ds_map(&beagle_vcf, &truth_dosage);

    let mut truth_vec = Vec::new();
    let mut reagle_vec = Vec::new();
    let mut beagle_vec = Vec::new();
    for (site, truth_val) in &truth_dosage {
        if let (Some(r), Some(b)) = (reagle_ds.get(site), beagle_ds.get(site)) {
            truth_vec.push(*truth_val);
            reagle_vec.push(*r);
            beagle_vec.push(*b);
        }
    }

    assert!(
        truth_vec.len() >= 500,
        "too few comparable imputed-only sites: {}",
        truth_vec.len()
    );

    let reagle_corr = pearson_corr(&truth_vec, &reagle_vec).expect("reagle corr");
    let beagle_corr = pearson_corr(&truth_vec, &beagle_vec).expect("beagle corr");

    println!(
        "Comparable sites: {} | Reagle corr: {:.6} | Beagle corr: {:.6}",
        truth_vec.len(),
        reagle_corr,
        beagle_corr
    );

    assert!(
        reagle_corr > beagle_corr,
        "Expected Reagle dosage correlation > Beagle on imputed-only sites; reagle={:.6}, beagle={:.6}",
        reagle_corr,
        beagle_corr
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

fn run_in(dir: &Path, cmd: &str, args: impl IntoIterator<Item = impl AsRef<OsStr>>) {
    let output = Command::new(cmd)
        .current_dir(dir)
        .args(args)
        .output()
        .unwrap_or_else(|e| panic!("failed to execute {cmd} in {}: {e}", dir.display()));
    if !output.status.success() {
        panic!(
            "command failed in {}: {}\nstdout:\n{}\nstderr:\n{}",
            dir.display(),
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

fn load_input_sites(input_vcf: &Path) -> HashSet<String> {
    let text = run_capture(
        "bcftools",
        [
            "query",
            "-f",
            "%CHROM\t%POS\n",
            input_vcf.to_str().expect("input path utf8"),
        ],
    );
    let mut sites = HashSet::new();
    for line in text.lines() {
        let mut fields = line.split('\t');
        let Some(chrom) = fields.next() else { continue };
        let Some(pos) = fields.next() else { continue };
        sites.insert(format!("{chrom}:{pos}"));
    }
    sites
}

fn load_truth_dosage(truth_vcf: &Path, input_sites: &HashSet<String>) -> HashMap<String, f64> {
    let text = run_capture(
        "bcftools",
        [
            "query",
            "-f",
            "%CHROM\t%POS[\t%GT]\n",
            truth_vcf.to_str().expect("truth path utf8"),
        ],
    );
    let mut truth = HashMap::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 3 {
            continue;
        }
        let site = format!("{}:{}", parts[0], parts[1]);
        if input_sites.contains(&site) {
            continue;
        }
        let Some(gt) = parts.get(2) else { continue };
        let Some(dosage) = gt_to_nonref_dosage(gt) else {
            continue;
        };
        truth.insert(site, dosage);
    }
    truth
}

fn load_ds_map(imputed_vcf: &Path, truth_sites: &HashMap<String, f64>) -> HashMap<String, f64> {
    let text = run_capture(
        "bcftools",
        [
            "query",
            "-f",
            "%CHROM\t%POS[\t%DS]\n",
            imputed_vcf.to_str().expect("imputed path utf8"),
        ],
    );
    let mut ds_map = HashMap::new();
    for line in text.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 3 {
            continue;
        }
        let site = format!("{}:{}", parts[0], parts[1]);
        if !truth_sites.contains_key(&site) {
            continue;
        }
        let Some(ds_str) = parts.get(2) else { continue };
        let Some(ds) = parse_ds(ds_str) else {
            continue;
        };
        ds_map.insert(site, ds);
    }
    ds_map
}

fn gt_to_nonref_dosage(gt: &str) -> Option<f64> {
    if gt == "." || gt == "./." || gt == ".|." {
        return None;
    }
    let sep = if gt.contains('|') { '|' } else { '/' };
    let mut it = gt.split(sep);
    let a = it.next()?;
    let b = it.next()?;
    let av = a.parse::<i32>().ok()?;
    let bv = b.parse::<i32>().ok()?;
    let left = if av != 0 { 1.0 } else { 0.0 };
    let right = if bv != 0 { 1.0 } else { 0.0 };
    Some(left + right)
}

fn parse_ds(ds: &str) -> Option<f64> {
    if ds.is_empty() || ds == "." {
        return None;
    }
    if ds.contains(',') {
        let mut sum = 0.0_f64;
        for part in ds.split(',') {
            if part.is_empty() || part == "." {
                continue;
            }
            let val = part.parse::<f64>().ok()?;
            sum += val;
        }
        return Some(sum);
    }
    ds.parse::<f64>().ok()
}

fn pearson_corr(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let n = x.len() as f64;
    let mut sum_x = 0.0_f64;
    let mut sum_y = 0.0_f64;
    let mut sum_xx = 0.0_f64;
    let mut sum_yy = 0.0_f64;
    let mut sum_xy = 0.0_f64;

    for (&a, &b) in x.iter().zip(y.iter()) {
        sum_x += a;
        sum_y += b;
        sum_xx += a * a;
        sum_yy += b * b;
        sum_xy += a * b;
    }

    let cov = sum_xy - (sum_x * sum_y / n);
    let var_x = sum_xx - (sum_x * sum_x / n);
    let var_y = sum_yy - (sum_y * sum_y / n);
    if var_x <= 0.0 || var_y <= 0.0 {
        return None;
    }
    Some(cov / (var_x.sqrt() * var_y.sqrt()))
}
