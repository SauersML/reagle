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

    let reagle_metrics_prefix = work_dir.path().join("reagle_py");
    let beagle_metrics_prefix = work_dir.path().join("beagle_py");
    run_python_calculate_metrics(
        &manifest_dir,
        &truth_vcf,
        &reagle_vcf,
        &reagle_metrics_prefix,
        &ref_vcf,
    );
    run_python_calculate_metrics(
        &manifest_dir,
        &truth_vcf,
        &beagle_vcf,
        &beagle_metrics_prefix,
        &ref_vcf,
    );

    let reagle_r2 = read_r_squared(&PathBuf::from(format!(
        "{}_metrics.json",
        reagle_metrics_prefix.display()
    )));
    let beagle_r2 = read_r_squared(&PathBuf::from(format!(
        "{}_metrics.json",
        beagle_metrics_prefix.display()
    )));

    println!(
        "Reagle r_squared: {:.6} | Beagle r_squared: {:.6}",
        reagle_r2, beagle_r2
    );

    assert!(
        reagle_r2 > beagle_r2,
        "Expected Reagle r_squared > Beagle r_squared; reagle={:.6}, beagle={:.6}",
        reagle_r2,
        beagle_r2
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

fn run_python_calculate_metrics(
    repo_root: &Path,
    truth_vcf: &Path,
    imputed_vcf: &Path,
    output_prefix: &Path,
    reference_vcf: &Path,
) {
    let script = r#"
import importlib.util
import pathlib
import sys

module_path = pathlib.Path(sys.argv[1])
truth_vcf = sys.argv[2]
imputed_vcf = sys.argv[3]
output_prefix = sys.argv[4]
reference_vcf = sys.argv[5]

spec = importlib.util.spec_from_file_location("integration_test_module", str(module_path))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.calculate_metrics(truth_vcf, imputed_vcf, output_prefix, reference_vcf=reference_vcf)
"#;

    run_in(
        repo_root,
        "python3",
        [
            "-c",
            script,
            "tests/integration_test.py",
            truth_vcf.to_str().expect("truth path utf8"),
            imputed_vcf.to_str().expect("imputed path utf8"),
            output_prefix.to_str().expect("output prefix path utf8"),
            reference_vcf.to_str().expect("reference path utf8"),
        ],
    );
}

fn read_r_squared(metrics_json: &Path) -> f64 {
    let raw = fs::read_to_string(metrics_json).unwrap_or_else(|e| {
        panic!(
            "failed reading metrics json {}: {e}",
            metrics_json.display()
        )
    });
    let parsed: Value = serde_json::from_str(&raw).expect("parse metrics json");
    parsed
        .get("r_squared")
        .and_then(Value::as_f64)
        .unwrap_or_else(|| panic!("r_squared missing or non-numeric in {}", metrics_json.display()))
}
