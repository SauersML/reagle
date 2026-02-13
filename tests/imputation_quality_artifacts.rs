use std::ffi::OsStr;
use std::fs;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::process::{Child, Stdio};
use std::time::UNIX_EPOCH;

use serde_json::Value;

const BEAGLE_URL: &str = "https://faculty.washington.edu/browning/beagle/beagle.27Feb25.75f.jar";
const WORKFLOW_NAME: &str = "imputation_quality_report.yml";
const RUN_SCAN_LIMIT: &str = "60";
const CHR22: &str = "chr22";
const METRIC_PROGRESS_INTERVAL: usize = 100_000;
const ARTIFACT_SELECTION_CACHE_FILE: &str = "artifact_selection.tsv";

#[derive(Clone, Copy)]
struct TestScope {
    label: &'static str,
    chr22_fraction_num: usize,
    chr22_fraction_den: usize,
}

impl TestScope {
    fn full() -> Self {
        Self {
            label: "full-chr22",
            chr22_fraction_num: 1,
            chr22_fraction_den: 1,
        }
    }

    fn fast_chr22_10pct() -> Self {
        Self {
            label: "fast-chr22-10pct",
            chr22_fraction_num: 1,
            chr22_fraction_den: 10,
        }
    }

    fn is_fractional(self) -> bool {
        self.chr22_fraction_num != self.chr22_fraction_den
    }
}

#[test]
fn kat_23andme_artifact_imputation_quality_rust_beats_beagle_on_dosage_corr() {
    run_imputation_quality_test(TestScope::full());
}

#[test]
fn kat_23andme_artifact_imputation_quality_rust_beats_beagle_on_dosage_corr_fast_chr22_10pct() {
    run_imputation_quality_test(TestScope::fast_chr22_10pct());
}

fn run_imputation_quality_test(scope: TestScope) {
    ensure_cmd_exists("gh");
    ensure_cmd_exists("bcftools");
    ensure_cmd_exists("java");
    ensure_cmd_exists("curl");

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let test_tmp_root = manifest_dir
        .join(".tmp")
        .join("imputation_quality_artifacts");
    fs::create_dir_all(&test_tmp_root).expect("create test tmp root");
    let cache_root = test_tmp_root.join("cache");
    fs::create_dir_all(&cache_root).expect("create cache root");
    let work_dir = tempfile::Builder::new()
        .prefix(&format!("{}-", scope.label))
        .tempdir_in(&test_tmp_root)
        .expect("create temp work dir");
    let dataset_dir = work_dir.path().join("dataset");
    fs::create_dir_all(&dataset_dir).expect("create dataset dir");

    println!("[{}] resolving repo and artifacts", scope.label);
    let repo = detect_repo_slug(&manifest_dir).expect("repo slug resolution failed");
    let (reference_dir, outputs_dir) = download_latest_quality_artifacts(&repo, &cache_root);
    println!(
        "[{}] artifact dirs ready: ref={} outputs={}",
        scope.label,
        reference_dir.display(),
        outputs_dir.display()
    );
    stage_required_inputs(&reference_dir, &outputs_dir, work_dir.path());

    let mut ref_vcf = dataset_dir.join("ref.vcf.gz");
    let mut input_vcf = dataset_dir.join("input.vcf.gz");
    let mut truth_vcf = dataset_dir.join("truth.vcf.gz");
    assert!(ref_vcf.exists(), "missing ref.vcf.gz after staging");
    assert!(input_vcf.exists(), "missing input.vcf.gz after staging");
    assert!(truth_vcf.exists(), "missing truth.vcf.gz after staging");

    if scope.is_fractional() {
        let subset_key = stable_key(&[
            "subset".to_string(),
            scope.label.to_string(),
            scope.chr22_fraction_num.to_string(),
            scope.chr22_fraction_den.to_string(),
            file_fingerprint(&ref_vcf),
            file_fingerprint(&input_vcf),
            file_fingerprint(&truth_vcf),
        ]);
        if let Some((cached_ref, cached_input, cached_truth)) =
            restore_cached_subset(&cache_root, &dataset_dir, &subset_key)
        {
            println!("[{}] reusing cached chr22 subset", scope.label);
            ref_vcf = cached_ref;
            input_vcf = cached_input;
            truth_vcf = cached_truth;
        } else {
            println!("[{}] computing 10% chr22 bounds", scope.label);
            let (region, start, end) = compute_chr22_fraction_region(
                &truth_vcf,
                scope.chr22_fraction_num,
                scope.chr22_fraction_den,
            );
            println!(
                "Subsetting {} to region {}:{}-{} ({}/{})",
                CHR22, CHR22, start, end, scope.chr22_fraction_num, scope.chr22_fraction_den
            );
            ref_vcf =
                subset_vcf_to_region(&ref_vcf, &dataset_dir.join("ref.subset.vcf.gz"), &region);
            input_vcf = subset_vcf_to_region(
                &input_vcf,
                &dataset_dir.join("input.subset.vcf.gz"),
                &region,
            );
            truth_vcf = subset_vcf_to_region(
                &truth_vcf,
                &dataset_dir.join("truth.subset.vcf.gz"),
                &region,
            );
            cache_subset_inputs(&cache_root, &subset_key, &ref_vcf, &input_vcf, &truth_vcf);
        }
    }

    assert!(
        ref_vcf.exists() && input_vcf.exists() && truth_vcf.exists(),
        "subset/staged inputs missing after preparation"
    );

    let reagle_bin = manifest_dir.join("target").join("release").join("reagle");
    assert!(
        reagle_bin.exists(),
        "reagle binary missing at {} (build with `cargo test --release --test imputation_quality_artifacts` or `cargo build --release`)",
        reagle_bin.display()
    );

    let beagle_jar = work_dir.path().join("beagle.jar");
    prepare_beagle_jar(&cache_root, &beagle_jar);
    assert!(beagle_jar.exists(), "failed to download beagle.jar");

    let reagle_out_prefix = work_dir.path().join("reagle_out");
    let beagle_out_prefix = work_dir.path().join("beagle_out");
    let reagle_vcf = reagle_out_prefix.with_extension("vcf.gz");
    let beagle_vcf = beagle_out_prefix.with_extension("vcf.gz");

    let scope_key = format!(
        "{}-{}-{}",
        scope.label, scope.chr22_fraction_num, scope.chr22_fraction_den
    );
    let dataset_key = stable_key(&[
        scope_key,
        file_fingerprint(&ref_vcf),
        file_fingerprint(&input_vcf),
        file_fingerprint(&truth_vcf),
    ]);
    let reagle_key = stable_key(&[dataset_key.clone(), file_fingerprint(&reagle_bin)]);
    let beagle_key = stable_key(&[dataset_key, file_fingerprint(&beagle_jar)]);

    if restore_cached_output(cache_root.as_path(), "reagle", &reagle_key, &reagle_vcf) {
        println!("[{}] reused cached reagle output", scope.label);
    } else {
        println!("[{}] running reagle", scope.label);
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
        cache_output(cache_root.as_path(), "reagle", &reagle_key, &reagle_vcf);
    }

    if restore_cached_output(cache_root.as_path(), "beagle", &beagle_key, &beagle_vcf) {
        println!("[{}] reused cached beagle output", scope.label);
    } else {
        println!("[{}] running beagle", scope.label);
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
        cache_output(cache_root.as_path(), "beagle", &beagle_key, &beagle_vcf);
    }

    assert!(
        reagle_vcf.exists(),
        "missing reagle output {}",
        reagle_vcf.display()
    );
    assert!(
        beagle_vcf.exists(),
        "missing beagle output {}",
        beagle_vcf.display()
    );

    harmonize_sample_names_if_needed(work_dir.path(), &[&truth_vcf, &reagle_vcf, &beagle_vcf]);

    println!("[{}] computing dosage r2 metrics", scope.label);
    let reagle_metrics = compute_metrics_streaming(&truth_vcf, &reagle_vcf, "reagle");
    let beagle_metrics = compute_metrics_streaming(&truth_vcf, &beagle_vcf, "beagle");

    println!(
        "Native dosage r_squared | Reagle: {:.6} (n={}) | Beagle: {:.6} (n={})",
        reagle_metrics.r2, reagle_metrics.r2_n, beagle_metrics.r2, beagle_metrics.r2_n
    );
    assert!(
        reagle_metrics.r2 > beagle_metrics.r2,
        "Expected Reagle r_squared > Beagle r_squared; reagle={:.6}, beagle={:.6}",
        reagle_metrics.r2,
        beagle_metrics.r2
    );
}

fn prepare_beagle_jar(cache_root: &Path, dst_jar: &Path) {
    let cached_jar = cache_root.join("beagle.jar");
    if !cached_jar.exists() {
        println!("[cache] downloading beagle jar");
        let tmp = cache_root.join("beagle.jar.download");
        run(
            "curl",
            [
                "-L",
                "-o",
                tmp.to_str().expect("tmp beagle path utf8"),
                BEAGLE_URL,
            ],
        );
        fs::rename(&tmp, &cached_jar).unwrap_or_else(|e| {
            panic!(
                "failed to move cached beagle jar {} -> {}: {e}",
                tmp.display(),
                cached_jar.display()
            )
        });
    } else {
        println!("[cache] reusing beagle jar {}", cached_jar.display());
    }
    link_or_copy_file(&cached_jar, dst_jar);
}

fn stable_key(parts: &[String]) -> String {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for p in parts {
        p.hash(&mut hasher);
    }
    format!("{:016x}", hasher.finish())
}

fn file_fingerprint(path: &Path) -> String {
    let meta = fs::metadata(path)
        .unwrap_or_else(|e| panic!("failed reading metadata for {}: {e}", path.display()));
    let modified_secs = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs())
        .unwrap_or(0);
    format!("{}:{}", meta.len(), modified_secs)
}

fn model_cache_dir(cache_root: &Path, model: &str) -> PathBuf {
    let dir = cache_root.join("model_outputs").join(model);
    fs::create_dir_all(&dir).expect("create model output cache dir");
    dir
}

fn model_cache_vcf_path(cache_root: &Path, model: &str, key: &str) -> PathBuf {
    model_cache_dir(cache_root, model).join(format!("{key}.vcf.gz"))
}

fn restore_cached_output(cache_root: &Path, model: &str, key: &str, dst_vcf: &Path) -> bool {
    let cached_vcf = model_cache_vcf_path(cache_root, model, key);
    if !cached_vcf.exists() {
        return false;
    }
    println!("[cache] reusing {model} output {}", cached_vcf.display());
    copy_with_index(&cached_vcf, dst_vcf);
    true
}

fn cache_output(cache_root: &Path, model: &str, key: &str, src_vcf: &Path) {
    let cached_vcf = model_cache_vcf_path(cache_root, model, key);
    println!("[cache] storing {model} output {}", cached_vcf.display());
    copy_with_index(src_vcf, &cached_vcf);
}

fn subset_cache_dir(cache_root: &Path, key: &str) -> PathBuf {
    cache_root.join("subsets").join(key)
}

fn restore_cached_subset(
    cache_root: &Path,
    dataset_dir: &Path,
    key: &str,
) -> Option<(PathBuf, PathBuf, PathBuf)> {
    let dir = subset_cache_dir(cache_root, key);
    if !dir.exists() {
        return None;
    }
    let ref_cached = dir.join("ref.vcf.gz");
    let input_cached = dir.join("input.vcf.gz");
    let truth_cached = dir.join("truth.vcf.gz");
    if !ref_cached.exists() || !input_cached.exists() || !truth_cached.exists() {
        return None;
    }

    let ref_dst = dataset_dir.join("ref.subset.vcf.gz");
    let input_dst = dataset_dir.join("input.subset.vcf.gz");
    let truth_dst = dataset_dir.join("truth.subset.vcf.gz");
    copy_with_index(&ref_cached, &ref_dst);
    copy_with_index(&input_cached, &input_dst);
    copy_with_index(&truth_cached, &truth_dst);
    Some((ref_dst, input_dst, truth_dst))
}

fn cache_subset_inputs(cache_root: &Path, key: &str, ref_vcf: &Path, input_vcf: &Path, truth_vcf: &Path) {
    let dir = subset_cache_dir(cache_root, key);
    fs::create_dir_all(&dir).expect("create subset cache dir");
    copy_with_index(ref_vcf, &dir.join("ref.vcf.gz"));
    copy_with_index(input_vcf, &dir.join("input.vcf.gz"));
    copy_with_index(truth_vcf, &dir.join("truth.vcf.gz"));
}

fn compute_chr22_fraction_region(vcf: &Path, num: usize, den: usize) -> (String, i64, i64) {
    assert!(num > 0, "fraction numerator must be > 0");
    assert!(den > 0, "fraction denominator must be > 0");
    assert!(num <= den, "fraction must be <= 1");

    let (min_pos, max_pos) = query_chr22_bounds(vcf);
    let span = max_pos - min_pos + 1;
    let scaled_span = ((span as i128 * num as i128) + den as i128 - 1) / den as i128;
    let target_span = std::cmp::max(1_i64, scaled_span as i64);
    let end = min_pos + target_span - 1;
    let end_clamped = std::cmp::min(end, max_pos);
    (format!("{CHR22}:{min_pos}-{end_clamped}"), min_pos, end_clamped)
}

fn query_chr22_bounds(vcf: &Path) -> (i64, i64) {
    let mut child = Command::new("bcftools")
        .args([
            "query",
            "-r",
            CHR22,
            "-f",
            "%POS\n",
            vcf.to_str().expect("vcf path utf8"),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn bcftools query for chr22 bounds");

    let stdout = child.stdout.take().expect("bcftools stdout missing");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    let mut min_pos: Option<i64> = None;
    let mut max_pos: Option<i64> = None;
    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .expect("read bcftools chr22 bounds output");
        if n == 0 {
            break;
        }
        let pos = match line.trim().parse::<i64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        min_pos = Some(min_pos.map_or(pos, |v| v.min(pos)));
        max_pos = Some(max_pos.map_or(pos, |v| v.max(pos)));
    }

    let output = child
        .wait_with_output()
        .expect("wait bcftools chr22 bounds child");
    assert!(
        output.status.success(),
        "bcftools query for chr22 bounds failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    match (min_pos, max_pos) {
        (Some(min_v), Some(max_v)) if min_v <= max_v => (min_v, max_v),
        _ => panic!("no chr22 positions found in {}", vcf.display()),
    }
}

fn subset_vcf_to_region(src_vcf: &Path, dst_vcf: &Path, region: &str) -> PathBuf {
    run(
        "bcftools",
        [
            "view",
            "-r",
            region,
            "-Oz",
            src_vcf.to_str().expect("src vcf path utf8"),
            "-o",
            dst_vcf.to_str().expect("dst vcf path utf8"),
        ],
    );
    run(
        "bcftools",
        ["index", "-f", dst_vcf.to_str().expect("dst vcf path utf8")],
    );
    dst_vcf.to_path_buf()
}

fn ensure_cmd_exists(cmd: &str) {
    let status = Command::new("which")
        .arg(cmd)
        .status()
        .expect("which command failed");
    assert!(
        status.success(),
        "required command not found in PATH: {cmd}"
    );
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

fn download_latest_quality_artifacts(repo: &str, cache_root: &Path) -> (PathBuf, PathBuf) {
    if let Some((reference_dir, outputs_dir)) = read_artifact_selection_cache(repo, cache_root) {
        println!(
            "[cache] using cached artifact selection: ref={} outputs={}",
            reference_dir.display(),
            outputs_dir.display()
        );
        return (reference_dir, outputs_dir);
    }

    if let Some((reference_dir, outputs_dir)) = find_cached_quality_artifacts(cache_root) {
        println!(
            "[cache] using cached artifacts without gh lookup: ref={} outputs={}",
            reference_dir.display(),
            outputs_dir.display()
        );
        return (reference_dir, outputs_dir);
    }

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
    let mut output_candidates: Vec<(i64, String)> = Vec::new();
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
        let mut outputs_kat_name: Option<String> = None;
        for artifact in artifact_list {
            let Some(name) = artifact.get("name").and_then(Value::as_str) else {
                continue;
            };
            if name == "reference-panel" {
                has_reference = true;
            }
            if outputs_kat_name.is_none() && name.starts_with("outputs-Kat-") {
                outputs_kat_name = Some(name.to_string());
            }
        }
        if has_reference {
            reference_run_id = Some(run_id);
        }
        if let Some(outputs_name) = outputs_kat_name {
            output_candidates.push((run_id, outputs_name));
        }
    }
    let ref_run_id = reference_run_id.expect("no recent run contains reference-panel artifact");
    let reference_dir = ensure_artifact_dir(
        cache_root,
        repo,
        ref_run_id,
        "reference-panel",
        "reference",
    );
    let mut best_input: Option<(i64, PathBuf)> = None;
    let mut best_truth: Option<(i64, PathBuf)> = None;
    let mut target_only_candidates: Vec<(i64, PathBuf, PathBuf, usize)> = Vec::new();

    for (run_id, outputs_name) in output_candidates {
        let candidate_dir =
            ensure_artifact_dir(cache_root, repo, run_id, outputs_name.as_str(), "outputs");

        let input_path = find_input_file(&candidate_dir);
        let has_input = input_path.is_some();
        let has_truth = find_truth_file(&candidate_dir).is_some();

        if has_input && has_truth {
            write_artifact_selection_cache(repo, cache_root, ref_run_id, run_id, outputs_name.as_str());
            return (reference_dir.clone(), candidate_dir);
        }
        if has_input {
            let replace = best_input
                .as_ref()
                .map(|(best_run, _)| run_id > *best_run)
                .unwrap_or(true);
            if replace {
                best_input = Some((run_id, candidate_dir.clone()));
            }
        }
        if has_truth {
            let replace = best_truth
                .as_ref()
                .map(|(best_run, _)| run_id > *best_run)
                .unwrap_or(true);
            if replace {
                best_truth = Some((run_id, candidate_dir.clone()));
            }
        }
        if has_input && !has_truth {
            if let Some(input_path) = input_path {
                if input_path
                    .file_name()
                    .and_then(OsStr::to_str)
                    .map(|n| n == "target.vcf.gz")
                    .unwrap_or(false)
                {
                    let n_records = count_vcf_records(&input_path).unwrap_or(0);
                    if n_records > 0 {
                        target_only_candidates.push((
                            run_id,
                            candidate_dir.clone(),
                            input_path,
                            n_records,
                        ));
                    }
                }
            }
        }
    }
    if let (Some((input_run, input_dir)), Some((truth_run, truth_dir))) =
        (best_input.clone(), best_truth.clone())
    {
        let merged = build_outputs_union_dir(
            cache_root,
            &input_dir,
            &truth_dir,
            input_run,
            truth_run,
        );
        return (reference_dir, merged);
    }
    if best_truth.is_none() && target_only_candidates.len() >= 2 {
        target_only_candidates.sort_by(|a, b| {
            b.3.cmp(&a.3).then_with(|| b.0.cmp(&a.0))
        });
        let truth_pick = target_only_candidates
            .first()
            .expect("truth candidate should exist");
        let input_pick = target_only_candidates
            .last()
            .expect("input candidate should exist");
        let merged = build_outputs_union_dir_from_sources(
            cache_root,
            &input_pick.2,
            &truth_pick.2,
            input_pick.0,
            truth_pick.0,
        );
        return (reference_dir, merged);
    }
    panic!(
        "no recent {} run had outputs-Kat-* containing target/input.vcf.gz + truth.vcf.gz",
        WORKFLOW_NAME
    );
}

fn artifact_selection_cache_path(cache_root: &Path) -> PathBuf {
    cache_root.join(ARTIFACT_SELECTION_CACHE_FILE)
}

fn read_artifact_selection_cache(repo: &str, cache_root: &Path) -> Option<(PathBuf, PathBuf)> {
    let cache_path = artifact_selection_cache_path(cache_root);
    let content = fs::read_to_string(&cache_path).ok()?;
    let mut fields = content.trim().split('\t');
    let cached_repo = fields.next()?;
    let cached_workflow = fields.next()?;
    let ref_run_id = fields.next()?.parse::<i64>().ok()?;
    let outputs_run_id = fields.next()?.parse::<i64>().ok()?;
    let outputs_name = fields.next()?;
    if fields.next().is_some() || cached_repo != repo || cached_workflow != WORKFLOW_NAME {
        return None;
    }
    let reference_dir = cache_root.join(format!("reference-run-{ref_run_id}-reference-panel"));
    let outputs_dir = cache_root.join(format!(
        "outputs-run-{outputs_run_id}-{}",
        outputs_name.replace('/', "_")
    ));
    let ref_ok = reference_dir.join(".complete").exists()
        && find_file_recursive(&reference_dir, "ref.vcf.gz").is_some();
    let outputs_ok = outputs_dir.join(".complete").exists()
        && (find_file_recursive(&outputs_dir, "target.vcf.gz").is_some()
            || find_file_recursive(&outputs_dir, "input.vcf.gz").is_some())
        && find_file_recursive(&outputs_dir, "truth.vcf.gz").is_some();
    if ref_ok && outputs_ok {
        Some((reference_dir, outputs_dir))
    } else {
        None
    }
}

fn write_artifact_selection_cache(
    repo: &str,
    cache_root: &Path,
    ref_run_id: i64,
    outputs_run_id: i64,
    outputs_name: &str,
) {
    let cache_path = artifact_selection_cache_path(cache_root);
    let tmp_path = cache_root.join(format!("{}.tmp", ARTIFACT_SELECTION_CACHE_FILE));
    let payload = format!(
        "{repo}\t{workflow}\t{ref_run_id}\t{outputs_run_id}\t{outputs_name}\n",
        workflow = WORKFLOW_NAME
    );
    fs::write(&tmp_path, payload).unwrap_or_else(|e| {
        panic!(
            "failed writing artifact cache temp {}: {e}",
            tmp_path.display()
        )
    });
    fs::rename(&tmp_path, &cache_path).unwrap_or_else(|e| {
        panic!(
            "failed finalizing artifact cache {} -> {}: {e}",
            tmp_path.display(),
            cache_path.display()
        )
    });
}

fn find_cached_quality_artifacts(cache_root: &Path) -> Option<(PathBuf, PathBuf)> {
    let entries = fs::read_dir(cache_root).ok()?;
    let mut best_ref: Option<(i64, PathBuf)> = None;
    let mut best_out: Option<(i64, PathBuf)> = None;
    let mut best_input: Option<(i64, PathBuf)> = None;
    let mut best_truth: Option<(i64, PathBuf)> = None;
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let name = match path.file_name().and_then(OsStr::to_str) {
            Some(v) => v,
            None => continue,
        };
        let done = path.join(".complete");
        if !done.exists() {
            continue;
        }
        if name.starts_with("reference-run-") {
            let Some(run_id) = parse_cached_run_id(name, "reference-run-") else {
                continue;
            };
            if find_file_recursive(&path, "ref.vcf.gz").is_some() {
                let replace = best_ref
                    .as_ref()
                    .map(|(best_id, _)| run_id > *best_id)
                    .unwrap_or(true);
                if replace {
                    best_ref = Some((run_id, path));
                }
            }
            continue;
        }
        if name.starts_with("outputs-run-") {
            let Some(run_id) = parse_cached_run_id(name, "outputs-run-") else {
                continue;
            };
            let has_input = find_input_file(&path).is_some();
            let has_truth = find_truth_file(&path).is_some();
            if has_input && has_truth {
                let replace = best_out
                    .as_ref()
                    .map(|(best_id, _)| run_id > *best_id)
                    .unwrap_or(true);
                if replace {
                    best_out = Some((run_id, path));
                }
            } else {
                if has_input {
                    let replace = best_input
                        .as_ref()
                        .map(|(best_id, _)| run_id > *best_id)
                        .unwrap_or(true);
                    if replace {
                        best_input = Some((run_id, path.clone()));
                    }
                }
                if has_truth {
                    let replace = best_truth
                        .as_ref()
                        .map(|(best_id, _)| run_id > *best_id)
                        .unwrap_or(true);
                    if replace {
                        best_truth = Some((run_id, path.clone()));
                    }
                }
            }
        }
    }
    match (best_ref, best_out) {
        (Some((_, ref_dir)), Some((_, out_dir))) => Some((ref_dir, out_dir)),
        (Some((_, ref_dir)), None) => {
            if let (Some((input_run, input_dir)), Some((truth_run, truth_dir))) =
                (best_input, best_truth)
            {
                let merged = build_outputs_union_dir(
                    cache_root,
                    &input_dir,
                    &truth_dir,
                    input_run,
                    truth_run,
                );
                Some((ref_dir, merged))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn parse_cached_run_id(name: &str, prefix: &str) -> Option<i64> {
    let rest = name.strip_prefix(prefix)?;
    let run_token = rest.split('-').next()?;
    run_token.parse::<i64>().ok()
}

fn find_input_file(outputs_dir: &Path) -> Option<PathBuf> {
    find_file_recursive(outputs_dir, "target.vcf.gz")
        .or_else(|| find_file_recursive(outputs_dir, "input.vcf.gz"))
}

fn find_truth_file(outputs_dir: &Path) -> Option<PathBuf> {
    find_file_recursive(outputs_dir, "truth.vcf.gz")
}

fn build_outputs_union_dir(
    cache_root: &Path,
    input_dir: &Path,
    truth_dir: &Path,
    input_run_id: i64,
    truth_run_id: i64,
) -> PathBuf {
    let final_dir = cache_root.join(format!(
        "outputs-merged-input{input_run_id}-truth{truth_run_id}"
    ));
    if final_dir.join(".complete").exists() {
        return final_dir;
    }
    if final_dir.exists() {
        let _ = fs::remove_dir_all(&final_dir);
    }
    let tmp_dir = cache_root.join(format!(
        "outputs-merged-input{input_run_id}-truth{truth_run_id}.downloading"
    ));
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).expect("create merged outputs temp dir");

    let input_src = find_input_file(input_dir).unwrap_or_else(|| {
        panic!(
            "missing target/input in cached outputs dir {}",
            input_dir.display()
        )
    });
    let truth_src = find_truth_file(truth_dir).unwrap_or_else(|| {
        panic!("missing truth in cached outputs dir {}", truth_dir.display())
    });
    build_outputs_union_dir_from_sources(
        cache_root,
        &input_src,
        &truth_src,
        input_run_id,
        truth_run_id,
    )
}

fn build_outputs_union_dir_from_sources(
    cache_root: &Path,
    input_src: &Path,
    truth_src: &Path,
    input_run_id: i64,
    truth_run_id: i64,
) -> PathBuf {
    let final_dir = cache_root.join(format!(
        "outputs-merged-input{input_run_id}-truth{truth_run_id}"
    ));
    if final_dir.join(".complete").exists() {
        return final_dir;
    }
    if final_dir.exists() {
        let _ = fs::remove_dir_all(&final_dir);
    }
    let tmp_dir = cache_root.join(format!(
        "outputs-merged-input{input_run_id}-truth{truth_run_id}.downloading"
    ));
    let _ = fs::remove_dir_all(&tmp_dir);
    fs::create_dir_all(&tmp_dir).expect("create merged outputs temp dir");

    copy_with_index(&input_src, &tmp_dir.join("target.vcf.gz"));
    copy_with_index(&truth_src, &tmp_dir.join("truth.vcf.gz"));
    fs::write(tmp_dir.join(".complete"), "ok\n").expect("write merged outputs marker");
    fs::rename(&tmp_dir, &final_dir).unwrap_or_else(|e| {
        panic!(
            "failed moving merged outputs {} -> {}: {e}",
            tmp_dir.display(),
            final_dir.display()
        )
    });
    final_dir
}

fn count_vcf_records(vcf: &Path) -> Option<usize> {
    let out = run_capture(
        "bcftools",
        ["index", "-n", vcf.to_str().expect("vcf path utf8")],
    );
    out.trim().parse::<usize>().ok()
}

fn ensure_artifact_dir(
    cache_root: &Path,
    repo: &str,
    run_id: i64,
    artifact_name: &str,
    kind: &str,
) -> PathBuf {
    let safe_name = artifact_name.replace('/', "_");
    let final_dir = cache_root.join(format!("{kind}-run-{run_id}-{safe_name}"));
    let done_marker = final_dir.join(".complete");
    if done_marker.exists() {
        println!(
            "[cache] reusing {} artifact run {} ({})",
            kind, run_id, artifact_name
        );
        return final_dir;
    }
    if final_dir.exists() {
        fs::remove_dir_all(&final_dir).unwrap_or_else(|e| {
            panic!(
                "failed removing stale artifact cache dir {}: {e}",
                final_dir.display()
            )
        });
    }

    let tmp_dir = cache_root.join(format!("{kind}-run-{run_id}-{safe_name}.downloading"));
    if tmp_dir.exists() {
        fs::remove_dir_all(&tmp_dir).unwrap_or_else(|e| {
            panic!(
                "failed removing stale artifact temp dir {}: {e}",
                tmp_dir.display()
            )
        });
    }
    fs::create_dir_all(&tmp_dir).expect("create artifact tmp dir");
    println!(
        "[cache] downloading {} artifact run {} ({})",
        kind, run_id, artifact_name
    );
    run(
        "gh",
        [
            "run",
            "download",
            "--repo",
            repo,
            &run_id.to_string(),
            "-n",
            artifact_name,
            "-D",
            tmp_dir.to_str().expect("artifact tmp dir utf8"),
        ],
    );
    fs::write(tmp_dir.join(".complete"), "ok\n").expect("write artifact marker");
    if final_dir.exists() {
        fs::remove_dir_all(&tmp_dir).unwrap_or_else(|e| {
            panic!(
                "failed removing redundant artifact temp dir {}: {e}",
                tmp_dir.display()
            )
        });
        return final_dir;
    }
    fs::rename(&tmp_dir, &final_dir).unwrap_or_else(|e| {
        panic!(
            "failed to move artifact cache {} -> {}: {e}",
            tmp_dir.display(),
            final_dir.display()
        )
    });
    final_dir
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
    link_or_copy_file(src_vcf, dst_vcf);

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
            link_or_copy_file(path, &dst);
            return;
        }
    }
}

fn link_or_copy_file(src: &Path, dst: &Path) {
    if dst.exists() {
        fs::remove_file(dst).unwrap_or_else(|e| {
            panic!("failed removing existing {}: {e}", dst.display())
        });
    }
    if fs::hard_link(src, dst).is_ok() {
        return;
    }
    fs::copy(src, dst).unwrap_or_else(|e| {
        panic!(
            "failed copying {} -> {}: {e}",
            src.display(),
            dst.display()
        )
    });
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
}

fn next_truth_site(reader: &mut BufReader<std::process::ChildStdout>) -> Option<TruthSite> {
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .expect("read truth query output");
        if n == 0 {
            return None;
        }
        let mut fields = line.trim_end().split('\t');
        let Some(chrom) = fields.next() else { continue };
        let Some(pos_s) = fields.next() else { continue };
        let Some(ref_allele) = fields.next() else { continue };
        let Some(alt_allele) = fields.next() else { continue };
        let Some(gt) = fields.next() else { continue };
        let pos = match pos_s.parse::<i64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        return Some(TruthSite {
            chrom: chrom.to_string(),
            pos,
            ref_allele: ref_allele.to_string(),
            alt_allele: alt_allele.to_string(),
            gt: gt.to_string(),
        });
    }
}

fn next_imputed_site(reader: &mut BufReader<std::process::ChildStdout>) -> Option<ImputedSite> {
    let mut line = String::new();
    loop {
        line.clear();
        let n = reader
            .read_line(&mut line)
            .expect("read imputed query output");
        if n == 0 {
            return None;
        }
        let mut fields = line.trim_end().split('\t');
        let Some(chrom) = fields.next() else { continue };
        let Some(pos_s) = fields.next() else { continue };
        let Some(ref_allele) = fields.next() else { continue };
        let Some(alt_allele) = fields.next() else { continue };
        let Some(sample) = fields.next() else { continue };
        let pos = match pos_s.parse::<i64>() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let (gt, ds) = parse_imputed_sample(sample);
        return Some(ImputedSite {
            chrom: chrom.to_string(),
            pos,
            ref_allele: ref_allele.to_string(),
            alt_allele: alt_allele.to_string(),
            gt,
            ds,
        });
    }
}

fn parse_imputed_sample(sample: &str) -> (String, Option<f64>) {
    let mut parts = sample.splitn(3, ':');
    let gt = parts.next().unwrap_or(".").to_string();
    let ds = parts.next().and_then(parse_ds);
    let gp_ds = parts.next().and_then(ds_from_gp_str_biallelic);
    (gt, ds.or(gp_ds))
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

fn ds_from_gp_str_biallelic(v: &str) -> Option<f64> {
    if v.is_empty() || v == "." {
        return None;
    }
    let mut toks = v.split(',');
    let p0 = toks.next()?.parse::<f64>().ok()?;
    let p1 = toks.next()?.parse::<f64>().ok()?;
    let p2 = toks.next()?.parse::<f64>().ok()?;
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

struct MetricsResult {
    r2: f64,
    r2_n: usize,
}

fn compute_metrics_streaming(truth_vcf: &Path, imputed_vcf: &Path, label: &str) -> MetricsResult {
    let (truth_child, mut truth_reader) =
        spawn_bcftools_query(truth_vcf, "%CHROM\t%POS\t%REF\t%ALT[\t%GT]\n");
    let (imputed_child, mut imputed_reader) =
        spawn_bcftools_query(imputed_vcf, "%CHROM\t%POS\t%REF\t%ALT[\t%GT:%DS:%GP]\n");

    let mut sum_t = 0.0_f64;
    let mut sum_i = 0.0_f64;
    let mut sum_ti = 0.0_f64;
    let mut sum_tt = 0.0_f64;
    let mut sum_ii = 0.0_f64;
    let mut dosage_n: usize = 0;
    let mut next_progress: usize = METRIC_PROGRESS_INTERVAL;

    let mut t_site = next_truth_site(&mut truth_reader);
    let mut i_site = next_imputed_site(&mut imputed_reader);

    while let (Some(t), Some(i)) = (t_site.as_ref(), i_site.as_ref()) {
        match compare_site_key(t, i) {
            std::cmp::Ordering::Less => t_site = next_truth_site(&mut truth_reader),
            std::cmp::Ordering::Greater => i_site = next_imputed_site(&mut imputed_reader),
            std::cmp::Ordering::Equal => {
                let biallelic_truth = !t.alt_allele.contains(',');
                let biallelic_imp = !i.alt_allele.contains(',');
                let swapped = biallelic_truth
                    && biallelic_imp
                    && t.ref_allele == i.alt_allele
                    && t.alt_allele == i.ref_allele;
                let allele_compatible =
                    (t.ref_allele == i.ref_allele && t.alt_allele == i.alt_allele) || swapped;

                if allele_compatible {
                    let mut i_dos = i
                        .ds
                        .or_else(|| gt_to_nonref_dosage(&i.gt));
                    if let (Some(t_dos), Some(mut i_dos_val)) =
                        (gt_to_nonref_dosage(&t.gt), i_dos.take())
                    {
                        if swapped {
                            i_dos_val = 2.0 - i_dos_val;
                        }
                        sum_t += t_dos;
                        sum_i += i_dos_val;
                        sum_ti += t_dos * i_dos_val;
                        sum_tt += t_dos * t_dos;
                        sum_ii += i_dos_val * i_dos_val;
                        dosage_n += 1;
                    }
                }

                if dosage_n >= next_progress {
                    println!("[metrics:{label}] r2 comparable={}", dosage_n);
                    next_progress = next_progress.saturating_add(METRIC_PROGRESS_INTERVAL);
                }

                t_site = next_truth_site(&mut truth_reader);
                i_site = next_imputed_site(&mut imputed_reader);
            }
        }
    }

    wait_child_ok(truth_child, "bcftools truth query");
    wait_child_ok(imputed_child, "bcftools imputed query");

    assert!(
        dosage_n > 1,
        "insufficient comparable points for r_squared (n={dosage_n})"
    );
    let n_f = dosage_n as f64;
    let mean_t = sum_t / n_f;
    let mean_i = sum_i / n_f;
    let cov = sum_ti / n_f - mean_t * mean_i;
    let var_t = sum_tt / n_f - mean_t * mean_t;
    let var_i = sum_ii / n_f - mean_i * mean_i;
    assert!(
        var_t > 0.0 && var_i > 0.0,
        "non-positive variance for r_squared (var_t={var_t}, var_i={var_i}, n={dosage_n})"
    );
    let r = cov / (var_t * var_i).sqrt();

    MetricsResult {
        r2: r * r,
        r2_n: dosage_n,
    }
}

fn sample_names(vcf: &Path) -> Vec<String> {
    let stdout = run_capture("bcftools", ["query", "-l", vcf.to_str().expect("vcf path utf8")]);
    stdout
        .lines()
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .map(ToString::to_string)
        .collect()
}

fn harmonize_sample_names_if_needed(work_dir: &Path, vcfs: &[&Path]) {
    let mut all_names: Vec<Vec<String>> = Vec::with_capacity(vcfs.len());
    for vcf in vcfs {
        all_names.push(sample_names(vcf));
    }
    let all_single_sample = all_names.iter().all(|names| names.len() == 1);
    let all_named_sample = all_names
        .iter()
        .all(|names| names.first().map(|v| v == "SAMPLE").unwrap_or(false));
    if all_single_sample && all_named_sample {
        println!("[sample] sample names already harmonized; skipping reheader/index");
        return;
    }
    if all_single_sample {
        let first = all_names
            .first()
            .and_then(|v| v.first())
            .expect("single-sample names should not be empty");
        let all_same_single = all_names
            .iter()
            .all(|names| names.first().map(|v| v == first).unwrap_or(false));
        if all_same_single {
            println!(
                "[sample] all files share single sample name {}; skipping reheader/index",
                first
            );
            return;
        }
    }
    println!("[sample] harmonizing sample names via reheader/index");

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
