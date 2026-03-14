#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import re
import shlex
import subprocess
import sys
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Any

CI_WORKFLOW_FILE = "CI.yml"
IQA_WORKFLOW_FILE = "imputation_quality_report.yml"
REFERENCE_JOB_NAME = "reference-comparison"
CHR21_JOB_NAME = "reference-comparison-chr21"
CHR21_TARGET_TEST_NAME = "test_reference_comparison_full_chr21_ref1000_target10"
CHR21_ASSERTION_PREFIX = "Reagle worse than Beagle on "
TEST_RESULT_RE = re.compile(r"test result: \w+\. (\d+) passed; (\d+) failed;")
BEAGLE_REF_TEST_RE = re.compile(r"^test (test_java_beagle_[A-Za-z0-9_]+) \.\.\. (ok|FAILED)$")
SEED_RE = re.compile(r"Seed\s+(\d+):\s+Java\s+([0-9.]+)%\s*,\s*Rust\s+([0-9.]+)%")
ASSERTION_RE = re.compile(
    r"^Reagle worse than Beagle on (?P<metric>.+?): "
    r"reagle=(?P<reagle>[\d.eE+-]+), beagle=(?P<beagle>[\d.eE+-]+)$"
)
CHR21_TIMING_RE = re.compile(r"Timing: BEAGLE=([\d.]+)s REAGLE=([\d.]+)s")
CHR21_TOOL_RE = re.compile(
    r"^(BEAGLE|REAGLE): "
    r"sites=(\d+) "
    r"genotypes=(\d+) "
    r"r2=(Some\([^)]+\)|None) "
    r"iqs=(Some\([^)]+\)|None) "
    r"hellinger=(Some\([^)]+\)|None) "
    r"SER=(Some\([^)]+\)|None) "
    r"\((\d+)/(\d+)\) "
    r"phase_conc=(Some\([^)]+\)|None) "
    r"\((\d+)/(\d+)\)$"
)


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def run_text(command: list[str]) -> str:
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(f"{shlex.join(command)} failed: {stderr or 'unknown error'}")
    return proc.stdout


def gh_json(args: list[str]) -> Any:
    return json.loads(run_text(["gh", *args]))


def gh_api_json(path: str, params: dict[str, Any] | None = None) -> Any:
    endpoint = path.lstrip("/")
    if params:
        endpoint = f"{endpoint}?{urllib.parse.urlencode(params, doseq=True)}"
    return gh_json(["api", endpoint])


def api_paginate_list(
    path: str,
    *,
    params: dict[str, Any] | None = None,
    list_key: str | None = None,
    stop_before: dt.datetime | None = None,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page = 1

    while True:
        query: dict[str, Any] = {"per_page": 100, "page": page}
        if params:
            query.update(params)
        data = gh_api_json(path, query)
        chunk = data if list_key is None else data.get(list_key, [])
        if not chunk:
            break

        items.extend(chunk)

        if stop_before is not None:
            oldest = parse_run_created_at(chunk[-1])
            if oldest is not None and oldest < stop_before:
                break

        if len(chunk) < 100:
            break
        page += 1

    return items


def parse_iso(ts: str | None) -> dt.datetime | None:
    if not ts:
        return None
    raw = ts
    if "." in raw:
        base, frac = raw.split(".", 1)
        frac = frac.rstrip("Z")[:6]
        raw = f"{base}.{frac}Z"
        fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
    else:
        fmt = "%Y-%m-%dT%H:%M:%SZ"
    return dt.datetime.strptime(raw, fmt).replace(tzinfo=dt.timezone.utc)


def parse_run_created_at(run: dict[str, Any]) -> dt.datetime | None:
    return parse_iso(run.get("created_at") or run.get("createdAt"))


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def flatten_numeric_values(value: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_numeric_values(value[key], next_prefix))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            out.update(flatten_numeric_values(item, next_prefix))
    elif is_numeric(value):
        val = float(value)
        if math.isfinite(val):
            out[prefix] = val
    return out


def metric_key_is_relevant(key: str) -> bool:
    if key.startswith("ds_calibration["):
        return False
    if key.startswith("confusion_matrix["):
        return False
    if key.startswith("r2_stats."):
        return False
    if key.startswith("rare_r2_stats."):
        return False
    if ".agg_stats." in key:
        return False
    return True


def filter_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in metrics.items() if metric_key_is_relevant(key)}


def subtract_numeric_maps(current: dict[str, float], base: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in sorted(set(current.keys()) & set(base.keys())):
        out[key] = current[key] - base[key]
    return out


def mean_numeric_maps(maps: list[dict[str, float]]) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for item in maps:
        for key, value in item.items():
            sums[key] += value
            counts[key] += 1
    return {key: sums[key] / counts[key] for key in sorted(sums.keys()) if counts[key] > 0}


def strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", value)


def log_payload(raw_line: str) -> str:
    line = strip_ansi(raw_line.rstrip("\n")).lstrip("\ufeff")
    parts = line.split("\t", 2)
    return parts[2] if len(parts) == 3 else line


def normalize_log_line(raw_line: str) -> str:
    payload = log_payload(raw_line)
    payload = re.sub(r"^\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s*", "", payload)
    return payload.strip()


def parse_log_timestamp(raw_line: str) -> dt.datetime | None:
    payload = log_payload(raw_line)
    match = re.match(r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s", payload)
    if not match:
        return None
    return parse_iso(match.group("ts"))


def parse_optional_float(value: str) -> float | None:
    if value == "None":
        return None
    match = re.fullmatch(r"Some\(([-+0-9.eE]+)\)", value)
    if not match:
        return None
    return float(match.group(1))


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": run.get("status"),
        "conclusion": run.get("conclusion"),
        "created_at": run.get("created_at") or run.get("createdAt"),
    }


def summarize_job_state(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": job.get("status"),
        "conclusion": job.get("conclusion"),
    }


def detect_repo(explicit_repo: str) -> str:
    if explicit_repo:
        return explicit_repo
    data = gh_json(["repo", "view", "--json", "nameWithOwner"])
    repo = data.get("nameWithOwner")
    if not repo:
        fail("could not detect repo from gh")
    return repo


def list_open_prs(repo: str, limit: int | None) -> list[dict[str, Any]]:
    prs = api_paginate_list(f"repos/{repo}/pulls", params={"state": "open"}, list_key=None)
    prs = [
        {
            "number": item["number"],
            "title": item["title"],
            "url": item["html_url"],
            "created_at": item["created_at"],
            "head_branch": item["head"]["ref"],
        }
        for item in prs
    ]
    prs.sort(key=lambda item: item["number"], reverse=True)
    if limit is not None:
        return prs[:limit]
    return prs


def list_workflow_runs(
    repo: str,
    workflow_file: str,
    *,
    branch: str | None = None,
    stop_before: dt.datetime | None = None,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {}
    if branch:
        params["branch"] = branch
    return api_paginate_list(
        f"repos/{repo}/actions/workflows/{workflow_file}/runs",
        params=params,
        list_key="workflow_runs",
        stop_before=stop_before,
    )


def latest_workflow_run(repo: str, workflow_file: str, branch: str) -> dict[str, Any] | None:
    runs = gh_json(
        [
            "run",
            "list",
            "-R",
            repo,
            "--workflow",
            workflow_file,
            "--branch",
            branch,
            "--limit",
            "1",
            "--json",
            "databaseId,displayTitle,headBranch,headSha,status,conclusion,event,createdAt,updatedAt,url,number,name",
        ]
    )
    return runs[0] if runs else None


def gh_run_details(repo: str, run_id: int) -> dict[str, Any]:
    return gh_json(
        [
            "run",
            "view",
            str(run_id),
            "-R",
            repo,
            "--json",
            "databaseId,name,displayTitle,headBranch,headSha,status,conclusion,event,createdAt,updatedAt,startedAt,url,number,jobs",
        ]
    )


def run_database_id(run: dict[str, Any]) -> int:
    raw_id = run.get("databaseId")
    if raw_id is None:
        raw_id = run.get("id")
    if raw_id is None:
        raise KeyError("run is missing databaseId/id")
    return int(raw_id)


def list_run_artifacts(repo: str, run_id: int) -> list[dict[str, Any]]:
    return api_paginate_list(
        f"repos/{repo}/actions/runs/{run_id}/artifacts",
        list_key="artifacts",
    )


def cached_metric_artifacts(run_id: int, cache_dir: Path) -> list[dict[str, Any]]:
    run_dir = cache_dir / f"run_{run_id}"
    if not run_dir.exists():
        return []

    artifacts: list[dict[str, Any]] = []
    for artifact_dir in sorted(run_dir.iterdir()):
        if not artifact_dir.is_dir() or not artifact_dir.name.startswith("metrics-"):
            continue
        reagle_json = list(artifact_dir.rglob("reagle_metrics.json"))
        beagle_json = list(artifact_dir.rglob("beagle_metrics.json"))
        if reagle_json and beagle_json:
            artifacts.append(
                {
                    "name": artifact_dir.name,
                    "expired": False,
                }
            )
    return artifacts


def load_job_log(repo: str, run_id: int, job_id: int, cache_dir: Path) -> str:
    run_dir = cache_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"job_{job_id}.log"
    if log_path.exists():
        return log_path.read_text(encoding="utf-8", errors="replace")

    text = run_text(
        [
            "gh",
            "run",
            "view",
            str(run_id),
            "-R",
            repo,
            "--job",
            str(job_id),
            "--log",
        ]
    )
    log_path.write_text(text, encoding="utf-8")
    return text


def download_metric_artifact(repo: str, run_id: int, artifact_name: str, cache_dir: Path) -> Path:
    artifact_dir = cache_dir / f"run_{run_id}" / safe_name(artifact_name)
    reagle_json = list(artifact_dir.rglob("reagle_metrics.json"))
    beagle_json = list(artifact_dir.rglob("beagle_metrics.json"))
    if reagle_json and beagle_json:
        return artifact_dir

    artifact_dir.mkdir(parents=True, exist_ok=True)
    download_dir = artifact_dir
    if any(artifact_dir.iterdir()):
        slot = 1
        while True:
            candidate = artifact_dir / f"download_{slot}"
            if not candidate.exists():
                download_dir = candidate
                break
            slot += 1
        download_dir.mkdir(parents=True, exist_ok=True)
    run_text(
        [
            "gh",
            "run",
            "download",
            str(run_id),
            "-R",
            repo,
            "-n",
            artifact_name,
            "-D",
            str(download_dir),
        ]
    )
    return artifact_dir


def parse_test_results(log_text: str) -> dict[str, Any]:
    passed = 0
    failed = 0

    for raw_line in log_text.splitlines():
        line = normalize_log_line(raw_line)
        match = TEST_RESULT_RE.search(line)
        if not match:
            continue
        line_passed = int(match.group(1))
        line_failed = int(match.group(2))
        passed += line_passed
        failed += line_failed

    total = passed + failed
    return {
        "passed": passed,
        "failed": failed,
        "total": total,
    }


def parse_beagle_reference_stats(log_text: str) -> dict[str, Any] | None:
    passed = 0
    failed = 0

    for raw_line in log_text.splitlines():
        line = normalize_log_line(raw_line)
        match = BEAGLE_REF_TEST_RE.match(line)
        if not match:
            continue
        if match.group(2) == "ok":
            passed += 1
        else:
            failed += 1

    total = passed + failed
    if total == 0:
        return None

    return {
        "passed": passed,
        "failed": failed,
        "total": total,
        "pass_rate": (passed / total) if total else None,
    }


def parse_seed_metrics(log_text: str) -> dict[str, Any]:
    count = 0
    total_delta = 0.0
    for raw_line in log_text.splitlines():
        line = normalize_log_line(raw_line)
        match = SEED_RE.search(line)
        if not match:
            continue
        java_pct = float(match.group(2))
        rust_pct = float(match.group(3))
        total_delta += rust_pct - java_pct
        count += 1

    mean_delta = None
    if count:
        mean_delta = total_delta / count

    return {
        "count": count,
        "mean_delta_percent": mean_delta,
    }


def parse_chr21_target_result(log_text: str) -> dict[str, Any]:
    lines = [normalize_log_line(line) for line in log_text.splitlines()]
    result: dict[str, Any] = {
        "status": "missing",
        "assertion": None,
    }

    for idx, line in enumerate(lines):
        if CHR21_TARGET_TEST_NAME not in line:
            continue

        window_end = min(len(lines), idx + 250)
        for j in range(idx, window_end):
            current = lines[j]
            if current == f"test {CHR21_TARGET_TEST_NAME} ... ok":
                return {"status": "pass", "assertion": None}

            if current.startswith(f"thread '{CHR21_TARGET_TEST_NAME}'") and "panicked at" in current:
                for candidate in lines[j + 1 : min(window_end, j + 8)]:
                    if not candidate or candidate.startswith("note:"):
                        continue
                    assertion = None
                    match = ASSERTION_RE.match(candidate)
                    if match:
                        assertion = {
                            "metric": match.group("metric"),
                            "reagle": float(match.group("reagle")),
                            "beagle": float(match.group("beagle")),
                            "delta": float(match.group("reagle")) - float(match.group("beagle")),
                        }
                    return {
                        "status": "fail",
                        "assertion": assertion,
                    }

            if current.startswith(CHR21_ASSERTION_PREFIX):
                match = ASSERTION_RE.match(current)
                assertion = None
                if match:
                    assertion = {
                        "metric": match.group("metric"),
                        "reagle": float(match.group("reagle")),
                        "beagle": float(match.group("beagle")),
                        "delta": float(match.group("reagle")) - float(match.group("beagle")),
                    }
                return {
                    "status": "fail",
                    "assertion": assertion,
                }

    return result


def parse_chr21_metrics(log_text: str) -> dict[str, Any]:
    timing: dict[str, float] = {}
    tool_metrics: dict[str, dict[str, Any]] = {}

    for raw_line in log_text.splitlines():
        line = normalize_log_line(raw_line)

        timing_match = CHR21_TIMING_RE.match(line)
        if timing_match:
            beagle_runtime = float(timing_match.group(1))
            reagle_runtime = float(timing_match.group(2))
            timing = {
                "beagle_runtime_sec": beagle_runtime,
                "reagle_runtime_sec": reagle_runtime,
                "reagle_minus_beagle_runtime_sec": reagle_runtime - beagle_runtime,
            }
            continue

        metric_match = CHR21_TOOL_RE.match(line)
        if not metric_match:
            continue

        tool_name = metric_match.group(1).lower()
        switch_errors = int(metric_match.group(8))
        switch_opportunities = int(metric_match.group(9))
        phase_concordant = int(metric_match.group(11))
        phase_total = int(metric_match.group(12))
        tool_metrics[tool_name] = {
            "sites_compared": int(metric_match.group(2)),
            "genotypes_compared": int(metric_match.group(3)),
            "r_squared": parse_optional_float(metric_match.group(4)),
            "iqs": parse_optional_float(metric_match.group(5)),
            "hellinger_score": parse_optional_float(metric_match.group(6)),
            "switch_error_rate": parse_optional_float(metric_match.group(7)),
            "switch_errors": switch_errors,
            "switch_opportunities": switch_opportunities,
            "phase_concordance": parse_optional_float(metric_match.group(10)),
            "phase_concordant": phase_concordant,
            "phase_total": phase_total,
        }

    deltas: dict[str, float] = {}
    reagle = tool_metrics.get("reagle")
    beagle = tool_metrics.get("beagle")
    if reagle and beagle:
        for key in sorted(set(reagle.keys()) & set(beagle.keys())):
            if is_numeric(reagle[key]) and is_numeric(beagle[key]):
                deltas[key] = float(reagle[key]) - float(beagle[key])

    return {
        "timing": timing,
        "beagle": tool_metrics.get("beagle"),
        "reagle": tool_metrics.get("reagle"),
        "reagle_minus_beagle": deltas,
    }


def parse_iqa_job_log(log_text: str) -> dict[str, Any]:
    reagle_start: dt.datetime | None = None
    beagle_start: dt.datetime | None = None
    artifact_name: str | None = None

    for raw_line in log_text.splitlines():
        normalized = normalize_log_line(raw_line)
        timestamp = parse_log_timestamp(raw_line)

        if "=== Running Reagle ===" in normalized and reagle_start is None:
            reagle_start = timestamp
        if "=== Running Beagle ===" in normalized and reagle_start is not None and beagle_start is None:
            beagle_start = timestamp

        artifact_match = re.search(r"Artifact (metrics-[A-Za-z0-9_.-]+)(?:\.zip)?", normalized)
        if artifact_match:
            artifact_name = artifact_match.group(1)
            continue

        name_match = re.search(r"\bname:\s*(metrics-[A-Za-z0-9_.-]+)\b", normalized)
        if artifact_name is None and name_match:
            artifact_name = name_match.group(1)

    reagle_step_seconds = None
    if reagle_start is not None and beagle_start is not None:
        delta = (beagle_start - reagle_start).total_seconds()
        if delta >= 0:
            reagle_step_seconds = delta

    return {
        "artifact_name": artifact_name,
        "reagle_step_seconds": reagle_step_seconds,
    }


def load_metrics_pair(
    repo: str,
    run_id: int,
    artifact: dict[str, Any],
    cache_dir: Path,
) -> dict[str, Any]:
    artifact_name = artifact["name"]
    artifact_dir = download_metric_artifact(repo, run_id, artifact_name, cache_dir)
    reagle_candidates = sorted(artifact_dir.rglob("reagle_metrics.json"))
    beagle_candidates = sorted(artifact_dir.rglob("beagle_metrics.json"))

    if not reagle_candidates or not beagle_candidates:
        raise RuntimeError(f"artifact {artifact_name} in run {run_id} does not contain both metrics JSON files")

    reagle_raw = json.loads(reagle_candidates[0].read_text(encoding="utf-8"))
    beagle_raw = json.loads(beagle_candidates[0].read_text(encoding="utf-8"))
    reagle_flat = filter_metrics(flatten_numeric_values(reagle_raw))
    beagle_flat = filter_metrics(flatten_numeric_values(beagle_raw))

    return {
        "name": artifact_name,
        "reagle_flat": reagle_flat,
        "beagle_flat": beagle_flat,
        "reagle_minus_beagle_flat": subtract_numeric_maps(reagle_flat, beagle_flat),
    }


def collect_iqa_artifacts(
    repo: str,
    run_details: dict[str, Any],
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    run_id = run_database_id(run_details)
    jobs = [job for job in run_details.get("jobs", []) if (job.get("name") or "").startswith("impute-and-measure")]
    job_by_artifact: dict[str, dict[str, Any]] = {}

    for job in jobs:
        job_id = int(job["databaseId"])
        try:
            log_text = load_job_log(repo, run_id, job_id, cache_dir)
        except Exception:
            continue
        parsed = parse_iqa_job_log(log_text)
        artifact_name = parsed.get("artifact_name")
        if artifact_name:
            job_by_artifact[artifact_name] = {
                "reagle_step_seconds": parsed.get("reagle_step_seconds"),
            }

    artifact_reports: list[dict[str, Any]] = []
    artifacts = cached_metric_artifacts(run_id, cache_dir)
    if not artifacts:
        artifacts = [
            artifact
            for artifact in list_run_artifacts(repo, run_id)
            if artifact.get("name", "").startswith("metrics-") and not artifact.get("expired", False)
        ]
    artifacts.sort(key=lambda item: item["name"])

    for artifact in artifacts:
        report = load_metrics_pair(repo, run_id, artifact, cache_dir)
        log_data = job_by_artifact.get(report["name"])
        if log_data is not None:
            if log_data.get("reagle_step_seconds") is not None:
                report["reagle_step_seconds"] = log_data["reagle_step_seconds"]
                report["reagle_flat"]["reagle_step_seconds"] = float(log_data["reagle_step_seconds"])
        artifact_reports.append(report)

    return artifact_reports, job_by_artifact


def run_has_metric_artifacts(
    repo: str,
    run: dict[str, Any],
    base_cache: dict[int, dict[str, Any]],
    cache_dir: Path,
) -> bool:
    run_id = int(run["id"])
    cached = base_cache.get(run_id)
    if cached is not None and "has_metric_artifacts" in cached:
        return bool(cached["has_metric_artifacts"])

    cached_artifacts = cached_metric_artifacts(run_id, cache_dir)
    if cached_artifacts:
        has_metrics = True
    else:
        has_metrics = any(
            artifact.get("name", "").startswith("metrics-") and not artifact.get("expired", False)
            for artifact in list_run_artifacts(repo, run_id)
        )
    entry = cached or {}
    entry.setdefault("run", summarize_run(run))
    entry["has_metric_artifacts"] = has_metrics
    base_cache[run_id] = entry
    return has_metrics


def find_baseline_main_run(
    repo: str,
    main_runs: list[dict[str, Any]],
    pr_created_at: str,
    base_cache: dict[int, dict[str, Any]],
    cache_dir: Path,
) -> dict[str, Any] | None:
    pr_created = parse_iso(pr_created_at)
    if pr_created is None:
        return None

    for run in main_runs:
        created_at = parse_run_created_at(run)
        if created_at is None or created_at >= pr_created:
            continue
        if run_has_metric_artifacts(repo, run, base_cache, cache_dir):
            return run
    return None


def build_baseline_report(
    repo: str,
    pr_artifacts: list[dict[str, Any]],
    base_run: dict[str, Any] | None,
    base_cache: dict[int, dict[str, Any]],
    cache_dir: Path,
) -> dict[str, Any] | None:
    if base_run is None:
        return None

    base_run_id = int(base_run["id"])
    cached = base_cache.get(base_run_id)
    if cached is None or "artifacts" not in cached:
        base_artifacts, _ = collect_iqa_artifacts(repo, base_run, cache_dir)
        entry = cached or {}
        entry.update(
            {
            "run": summarize_run(base_run),
            "artifacts": {artifact["name"]: artifact for artifact in base_artifacts},
            "has_metric_artifacts": bool(base_artifacts),
            }
        )
        base_cache[base_run_id] = entry

    cached = base_cache[base_run_id]
    base_artifacts_by_name = cached["artifacts"]
    matched_artifact_names: list[str] = []
    reagle_delta_maps: list[dict[str, float]] = []
    beagle_delta_maps: list[dict[str, float]] = []

    for pr_artifact in pr_artifacts:
        name = pr_artifact["name"]
        base_artifact = base_artifacts_by_name.get(name)
        if base_artifact is None:
            continue

        reagle_delta = subtract_numeric_maps(pr_artifact["reagle_flat"], base_artifact["reagle_flat"])
        beagle_delta = subtract_numeric_maps(pr_artifact["beagle_flat"], base_artifact["beagle_flat"])
        reagle_delta_maps.append(reagle_delta)
        beagle_delta_maps.append(beagle_delta)
        matched_artifact_names.append(name)

    return {
        "run": cached["run"],
        "matched_artifact_names": matched_artifact_names,
        "reagle_delta_flat_mean": mean_numeric_maps(reagle_delta_maps),
        "beagle_delta_flat_mean": mean_numeric_maps(beagle_delta_maps),
    }


def collect_ci_report(repo: str, pr: dict[str, Any], cache_dir: Path) -> dict[str, Any] | None:
    latest = latest_workflow_run(repo, CI_WORKFLOW_FILE, pr["head_branch"])
    if latest is None:
        return None

    run_id = int(latest["databaseId"])
    run_details = gh_run_details(repo, run_id)
    jobs = sorted(run_details.get("jobs", []), key=lambda item: item.get("name", ""))
    jobs_by_name = {job["name"]: job for job in jobs}

    parsed_logs: dict[str, dict[str, Any]] = {}
    total_passed = 0
    total_failed = 0

    for job_name in [REFERENCE_JOB_NAME, CHR21_JOB_NAME]:
        job = jobs_by_name.get(job_name)
        if job is None:
            continue
        try:
            log_text = load_job_log(repo, run_id, int(job["databaseId"]), cache_dir)
        except Exception as exc:
            parsed_logs[job_name] = {"error": str(exc), **summarize_job_state(job)}
            continue

        test_results = parse_test_results(log_text)
        total_passed += test_results["passed"]
        total_failed += test_results["failed"]
        parsed_logs[job_name] = {**summarize_job_state(job), "test_results": test_results}

        if job_name == REFERENCE_JOB_NAME:
            parsed_logs[job_name]["seed_metrics"] = parse_seed_metrics(log_text)
            parsed_logs[job_name]["beagle_reference"] = parse_beagle_reference_stats(log_text)
        if job_name == CHR21_JOB_NAME:
            parsed_logs[job_name]["target_test"] = parse_chr21_target_result(log_text)
            parsed_logs[job_name]["metrics"] = parse_chr21_metrics(log_text)

    aggregate = {
        "passed": total_passed,
        "failed": total_failed,
        "total": total_passed + total_failed,
    }

    return {
        "run": summarize_run(run_details),
        "aggregate_test_results": aggregate,
        "reference_comparison": parsed_logs.get(REFERENCE_JOB_NAME),
        "reference_comparison_chr21": parsed_logs.get(CHR21_JOB_NAME),
    }


def collect_iqa_report(
    repo: str,
    pr: dict[str, Any],
    main_runs: list[dict[str, Any]],
    base_cache: dict[int, dict[str, Any]],
    cache_dir: Path,
) -> dict[str, Any] | None:
    latest = latest_workflow_run(repo, IQA_WORKFLOW_FILE, pr["head_branch"])
    if latest is None:
        return None

    run_id = int(latest["databaseId"])
    run_details = gh_run_details(repo, run_id)
    artifact_reports, _ = collect_iqa_artifacts(repo, run_details, cache_dir)
    baseline_run = find_baseline_main_run(repo, main_runs, pr["created_at"], base_cache, cache_dir)
    baseline = build_baseline_report(repo, artifact_reports, baseline_run, base_cache, cache_dir)

    return {
        "run": summarize_run(run_details),
        "metric_artifacts": artifact_reports,
        "baseline_main": baseline,
    }


def collect_pr_report(
    repo: str,
    pr: dict[str, Any],
    main_runs: list[dict[str, Any]],
    base_cache: dict[int, dict[str, Any]],
    cache_dir: Path,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "pr": pr,
        "ci": None,
        "iqa": None,
    }
    errors: list[str] = []

    try:
        report["ci"] = collect_ci_report(repo, pr, cache_dir)
    except Exception as exc:
        errors.append(f"ci: {exc}")

    try:
        report["iqa"] = collect_iqa_report(repo, pr, main_runs, base_cache, cache_dir)
    except Exception as exc:
        errors.append(f"iqa: {exc}")

    if errors:
        report["errors"] = errors

    return report


def format_optional(value: float | None, fmt: str) -> str:
    if value is None:
        return "-"
    return fmt.format(value)


def headline_concord_delta(report: dict[str, Any]) -> str:
    ci = report.get("ci") or {}
    ref = ci.get("reference_comparison") or {}
    seeds = ref.get("seed_metrics") or {}
    mean = seeds.get("mean_delta_percent")
    return format_optional(mean, "{:+.2f}%")


def headline_beagle_ref(report: dict[str, Any]) -> str:
    ci = report.get("ci") or {}
    ref = ci.get("reference_comparison") or {}
    stats = ref.get("beagle_reference") or {}
    passed = stats.get("passed")
    total = stats.get("total")
    if not isinstance(passed, int) or not isinstance(total, int) or total <= 0:
        return "-"
    return f"{passed}/{total}"


def headline_chr21_delta(report: dict[str, Any], key: str, fmt: str) -> str:
    ci = report.get("ci") or {}
    chr21 = ci.get("reference_comparison_chr21") or {}
    metrics = chr21.get("metrics") or {}
    deltas = metrics.get("reagle_minus_beagle") or {}
    timing = metrics.get("timing") or {}
    value = deltas.get(key)
    if value is None:
        value = timing.get(key)
    if not is_numeric(value):
        return "-"
    return format_optional(float(value), fmt)


def print_summary(report: dict[str, Any]) -> None:
    rows = report["pull_requests"]
    print(
        f"{'PR':<6} {'Concord Δ':<10} {'Beagle Ref':<11} {'ΔR²':<10} "
        f"{'ΔIQS':<10} {'ΔHell':<10} {'ΔPhase':<10} {'ΔSER':<10} {'ΔTime(s)':<10} Title"
    )
    print("-" * 133)

    for row in rows:
        print(
            f"#{row['pr']['number']:<5} "
            f"{headline_concord_delta(row):<10} "
            f"{headline_beagle_ref(row):<11} "
            f"{headline_chr21_delta(row, 'r_squared', '{:+.4f}'):<10} "
            f"{headline_chr21_delta(row, 'iqs', '{:+.4f}'):<10} "
            f"{headline_chr21_delta(row, 'hellinger_score', '{:+.4f}'):<10} "
            f"{headline_chr21_delta(row, 'phase_concordance', '{:+.4f}'):<10} "
            f"{headline_chr21_delta(row, 'switch_error_rate', '{:+.4f}'):<10} "
            f"{headline_chr21_delta(row, 'reagle_minus_beagle_runtime_sec', '{:+.2f}'):<10} "
            f"{row['pr']['title']}"
        )


def build_report(repo: str, prs: list[dict[str, Any]], cache_dir: Path, max_workers: int) -> dict[str, Any]:
    if prs:
        oldest_pr_created_at = min(parse_iso(pr["created_at"]) for pr in prs if parse_iso(pr["created_at"]) is not None)
    else:
        oldest_pr_created_at = None

    main_runs = list_workflow_runs(
        repo,
        IQA_WORKFLOW_FILE,
        branch="main",
        stop_before=oldest_pr_created_at,
    )

    base_cache: dict[int, dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(collect_pr_report, repo, pr, main_runs, base_cache, cache_dir): pr
            for pr in prs
        }
        for future in concurrent.futures.as_completed(future_map):
            results.append(future.result())

    results.sort(key=lambda item: item["pr"]["number"], reverse=True)
    return {
        "repo": repo,
        "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pull_request_count": len(results),
        "pull_requests": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect every open-PR metric currently exposed by CI and Imputation Quality Assessment."
    )
    parser.add_argument("--repo", default="", help="GitHub repo slug owner/repo. Defaults to the current gh repo.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of open PRs to inspect.")
    parser.add_argument(
        "--cache-dir",
        default="/tmp/open_pr_metrics_cache",
        help="Directory used for cached job logs and downloaded artifacts.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum concurrent PR collectors. Use 1 to avoid GitHub Actions API throttling.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional path to write the full JSON report. If omitted, JSON is written to stdout unless --summary is set.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact human summary instead of emitting JSON to stdout.",
    )
    args = parser.parse_args()

    repo = detect_repo(args.repo)
    cache_dir = Path(args.cache_dir)
    prs = list_open_prs(repo, args.limit)
    report = build_report(repo, prs, cache_dir, max(1, args.max_workers))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.summary:
        print_summary(report)
        return 0

    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
