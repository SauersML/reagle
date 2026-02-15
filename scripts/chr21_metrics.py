#!/usr/bin/env python3

import argparse
import io
import json
import re
import shutil
import subprocess
import sys
import zipfile

TARGET_WORKFLOW = "CI"
TARGET_JOB_NAME = "reference-comparison-chr21"
TARGET_TEST_NAME = "test_reference_comparison_full_chr21_ref1000_target10"
TARGET_SECTION_HEADER = "=== Fast Accuracy Metrics (chr21, ref=1000, target=10) ==="


def get_gh_bin():
    gh_bin = shutil.which("gh")
    if gh_bin:
        return gh_bin
    fallback = "/opt/homebrew/bin/gh"
    if shutil.which(fallback):
        return fallback
    return None


def gh_cmd(args):
    gh_bin = get_gh_bin()
    if not gh_bin:
        print("gh CLI not found in PATH.", file=sys.stderr)
        sys.exit(1)
    return [gh_bin] + args


def run_cmd(cmd):
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        if stderr:
            print(stderr, file=sys.stderr)
        return None


def run_cmd_bytes(cmd, allow_404=False):
    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
        return result.stdout
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="ignore") if exc.stderr else ""
        if allow_404 and "HTTP 404" in stderr:
            return None
        if stderr:
            print(stderr.strip(), file=sys.stderr)
        return None


def strip_ansi(value):
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", value)


def get_repo():
    output = run_cmd(gh_cmd(["repo", "view", "--json", "nameWithOwner"]))
    if not output:
        return None
    data = json.loads(output)
    return data.get("nameWithOwner")


def get_open_prs(limit):
    output = run_cmd(
        gh_cmd([
            "pr",
            "list",
            "--limit",
            str(limit),
            "--json",
            "number,title,headRefName",
        ])
    )
    if not output:
        return []
    return json.loads(output)


def get_latest_run_for_branch(branch_name):
    runs = get_recent_runs_for_branch(branch_name, 1)
    if not runs:
        return None
    return runs[0]


def get_recent_runs_for_branch(branch_name, limit):
    output = run_cmd(
        gh_cmd(
            [
                "run",
                "list",
                "--workflow",
                TARGET_WORKFLOW,
                "--branch",
                branch_name,
                "--limit",
                str(limit),
                "--json",
                "databaseId,status,conclusion,url,headBranch,headSha,createdAt,displayTitle",
            ]
        )
    )
    if not output:
        return []
    runs = json.loads(output)
    return runs if runs else []


def get_jobs(repo, run_id):
    output = run_cmd(
        gh_cmd(
            [
                "api",
                f"repos/{repo}/actions/runs/{run_id}/jobs",
                "--paginate",
            ]
        )
    )
    if not output:
        return []
    data = json.loads(output)
    return data.get("jobs", [])


def get_job_by_name(jobs, job_name):
    for job in jobs:
        if (job.get("name") or "") == job_name:
            return job
    return None


def get_job_log_text(repo, job_id):
    output = run_cmd_bytes(
        gh_cmd(["api", f"repos/{repo}/actions/jobs/{job_id}/logs"]),
        allow_404=True,
    )
    if not output:
        return ""

    if output.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(output)) as zf:
                chunks = []
                for name in zf.namelist():
                    data = zf.read(name)
                    chunks.append(strip_ansi(data.decode(errors="ignore")))
                return "\n".join(chunks)
        except zipfile.BadZipFile:
            return ""

    return strip_ansi(output.decode("utf-8-sig", errors="ignore"))


def parse_reagle_metrics_line(line):
    pattern = re.compile(
        r"REAGLE:\s+"
        r"sites=(?P<sites>\d+)\s+"
        r"genotypes=(?P<genotypes>\d+)\s+"
        r"r2=Some\((?P<r2>[0-9.eE+-]+)\)\s+"
        r"iqs=Some\((?P<iqs>[0-9.eE+-]+)\)\s+"
        r"hellinger=Some\((?P<hellinger>[0-9.eE+-]+)\)\s+"
        r"SER=Some\((?P<ser>[0-9.eE+-]+)\).*?"
        r"phase_conc=Some\((?P<phase_conc>[0-9.eE+-]+)\)"
    )
    match = pattern.search(line)
    if not match:
        return None
    out = match.groupdict()
    out["raw"] = line.strip()
    return out


def extract_target_reagle_metrics(log_text):
    if not log_text:
        return None, "empty log"

    lines = log_text.splitlines()

    # Primary strategy: anchor on test name, then section header, then REAGLE line.
    for idx, line in enumerate(lines):
        if TARGET_TEST_NAME not in line:
            continue

        header_idx = None
        for j in range(idx, min(len(lines), idx + 250)):
            if TARGET_SECTION_HEADER in lines[j]:
                header_idx = j
                break

        if header_idx is None:
            continue

        for k in range(header_idx, min(len(lines), header_idx + 30)):
            if "REAGLE:" in lines[k]:
                parsed = parse_reagle_metrics_line(lines[k])
                if parsed:
                    return parsed, "matched target test + section"

    # Fallback: any matching section + REAGLE line (last occurrence wins).
    last_candidate = None
    for idx, line in enumerate(lines):
        if TARGET_SECTION_HEADER not in line:
            continue
        for k in range(idx, min(len(lines), idx + 30)):
            if "REAGLE:" in lines[k]:
                parsed = parse_reagle_metrics_line(lines[k])
                if parsed:
                    last_candidate = parsed

    if last_candidate is not None:
        return last_candidate, "matched section fallback"

    return None, "target section/reagle line not found"


def parse_target_job_metrics_for_run(repo, run):
    run_id = run.get("databaseId")
    if not run_id:
        return None, "run missing databaseId", "-", "-"

    jobs = get_jobs(repo, run_id)
    job = get_job_by_name(jobs, TARGET_JOB_NAME)
    if not job:
        return None, f"job '{TARGET_JOB_NAME}' not found", "-", "-"

    job_id = job.get("id")
    job_status = job.get("status") or "-"
    job_conclusion = job.get("conclusion") or "-"
    if not job_id:
        return None, "target job missing id", job_status, job_conclusion

    log_text = get_job_log_text(repo, job_id)
    metrics, reason = extract_target_reagle_metrics(log_text)
    return metrics, reason, job_status, job_conclusion


def collect_row(repo, pr):
    number = pr.get("number")
    title = pr.get("title") or ""
    branch = pr.get("headRefName") or ""

    runs = get_recent_runs_for_branch(branch, 10)
    if not runs:
        return {
            "pr": number,
            "branch": branch,
            "status": "NO_RUN",
            "metrics": None,
            "note": "no CI run found",
            "title": title,
        }

    latest_run = runs[0]
    run_id = latest_run.get("databaseId")
    run_status = latest_run.get("status") or "-"
    run_conclusion = latest_run.get("conclusion") or "-"
    if not run_id:
        return {
            "pr": number,
            "branch": branch,
            "status": f"{run_status}/{run_conclusion}",
            "metrics": None,
            "note": "run missing databaseId",
            "title": title,
        }

    metrics, reason, latest_job_status, latest_job_conclusion = parse_target_job_metrics_for_run(repo, latest_run)
    if metrics:
        return {
            "pr": number,
            "branch": branch,
            "status": f"{latest_job_status}/{latest_job_conclusion}",
            "metrics": metrics,
            "note": reason,
            "title": title,
        }

    for candidate_run in runs[1:]:
        if (candidate_run.get("status") or "") != "in_progress":
            continue
        candidate_metrics, candidate_reason, candidate_job_status, candidate_job_conclusion = (
            parse_target_job_metrics_for_run(repo, candidate_run)
        )
        if candidate_metrics:
            return {
                "pr": number,
                "branch": branch,
                "status": f"{latest_job_status}/{latest_job_conclusion}",
                "metrics": candidate_metrics,
                "note": f"sourced from in-progress run {candidate_run.get('databaseId')} ({candidate_reason})",
                "title": title,
            }

    note = f"latest run: {reason}"
    for candidate_run in runs[1:]:
        if (candidate_run.get("status") or "") != "in_progress":
            continue
        _, candidate_reason, _, _ = parse_target_job_metrics_for_run(repo, candidate_run)
        note = f"{note}; in-progress run {candidate_run.get('databaseId')}: {candidate_reason}"
        break

    return {
        "pr": number,
        "branch": branch,
        "status": f"{run_status}/{run_conclusion}",
        "metrics": None,
        "note": note,
        "title": title,
    }


def print_results(rows):
    print(
        f"{'PR':<6} {'Status':<20} {'r2':<14} {'iqs':<14} {'hellinger':<14} {'SER':<14} {'phase_conc':<14} {'sites':<10} {'genotypes':<12} Branch"
    )
    print("-" * 150)

    for row in rows:
        metrics = row["metrics"]
        if not metrics:
            print(
                f"#{row['pr']:<5} {row['status']:<20} {'-':<14} {'-':<14} {'-':<14} {'-':<14} {'-':<14} {'-':<10} {'-':<12} {row['branch']}"
            )
            print(f"      note: {row['note']}")
            continue

        print(
            f"#{row['pr']:<5} {row['status']:<20} {metrics['r2']:<14} {metrics['iqs']:<14} {metrics['hellinger']:<14} {metrics['ser']:<14} {metrics['phase_conc']:<14} {metrics['sites']:<10} {metrics['genotypes']:<12} {row['branch']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "For each open PR, inspect latest CI run and print REAGLE chr21(ref=1000,target=10) "
            "metrics from reference-comparison-chr21 job logs."
        )
    )
    parser.add_argument("--limit", type=int, default=100, help="Max open PRs to inspect (default: 100)")
    args = parser.parse_args()

    repo = get_repo()
    if not repo:
        print("Failed to determine repo. Is gh authenticated?", file=sys.stderr)
        sys.exit(1)

    prs = get_open_prs(args.limit)
    if not prs:
        print("No open PRs found.")
        return

    rows = []
    for pr in prs:
        rows.append(collect_row(repo, pr))

    print_results(rows)


if __name__ == "__main__":
    main()
