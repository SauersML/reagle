#!/usr/bin/env python3

import argparse
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone

def get_gh_bin():
    gh_bin = shutil.which("gh")
    if gh_bin:
        return gh_bin
    fallback = "/opt/homebrew/bin/gh"
    if os.path.exists(fallback):
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
        print(exc.stderr.strip(), file=sys.stderr)
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

def iso_to_dt(value):
    if not value:
        return None
    # GitHub returns timestamps like 2024-01-01T12:34:56Z
    return datetime.fromisoformat(value.replace("Z", "+00:00"))

def format_dt(value):
    if not value:
        return "-"
    dt = iso_to_dt(value)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def format_duration(started_at, completed_at):
    start = iso_to_dt(started_at)
    end = iso_to_dt(completed_at)
    if not start or not end:
        return "-"
    seconds = int((end - start).total_seconds())
    if seconds < 0:
        return "-"
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"

def format_elapsed(started_at):
    start = iso_to_dt(started_at)
    if not start:
        return "-"
    end = datetime.now(timezone.utc)
    seconds = int((end - start).total_seconds())
    if seconds < 0:
        return "-"
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"

def get_repo():
    output = run_cmd(gh_cmd(["repo", "view", "--json", "nameWithOwner"]))
    if not output:
        return None
    data = json.loads(output)
    return data.get("nameWithOwner")

def get_runs(limit):
    output = run_cmd(gh_cmd([
        "run", "list",
        "--workflow", "CI.yml",
        "--branch", "main",
        "--limit", str(limit),
        "--json", "databaseId,createdAt,conclusion,status,event,displayTitle,headBranch,headSha,number"
    ]))
    if not output:
        return []
    runs = json.loads(output)
    # Ensure only main branch and non-PR runs
    return [
        run for run in runs
        if run.get("headBranch") == "main" and run.get("event") != "pull_request"
    ]

def get_jobs(repo, run_id):
    output = run_cmd(gh_cmd([
        "api",
        f"repos/{repo}/actions/runs/{run_id}/jobs",
        "--paginate"
    ]))
    if not output:
        return []
    data = json.loads(output)
    return data.get("jobs", [])

def find_rust_test_step(jobs, step_name):
    for job in jobs:
        steps = job.get("steps") or []
        for step in steps:
            if step.get("name") == step_name:
                return job, step
    return None, None

def job_step_status(job, step_name):
    steps = job.get("steps") or []
    for step in steps:
        if step.get("name") == step_name:
            return step.get("status"), step.get("conclusion")
    return None, None

def job_logs_contain(repo, job_id, needle):
    output = run_cmd_bytes(gh_cmd([
        "api",
        f"repos/{repo}/actions/jobs/{job_id}/logs"
    ]), allow_404=True)
    if not output:
        return False
    try:
        with zipfile.ZipFile(io.BytesIO(output)) as zf:
            for name in zf.namelist():
                data = zf.read(name)
                text = data.decode(errors="ignore")
                if needle in text:
                    return True
    except zipfile.BadZipFile:
        return False
    return False

def job_log_diagnostics(repo, job_id, needle):
    output = run_cmd_bytes([
        "gh", "api",
        f"repos/{repo}/actions/jobs/{job_id}/logs"
    ], allow_404=True)
    if not output:
        return False, "no logs (404 or empty)"
    if output.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(output)) as zf:
                names = zf.namelist()
                for name in names:
                    data = zf.read(name)
                    text = strip_ansi(data.decode(errors="ignore"))
                    if needle in text:
                        return True, f"hit in {name}"
                return False, f"scanned {len(names)} file(s)"
        except zipfile.BadZipFile:
            return False, "bad zip"
    text = strip_ansi(output.decode("utf-8-sig", errors="ignore"))
    if needle in text:
        return True, f"hit in plain log ({len(text)} chars)"
    snippet = text[:200].replace("\n", " ").strip()
    if not snippet:
        snippet = "<empty or binary response>"
    return False, f"plain log scanned ({len(text)} chars): {snippet}"

def strip_ansi(value):
    # Strip common ANSI escape sequences used for colored output.
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", value)

def build_table(rows, headers):
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt_row(values):
        return "  ".join(value.ljust(widths[idx]) for idx, value in enumerate(values))

    lines = [fmt_row(headers), fmt_row(["-" * w for w in widths])]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("CI_TIMING_LIMIT", "25")),
        help="Number of runs to inspect (default: 25 or CI_TIMING_LIMIT)",
    )
    args = parser.parse_args()

    repo = get_repo()
    if not repo:
        print("Failed to determine repo. Is gh authenticated?", file=sys.stderr)
        sys.exit(1)

    runs = get_runs(args.limit)
    if not runs:
        print("No runs found for CI.yml on main.")
        return

    rows = []
    diag_lines = []
    for run in runs:
        run_id = run.get("databaseId")
        jobs = get_jobs(repo, run_id) if run_id else []
        run_number = run.get("number", "-")
        run_sha = (run.get("headSha") or "-")[:8] if run.get("headSha") else "-"
        if not jobs:
            diag_lines.append(f"Run #{run_number}: no jobs found")
            continue

        rust_job, rust_step = find_rust_test_step(jobs, "Run Rust Tests")
        step_status = "-"
        step_conclusion = "-"
        duration = "-"
        if rust_step:
            step_status = rust_step.get("status") or "-"
            step_conclusion = rust_step.get("conclusion") or "-"
            duration = format_duration(rust_step.get("started_at"), rust_step.get("completed_at"))

        log_hit = False
        in_progress = False
        run_status = run.get("status") or "-"
        run_conclusion = run.get("conclusion") or "-"
        diag_lines.append(
            f"Run #{run_number}: {len(jobs)} job(s), status={run_status}, conclusion={run_conclusion}"
        )
        for job in jobs:
            if job.get("head_branch") != "main":
                continue
            job_id = job.get("id")
            job_name = job.get("name") or "-"
            job_status = job.get("status") or "-"
            job_conclusion = job.get("conclusion") or "-"
            step_status, step_conclusion = job_step_status(job, "Run Rust Tests")
            step_bits = ""
            if step_status or step_conclusion:
                step_bits = f", step=Run Rust Tests:{step_status or '-'}:{step_conclusion or '-'}"
            if step_status == "in_progress":
                in_progress = True
            log_hit, reason = job_log_diagnostics(
                repo, job_id, 'Finished `test` profile'
            )
            diag_lines.append(
                f"Run #{run_number}: job {job_id} ({job_name}) status={job_status} conclusion={job_conclusion}{step_bits} -> {reason}"
            )
            if log_hit:
                break
        if not log_hit and not in_progress:
            continue

        if not log_hit and in_progress and rust_step:
            duration = format_elapsed(rust_step.get("started_at"))
            step_status = "in_progress"
            step_conclusion = "-"

        rows.append([
            f"#{run_number}",
            run_sha,
            format_dt(run.get("createdAt")),
            run.get("status") or "-",
            run.get("conclusion") or "-",
            step_status,
            step_conclusion,
            duration,
        ])

    headers = [
        "Run",
        "SHA",
        "Created",
        "Run Status",
        "Run Conclusion",
        "Step Status",
        "Step Conclusion",
        "Rust Test Duration",
    ]

    if rows:
        print(build_table(rows, headers))
    else:
        print("No runs found with 'Finished `test` profile' in logs.")
    if diag_lines:
        print("\nDiagnostics:")
        print("\n".join(diag_lines))

if __name__ == "__main__":
    main()
