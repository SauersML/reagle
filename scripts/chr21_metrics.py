#!/usr/bin/env python3

import argparse
import concurrent.futures
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
ASSERTION_PREFIX = "Reagle worse than Beagle on "


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


def get_open_prs(repo, limit=None):
    prs = []
    page = 1

    while True:
        output = run_cmd(
            gh_cmd(
                [
                    "api",
                    f"repos/{repo}/pulls?state=open&per_page=100&page={page}",
                ]
            )
        )
        if not output:
            break

        page_items = json.loads(output)
        if not page_items:
            break

        for item in page_items:
            prs.append(
                {
                    "number": item["number"],
                    "title": item["title"],
                    "headRefName": item["head"]["ref"],
                }
            )
            if limit is not None and len(prs) >= limit:
                return prs

        page += 1

    return prs


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


def normalize_log_line(line):
    return re.sub(r"^\ufeff?\d{4}-\d{2}-\d{2}T[0-9:.]+Z\s*", "", line).strip()


def extract_target_result(log_text):
    if not log_text:
        return None, "empty log"

    lines = [normalize_log_line(line) for line in log_text.splitlines()]

    for idx, line in enumerate(lines):
        if TARGET_TEST_NAME not in line:
            continue

        window_end = min(len(lines), idx + 250)

        for j in range(idx, window_end):
            current = lines[j]
            if current == f"test {TARGET_TEST_NAME} ... ok":
                return "PASS", "target test passed"

            if current.startswith(f"thread '{TARGET_TEST_NAME}'") and "panicked at" in current:
                for k in range(j + 1, min(window_end, j + 8)):
                    candidate = lines[k]
                    if not candidate or candidate.startswith("note:"):
                        continue
                    return candidate, "panic assertion"

            if current.startswith(ASSERTION_PREFIX):
                return current, "matched assertion line"

    return None, "target test result not found"


def parse_target_job_result_for_run(repo, run):
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
    result, reason = extract_target_result(log_text)
    return result, reason, job_status, job_conclusion


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
            "result": None,
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
            "result": None,
            "note": "run missing databaseId",
            "title": title,
        }

    result, reason, latest_job_status, latest_job_conclusion = parse_target_job_result_for_run(repo, latest_run)
    if result:
        return {
            "pr": number,
            "branch": branch,
            "status": f"{latest_job_status}/{latest_job_conclusion}",
            "result": result,
            "note": reason,
            "title": title,
        }

    for candidate_run in runs[1:]:
        if (candidate_run.get("status") or "") != "in_progress":
            continue
        candidate_result, candidate_reason, _, _ = parse_target_job_result_for_run(repo, candidate_run)
        if candidate_result:
            return {
                "pr": number,
                "branch": branch,
                "status": f"{latest_job_status}/{latest_job_conclusion}",
                "result": candidate_result,
                "note": f"sourced from in-progress run {candidate_run.get('databaseId')} ({candidate_reason})",
                "title": title,
            }

    note = f"latest run: {reason}"
    for candidate_run in runs[1:]:
        if (candidate_run.get("status") or "") != "in_progress":
            continue
        _, candidate_reason, _, _ = parse_target_job_result_for_run(repo, candidate_run)
        note = f"{note}; in-progress run {candidate_run.get('databaseId')}: {candidate_reason}"
        break

    return {
        "pr": number,
        "branch": branch,
        "status": f"{run_status}/{run_conclusion}",
        "result": None,
        "note": note,
        "title": title,
    }


def print_results(rows):
    print(f"{'PR':<6} {'Status':<20} {'Branch':<42} Result")
    print("-" * 160)

    for row in rows:
        if not row["result"]:
            print(f"#{row['pr']:<5} {row['status']:<20} {row['branch']:<42} -")
            print(f"      note: {row['note']}")
            continue

        print(f"#{row['pr']:<5} {row['status']:<20} {row['branch']:<42} {row['result']}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "For each open PR, inspect the chr21 reference-comparison CI job and print the "
            "target test's assertion line or PASS."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max open PRs to inspect (default: all open PRs)",
    )
    args = parser.parse_args()

    repo = get_repo()
    if not repo:
        print("Failed to determine repo. Is gh authenticated?", file=sys.stderr)
        sys.exit(1)

    prs = get_open_prs(repo, args.limit)
    if not prs:
        print("No open PRs found.")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        rows = list(executor.map(lambda pr: collect_row(repo, pr), prs))

    print_results(rows)


if __name__ == "__main__":
    main()
