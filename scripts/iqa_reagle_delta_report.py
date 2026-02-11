#!/usr/bin/env python3
"""Compare REAGLE metrics across Imputation Quality Assessment runs vs a base run.

This script:
1. Resolves a base run from an Actions run URL, run id, or run number (e.g. #1298).
2. Reads REAGLE metrics from the four metrics-* artifacts in that base run.
3. Scans every subsequent "Imputation Quality Assessment" run (including in-progress/queued).
4. For any subsequent run artifact already available, computes deltas vs base.
5. Prints emoji-coded improvement/worsening per metric and saves a 4-panel delta plot.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    ("unphased_concordance", "Unphased concordance", "higher"),
    ("r_squared", "Overall R²", "higher"),
    ("hellinger_score", "Hellinger Score", "lower"),
    ("sen_mean", "SEN", "higher"),
    ("iqs", "IQS", "higher"),
    ("homref_accuracy", "HOMREF accuracy", "higher"),
    ("het_accuracy", "HET accuracy", "higher"),
    ("homalt_accuracy", "HOMALT accuracy", "higher"),
    ("reagle_step_seconds", "Reagle step seconds", "lower"),
]

EPS = 1e-12


def run_cmd(command: list[str]) -> str:
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{proc.stderr.strip()}")
    return proc.stdout.strip()


def run_cmd_maybe(command: list[str]) -> str | None:
    proc = subprocess.run(command, capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out if out else None


def detect_repo(default_repo: str) -> str:
    try:
        url = run_cmd(["git", "config", "--get", "remote.origin.url"])
    except Exception:
        return default_repo

    # Handles: https://github.com/owner/repo(.git) and git@github.com:owner/repo(.git)
    m = re.search(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+)(?:\.git)?$", url)
    if not m:
        return default_repo
    return f"{m.group('owner')}/{m.group('repo')}"


class GitHubClient:
    def __init__(self, repo: str):
        self.repo = repo
        self.token = run_cmd(["gh", "auth", "token"]).strip()
        if not self.token:
            raise RuntimeError("Could not get GitHub token from gh auth.")

    def _request_json(self, url: str) -> tuple[dict, str | None]:
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            link = resp.headers.get("Link")
            return data, link

    def get(self, path: str, params: dict | None = None) -> dict:
        base = f"https://api.github.com{path}"
        if params:
            q = urllib.parse.urlencode(params)
            base = f"{base}?{q}"
        data, _ = self._request_json(base)
        return data

    def paginate(self, path: str, params: dict | None = None, list_key: str | None = None) -> list[dict]:
        base = f"https://api.github.com{path}"
        if params:
            q = urllib.parse.urlencode(params)
            base = f"{base}?{q}"

        items: list[dict] = []
        next_url: str | None = base
        while next_url:
            data, link = self._request_json(next_url)
            if list_key is None:
                if isinstance(data, list):
                    chunk = data
                else:
                    raise RuntimeError("Expected list response with list_key=None")
            else:
                chunk = data.get(list_key, [])
            items.extend(chunk)
            next_url = parse_next_link(link)
        return items

def parse_next_link(link_header: str | None) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        section = part.strip()
        if 'rel="next"' in section:
            m = re.match(r"<([^>]+)>", section)
            if m:
                return m.group(1)
    return None


def parse_run_input(raw: str) -> tuple[str, int]:
    s = raw.strip()
    m_url = re.search(r"/actions/runs/(\d+)", s)
    if m_url:
        return "id", int(m_url.group(1))

    m_hash = re.match(r"#(\d+)$", s)
    if m_hash:
        return "run_number", int(m_hash.group(1))

    if s.isdigit():
        val = int(s)
        # Heuristic: run ids are large; run_number is usually much smaller.
        if val >= 1_000_000:
            return "id", val
        return "run_number", val

    raise ValueError(f"Unrecognized run input: {raw}")


def parse_iso_time(ts: str) -> dt.datetime:
    return dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def resolve_pr_for_run(
    repo: str, run: dict, pr_cache: dict[str, tuple[int | None, str | None]]
) -> tuple[int | None, str | None]:
    if run.get("event") != "pull_request":
        return None, None

    head_branch = run.get("head_branch") or run.get("headBranch")
    if not head_branch:
        return None, None

    if head_branch in pr_cache:
        return pr_cache[head_branch]

    pr_num: int | None = None
    pr_state: str | None = None
    q1 = run_cmd_maybe(
        [
            "gh",
            "pr",
            "list",
            "-R",
            repo,
            "--state",
            "all",
            "--head",
            str(head_branch),
            "--json",
            "number,state",
            "--jq",
            ".[0]",
        ]
    )
    if q1:
        obj1 = json.loads(q1)
        if isinstance(obj1, dict):
            num = obj1.get("number")
            state = obj1.get("state")
            if isinstance(num, int):
                pr_num = num
            if isinstance(state, str):
                pr_state = state
    else:
        head_sha = run.get("head_sha") or run.get("headSha")
        if head_sha:
            q2 = run_cmd_maybe(
                [
                    "gh",
                    "pr",
                    "list",
                    "-R",
                    repo,
                    "--state",
                    "all",
                    "--search",
                    f"{head_sha} in:commits",
                    "--json",
                    "number,state",
                    "--jq",
                    ".[0]",
                ]
            )
            if q2:
                obj2 = json.loads(q2)
                if isinstance(obj2, dict):
                    num = obj2.get("number")
                    state = obj2.get("state")
                    if isinstance(num, int):
                        pr_num = num
                    if isinstance(state, str):
                        pr_state = state

    pr_cache[head_branch] = (pr_num, pr_state)
    return pr_num, pr_state


def read_reagle_metrics_from_artifact(
    gh: GitHubClient,
    artifact: dict,
    run_id: int,
    cache_dir: Path,
) -> dict | None:
    art_name = artifact["name"]
    run_dir = cache_dir / f"run_{run_id}"
    art_dir = run_dir / safe_name(art_name)
    metrics_path = art_dir / "reagle_metrics.json"

    if metrics_path.exists():
        return json.loads(metrics_path.read_text())

    run_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)

    # Use gh CLI for artifact download to rely on its auth/session handling.
    # This extracts artifact contents under art_dir/<artifact-name>/...
    run_cmd(
        [
            "gh",
            "run",
            "download",
            str(run_id),
            "-R",
            gh.repo,
            "-n",
            art_name,
            "-D",
            str(art_dir),
        ]
    )

    candidates = list(art_dir.rglob("reagle_metrics.json"))
    if not candidates:
        return None

    metrics_path = candidates[0]
    return json.loads(metrics_path.read_text())


def metric_delta_status(metric_name: str, base_val: float | None, cur_val: float | None) -> tuple[str, float | None]:
    if base_val is None or cur_val is None:
        return "⚪", None

    delta = cur_val - base_val
    if abs(delta) <= EPS:
        return "⚪", delta

    direction = next(x[2] for x in METRICS if x[0] == metric_name)
    improved = delta > 0 if direction == "higher" else delta < 0
    return ("🟢" if improved else "🔴"), delta


def fmt(v: float | None) -> str:
    if v is None:
        return "N/A"
    if not math.isfinite(v):
        return "N/A"
    return f"{v:.6f}"


def resolve_base_run(gh: GitHubClient, workflow_name: str, run_input: str) -> dict:
    kind, val = parse_run_input(run_input)
    if kind == "id":
        run = gh.get(f"/repos/{gh.repo}/actions/runs/{val}")
        if run.get("name") != workflow_name:
            raise RuntimeError(
                f"Base run {val} is workflow '{run.get('name')}', expected '{workflow_name}'."
            )
        return run

    # run_number path
    workflow_id = find_workflow_id(gh, workflow_name)
    runs = gh.paginate(
        f"/repos/{gh.repo}/actions/workflows/{workflow_id}/runs",
        params={"per_page": 100},
        list_key="workflow_runs",
    )
    for run in runs:
        if run.get("run_number") == val:
            return run
    raise RuntimeError(f"Could not find workflow run number #{val} for '{workflow_name}'.")


def find_workflow_id(gh: GitHubClient, workflow_name: str) -> int:
    workflows = gh.paginate(
        f"/repos/{gh.repo}/actions/workflows",
        params={"per_page": 100},
        list_key="workflows",
    )
    for wf in workflows:
        if wf.get("name") == workflow_name:
            return int(wf["id"])
    raise RuntimeError(f"Workflow '{workflow_name}' not found in repo {gh.repo}.")


def list_run_artifacts(gh: GitHubClient, run_id: int) -> list[dict]:
    return gh.paginate(
        f"/repos/{gh.repo}/actions/runs/{run_id}/artifacts",
        params={"per_page": 100},
        list_key="artifacts",
    )


def list_run_jobs(gh: GitHubClient, run_id: int) -> list[dict]:
    return gh.paginate(
        f"/repos/{gh.repo}/actions/runs/{run_id}/jobs",
        params={"per_page": 100},
        list_key="jobs",
    )


def read_job_log(gh: GitHubClient, run_id: int, job_id: int, cache_dir: Path) -> str:
    run_dir = cache_dir / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"job_{job_id}.log"
    if log_path.exists():
        return log_path.read_text(errors="replace")

    text = run_cmd(
        [
            "gh",
            "run",
            "view",
            str(run_id),
            "-R",
            gh.repo,
            "--job",
            str(job_id),
            "--log",
        ]
    )
    log_path.write_text(text)
    return text


def _parse_ts_from_log_line(line: str) -> dt.datetime | None:
    # Expected segment after second tab:
    # 2026-02-10T08:42:01.2443380Z === Running Reagle ===
    parts = line.split("\t", 2)
    if len(parts) < 3:
        return None
    payload = parts[2]
    m = re.match(r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s", payload)
    if not m:
        return None
    ts = m.group("ts")
    # fromisoformat handles up to 6 digits; truncate fractional precision if needed.
    if "." in ts:
        base, frac_z = ts.split(".", 1)
        frac = frac_z[:-1]  # strip Z
        frac = frac[:6]
        ts = f"{base}.{frac}Z"
    return dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%fZ" if "." in ts else "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=dt.timezone.utc
    )


def parse_reagle_seconds_and_artifact(log_text: str) -> tuple[float | None, str | None]:
    reagle_start: dt.datetime | None = None
    beagle_start: dt.datetime | None = None
    artifact_name: str | None = None

    for line in log_text.splitlines():
        if "=== Running Reagle ===" in line and reagle_start is None:
            reagle_start = _parse_ts_from_log_line(line)
        if "=== Running Beagle ===" in line and reagle_start is not None and beagle_start is None:
            beagle_start = _parse_ts_from_log_line(line)

        # Prefer finalized upload line, fallback to any metrics-name line.
        m_art = re.search(r"Artifact (metrics-[A-Za-z0-9-]+)(?:\.zip)?", line)
        if m_art:
            artifact_name = m_art.group(1)
        elif artifact_name is None:
            m_name = re.search(r"\bname:\s*(metrics-[A-Za-z0-9-]+)\b", line)
            if m_name:
                artifact_name = m_name.group(1)

    if reagle_start is None or beagle_start is None:
        return None, artifact_name
    seconds = (beagle_start - reagle_start).total_seconds()
    if seconds < 0:
        return None, artifact_name
    return seconds, artifact_name


def parse_reagle_metrics_from_log(log_text: str) -> dict | None:
    """Extract top-level reagle metrics from job log text.

    The integration_test.py metrics stage prints lines like:
      key: reagle=0.775865 beagle=0.890333 delta=-0.114468
    We extract the reagle value for each known metric key.
    Only top-level lines are matched (lines containing by_maf. are skipped).
    """
    metric_keys = {m[0] for m in METRICS}
    result: dict[str, float] = {}
    pattern = re.compile(r"\s+(\S+):\s+reagle=([\d.eE+-]+)\s+beagle=")
    for line in log_text.splitlines():
        if "by_maf." in line:
            continue
        m = pattern.search(line)
        if m:
            key = m.group(1)
            if key in metric_keys:
                try:
                    result[key] = float(m.group(2))
                except ValueError:
                    pass
    return result if result else None


def extract_metrics_from_logs(
    gh: GitHubClient,
    run: dict,
    cache_dir: Path,
) -> dict[str, dict]:
    """Fallback: extract metrics from job logs when artifacts are unavailable."""
    run_id = int(run["id"])
    jobs = list_run_jobs(gh, run_id)
    out: dict[str, dict] = {}
    for job in jobs:
        name = job.get("name", "")
        if not name.startswith("impute-and-measure"):
            continue
        job_id = int(job["id"])
        try:
            log_text = read_job_log(gh, run_id, job_id, cache_dir)
        except Exception:
            continue
        seconds, artifact_name = parse_reagle_seconds_and_artifact(log_text)
        metrics = parse_reagle_metrics_from_log(log_text)
        if metrics is not None:
            if seconds is not None:
                metrics["reagle_step_seconds"] = seconds
            # Use artifact name if detected, otherwise synthesize from job name.
            art_key = artifact_name or f"metrics-{safe_name(name)}"
            out[art_key] = metrics
    return out


def collect_reagle_step_seconds_by_artifact(
    gh: GitHubClient,
    run: dict,
    cache_dir: Path,
) -> dict[str, float]:
    run_id = int(run["id"])
    jobs = list_run_jobs(gh, run_id)
    out: dict[str, float] = {}

    for job in jobs:
        name = job.get("name", "")
        if not name.startswith("impute-and-measure"):
            continue
        job_id = int(job["id"])
        try:
            log_text = read_job_log(gh, run_id, job_id, cache_dir)
        except Exception:
            continue
        seconds, artifact_name = parse_reagle_seconds_and_artifact(log_text)
        if artifact_name and seconds is not None:
            out[artifact_name] = seconds
    return out


def collect_run_metrics(
    gh: GitHubClient,
    run: dict,
    target_artifact_names: list[str],
    cache_dir: Path,
) -> dict[str, dict | None]:
    reagle_secs_by_artifact = collect_reagle_step_seconds_by_artifact(gh, run, cache_dir)
    artifacts = list_run_artifacts(gh, int(run["id"]))
    by_name = {a["name"]: a for a in artifacts if not a.get("expired", False)}
    out: dict[str, dict | None] = {}
    for name in target_artifact_names:
        art = by_name.get(name)
        if art is None:
            out[name] = None
            continue
        try:
            metrics = read_reagle_metrics_from_artifact(gh, art, int(run["id"]), cache_dir)
            if metrics is not None:
                metrics["reagle_step_seconds"] = reagle_secs_by_artifact.get(name)
            out[name] = metrics
        except Exception as exc:
            print(f"WARN: Failed reading artifact '{name}' from run {run['id']}: {exc}")
            out[name] = None
    return out


def print_report(base_run: dict, base_metrics: dict[str, dict], comparisons: list[dict], artifact_names: list[str]) -> None:
    print("=" * 110)
    base_pr = base_run.get("pr_number")
    base_pr_txt = f", pr=#{base_pr}" if base_pr is not None else ""
    print(
        f"BASE RUN: {base_run['id']} (run_number={base_run.get('run_number')}{base_pr_txt}, "
        f"status={base_run.get('status')}, conclusion={base_run.get('conclusion')})"
    )
    print(f"URL: {base_run.get('html_url')}")
    print("=" * 110)

    print("\nBASE REFERENCE (by artifact/job):")
    for art in artifact_names:
        print(f"\n  {art}")
        b = base_metrics[art]
        for key, label, _ in METRICS:
            print(f"    {label}: {fmt(b.get(key))}")

    print("\n" + "=" * 110)
    print("SUBSEQUENT RUNS (job-level output)")
    print("=" * 110)
    for entry in comparisons:
        run = entry["run"]
        pr_txt = f", pr=#{run.get('pr_number')}" if run.get("pr_number") is not None else ""
        print(
            f"\nRun {run['id']} (#{run.get('run_number')}, "
            f"{run.get('status')}/{run.get('conclusion')}{pr_txt})"
        )
        # Run-level mean deltas across available artifact jobs.
        print("  Mean delta across available metric jobs:")
        for key, label, direction in METRICS:
            deltas = []
            for art in artifact_names:
                b = base_metrics[art]
                m = entry["metrics"].get(art)
                if not m:
                    continue
                _, delta = metric_delta_status(key, b.get(key), m.get(key))
                if delta is not None:
                    deltas.append(delta)
            if not deltas:
                print(f"    ⚪ {label}: N/A (n=0)")
                continue
            mean_delta = sum(deltas) / len(deltas)
            if abs(mean_delta) <= EPS:
                emoji = "⚪"
                verdict = "neutral"
            else:
                improved = mean_delta > 0 if direction == "higher" else mean_delta < 0
                emoji = "🟢" if improved else "🔴"
                verdict = "good" if improved else "bad"
            print(f"    {emoji} {label}: {mean_delta:+.6f} ({verdict}, n={len(deltas)})")

        printed_any = False
        for art in artifact_names:
            m = entry["metrics"].get(art)
            if not m:
                continue
            printed_any = True
            print(f"  {art}")
            b = base_metrics[art]
            for key, label, _ in METRICS:
                base_val = b.get(key)
                cur_val = m.get(key)
                emoji, delta = metric_delta_status(key, base_val, cur_val)
                d = "N/A" if delta is None else f"{delta:+.6f}"
                print(f"    {emoji} {label}: {fmt(cur_val)}   Δ {d}")
        if not printed_any:
            print("  No matching metrics-* artifacts available yet for this run.")


def build_plot(
    base_metrics: dict[str, dict],
    comparisons: list[dict],
    artifact_names: list[str],
    out_png: Path,
) -> None:
    n_cols = len(artifact_names)
    fig, axes = plt.subplots(1, n_cols, figsize=(34, 11), sharey=True)
    if n_cols == 1:
        axes = [axes]

    y_labels = [label for _, label, _ in METRICS]
    y_idx = list(range(len(METRICS)))

    for i, art in enumerate(artifact_names):
        ax = axes[i]
        base = base_metrics[art]

        plotted = 0
        all_x = []

        for run_pos, entry in enumerate(comparisons):
            run = entry["run"]
            m = entry["metrics"].get(art)
            if not m:
                continue

            for j, (key, label, _) in enumerate(METRICS):
                base_val = base.get(key)
                cur_val = m.get(key)
                emoji, delta = metric_delta_status(key, base_val, cur_val)
                if delta is None:
                    continue

                # Plot speed as fractional percent change (0..1) instead of raw seconds delta.
                plot_x = delta
                if key == "reagle_step_seconds":
                    if base_val is None or cur_val is None or base_val <= 0:
                        continue
                    frac_change = abs((cur_val - base_val) / base_val)
                    plot_x = min(max(frac_change, 0.0), 1.0)

                y = j + ((run_pos % 9) - 4) * 0.045
                color = "#16a34a" if emoji == "🟢" else ("#dc2626" if emoji == "🔴" else "#6b7280")
                marker = "o" if run.get("status") == "completed" else "^"
                ax.scatter(plot_x, y, s=42, color=color, alpha=0.85, marker=marker, edgecolors="none")
                all_x.append(plot_x)
                plotted += 1

        ax.axvline(0.0, color="#111827", linewidth=1.0, alpha=0.6)
        ax.set_yticks(y_idx)
        ax.set_yticklabels(y_labels)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        short = art.replace("metrics-", "")
        if len(short) > 40:
            short = short[:37] + "..."
        ax.set_title(f"{short}\n(points={plotted})", fontsize=10)
        ax.set_xlabel("Delta vs base (speed shown as 0-1 fractional change)")

        if all_x:
            max_abs = max(abs(x) for x in all_x)
            lim = max(0.02, max_abs * 1.2)
            ax.set_xlim(-lim, lim)
        else:
            ax.text(0.5, 0.5, "No deltas available", transform=ax.transAxes, ha="center", va="center", color="#6b7280")

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="#16a34a", label="Improved", markersize=8),
        plt.Line2D([0], [0], marker="o", linestyle="", color="#dc2626", label="Worse", markersize=8),
        plt.Line2D([0], [0], marker="o", linestyle="", color="#6b7280", label="No change", markersize=8),
        plt.Line2D([0], [0], marker="o", linestyle="", color="#111827", label="Completed run", markersize=8),
        plt.Line2D([0], [0], marker="^", linestyle="", color="#111827", label="In-progress/queued run", markersize=8),
    ]

    fig.legend(handles=legend_handles, loc="upper center", ncol=5, frameon=False)
    fig.suptitle("REAGLE metric deltas vs base run (Imputation Quality Assessment)", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0.01, 0.04, 0.99, 0.93])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare REAGLE metrics across IQA runs against a base run.")
    parser.add_argument("base_run", help="Base run URL, run id, or run number (e.g. #1298).")
    parser.add_argument("--repo", default="", help="GitHub repo slug owner/repo (auto-detected by default).")
    parser.add_argument("--workflow", default="Imputation Quality Assessment", help="Workflow name.")
    parser.add_argument(
        "--output",
        default="scripts/iqa_reagle_delta_report.png",
        help="Output PNG path (single figure with four subplots).",
    )
    parser.add_argument(
        "--cache-dir",
        default="/tmp/iqa_reagle_delta_cache",
        help="Directory used to cache downloaded artifact contents.",
    )
    parser.add_argument(
        "--max-subsequent",
        type=int,
        default=0,
        help="Optional cap for number of subsequent runs to process (0 = all).",
    )

    args = parser.parse_args()

    repo = args.repo or detect_repo("SauersML/reagle")
    cache_dir = Path(args.cache_dir)
    out_png = Path(args.output)

    print(f"Repo: {repo}")
    print(f"Workflow: {args.workflow}")

    gh = GitHubClient(repo)
    base_run = resolve_base_run(gh, args.workflow, args.base_run)
    pr_cache: dict[str, tuple[int | None, str | None]] = {}
    base_pr_num, base_pr_state = resolve_pr_for_run(repo, base_run, pr_cache)
    base_run["pr_number"] = base_pr_num
    base_run["pr_state"] = base_pr_state
    base_created = parse_iso_time(base_run["created_at"])

    workflow_id = find_workflow_id(gh, args.workflow)
    all_runs = gh.paginate(
        f"/repos/{repo}/actions/workflows/{workflow_id}/runs",
        params={"per_page": 100},
        list_key="workflow_runs",
    )

    # Base artifact names are the 4 metrics-* artifacts from this run.
    base_arts = [
        a for a in list_run_artifacts(gh, int(base_run["id"]))
        if a.get("name", "").startswith("metrics-") and not a.get("expired", False)
    ]
    base_artifact_names = sorted(a["name"] for a in base_arts)

    if len(base_artifact_names) == 0:
        print("No metrics-* artifacts found in base run; falling back to log extraction...")
        log_metrics = extract_metrics_from_logs(gh, base_run, cache_dir)
        if not log_metrics:
            raise RuntimeError(
                "No metrics-* artifacts found in base run and could not extract metrics from logs."
            )
        base_artifact_names = sorted(log_metrics.keys())
        base_metrics = {name: metrics for name, metrics in log_metrics.items()}
        print(f"Extracted metrics from logs for {len(base_artifact_names)} job(s): {base_artifact_names}")
    else:
        if len(base_artifact_names) != 4:
            print(f"WARN: Expected 4 metrics artifacts, found {len(base_artifact_names)}")
        base_metrics = collect_run_metrics(gh, base_run, base_artifact_names, cache_dir)
        missing_base = [name for name, m in base_metrics.items() if m is None]
        if missing_base:
            raise RuntimeError(f"Base run is missing REAGLE metrics in artifacts: {missing_base}")

    subsequent = [
        r for r in all_runs
        if parse_iso_time(r["created_at"]) > base_created
    ]
    subsequent.sort(key=lambda r: parse_iso_time(r["created_at"]))
    if args.max_subsequent > 0:
        subsequent = subsequent[: args.max_subsequent]

    print(f"Base run id: {base_run['id']} created_at={base_run['created_at']}")
    print(f"Found {len(subsequent)} subsequent runs to inspect")

    comparisons: list[dict] = []
    for idx, run in enumerate(subsequent, start=1):
        run_pr_num, run_pr_state = resolve_pr_for_run(repo, run, pr_cache)
        run["pr_number"] = run_pr_num
        run["pr_state"] = run_pr_state
        if run.get("event") == "pull_request" and run_pr_state != "OPEN":
            continue
        pr_txt = f", pr=#{run.get('pr_number')}" if run.get("pr_number") is not None else ""
        print(
            f"[{idx}/{len(subsequent)}] run {run['id']} "
            f"(#{run.get('run_number')}, {run.get('status')}/{run.get('conclusion')}{pr_txt})"
        )
        metrics = collect_run_metrics(gh, run, base_artifact_names, cache_dir)
        comparisons.append({"run": run, "metrics": metrics})

    print_report(base_run, base_metrics, comparisons, base_artifact_names)
    build_plot(base_metrics, comparisons, base_artifact_names, out_png)
    print(f"\nSaved plot: {out_png}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
