#!/usr/bin/env python3
"""Compare REAGLE metrics for currently open PRs vs dynamic per-PR main baselines.

Rules implemented:
1. No required CLI args.
2. Only currently open PRs are considered.
3. Each open PR uses only its most recent IQA workflow run.
4. The baseline for that PR is the most recent IQA run on main before PR creation time.
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
from typing import Any

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


def detect_repo(default_repo: str) -> str:
    try:
        url = run_cmd(["git", "config", "--get", "remote.origin.url"])
    except Exception:
        return default_repo

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

    def _request_json(self, url: str) -> tuple[Any, str | None]:
        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/vnd.github+json")
        req.add_header("X-GitHub-Api-Version", "2022-11-28")
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            link = resp.headers.get("Link")
            return data, link

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        base = f"https://api.github.com{path}"
        if params:
            q = urllib.parse.urlencode(params)
            base = f"{base}?{q}"
        data, _ = self._request_json(base)
        return data

    def paginate(self, path: str, params: dict[str, Any] | None = None, list_key: str | None = None) -> list[dict]:
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


def parse_iso_time(ts: str) -> dt.datetime:
    return dt.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def human_utc(ts: str) -> str:
    t = parse_iso_time(ts)
    return t.strftime("%Y-%m-%d %H:%M:%S UTC")


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


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
    parts = line.split("\t", 2)
    if len(parts) < 3:
        return None
    payload = parts[2]
    m = re.match(r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)\s", payload)
    if not m:
        return None
    ts = m.group("ts")
    if "." in ts:
        base, frac_z = ts.split(".", 1)
        frac = frac_z[:-1]
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

    if all(v is None for v in out.values()):
        log_metrics = extract_metrics_from_logs(gh, run, cache_dir)
        if log_metrics:
            for name in target_artifact_names:
                if name in log_metrics:
                    out[name] = log_metrics[name]
                else:
                    for lv in log_metrics.values():
                        if out.get(name) is None:
                            out[name] = lv
                            break
    return out


def list_open_prs(gh: GitHubClient) -> list[dict]:
    return gh.paginate(
        f"/repos/{gh.repo}/pulls",
        params={"state": "open", "per_page": 100},
        list_key=None,
    )


def resolve_run_pr_numbers(run: dict, open_prs_by_number: dict[int, dict], open_prs_by_head: dict[str, int]) -> list[int]:
    nums: list[int] = []
    for pr_stub in run.get("pull_requests", []):
        num = pr_stub.get("number")
        if isinstance(num, int) and num in open_prs_by_number:
            nums.append(num)

    if nums:
        return sorted(set(nums))

    head_branch = run.get("head_branch")
    if isinstance(head_branch, str):
        num = open_prs_by_head.get(head_branch)
        if num is not None:
            return [num]

    return []


def latest_pr_runs(
    pr_runs: list[dict],
    open_prs_by_number: dict[int, dict],
    open_prs_by_head: dict[str, int],
) -> dict[int, dict]:
    latest: dict[int, dict] = {}
    for run in pr_runs:
        pr_nums = resolve_run_pr_numbers(run, open_prs_by_number, open_prs_by_head)
        if not pr_nums:
            continue
        run_ts = parse_iso_time(run["created_at"])
        for pr_num in pr_nums:
            cur = latest.get(pr_num)
            if cur is None or run_ts > parse_iso_time(cur["created_at"]):
                latest[pr_num] = run
    return latest


def find_main_base_run_before(main_runs: list[dict], pr_created: dt.datetime) -> dict | None:
    best: dict | None = None
    best_ts: dt.datetime | None = None
    for run in main_runs:
        ts = parse_iso_time(run["created_at"])
        if ts >= pr_created:
            continue
        if best is None or (best_ts is not None and ts > best_ts):
            best = run
            best_ts = ts
    return best


def load_base_metrics(
    gh: GitHubClient,
    base_run: dict,
    cache_dir: Path,
    base_cache: dict[int, tuple[list[str], dict[str, dict]]],
) -> tuple[list[str], dict[str, dict]]:
    run_id = int(base_run["id"])
    if run_id in base_cache:
        return base_cache[run_id]

    base_arts = [
        a
        for a in list_run_artifacts(gh, run_id)
        if a.get("name", "").startswith("metrics-") and not a.get("expired", False)
    ]
    base_artifact_names = sorted(a["name"] for a in base_arts)

    if len(base_artifact_names) == 0:
        log_metrics = extract_metrics_from_logs(gh, base_run, cache_dir)
        if not log_metrics:
            raise RuntimeError(f"No metrics found for base run {run_id}.")
        base_artifact_names = sorted(log_metrics.keys())
        base_metrics = {name: metrics for name, metrics in log_metrics.items()}
    else:
        base_metrics_raw = collect_run_metrics(gh, base_run, base_artifact_names, cache_dir)
        missing_base = [name for name, m in base_metrics_raw.items() if m is None]
        if missing_base:
            raise RuntimeError(f"Base run {run_id} missing metrics in artifacts: {missing_base}")
        base_metrics = {name: m for name, m in base_metrics_raw.items() if m is not None}

    base_cache[run_id] = (base_artifact_names, base_metrics)
    return base_artifact_names, base_metrics


def print_pr_report(comparisons: list[dict]) -> None:
    for entry in comparisons:
        pr = entry["pr"]
        pr_run = entry["pr_run"]
        base_run = entry["base_run"]
        base_metrics = entry["base_metrics"]
        pr_metrics = entry["pr_metrics"]
        artifact_names = entry["artifact_names"]

        print("=" * 110)
        print(f"PR #{pr['number']}: {pr.get('title', '')}")
        print(f"PR URL: {pr.get('html_url')}")
        print(f"PR created_at: {pr.get('created_at')} ({human_utc(pr['created_at'])})")
        print(
            f"PR run: {pr_run['id']} (#{pr_run.get('run_number')}, "
            f"{pr_run.get('status')}/{pr_run.get('conclusion')})"
        )
        print(
            f"PR run created_at: {pr_run.get('created_at')} "
            f"({human_utc(pr_run['created_at'])})"
        )
        print(f"PR run commit: {pr_run.get('head_sha', 'N/A')}")
        print(
            f"Base run: {base_run['id']} (#{base_run.get('run_number')}, "
            f"created_at={base_run.get('created_at')} ({human_utc(base_run['created_at'])})"
        )
        print(f"Base run commit: {base_run.get('head_sha', 'N/A')}")

        print("  Mean delta across available metric jobs:")
        for key, label, direction in METRICS:
            deltas = []
            for art in artifact_names:
                b = base_metrics.get(art)
                m = pr_metrics.get(art)
                if not b or not m:
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
            m = pr_metrics.get(art)
            b = base_metrics.get(art)
            if not m or not b:
                continue
            printed_any = True
            print(f"  {art}")
            for key, label, _ in METRICS:
                base_val = b.get(key)
                cur_val = m.get(key)
                emoji, delta = metric_delta_status(key, base_val, cur_val)
                d = "N/A" if delta is None else f"{delta:+.6f}"
                print(f"    {emoji} {label}: {fmt(cur_val)}   Δ {d}")
        if not printed_any:
            print("  No matching metrics available for this PR run.")


def build_plot(comparisons: list[dict], out_png: Path) -> None:
    artifact_names = sorted(
        {
            art
            for entry in comparisons
            for art in entry["artifact_names"]
        }
    )
    if not artifact_names:
        return

    n_cols = len(artifact_names)
    fig, axes = plt.subplots(1, n_cols, figsize=(8 * n_cols, 11), sharey=True)
    if n_cols == 1:
        axes = [axes]

    y_labels = [label for _, label, _ in METRICS]
    y_idx = list(range(len(METRICS)))

    for i, art in enumerate(artifact_names):
        ax = axes[i]
        plotted = 0
        all_x: list[float] = []

        for pos, entry in enumerate(comparisons):
            pr = entry["pr"]
            base = entry["base_metrics"].get(art)
            cur = entry["pr_metrics"].get(art)
            pr_run = entry["pr_run"]
            if not base or not cur:
                continue

            for j, (key, _, _) in enumerate(METRICS):
                base_val = base.get(key)
                cur_val = cur.get(key)
                emoji, delta = metric_delta_status(key, base_val, cur_val)
                if delta is None:
                    continue

                plot_x = delta
                if key == "reagle_step_seconds":
                    if base_val is None or cur_val is None or base_val <= 0:
                        continue
                    frac_change = abs((cur_val - base_val) / base_val)
                    plot_x = min(max(frac_change, 0.0), 1.0)

                y = j + ((pos % 9) - 4) * 0.045
                color = "#16a34a" if emoji == "🟢" else ("#dc2626" if emoji == "🔴" else "#6b7280")
                marker = "o" if pr_run.get("status") == "completed" else "^"
                ax.scatter(plot_x, y, s=42, color=color, alpha=0.85, marker=marker, edgecolors="none")
                ax.text(plot_x, y + 0.02, f"PR{pr['number']}", fontsize=7, alpha=0.65)
                all_x.append(plot_x)
                plotted += 1

        ax.axvline(0.0, color="#111827", linewidth=1.0, alpha=0.6)
        ax.set_yticks(y_idx)
        ax.set_yticklabels(y_labels)
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        short = art.replace("metrics-", "")
        if len(short) > 40:
            short = short[:37] + "..."
        ax.set_title(f"{short}\\n(points={plotted})", fontsize=10)
        ax.set_xlabel("Delta vs PR-specific main baseline (speed shown as 0-1 fractional change)")

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
    fig.suptitle("REAGLE metric deltas per open PR vs dynamic pre-PR main baseline", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0.01, 0.04, 0.99, 0.93])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare REAGLE metrics for open PRs vs dynamic pre-PR main baselines.")
    parser.add_argument("--repo", default="", help="GitHub repo slug owner/repo (auto-detected by default).")
    parser.add_argument("--workflow", default="Imputation Quality Assessment", help="Workflow name.")
    parser.add_argument(
        "--output",
        default="scripts/iqa_reagle_delta_report.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--cache-dir",
        default="/tmp/iqa_reagle_delta_cache",
        help="Directory used to cache downloaded artifact contents.",
    )
    args = parser.parse_args()

    repo = args.repo or detect_repo("SauersML/reagle")
    cache_dir = Path(args.cache_dir)
    out_png = Path(args.output)

    gh = GitHubClient(repo)
    workflow_id = find_workflow_id(gh, args.workflow)

    open_prs = list_open_prs(gh)
    if not open_prs:
        return 0

    open_prs_by_number = {int(pr["number"]): pr for pr in open_prs}
    open_prs_by_head = {str(pr.get("head", {}).get("ref", "")): int(pr["number"]) for pr in open_prs}

    pr_runs = gh.paginate(
        f"/repos/{repo}/actions/workflows/{workflow_id}/runs",
        params={"per_page": 100, "event": "pull_request"},
        list_key="workflow_runs",
    )
    latest_by_pr = latest_pr_runs(pr_runs, open_prs_by_number, open_prs_by_head)

    main_runs = gh.paginate(
        f"/repos/{repo}/actions/workflows/{workflow_id}/runs",
        params={"per_page": 100, "branch": "main"},
        list_key="workflow_runs",
    )

    base_cache: dict[int, tuple[list[str], dict[str, dict]]] = {}
    comparisons: list[dict] = []

    for pr_num in sorted(open_prs_by_number.keys()):
        pr = open_prs_by_number[pr_num]
        pr_run = latest_by_pr.get(pr_num)
        if pr_run is None:
            continue

        pr_created = parse_iso_time(pr["created_at"])
        base_run = find_main_base_run_before(main_runs, pr_created)
        if base_run is None:
            continue

        try:
            artifact_names, base_metrics = load_base_metrics(gh, base_run, cache_dir, base_cache)
        except Exception:
            continue

        pr_metrics = collect_run_metrics(gh, pr_run, artifact_names, cache_dir)

        comparisons.append(
            {
                "pr": pr,
                "pr_run": pr_run,
                "base_run": base_run,
                "artifact_names": artifact_names,
                "base_metrics": base_metrics,
                "pr_metrics": pr_metrics,
            }
        )

    if not comparisons:
        return 0

    print_pr_report(comparisons)
    build_plot(comparisons, out_png)
    print(f"\nSaved plot: {out_png}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
