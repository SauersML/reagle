#!/usr/bin/env python3
"""
Minimal parallel experiment runner for HPC nodes.

- Input: plain text file with one shell command per line.
- Output: one log file per experiment + summary.tsv.
- Behavior: runs up to --jobs commands in parallel and prints live progress.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
import csv
from datetime import datetime
from pathlib import Path
from glob import glob


class Task:
    def __init__(self, idx, cmd, exp_dir, log_path, overrides=None):
        self.idx = idx
        self.cmd = cmd
        self.exp_dir = exp_dir
        self.log_path = log_path
        self.overrides = overrides or {}
        self.start_ts = None
        self.end_ts = None
        self.returncode = None
        self.pid = None
        self.proc = None
        self.log_handle = None
        self.max_rss_gb = 0.0


def shell_join(tokens):
    return " ".join(shlex.quote(str(t)) for t in tokens)


def choose_default_cargo_scratch_root():
    shm = Path("/dev/shm")
    if shm.exists() and os.access(str(shm), os.W_OK):
        return shm / "reagle-cargo"
    return Path("/tmp/reagle-cargo")


def build_cargo_local_env(workdir, scratch_root):
    user = os.environ.get("USER", "user")
    wd_hash = hashlib.sha1(str(workdir).encode("utf-8")).hexdigest()[:12]
    base = Path(scratch_root) / user / ("wd-" + wd_hash)
    cargo_home = base / "cargo-home"
    cargo_target = base / "target"
    cargo_home.mkdir(parents=True, exist_ok=True)
    cargo_target.mkdir(parents=True, exist_ok=True)
    return {
        "CARGO_HOME": str(cargo_home),
        "CARGO_TARGET_DIR": str(cargo_target),
    }


def _sanitize_path_list(raw):
    parts = [p for p in raw.split(":") if p]
    kept = []
    for p in parts:
        if os.path.isdir(p):
            kept.append(p)
    return kept


def sanitize_link_env(env):
    changed = []
    for key in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        raw = env.get(key, "")
        if not raw:
            continue
        kept = _sanitize_path_list(raw)
        new_val = ":".join(kept)
        if new_val != raw:
            env[key] = new_val
            changed.append(key)

    # If stale RUSTFLAGS force link against missing BLAS, strip those flags.
    rustflags = env.get("RUSTFLAGS", "")
    if rustflags:
        has_openblas = False
        search_dirs = []
        for key in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
            search_dirs.extend([p for p in env.get(key, "").split(":") if p])
        for d in search_dirs:
            if glob(os.path.join(d, "libopenblas.so*")):
                has_openblas = True
                break
        if not has_openblas:
            toks = shlex.split(rustflags)
            filtered = [t for t in toks if t not in ("-lopenblas", "-llapack")]
            if filtered != toks:
                env["RUSTFLAGS"] = shell_join(filtered)
                changed.append("RUSTFLAGS")

    return changed


def now_stamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def load_commands(path):
    lines = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def parse_value_literal(raw):
    s = raw.strip()
    sl = s.lower()
    if sl == "true":
        return "true"
    if sl == "false":
        return "false"
    if re.match(r"^-?\d+$", s):
        return s
    if re.match(r"^-?\d+\.\d+([eE]-?\d+)?$", s):
        return s
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        # Normalize single quotes to TOML double-quoted strings.
        if s.startswith("'"):
            inner = s[1:-1].replace("\\", "\\\\").replace('"', '\\"')
            return '"' + inner + '"'
        return s
    escaped = s.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + escaped + '"'


def parse_override_spec(spec):
    out = {}
    if not spec.strip():
        return out
    for item in spec.split(","):
        token = item.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError("invalid override token '{}' (expected key=value)".format(token))
        k, v = token.split("=", 1)
        key = k.strip()
        if not key:
            raise ValueError("empty key in override token '{}'".format(token))
        out[key] = parse_value_literal(v)
    return out


def render_toml_with_overrides(base_toml_path, overrides):
    text = base_toml_path.read_text()
    lines = text.splitlines()
    remaining = dict(overrides)
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("["):
            out.append(line)
            continue
        replaced = False
        for key in list(remaining.keys()):
            # Top-level key match only.
            m = re.match(r"^(\s*" + re.escape(key) + r"\s*=\s*).*$", line)
            if m:
                out.append(m.group(1) + remaining[key])
                remaining.pop(key, None)
                replaced = True
                break
        if not replaced:
            out.append(line)
    if remaining:
        out.append("")
        out.append("# Added by experiment_runner overrides")
        for key in sorted(remaining.keys()):
            out.append("{} = {}".format(key, remaining[key]))
    return "\n".join(out) + "\n"


def split_command_and_overrides(line, sep):
    if sep not in line:
        return {}, line
    left, right = line.split(sep, 1)
    overrides = parse_override_spec(left)
    cmd = right.strip()
    if not cmd:
        raise ValueError("empty command after override separator")
    return overrides, cmd


def parse_cargo_test_command(cmd):
    tokens = shlex.split(cmd)
    if not tokens:
        return None
    i = 0
    env_assign_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")
    while i < len(tokens) and env_assign_re.match(tokens[i]):
        i += 1
    if i + 1 >= len(tokens) or tokens[i] != "cargo" or tokens[i + 1] != "test":
        return None

    rest = tokens[i + 2 :]
    if "--" in rest:
        sep = rest.index("--")
        cargo_args = rest[:sep]
        test_args = rest[sep + 1 :]
    else:
        cargo_args = rest
        test_args = []

    test_name = None
    manifest_path = None
    has_release = False
    has_no_run = False
    has_message_json = False
    j = 0
    while j < len(cargo_args):
        tok = cargo_args[j]
        if tok == "--test" and j + 1 < len(cargo_args):
            test_name = cargo_args[j + 1]
            j += 2
            continue
        if tok.startswith("--test="):
            test_name = tok.split("=", 1)[1]
            j += 1
            continue
        if tok == "--manifest-path" and j + 1 < len(cargo_args):
            manifest_path = cargo_args[j + 1]
            j += 2
            continue
        if tok.startswith("--manifest-path="):
            manifest_path = tok.split("=", 1)[1]
            j += 1
            continue
        if tok == "--release":
            has_release = True
        if tok == "--no-run":
            has_no_run = True
        if tok.startswith("--message-format=") and "json" in tok:
            has_message_json = True
        if tok == "--message-format" and j + 1 < len(cargo_args):
            if "json" in cargo_args[j + 1]:
                has_message_json = True
            j += 2
            continue
        j += 1

    if not test_name:
        return None

    return {
        "tokens": tokens,
        "cargo_index": i,
        "env_prefix_tokens": tokens[:i],
        "cargo_args": cargo_args,
        "test_args": test_args,
        "test_name": test_name,
        "manifest_path": manifest_path,
        "has_release": has_release,
        "has_no_run": has_no_run,
        "has_message_json": has_message_json,
    }


def _expand_manifest_path(path_token, workdir):
    if not path_token:
        return None
    p = path_token.replace("$WORKDIR", str(workdir))
    p = p.replace("${WORKDIR}", str(workdir))
    return p


def prepare_cargo_test_executable(parsed, workdir, force_release, extra_env=None):
    cargo_args = list(parsed["cargo_args"])
    if force_release and not parsed["has_release"]:
        cargo_args.append("--release")
    if not parsed["has_no_run"]:
        cargo_args.append("--no-run")
    if not parsed["has_message_json"]:
        cargo_args.extend(["--message-format=json"])

    manifest_path = _expand_manifest_path(parsed["manifest_path"], workdir)
    if not manifest_path:
        manifest_path = str(Path(workdir) / "Cargo.toml")
        cargo_args.extend(["--manifest-path", manifest_path])

    cmd = ["cargo", "test"] + cargo_args
    env = os.environ.copy()
    env["WORKDIR"] = str(workdir)
    if extra_env:
        env.update(extra_env)
    sanitize_link_env(env)
    proc = subprocess.run(
        cmd,
        cwd=str(workdir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "cargo prebuild failed ({}):\n{}".format(proc.returncode, proc.stdout[-3000:])
        )

    exe = None
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("reason") != "compiler-artifact":
            continue
        target = obj.get("target", {})
        if target.get("name") == parsed["test_name"] and target.get("test"):
            maybe_exe = obj.get("executable")
            if maybe_exe:
                exe = maybe_exe
    if not exe:
        raise RuntimeError("could not locate compiled test executable in cargo json output")
    return exe


def ensure_cargo_manifest_path(cmd, workdir):
    parsed = parse_cargo_test_command(cmd)
    if parsed is None:
        return cmd
    if parsed["manifest_path"]:
        return cmd

    tokens = list(parsed["tokens"])
    insert_at = parsed["cargo_index"] + 2
    tokens[insert_at:insert_at] = ["--manifest-path", str(Path(workdir) / "Cargo.toml")]
    return shell_join(tokens)


def read_meminfo_gb():
    total = None
    avail = None
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        total = int(parts[1]) / (1024.0 * 1024.0)
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        avail = int(parts[1]) / (1024.0 * 1024.0)
    except Exception:
        return None, None
    return total, avail


def read_available_memory_gb():
    _, avail = read_meminfo_gb()
    return avail


def process_rss_gb(pid):
    try:
        with open("/proc/{}/status".format(pid), "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        kb = int(parts[1])
                        return kb / (1024.0 * 1024.0)
    except Exception:
        return 0.0
    return 0.0


def plan_parallelism(
    n_commands,
    requested_jobs,
    requested_threads_per_job,
    reserve_cores,
    target_threads_per_job,
    min_threads_per_job,
    max_threads_per_job,
    max_jobs,
    mem_per_job_gb,
):
    total_cores = max(1, os.cpu_count() or 1)
    usable_cores = max(1, total_cores - max(0, reserve_cores))
    jobs_cap = n_commands
    if max_jobs > 0:
        jobs_cap = min(jobs_cap, max_jobs)

    if requested_jobs is not None and requested_jobs > 0:
        jobs = min(jobs_cap, requested_jobs)
    else:
        jobs = jobs_cap

    if requested_threads_per_job > 0:
        threads = requested_threads_per_job
    else:
        threads = max(min_threads_per_job, target_threads_per_job)

    if max_threads_per_job > 0:
        threads = min(threads, max_threads_per_job)
    threads = max(1, threads)

    if jobs > 0:
        jobs = min(jobs, max(1, usable_cores // threads))
    else:
        jobs = 1

    avail_mem_gb = read_available_memory_gb()
    if mem_per_job_gb > 0 and avail_mem_gb is not None:
        jobs_by_mem = max(1, int(avail_mem_gb // mem_per_job_gb))
        jobs = min(jobs, jobs_by_mem)

    jobs = max(1, min(jobs, jobs_cap))

    return {
        "total_cores": total_cores,
        "usable_cores": usable_cores,
        "jobs": jobs,
        "threads_per_job": threads,
        "avail_mem_gb": avail_mem_gb,
    }


def compute_required_free_gb(total_mem_gb, safety_min_free_gb, safety_free_fraction):
    if total_mem_gb is None or total_mem_gb <= 0:
        return max(4.0, safety_min_free_gb)
    target = max(safety_min_free_gb, total_mem_gb * safety_free_fraction)
    # Prevent impossible thresholds on smaller machines.
    upper = max(4.0, total_mem_gb * 0.8)
    return min(target, upper)


def estimate_next_job_mem_gb(tasks, fallback_job_mem_gb, growth_factor):
    peak = 0.0
    for t in tasks:
        if t.max_rss_gb > peak:
            peak = t.max_rss_gb
    if peak <= 0.0:
        return fallback_job_mem_gb
    return max(fallback_job_mem_gb, peak * growth_factor)


def running_rss_total_gb(running):
    total = 0.0
    for task in running.values():
        if task.pid:
            rss = process_rss_gb(task.pid)
            if rss > task.max_rss_gb:
                task.max_rss_gb = rss
            total += rss
    return total


def fmt_duration(seconds):
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def truncate(text, width):
    if len(text) <= width:
        return text
    return text[: max(0, width - 3)] + "..."


def clear_screen():
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def write_summary(tasks, outdir):
    summary_path = outdir / "summary.tsv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("id\tstatus\texit_code\tduration_sec\tlog\tcommand\n")
        for task in tasks:
            if task.returncode is None:
                status = "UNKNOWN"
            elif task.returncode == 0:
                status = "OK"
            else:
                status = "FAIL"
            duration = ""
            if task.start_ts is not None and task.end_ts is not None:
                duration = f"{task.end_ts - task.start_ts:.2f}"
            rc = "" if task.returncode is None else str(task.returncode)
            f.write(
                f"{task.idx:04d}\t{status}\t{rc}\t{duration}\t{task.log_path}\t{task.cmd}\n"
            )


def parse_simple_toml(path):
    cfg = {}
    if not path.exists():
        return cfg
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()
    return cfg


def write_combined_metrics(tasks, outdir):
    metric_name = "chr21_fast_metrics.tsv"
    rows = []
    config_keys = set()
    metric_keys = set()

    for task in tasks:
        metrics_path = task.exp_dir / metric_name
        if not metrics_path.exists():
            continue

        cfg = parse_simple_toml(task.exp_dir / "reagle.toml")
        for key in cfg.keys():
            config_keys.add(key)

        by_tool = {}
        with metrics_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for rec in reader:
                tool = (rec.get("tool") or "").strip().lower()
                if not tool:
                    continue
                by_tool[tool] = rec
                for k in rec.keys():
                    if k != "tool":
                        metric_keys.add(k)

        rows.append(
            {
                "task": task,
                "cfg": cfg,
                "metrics": by_tool,
                "metrics_path": metrics_path,
            }
        )

    if not rows:
        return None

    config_cols = sorted(config_keys)
    metric_cols = sorted(metric_keys)
    out_path = outdir / "combined_metrics.tsv"

    header = ["id", "exp_dir", "status", "command", "metrics_path"]
    header.extend("cfg__" + k for k in config_cols)
    for tool in ("beagle", "reagle"):
        header.extend("{}__{}".format(tool, k) for k in metric_cols)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(header)
        for row in sorted(rows, key=lambda r: r["task"].idx):
            task = row["task"]
            status = "UNKNOWN"
            if task.returncode is not None:
                status = "OK" if task.returncode == 0 else "FAIL"
            out = [
                "{:04d}".format(task.idx),
                str(task.exp_dir),
                status,
                task.cmd,
                str(row["metrics_path"]),
            ]
            cfg = row["cfg"]
            out.extend(cfg.get(k, "") for k in config_cols)
            for tool in ("beagle", "reagle"):
                m = row["metrics"].get(tool, {})
                out.extend(m.get(k, "") for k in metric_cols)
            writer.writerow(out)

    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Run many experiment commands in parallel with live monitoring."
    )
    parser.add_argument("--commands", required=True, type=Path, help="Text file with commands.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Max concurrent experiments (default: auto-plan from cores/memory).",
    )
    parser.add_argument("--outdir", type=Path, default=None, help="Run output directory.")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="Working directory for all experiment commands (default: current directory).",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--base-toml",
        type=Path,
        default=None,
        help="Base TOML config to clone per experiment (written to each exp dir as reagle.toml).",
    )
    parser.add_argument(
        "--override-sep",
        type=str,
        default=":::",
        help="Separator for inline overrides: key=val,key2=val ::: command",
    )
    parser.add_argument(
        "--threads-per-job",
        type=int,
        default=0,
        help="If >0, force OMP_NUM_THREADS and RAYON_NUM_THREADS per command.",
    )
    parser.add_argument(
        "--reserve-cores",
        type=int,
        default=0,
        help="Cores to keep free for OS/other jobs when auto-planning.",
    )
    parser.add_argument(
        "--target-threads-per-job",
        type=int,
        default=8,
        help="Auto-planner target threads per experiment when --threads-per-job is not set.",
    )
    parser.add_argument(
        "--min-threads-per-job",
        type=int,
        default=1,
        help="Lower bound for auto-planned threads per experiment.",
    )
    parser.add_argument(
        "--max-threads-per-job",
        type=int,
        default=0,
        help="Upper bound for auto-planned threads per experiment (0 means no cap).",
    )
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=0,
        help="Upper bound for auto-planned jobs (0 means no cap).",
    )
    parser.add_argument(
        "--mem-per-job-gb",
        type=float,
        default=0.0,
        help="Optional memory estimate per experiment. Auto-planner caps jobs by MemAvailable/estimate.",
    )
    parser.add_argument(
        "--safety-min-free-gb",
        type=float,
        default=96.0,
        help="Always keep at least this much MemAvailable before launching new jobs.",
    )
    parser.add_argument(
        "--safety-free-fraction",
        type=float,
        default=0.20,
        help="Always keep at least this fraction of total memory free before launching new jobs.",
    )
    parser.add_argument(
        "--job-mem-growth-factor",
        type=float,
        default=2.5,
        help="Assumed growth from observed RSS peak when estimating next launch memory risk.",
    )
    parser.add_argument(
        "--default-job-mem-gb",
        type=float,
        default=16.0,
        help="Fallback per-job memory estimate until observed RSS is available.",
    )
    parser.add_argument(
        "--cache-cargo-tests",
        action="store_true",
        default=True,
        help="Prebuild and reuse cargo test executables across experiments.",
    )
    parser.add_argument(
        "--no-cache-cargo-tests",
        dest="cache_cargo_tests",
        action="store_false",
        help="Disable cargo test prebuild/reuse optimization.",
    )
    parser.add_argument(
        "--cache-cargo-tests-release",
        action="store_true",
        default=True,
        help="Force --release during cargo test prebuild when caching is enabled.",
    )
    parser.add_argument(
        "--no-cache-cargo-tests-release",
        dest="cache_cargo_tests_release",
        action="store_false",
        help="Do not force --release in cached cargo test prebuild.",
    )
    parser.add_argument(
        "--cargo-local-cache",
        action="store_true",
        default=True,
        help="Use node-local scratch for CARGO_HOME/CARGO_TARGET_DIR (default: on).",
    )
    parser.add_argument(
        "--no-cargo-local-cache",
        dest="cargo_local_cache",
        action="store_false",
        help="Disable node-local Cargo cache/target directories.",
    )
    parser.add_argument(
        "--cargo-scratch-root",
        type=Path,
        default=None,
        help="Root dir for local Cargo cache/target (default: /dev/shm/reagle-cargo or /tmp/reagle-cargo).",
    )
    args = parser.parse_args()

    commands = load_commands(args.commands)
    if not commands:
        print(f"No commands found in {args.commands}")
        return 1

    outdir = args.outdir or Path("runs") / f"run-{now_stamp()}"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.base_toml is not None and not args.base_toml.exists():
        print("Base TOML not found: {}".format(args.base_toml))
        return 1

    plan = plan_parallelism(
        n_commands=len(commands),
        requested_jobs=args.jobs,
        requested_threads_per_job=args.threads_per_job,
        reserve_cores=args.reserve_cores,
        target_threads_per_job=max(1, args.target_threads_per_job),
        min_threads_per_job=max(1, args.min_threads_per_job),
        max_threads_per_job=max(0, args.max_threads_per_job),
        max_jobs=max(0, args.max_jobs),
        mem_per_job_gb=max(0.0, args.mem_per_job_gb),
    )
    run_jobs = plan["jobs"]
    run_threads = plan["threads_per_job"]
    total_mem_gb, avail_mem_gb = read_meminfo_gb()

    mem_note = ""
    if avail_mem_gb is not None:
        mem_note = " avail_mem_gb={:.1f}".format(avail_mem_gb)
    print(
        "Plan: jobs={} threads/job={} usable_cores={} total_cores={}{}".format(
            run_jobs, run_threads, plan["usable_cores"], plan["total_cores"], mem_note
        )
    )

    cargo_local_env = None
    if args.cargo_local_cache:
        scratch_root = args.cargo_scratch_root or choose_default_cargo_scratch_root()
        cargo_local_env = build_cargo_local_env(args.workdir, scratch_root)
        print(
            "Cargo local cache: CARGO_HOME={} CARGO_TARGET_DIR={}".format(
                cargo_local_env["CARGO_HOME"], cargo_local_env["CARGO_TARGET_DIR"]
            )
        )

    startup_env = os.environ.copy()
    if cargo_local_env:
        startup_env.update(cargo_local_env)
    sanitized_keys = sanitize_link_env(startup_env)
    if sanitized_keys:
        print("Sanitized linker env keys: {}".format(",".join(sorted(set(sanitized_keys)))))

    tasks = []
    cargo_test_cache = {}
    for i, raw_cmd in enumerate(commands, start=1):
        exp_dir = outdir / f"exp-{i:04d}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        overrides = {}
        cmd = raw_cmd
        if args.base_toml is not None:
            try:
                overrides, cmd = split_command_and_overrides(raw_cmd, args.override_sep)
            except ValueError as e:
                print("Invalid overrides on command {}: {}".format(i, e))
                return 1
            rendered = render_toml_with_overrides(args.base_toml, overrides)
            (exp_dir / "reagle.toml").write_text(rendered, encoding="utf-8")

        # Always resolve cargo workspace correctly when running inside exp dirs.
        cmd = ensure_cargo_manifest_path(cmd, args.workdir)

        if args.cache_cargo_tests:
            parsed = parse_cargo_test_command(cmd)
            if parsed is not None:
                cache_key = (
                    tuple(parsed["cargo_args"]),
                    parsed["test_name"],
                    bool(args.cache_cargo_tests_release),
                    _expand_manifest_path(parsed["manifest_path"], args.workdir)
                    or str(Path(args.workdir) / "Cargo.toml"),
                )
                exe = cargo_test_cache.get(cache_key)
                if exe is None:
                    print(
                        "Prebuilding cargo test binary for '{}' (release={})...".format(
                            parsed["test_name"], args.cache_cargo_tests_release
                        )
                    )
                    try:
                        exe = prepare_cargo_test_executable(
                            parsed,
                            workdir=args.workdir,
                            force_release=args.cache_cargo_tests_release,
                            extra_env=cargo_local_env,
                        )
                    except RuntimeError as e:
                        print("Cargo test prebuild failed on command {}: {}".format(i, e))
                        return 1
                    cargo_test_cache[cache_key] = exe
                rewritten_tokens = list(parsed["env_prefix_tokens"]) + [exe] + list(parsed["test_args"])
                cmd = shell_join(rewritten_tokens)

        (exp_dir / "command.sh").write_text(cmd + "\n", encoding="utf-8")
        tasks.append(
            Task(idx=i, cmd=cmd, exp_dir=exp_dir, log_path=exp_dir / "run.log", overrides=overrides)
        )

    pending = tasks[:]
    running = {}
    done = []
    stop_requested = False

    def handle_stop(sig, frame):  # noqa: ANN001
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    fallback_job_mem_gb = max(1.0, args.default_job_mem_gb)
    if args.mem_per_job_gb > 0:
        fallback_job_mem_gb = args.mem_per_job_gb
    safety_min_free_gb = max(1.0, args.safety_min_free_gb)
    safety_free_fraction = min(0.95, max(0.01, args.safety_free_fraction))
    growth_factor = max(1.0, args.job_mem_growth_factor)
    launch_paused_for_mem = False

    while pending or running:
        while not stop_requested and pending and len(running) < run_jobs:
            total_mem_gb, mem_avail_gb = read_meminfo_gb()
            running_rss_gb = running_rss_total_gb(running)
            est_next_job_gb = estimate_next_job_mem_gb(tasks, fallback_job_mem_gb, growth_factor)
            required_free_gb = compute_required_free_gb(
                total_mem_gb=total_mem_gb,
                safety_min_free_gb=safety_min_free_gb,
                safety_free_fraction=safety_free_fraction,
            )
            if (
                mem_avail_gb is not None
                and (mem_avail_gb - est_next_job_gb) < required_free_gb
            ):
                launch_paused_for_mem = True
                break
            launch_paused_for_mem = False
            task = pending.pop(0)
            env = os.environ.copy()
            if run_threads > 0:
                env["OMP_NUM_THREADS"] = str(run_threads)
                env["RAYON_NUM_THREADS"] = str(run_threads)
            env["WORKDIR"] = str(args.workdir)
            env["EXP_DIR"] = str(task.exp_dir)
            env["TASK_ID"] = str(task.idx)
            if cargo_local_env:
                env.update(cargo_local_env)
            sanitize_link_env(env)

            logf = task.log_path.open("w", encoding="utf-8")
            run_cwd = task.exp_dir if args.base_toml is not None else args.workdir
            proc = subprocess.Popen(
                ["bash", "-lc", task.cmd],
                cwd=str(run_cwd),
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
            )
            task.proc = proc
            task.log_handle = logf
            task.pid = proc.pid
            task.start_ts = time.time()
            running[task.idx] = task

        finished_ids = []
        for task_id, task in running.items():
            rc = task.proc.poll() if task.proc else None
            if rc is not None:
                task.returncode = rc
                task.end_ts = time.time()
                if task.log_handle is not None:
                    task.log_handle.close()
                    task.log_handle = None
                done.append(task)
                finished_ids.append(task_id)

        for task_id in finished_ids:
            running.pop(task_id, None)

        ok = sum(1 for t in done if t.returncode == 0)
        fail = sum(1 for t in done if t.returncode and t.returncode != 0)

        clear_screen()
        total_mem_gb, mem_avail_gb = read_meminfo_gb()
        running_rss_gb = running_rss_total_gb(running)
        est_next_job_gb = estimate_next_job_mem_gb(tasks, fallback_job_mem_gb, growth_factor)
        required_free_gb = compute_required_free_gb(
            total_mem_gb=total_mem_gb,
            safety_min_free_gb=safety_min_free_gb,
            safety_free_fraction=safety_free_fraction,
        )
        print(f"Run directory: {outdir}")
        print(
            f"Total={len(tasks)}  Pending={len(pending)}  Running={len(running)}  "
            f"Done={len(done)}  OK={ok}  FAIL={fail}"
        )
        if mem_avail_gb is not None:
            print(
                "Mem guard: avail={:.1f}G running_rss={:.1f}G est_next={:.1f}G required_free={:.1f}G".format(
                    mem_avail_gb, running_rss_gb, est_next_job_gb, required_free_gb
                )
            )
        if launch_paused_for_mem:
            print("Launch paused: memory safety guard active.")
        if stop_requested:
            print("Stop requested: waiting for running jobs to finish.")
        print("")
        print("Active jobs:")
        if not running:
            print("  (none)")
        else:
            for task in sorted(running.values(), key=lambda x: x.idx)[:15]:
                elapsed = 0.0 if task.start_ts is None else (time.time() - task.start_ts)
                cmd_preview = truncate(task.cmd, 100)
                print(
                    f"  [{task.idx:04d}] pid={task.pid}  elapsed={fmt_duration(elapsed)}  cmd={cmd_preview}"
                )

        if stop_requested and not running:
            # User requested stop; do not loop forever on pending queue.
            for task in pending:
                task.returncode = 130
                task.start_ts = task.start_ts or time.time()
                task.end_ts = time.time()
                done.append(task)
            pending.clear()
            break

        if pending or running:
            time.sleep(max(0.2, args.poll_seconds))

    write_summary(tasks, outdir)
    combined_metrics = write_combined_metrics(tasks, outdir)
    failed = [t for t in tasks if (t.returncode or 0) != 0]
    print("")
    print(f"Finished. Summary: {outdir / 'summary.tsv'}")
    if combined_metrics is not None:
        print(f"Combined metrics: {combined_metrics}")
    if failed:
        print(f"Failed experiments: {len(failed)}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
