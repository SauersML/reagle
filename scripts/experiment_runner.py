#!/usr/bin/env python3
"""
Minimal parallel experiment runner for HPC nodes.

- Input: plain text file with one shell command per line.
- Output: one log file per experiment + summary.tsv.
- Behavior: runs up to --jobs commands in parallel and prints live progress.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


class Task:
    def __init__(self, idx, cmd, exp_dir, log_path):
        self.idx = idx
        self.cmd = cmd
        self.exp_dir = exp_dir
        self.log_path = log_path
        self.start_ts = None
        self.end_ts = None
        self.returncode = None
        self.pid = None
        self.proc = None
        self.log_handle = None


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


def main():
    parser = argparse.ArgumentParser(
        description="Run many experiment commands in parallel with live monitoring."
    )
    parser.add_argument("--commands", required=True, type=Path, help="Text file with commands.")
    parser.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--outdir", type=Path, default=None, help="Run output directory.")
    parser.add_argument(
        "--workdir",
        type=Path,
        default=Path.cwd(),
        help="Working directory for all experiment commands (default: current directory).",
    )
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--threads-per-job",
        type=int,
        default=0,
        help="If >0, sets OMP_NUM_THREADS and RAYON_NUM_THREADS for each command.",
    )
    args = parser.parse_args()

    commands = load_commands(args.commands)
    if not commands:
        print(f"No commands found in {args.commands}")
        return 1

    outdir = args.outdir or Path("runs") / f"run-{now_stamp()}"
    outdir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for i, cmd in enumerate(commands, start=1):
        exp_dir = outdir / f"exp-{i:04d}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "command.sh").write_text(cmd + "\n", encoding="utf-8")
        tasks.append(Task(idx=i, cmd=cmd, exp_dir=exp_dir, log_path=exp_dir / "run.log"))

    pending = tasks[:]
    running = {}
    done = []
    stop_requested = False

    def handle_stop(sig, frame):  # noqa: ANN001
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    while pending or running:
        while not stop_requested and pending and len(running) < args.jobs:
            task = pending.pop(0)
            env = os.environ.copy()
            if args.threads_per_job > 0:
                env["OMP_NUM_THREADS"] = str(args.threads_per_job)
                env["RAYON_NUM_THREADS"] = str(args.threads_per_job)

            logf = task.log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                ["bash", "-lc", task.cmd],
                cwd=str(args.workdir),
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
        print(f"Run directory: {outdir}")
        print(
            f"Total={len(tasks)}  Pending={len(pending)}  Running={len(running)}  "
            f"Done={len(done)}  OK={ok}  FAIL={fail}"
        )
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

        if pending or running:
            time.sleep(max(0.2, args.poll_seconds))

    write_summary(tasks, outdir)
    failed = [t for t in tasks if (t.returncode or 0) != 0]
    print("")
    print(f"Finished. Summary: {outdir / 'summary.tsv'}")
    if failed:
        print(f"Failed experiments: {len(failed)}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
