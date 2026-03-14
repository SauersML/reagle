#!/usr/bin/env python3
"""
Long-running Codex supervisor for autonomous Reagle improvement.

Operational model:
- This Python process is the durable orchestrator.
- A Codex orchestrator turn runs after worker completions and on a periodic
  cadence to inspect PRs, evaluate state, merge, close, or resolve follow-up.
- Every attempt gets its own git worktree and `codex/` branch.
- Worker Codex runs are long-lived non-interactive turns that may take many
  hours to produce candidate changes.
- The Python layer handles lifecycle, state, worktrees, and execution only.

This is intentionally closer to Symphony's orchestration pattern than to a
single immortal agent session. The loop can run for weeks because state,
branching, retries, and PR reconciliation live outside the agent.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import fcntl
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from typing import Any

STATE_VERSION = 1
DEFAULT_BASE_BRANCH = "main"
DEFAULT_STATE_DIR = ".codex-autopilot"
DEFAULT_BRANCH_PREFIX = "codex/autopilot"
DEFAULT_MONITOR_SECONDS = 15
DEFAULT_ORCHESTRATOR_INTERVAL_SECONDS = 60 * 60
DEFAULT_ORCHESTRATOR_TIMEOUT_SECONDS = 2 * 60 * 60
DEFAULT_WORKER_COUNT = 2
DEFAULT_RESTART_BACKOFF_SECONDS = 5

OVERALL_GOALS_CONTEXT = """
Overall goals:
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
THE GOAL is to have the BEST AND MOST ACCURATE imputation (and phasing) OUT OF ANY TOOL, while being VERY FAST (and no OOM). This project is for genotype phasing and imputation. We care most about accuracy, then very fast speed, with the constraint that it must not run out of memory with ~15 GB. We care about real-world usage above any test. The goal of the project should be to maximize universal future wellbeing above any one particular present task. This should guide our decision-making. This happens through taking proper long-term solutions. How? Well, better phasing and imputation leads to downstream better, e.g., predictive power for disease prediction, disease associations, science, etc., which improves lives tangibly. This means we must always do what is best for real-world accuracy and performance.
"""

PROJECT_CONTEXT = """
Project context:
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
Large changes are not bad (bigness of change is orthogonal to goodness), and should not be avoided. We don’t care about backwards compatibility. Model changes or algorithm changes are not bad. Instead of having a bias towards the status quo, we should have a bias for what is most likely to be correct when reasoning from first principles. Don’t assume the current code has been well-tested or is already deployed. It’s in the prototype stage, and core ideas may change. There may not always be correct or verified reasons for the decisions made in the code.
"""

WORKER_DIRECTION_CONTEXT = """
Current goal:
=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
In order to achieve our overall goals, we will implement or change something in order to improve. You should read code before deciding what to do. YOU are responsible for choosing an idea that will ACTUALLY WORK to help achieve our overall goal(s). Some ideas may be bad, and some may be good. A low-risk, low-reward change is bad since it wastes time. We want serious, large improvements. You will plan, and then, you will FULLY implement the best possible, ideal version of that to absolute completion, perfectly. Do everything that you think will improve accuracy.
"""

EVALUATION_CONTEXT = """
You might want to increase r2 and iqs while minimizing switch error rate and hellinger and total time, for example, but you should look at all metrics holistically. Changes are good if they improve most metrics! It’s okay for a change to get worse on some if it gets better on others.

Important note: small tests are bad signals of real-world imputation since it’s unrealistic. For example, long-range haplotype structure is unmeasured if you test only a small chromosome chunk. Similarly, haplotype selection may not be assessed realistically if only a tiny panel is used.

Therefore, do not overfit or overindex on small tests, narrow chromosome slices, toy panels, or any single metric. Do not let one narrow benchmark or one metric dominate your judgment. Also, run big tests and use realistic evidence whenever possible.

The main goal is the accuracy metrics for phasing and imputation, and do not regress time. Release mode is good since it might take a while to finish. Try to measure and then beat previous results.

Get Reagle to beat other tools on various realistic tests. DO NOT modify the test or make Reagle wrap external tools. DO NOT tweak knobs. DO active experimentation.

Prefer tools already available in the environment. Do not rely on privileged or system-wide installs; if extra tooling is truly needed, prefer repo-local or user-space approaches.
"""

SCIENTIFIC_METHOD_CONTEXT = """
YOUR INFORMATIONAL SUBGOAL (NOT TRUE OBJECTIVE) is to do an ideal, beautiful, correct, root-cause fix of the following failure to achieve information for true goal. HOWEVER, your change should NOT MERELY fix the failing test, but it should be SO GOOD AND CORRECT that is an OBJECTIVE IMPROVEMENT OVERALL, clearly, even if the test NEVER EXISTED. So use the test to GUIDE your improvement / fix.

We should use the scientific method to reach conclusions. This can take the form of rejecting hypotheses, testing and verifying before concluding, questioning assumptions, or creating and actively testing hypotheses. For example, you may come up with an idea of what is happening from reasoning, then add prints to see if they align with your idea, then write a falsifiable test to see if the specific mechanism you hypothesized holds. We also think about confounding variables: a test may align with our expectations, but if other explanations can account for the result, it is not definitive. Multiple orthogonal lines of evidence, coupled with reasoning, are better. If something is going wrong, and we have a hypothesis about why, the null hypothesis is that we are wrong about why something is going wrong. We should assume most hypotheses are wrong unless we have active empirical evidence otherwise. (Of course, some ideas, such as features, engineering, or designing fixes for already-known issues do not require as much science, and reasoning is often sufficient. Science is most important when something has gone wrong, and we must discover why.) Importantly, this process involves iteration. You do actions serially, observing, refining, and improving hypotheses--it’s not a one-time action, and the results of future actions are dependent on your observations of past behavior and actions. This means you shouldn’t stop to talk to the user after forming your initial hypotheses--just begin testing and iterating.

Avoid overfitting to tests and their failures, and avoid adding heuristics for the purpose of passing tests. Always be asking why. Always try to attain deeper understanding. Heuristics can be unavoidable, but in general, we should use precise math and theory instead of a heuristic whenever possible.
"""

ENGINEERING_CONTEXT = """
Dead code in any form isn’t allowed, including underscore-prefixed variables, let _ = pattern when something is a no-op, and other patterns.

You can often proceed to implement or test without asking.

Never use environment variables, generally, unless forced to (e.g., some GitHub actions code requires them). Prefer being explicit. Similarly, you can include logging and prints in the production code without env vars or gates--just make sure it’s useful and non-spammy prints. If you truly need to see something that shouldn’t be in prod, just remove it after rather than gating it, or decide if you can make a version of it that can go in prod and still be useful to you.
Do not use sudo, do not assume privileged machine control, and do not start background daemons or other detached processes.
"""

ORCHESTRATOR_ROLE_CONTEXT = """
You are the orchestrator for the Reagle autopilot.

Review the current autopilot PRs and use your own judgment.
Run commands with `gh` and `git` freely to inspect PRs, checks, CI runs, reviews, mergeability, comments, logs, branches, commits, diffs, and any other GitHub or git state you need.
Only assess PRs whose CI and check runs have completed. If CI is still running or pending, leave that PR alone for now and come back on a later orchestrator pass.
CI is NON-BLOCKING. PR checks, test failures, and CI failures are informational evidence, not an automatic blocking gate, and are not necessarily caused by the changes in the PR.
Do not overfit your judgment to small tests, narrow benchmarks, or a single metric. Evaluate evidence holistically and prioritize realistic runs and real-world impact.
Be slightly biased toward merging when a PR looks net good. Change is not bad merely because it is large or different. Do not keep good work out of `main` just because it is ambitious.
Do not merge bad work. Use your judgment about whether the overall change is genuinely good for real-world accuracy, speed, robustness, and long-term direction.
Resolve conflicts manually by understanding the code and editing deliberately. Do not rely on automated conflict-resolution tools.
You are free to cherry-pick the good parts of PRs, split work apart, or recombine pieces across branches if that is the best path to getting the right changes into `main`.
Merge ready PRs, close bad ones, resolve conflicts, and do integration work when needed.
Use this worktree however you judge appropriate.
Worker launch and worker lifetime are handled elsewhere.
If you want prior worker reasoning, inspect the history file and attempt logs provided below.
"""

WORKER_ROLE_CONTEXT = """
You are an autonomous worker in the Reagle repository.

The host has already created your worktree and branch for this attempt. Work only on the assigned branch in this worktree.
Improve imputation and phasing accuracy using your own judgment.
Do not overfit to small tests, narrow benchmarks, or a single metric. Prefer realistic evidence and holistic improvement.
Parallelize expensive experiments, benchmarks, and test runs so you use the machine effectively across available CPU cores.
There is no time limit. You can work for days before finishing if needed. Do not rush to end the run.
Remote git and GitHub access are available. You may use `git` and `gh` when useful, including pushing branches and opening PRs.
If you decide the result is worth shipping, commit it, push the branch, and open a PR yourself.
Do not merge any PR into `main` from this worker run.
"""

FOLLOW_UP_ROLE_CONTEXT = """
This is a follow-up on your own in-progress attempt.

The host always sends a follow-up when a worker turn ends so you can reflect, refine, and keep improving the branch.
Continue the same branch and session from the current state of the worktree.
Use your own judgment about whether this work should still ship.
Do not overfit to small tests, narrow benchmarks, or a single metric. Re-evaluate holistically using realistic evidence.
There is no time limit. You can keep working for days before finishing if needed.
Remote git and GitHub access are available. You may use `git` and `gh` when useful, including pushing branches and opening PRs.
If you still think it should ship, continue and resolve the issue.
If you no longer think it should ship, say so plainly.
If it should ship, make sure the branch is committed, pushed, and has an open PR.
Do not merge into `main`.
"""


@dataclasses.dataclass
class Config:
    repo_root: Path
    state_dir: Path
    base_branch: str
    branch_prefix: str
    monitor_seconds: int
    orchestrator_interval_seconds: int
    orchestrator_timeout_seconds: int
    worker_count: int


@dataclasses.dataclass
class ActiveWorker:
    slot: int
    iteration: int
    branch_name: str
    worktree: Path
    attempt_dir: Path
    started_at: str
    session_id: str | None
    follow_up_index: int
    prompt_path: Path
    last_message_path: Path
    stdout_path: Path
    stderr_path: Path
    stdout_thread: threading.Thread
    stderr_thread: threading.Thread
    process: subprocess.Popen[str]


class SupervisorError(RuntimeError):
    pass


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def utc_now_iso() -> str:
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_command(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: int | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def checked_command(cmd: list[str], *, cwd: Path, timeout: int | None = None) -> str:
    proc = run_command(cmd, cwd=cwd, timeout=timeout)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        raise SupervisorError(
            f"{shell_join(cmd)} failed in {cwd}: {stderr or stdout or f'exit {proc.returncode}'}"
        )
    return proc.stdout


def sanitize_branch_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._/-]+", "-", value.strip())
    cleaned = cleaned.strip("-/.")
    return cleaned or "attempt"


def safe_branch_name(prefix: str, iteration: int) -> str:
    stamp = utc_now().strftime("%Y%m%d-%H%M%S")
    return f"{prefix}/{stamp}-{iteration:05d}"


def redact(text: str, limit: int = 4000) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 32] + "\n...[truncated by supervisor]..."


def compose_prompt(*sections: str) -> str:
    ordered_sections: list[str] = []
    seen: set[str] = set()
    for section in sections:
        normalized = textwrap.dedent(section).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered_sections.append(normalized)
    return "\n\n".join(ordered_sections) + "\n"


def worker_turn_paths(attempt_dir: Path, follow_up_index: int) -> tuple[Path, Path, Path, Path]:
    suffix = "" if follow_up_index == 0 else f".followup{follow_up_index}"
    prompt_path = attempt_dir / f"prompt{suffix}.md"
    last_message_path = attempt_dir / f"codex_last_message{suffix}.txt"
    stdout_path = attempt_dir / f"codex_stdout{suffix}.log"
    stderr_path = attempt_dir / f"codex_stderr{suffix}.log"
    return prompt_path, last_message_path, stdout_path, stderr_path


class CodexAutopilot:
    def __init__(self, config: Config):
        self.config = config
        self.repo_root = config.repo_root
        self.state_dir = config.state_dir
        self.state_path = self.state_dir / "state.json"
        self.history_path = self.state_dir / "history.jsonl"
        self.heartbeat_path = self.state_dir / "heartbeat.json"
        self.lock_path = self.state_dir / "lock"
        self.logs_dir = self.state_dir / "logs"
        self.master_log_path = self.logs_dir / "master.log"
        self.attempts_dir = self.state_dir / "attempts"
        self.worktrees_dir = self.state_dir / "worktrees"
        self.orchestrator_dir = self.state_dir / "orchestrator"
        self.orchestrator_worktree = self.worktrees_dir / "orchestrator"
        self.state = load_json(
            self.state_path,
            {
                "version": STATE_VERSION,
                "iteration": 0,
            },
        )
        self.active_workers: dict[int, ActiveWorker] = {}
        self._last_orchestrator_monotonic = 0.0
        self._lock_handle = None
        self._master_log_lock = threading.Lock()
        self._initial_cycle_pending = True

    def setup(self) -> None:
        for path in [
            self.state_dir,
            self.logs_dir,
            self.attempts_dir,
            self.orchestrator_dir,
            self.worktrees_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def acquire_lock(self) -> None:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.lock_path.open("w", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SupervisorError(f"another supervisor already holds {self.lock_path}") from exc
        self._lock_handle = handle
        handle.write(f"{os.getpid()}\n")
        handle.flush()

    def write_state(self) -> None:
        self.state["version"] = STATE_VERSION
        self.state["active_workers"] = [
            {
                "slot": worker.slot,
                "iteration": worker.iteration,
                "branch": worker.branch_name,
                "worktree": str(worker.worktree),
                "started_at": worker.started_at,
                "session_id": worker.session_id,
                "follow_up_index": worker.follow_up_index,
                "pid": worker.process.pid,
            }
            for worker in sorted(self.active_workers.values(), key=lambda item: item.slot)
        ]
        write_json(self.state_path, self.state)

    def update_heartbeat(self, status: str, extra: dict[str, Any] | None = None) -> None:
        payload = {
            "pid": os.getpid(),
            "status": status,
            "updated_at": utc_now_iso(),
            "active_workers": [
                {
                    "slot": worker.slot,
                    "iteration": worker.iteration,
                    "branch": worker.branch_name,
                    "session_id": worker.session_id,
                    "follow_up_index": worker.follow_up_index,
                    "pid": worker.process.pid,
                }
                for worker in sorted(self.active_workers.values(), key=lambda item: item.slot)
            ],
        }
        if extra:
            payload.update(extra)
        write_json(self.heartbeat_path, payload)

    def append_master_log_block(self, title: str, body: str) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        normalized = body.rstrip()
        block = f"[{utc_now_iso()}] {title}\n"
        if normalized:
            block += normalized + "\n"
        block += "\n"
        with self._master_log_lock:
            with self.master_log_path.open("a", encoding="utf-8") as handle:
                handle.write(block)
                handle.flush()

    def append_master_log_line(self, stream_name: str, line: str) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        with self._master_log_lock:
            with self.master_log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{utc_now_iso()}] {stream_name} {line}")
                if not line.endswith("\n"):
                    handle.write("\n")
                handle.flush()

    def stream_process_output(self, stream: Any, destination_path: Path, stream_name: str) -> None:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with destination_path.open("w", encoding="utf-8") as destination:
            while True:
                line = stream.readline()
                if line == "":
                    break
                destination.write(line)
                destination.flush()
                self.append_master_log_line(stream_name, line)
        stream.close()

    def spawn_logged_process(
        self,
        *,
        cmd: list[str],
        cwd: Path,
        prompt_path: Path,
        stdout_path: Path,
        stderr_path: Path,
        label: str,
    ) -> tuple[subprocess.Popen[str], threading.Thread, threading.Thread]:
        prompt_text = prompt_path.read_text(encoding="utf-8", errors="replace")
        self.append_master_log_block(
            f"{label} start",
            "\n".join(
                [
                    f"cwd: {cwd}",
                    f"command: {shell_join(cmd)}",
                    "",
                    "prompt:",
                    prompt_text,
                ]
            ),
        )

        stdin_handle = prompt_path.open("r", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdin=stdin_handle,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stdin_handle.close()
        if process.stdout is None or process.stderr is None:
            raise SupervisorError(f"{label} failed to allocate stdout/stderr pipes")

        stdout_thread = threading.Thread(
            target=self.stream_process_output,
            args=(process.stdout, stdout_path, f"{label} stdout"),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self.stream_process_output,
            args=(process.stderr, stderr_path, f"{label} stderr"),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        return process, stdout_thread, stderr_thread

    def join_process_threads(self, stdout_thread: threading.Thread, stderr_thread: threading.Thread) -> None:
        stdout_thread.join(timeout=30)
        stderr_thread.join(timeout=30)

    def handle_signal(self, signum: int, _frame: Any) -> None:
        try:
            signal_name = signal.Signals(signum).name
        except ValueError:
            signal_name = str(signum)
        message = f"received signal {signal_name}; ignoring it so the supervisor keeps running"
        logging.warning(message)
        self.append_master_log_block("signal ignored", message)

    def run(self) -> int:
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGTERM, self.handle_signal)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self.handle_signal)
        if hasattr(signal, "SIGQUIT"):
            signal.signal(signal.SIGQUIT, self.handle_signal)

        self.append_master_log_block("supervisor process start", f"pid: {os.getpid()}")

        while True:
            try:
                self.setup()
                if self._lock_handle is None:
                    self.acquire_lock()
                self.update_heartbeat("starting")
                if self._initial_cycle_pending:
                    self.update_heartbeat("initializing")
                    self.fill_worker_slots()
                    self.maybe_run_orchestrator(force=True)
                    self._initial_cycle_pending = False
                    self.update_heartbeat("idle")
                else:
                    self.tick()
            except Exception:
                logging.exception("supervisor cycle failed")
                self.update_heartbeat("error")
            sleep_for = max(1.0, self.config.monitor_seconds)
            self.update_heartbeat("sleeping", {"sleep_seconds": int(sleep_for)})
            time.sleep(sleep_for)

    def tick(self) -> None:
        self.update_heartbeat("tick")
        self.reap_finished_workers()
        self.fill_worker_slots()
        self.maybe_run_orchestrator(force=False)
        self.update_heartbeat("idle")

    def fetch_base_branch(self) -> None:
        checked_command(["git", "fetch", "origin", self.config.base_branch], cwd=self.repo_root)

    def current_origin_commit(self) -> str:
        return checked_command(
            ["git", "rev-parse", f"origin/{self.config.base_branch}"],
            cwd=self.repo_root,
        ).strip()

    def ensure_orchestrator_worktree(self) -> Path:
        commit = self.current_origin_commit()
        if not self.orchestrator_worktree.exists():
            checked_command(
                ["git", "worktree", "add", "--detach", str(self.orchestrator_worktree), commit],
                cwd=self.repo_root,
            )
            return self.orchestrator_worktree

        if self.git_status_dirty(self.orchestrator_worktree):
            self.append_master_log_block(
                "orchestrator worktree preserved",
                "\n".join(
                    [
                        f"worktree: {self.orchestrator_worktree}",
                        "The orchestrator worktree is dirty, so the supervisor is preserving it and reusing it instead of resetting to origin/main.",
                    ]
                ),
            )
            return self.orchestrator_worktree

        checked_command(["git", "fetch", "origin", self.config.base_branch], cwd=self.orchestrator_worktree)
        checked_command(["git", "checkout", "--detach", commit], cwd=self.orchestrator_worktree)
        return self.orchestrator_worktree

    def maybe_run_orchestrator(self, *, force: bool) -> None:
        now = time.monotonic()
        due = (now - self._last_orchestrator_monotonic) >= self.config.orchestrator_interval_seconds
        if not force and not due:
            return
        open_autopilot_prs = self.list_open_autopilot_prs()
        if not open_autopilot_prs:
            logging.info(
                "skipping orchestrator review; no open autopilot PRs due=%s force=%s",
                due,
                force,
            )
            return
        logging.info(
            "running orchestrator review; open autopilot PRs: %d due=%s force=%s",
            len(open_autopilot_prs),
            due,
            force,
        )
        self.run_orchestrator_tick(open_autopilot_prs)
        self._last_orchestrator_monotonic = time.monotonic()

    def run_orchestrator_tick(self, open_autopilot_prs: list[dict[str, Any]]) -> None:
        tick_dir = self.orchestrator_dir / utc_now().strftime("%Y%m%d-%H%M%S")
        tick_dir.mkdir(parents=True, exist_ok=True)
        worktree = self.ensure_orchestrator_worktree()
        result = self.run_orchestrator_turn(
            worktree=worktree,
            tick_dir=tick_dir,
            open_autopilot_prs=open_autopilot_prs,
        )
        record = {
            "kind": "orchestrator_tick",
            "started_at": utc_now_iso(),
            "worktree": str(worktree),
            "codex": result,
            "review": self.summarize_orchestrator_result(result, worktree),
        }
        append_jsonl(self.history_path, record)
        self.write_state()

    def fill_worker_slots(self) -> None:
        for slot in range(1, self.config.worker_count + 1):
            if slot in self.active_workers:
                continue
            self.launch_worker(slot)

    def launch_worker(self, slot: int) -> None:
        self.fetch_base_branch()
        iteration = int(self.state.get("iteration", 0)) + 1
        self.state["iteration"] = iteration
        branch_name = f"{safe_branch_name(self.config.branch_prefix, iteration)}-slot{slot}"
        attempt_dir = self.attempts_dir / sanitize_branch_component(branch_name.replace("/", "-"))
        attempt_dir.mkdir(parents=True, exist_ok=True)
        worktree = self.create_candidate_worktree(branch_name)
        prompt_path, last_message_path, stdout_path, stderr_path = worker_turn_paths(attempt_dir, 0)
        started_at = utc_now_iso()

        prompt = self.build_codex_prompt(branch_name=branch_name)
        self.write_text(prompt_path, prompt)

        cmd = [
            "codex",
            "-a",
            "never",
            "-s",
            "danger-full-access",
            "exec",
            "--json",
            "-C",
            str(worktree),
            "-o",
            str(last_message_path),
            "-",
        ]

        process, stdout_thread, stderr_thread = self.spawn_logged_process(
            cmd=cmd,
            cwd=self.repo_root,
            prompt_path=prompt_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            label=f"worker slot={slot} iteration={iteration} follow_up=0",
        )

        self.active_workers[slot] = ActiveWorker(
            slot=slot,
            iteration=iteration,
            branch_name=branch_name,
            worktree=worktree,
            attempt_dir=attempt_dir,
            started_at=started_at,
            session_id=None,
            follow_up_index=0,
            prompt_path=prompt_path,
            last_message_path=last_message_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdout_thread=stdout_thread,
            stderr_thread=stderr_thread,
            process=process,
        )
        self.write_state()
        logging.info(
            "launched worker slot %d iteration %d branch %s pid %d",
            slot,
            iteration,
            branch_name,
            process.pid,
        )

    def reap_finished_workers(self) -> bool:
        finished_any = False
        for slot in sorted(list(self.active_workers.keys())):
            worker = self.active_workers[slot]
            returncode = worker.process.poll()
            if returncode is None:
                continue
            finished_any = True
            next_worker: ActiveWorker | None = None
            try:
                next_worker = self.finalize_worker(worker, returncode)
            finally:
                if next_worker is not None:
                    self.active_workers[slot] = next_worker
                else:
                    self.active_workers.pop(slot, None)
                    if self.git_status_dirty(worker.worktree):
                        logging.info("preserving dirty worker worktree %s", worker.worktree)
                    else:
                        self.cleanup_worktree(worker.worktree)
                self.write_state()
        return finished_any

    def finalize_worker(self, worker: ActiveWorker, returncode: int) -> ActiveWorker | None:
        self.join_process_threads(worker.stdout_thread, worker.stderr_thread)
        codex_result = self.collect_worker_result(worker, returncode)
        last_message = codex_result.get("last_message") or ""
        existing_pr = self.lookup_open_pr_for_branch(worker.branch_name, worker.worktree)
        dirty = self.git_status_dirty(worker.worktree)
        record = {
            "kind": "attempt",
            "slot": worker.slot,
            "iteration": worker.iteration,
            "branch": worker.branch_name,
            "worktree": str(worker.worktree),
            "started_at": worker.started_at,
            "finished_at": utc_now_iso(),
            "codex": codex_result,
            "session_id": codex_result.get("session_id"),
            "follow_up_index": worker.follow_up_index,
            "returncode": returncode,
            "dirty": dirty,
            "pr_opened": existing_pr is not None,
            "diff_stat": self.git_diff_stat(worker.worktree),
        }

        next_worker = None
        follow_up_prompt = self.build_worker_follow_up_prompt(
            branch_name=worker.branch_name,
            last_message=last_message,
            existing_pr=existing_pr,
            returncode=returncode,
        )
        session_id = codex_result.get("session_id")
        if follow_up_prompt:
            next_worker = self.continue_worker_with_follow_up(worker, session_id=session_id, prompt=follow_up_prompt)
            record["follow_up"] = {
                "session_id": session_id,
                "follow_up_index": next_worker.follow_up_index,
                "prompt_path": str(next_worker.prompt_path),
                "last_message_path": str(next_worker.last_message_path),
            }

        if existing_pr is not None:
            record["pr"] = existing_pr
            logging.info(
                "worker slot %d iteration %d finished with returncode=%d and PR %s",
                worker.slot,
                worker.iteration,
                returncode,
                existing_pr.get("url"),
            )
        else:
            logging.info(
                "worker slot %d iteration %d finished with returncode=%d without opening a PR",
                worker.slot,
                worker.iteration,
                returncode,
            )
        if next_worker is not None:
            logging.info(
                "worker slot %d iteration %d continuing in resumed session %s",
                worker.slot,
                worker.iteration,
                next_worker.session_id,
            )

        self.append_master_log_block(
            f"worker slot={worker.slot} iteration={worker.iteration} follow_up={worker.follow_up_index} complete",
            "\n".join(
                [
                    f"returncode: {returncode}",
                    f"branch: {worker.branch_name}",
                    f"worktree: {worker.worktree}",
                    f"session_id: {codex_result.get('session_id') or 'none'}",
                    f"pr: {existing_pr.get('url') if existing_pr is not None else 'none'}",
                    "",
                    "last_message:",
                    codex_result.get("last_message") or "",
                ]
            ),
        )
        append_jsonl(self.history_path, record)
        return next_worker

    def collect_worker_result(self, worker: ActiveWorker, returncode: int) -> dict[str, Any]:
        if worker.follow_up_index > 0 and worker.session_id:
            cmd = [
                "codex",
                "-a",
                "never",
                "exec",
                "resume",
                "--json",
                "-o",
                str(worker.last_message_path),
                worker.session_id,
                "-",
            ]
        else:
            cmd = [
                "codex",
                "-a",
                "never",
                "-s",
                "danger-full-access",
                "exec",
                "--json",
                "-C",
                str(worker.worktree),
                "-o",
                str(worker.last_message_path),
                "-",
            ]
        result: dict[str, Any] = {
            "command": shell_join(cmd),
            "returncode": returncode,
            "stdout_log": str(worker.stdout_path),
            "stderr_log": str(worker.stderr_path),
            "prompt_path": str(worker.prompt_path),
            "last_message_path": str(worker.last_message_path),
        }
        if worker.last_message_path.exists():
            result["last_message"] = worker.last_message_path.read_text(encoding="utf-8", errors="replace")
        result["session_id"] = self.extract_session_id(worker.stdout_path) or worker.session_id
        if returncode != 0:
            stderr_text = worker.stderr_path.read_text(encoding="utf-8", errors="replace") if worker.stderr_path.exists() else ""
            stdout_text = worker.stdout_path.read_text(encoding="utf-8", errors="replace") if worker.stdout_path.exists() else ""
            result["stderr_tail"] = redact(stderr_text or stdout_text)
        return result

    def cleanup_worktree(self, worktree: Path) -> None:
        if not worktree.exists():
            return
        proc = run_command(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=self.repo_root,
            timeout=5 * 60,
        )
        if proc.returncode != 0:
            logging.warning("failed to remove worktree %s: %s", worktree, redact(proc.stderr or proc.stdout))

    def create_candidate_worktree(self, branch_name: str) -> Path:
        worktree = self.worktrees_dir / sanitize_branch_component(branch_name.replace("/", "-"))
        checked_command(
            [
                "git",
                "worktree",
                "add",
                "-b",
                branch_name,
                str(worktree),
                f"origin/{self.config.base_branch}",
            ],
            cwd=self.repo_root,
        )
        return worktree

    def list_open_autopilot_prs(self) -> list[dict[str, Any]]:
        proc = run_command(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "open",
                "--limit",
                "100",
                "--json",
                "number,title,headRefName,url",
            ],
            cwd=self.repo_root,
            timeout=5 * 60,
        )
        if proc.returncode != 0:
            logging.warning("gh pr list failed: %s", redact(proc.stderr or proc.stdout))
            return []
        try:
            rows = json.loads(proc.stdout or "[]")
        except json.JSONDecodeError:
            logging.warning("failed to parse gh pr list output")
            return []
        prefix = self.config.branch_prefix
        return [row for row in rows if (row.get("headRefName") or "").startswith(prefix)]

    def extract_session_id(self, stdout_path: Path) -> str | None:
        if not stdout_path.exists():
            return None
        for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("type") != "thread.started":
                continue
            thread_id = payload.get("thread_id")
            if isinstance(thread_id, str) and thread_id:
                return thread_id
        return None

    def build_worker_follow_up_prompt(
        self,
        *,
        branch_name: str,
        last_message: str,
        existing_pr: dict[str, Any] | None,
        returncode: int,
    ) -> str:
        reasons: list[str] = []
        if existing_pr is None:
            reasons.append("there is still no open PR for this branch")
        if returncode != 0:
            reasons.append(f"the previous Codex turn exited with returncode {returncode}")

        lines = [
            "The host is sending this follow-up after the last worker turn completed.",
        ]
        if reasons:
            lines.append("")
            lines.append("Specific things to account for:")
            for reason in reasons:
                lines.append(f"- {reason}.")
        else:
            lines.append("")
            lines.append("There is no contradiction being forced on you here. This follow-up exists so you can reflect on the branch, refine the work, inspect any PR/CI state, and keep improving if that seems worthwhile.")
        lines.extend(
            [
                "",
                f"Branch: {branch_name}",
                f"Previous turn returncode: {returncode}",
            ]
        )
        if existing_pr is not None:
            lines.extend(
                [
                    "",
                    f"There is already an open PR for this branch: {existing_pr.get('url')}",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "There is currently no open PR detected for this branch.",
                ]
            )
        if last_message.strip():
            lines.extend(
                [
                    "",
                    "Previous final message:",
                    last_message,
                ]
            )
        follow_up_state = "\n".join(lines)
        return compose_prompt(
            FOLLOW_UP_ROLE_CONTEXT,
            OVERALL_GOALS_CONTEXT,
            PROJECT_CONTEXT,
            WORKER_DIRECTION_CONTEXT,
            EVALUATION_CONTEXT,
            SCIENTIFIC_METHOD_CONTEXT,
            ENGINEERING_CONTEXT,
            follow_up_state,
            "In your final message, make your shipping decision and validation outcome clear. Plain language is fine.",
        )

    def continue_worker_with_follow_up(
        self,
        worker: ActiveWorker,
        *,
        session_id: Any,
        prompt: str,
    ) -> ActiveWorker:
        next_follow_up_index = worker.follow_up_index + 1
        prompt_path, last_message_path, stdout_path, stderr_path = worker_turn_paths(worker.attempt_dir, next_follow_up_index)
        self.write_text(prompt_path, prompt)

        resume_session_id = session_id if isinstance(session_id, str) and session_id else None
        if resume_session_id is not None:
            cmd = [
                "codex",
                "-a",
                "never",
                "exec",
                "resume",
                "--json",
                "-o",
                str(last_message_path),
                resume_session_id,
                "-",
            ]
            cwd = worker.worktree
        else:
            cmd = [
                "codex",
                "-a",
                "never",
                "-s",
                "danger-full-access",
                "exec",
                "--json",
                "-C",
                str(worker.worktree),
                "-o",
                str(last_message_path),
                "-",
            ]
            cwd = self.repo_root

        process, stdout_thread, stderr_thread = self.spawn_logged_process(
            cmd=cmd,
            cwd=cwd,
            prompt_path=prompt_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            label=(
                f"worker slot={worker.slot} iteration={worker.iteration} "
                f"follow_up={next_follow_up_index}"
            ),
        )

        logging.info(
            "continued worker slot %d iteration %d branch %s session %s follow_up_index=%d pid %d",
            worker.slot,
            worker.iteration,
            worker.branch_name,
            resume_session_id,
            next_follow_up_index,
            process.pid,
        )
        return ActiveWorker(
            slot=worker.slot,
            iteration=worker.iteration,
            branch_name=worker.branch_name,
            worktree=worker.worktree,
            attempt_dir=worker.attempt_dir,
            started_at=utc_now_iso(),
            session_id=resume_session_id,
            follow_up_index=next_follow_up_index,
            prompt_path=prompt_path,
            last_message_path=last_message_path,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdout_thread=stdout_thread,
            stderr_thread=stderr_thread,
            process=process,
        )

    def write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def run_orchestrator_turn(
        self,
        *,
        worktree: Path,
        tick_dir: Path,
        open_autopilot_prs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt_path = tick_dir / "prompt.md"
        stdout_path = tick_dir / "orchestrator_stdout.log"
        stderr_path = tick_dir / "orchestrator_stderr.log"

        prompt = self.build_orchestrator_prompt(open_autopilot_prs=open_autopilot_prs)
        self.write_text(prompt_path, prompt)

        cmd = [
            "codex",
            "-a",
            "never",
            "-s",
            "danger-full-access",
            "exec",
            "-C",
            str(worktree),
            "-",
        ]

        self.append_master_log_block(
            "orchestrator start",
            "\n".join(
                [
                    f"cwd: {self.repo_root}",
                    f"worktree: {worktree}",
                    f"command: {shell_join(cmd)}",
                    "",
                    "prompt:",
                    prompt,
                ]
            ),
        )

        try:
            proc = run_command(
                cmd,
                cwd=self.repo_root,
                timeout=self.config.orchestrator_timeout_seconds,
                input_text=prompt,
            )
            stdout_text = proc.stdout or ""
            stderr_text = proc.stderr or ""
            returncode: int | None = proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout_text = exc.stdout or ""
            stderr_text = exc.stderr or ""
            returncode = None

        self.write_text(stdout_path, stdout_text)
        self.write_text(stderr_path, stderr_text)

        result: dict[str, Any] = {
            "command": shell_join(cmd),
            "returncode": returncode,
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "prompt_path": str(prompt_path),
        }
        if returncode is None:
            result["timed_out"] = True
            result["stderr_tail"] = redact(stderr_text or stdout_text)
        elif returncode != 0:
            result["stderr_tail"] = redact(stderr_text or stdout_text)
        elif stdout_text:
            result["stdout_tail"] = redact(stdout_text)

        self.append_master_log_block(
            "orchestrator complete",
            "\n".join(
                [
                    f"returncode: {returncode if returncode is not None else 'timeout'}",
                    f"worktree: {worktree}",
                    "",
                    "stdout:",
                    stdout_text,
                    "",
                    "stderr:",
                    stderr_text,
                ]
            ),
        )
        return result

    def format_open_pr_signal(self, rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "No open autopilot PRs."

        lines = []
        for row in rows[:5]:
            lines.append(
                "- PR #{number} {title} | branch={branch} | {url}".format(
                    number=row.get("number"),
                    title=(row.get("title") or "")[:100],
                    branch=row.get("headRefName"),
                    url=row.get("url"),
                )
            )
        return "\n".join(lines)

    def build_orchestrator_prompt(self, *, open_autopilot_prs: list[dict[str, Any]]) -> str:
        return compose_prompt(
            ORCHESTRATOR_ROLE_CONTEXT,
            OVERALL_GOALS_CONTEXT,
            PROJECT_CONTEXT,
            EVALUATION_CONTEXT,
            SCIENTIFIC_METHOD_CONTEXT,
            ENGINEERING_CONTEXT,
            f"""
            Runtime context:

            History file: `{self.history_path}`
            Attempt directory: `{self.attempts_dir}`

            Open autopilot PRs:
            {self.format_open_pr_signal(open_autopilot_prs)}
            """,
        )

    def build_codex_prompt(self, *, branch_name: str) -> str:
        return compose_prompt(
            WORKER_ROLE_CONTEXT,
            OVERALL_GOALS_CONTEXT,
            PROJECT_CONTEXT,
            WORKER_DIRECTION_CONTEXT,
            EVALUATION_CONTEXT,
            SCIENTIFIC_METHOD_CONTEXT,
            ENGINEERING_CONTEXT,
            f"""
            Assigned branch: `{branch_name}`

            Explore the repository, run the tests or benchmarks you think matter, and make the strongest change you can justify.
            In your final message, make your shipping decision and validation outcome clear. Plain language is fine.
            """,
        )

    def summarize_orchestrator_result(self, orchestrator_result: dict[str, Any], worktree: Path) -> dict[str, Any]:
        dirty = self.git_status_dirty(worktree)
        return {
            "returncode": orchestrator_result.get("returncode"),
            "dirty": dirty,
            "stdout_tail": orchestrator_result.get("stdout_tail"),
            "stderr_tail": orchestrator_result.get("stderr_tail"),
        }

    def git_status_dirty(self, worktree: Path) -> bool:
        proc = run_command(["git", "status", "--short"], cwd=worktree)
        return bool((proc.stdout or "").strip())

    def git_diff_stat(self, worktree: Path) -> str:
        proc = run_command(["git", "diff", "--stat"], cwd=worktree)
        return redact(proc.stdout or "")

    def lookup_open_pr_for_branch(self, branch_name: str, worktree: Path) -> dict[str, Any] | None:
        proc = run_command(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "open",
                "--head",
                branch_name,
                "--json",
                "number,url,title",
            ],
            cwd=worktree,
            timeout=5 * 60,
        )
        if proc.returncode != 0:
            return None
        try:
            rows = json.loads(proc.stdout or "[]")
        except json.JSONDecodeError:
            return None
        if not rows:
            return None
        return rows[0]

def build_default_config() -> Config:
    repo_root = Path(__file__).resolve().parent.parent
    state_dir = repo_root / DEFAULT_STATE_DIR
    return Config(
        repo_root=repo_root,
        state_dir=state_dir,
        base_branch=DEFAULT_BASE_BRANCH,
        branch_prefix=DEFAULT_BRANCH_PREFIX,
        monitor_seconds=DEFAULT_MONITOR_SECONDS,
        orchestrator_interval_seconds=DEFAULT_ORCHESTRATOR_INTERVAL_SECONDS,
        orchestrator_timeout_seconds=DEFAULT_ORCHESTRATOR_TIMEOUT_SECONDS,
        worker_count=DEFAULT_WORKER_COUNT,
    )


def configure_logging(state_dir: Path) -> None:
    logs_dir = state_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "master.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ],
        force=True,
    )


def main() -> int:
    while True:
        try:
            config = build_default_config()
            configure_logging(config.state_dir)
            supervisor = CodexAutopilot(config)
            result = supervisor.run()
            logging.error("supervisor.run returned unexpectedly with result=%r; restarting", result)
        except BaseException:
            try:
                logging.exception("top-level supervisor wrapper caught a fatal error; restarting")
            except Exception:
                print("top-level supervisor wrapper caught a fatal error; restarting", file=sys.stderr)
        time.sleep(DEFAULT_RESTART_BACKOFF_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
