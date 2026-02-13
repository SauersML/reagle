import os
import sys
import json
import time
import subprocess
import requests
import re

JULES_API_URL = "https://jules.googleapis.com"
MAX_RETRIES = 2
RETRY_DELAY = 60

PROMPT_TEXT = """
The goal is to have the best and most accurate imputation (and phasing) out of any tool, while being extraordinarily fast (and no OOM).
This project is for genotype phasing and imputation. We care most about accuracy, then very fast speed, with the constraint that it must not run out of memory with ~15 GB.
We care about real-world usage above any test. The goal of the project should be to maximize universal future wellbeing above any one particular present task.
This should guide our decision-making. This happens through taking proper long-term solutions. How?
Well, better phasing and imputation leads to downstream better, e.g., predictive power for disease prediction, disease associations, science, etc., which improve lives tangibly.
This means we must always do what is best for real-world accuracy and performance.

Do not edit tests, loosen thresholds, or edit constants.

Large changes are not bad (bigness of change is orthogonal to goodness), and should not be avoided. We don’t care about backwards compatibility.
Model changes or algorithm changes are not bad.
Instead of having a bias towards the status quo, we should have a bias for what is most likely to be correct when reasoning from first principles.
Don’t assume the current code has been well-tested or is already deployed. It’s in the prototype stage, and core ideas may change.
There may not always be correct or verified reasons for the decisions made in the code.

RULES:
- Always make a real improvement and finish in one PR.
- Use the provided build log as a signal. If there are compile errors or failing tests, you might diagnose root cause and fix the implementation.
- If everything passes, improve performance, memory, correctness, and add or strengthen tests.
- Never weaken, delete, or #[ignore] tests to make them pass. Failing tests are valuable information.
- Only change a test if the test itself is genuinely wrong (not just failing).
- Commit progress if code compiles, even with failing tests.
- Correctness matters: this code affects real health outcomes via polygenic scores.
- No hacks. Do the proper fix.
- Avoid small, unimpactful changes such as knob tuning or tweaking.

SCOPE:
- DO NOT modify rust-toolchain.toml if it exists.
- Focus ONLY on src/*.rs files and Cargo.toml.
- Run cargo check --all-targets and cargo test to validate changes.
- You are encouraged to proactively search official Rust documentation when needed.

Remember to keep the overall goals in mind as we implement. Do this yourself in full now without stopping until it is fully finished.
We must have the perfect, ideal, best version for our goals. We must do it in the best way possible.

OUTPUT:
- Produce a patch that meaningfully improves the project.
""".strip()

def strip_ansi(text: str) -> str:
    """Removes ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


def run_command(cmd, check: bool = False):
    """Run a shell command and return stdout, stderr, return code."""
    if isinstance(cmd, list):
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    else:
        print(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)

    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(result.stdout)
        print(result.stderr)
        sys.exit(result.returncode)

    return result.stdout.strip(), result.stderr.strip(), result.returncode


def filter_noise(logs: str) -> str:
    """Remove noisy lines that don't help with debugging."""
    noise_patterns = [
        "Downloading crates ...",
        "Downloaded ",
        "Compiling proc-macro",
        "Compiling unicode-",
        "Compiling syn ",
        "Compiling quote ",
        "Compiling serde_derive",
        "Compiling memchr",
        "Compiling libc",
        "Compiling cfg-if",
        "Compiling autocfg",
        "Finished ",
        "Fresh ",
    ]

    out = []
    for line in logs.split("\n"):
        if any(pat in line for pat in noise_patterns):
            continue
        out.append(line)
    return "\n".join(out)


def get_run_info():
    """
    Retrieves status and logs.
    Expects LOCAL_LOG_FILE and LOCAL_BUILD_STATUS to be set by workflow.
    """
    print("\n--- Getting Run Info ---")
    local_log_file = os.environ.get("LOCAL_LOG_FILE")
    local_status = os.environ.get("LOCAL_BUILD_STATUS")

    if local_log_file and local_status:
        print(f"Using local log file: {local_log_file} with status: {local_status}")
        try:
            with open(local_log_file, "r") as f:
                raw_logs = f.read()

            logs = filter_noise(strip_ansi(raw_logs))

            max_len = 300000
            if len(logs) > max_len:
                print(f"Log size {len(logs)} exceeds {max_len}. Keeping last {max_len} characters.")
                logs = "..." + logs[-max_len:]

            return local_status, logs
        except Exception as e:
            print(f"Error reading local log file: {e}")
            return local_status, f"Error reading log file: {e}"

    print("No local log file provided. This script is expected to run with LOCAL_LOG_FILE set.")
    return "unknown", "No logs available."


def verify_rust_build() -> bool:
    """Run cargo check + cargo test. Return True if both pass."""
    print("\n--- Verifying Rust Build ---")

    _, stderr, code = run_command("cargo check --all-targets")
    if code != 0:
        print("Rust build failed.")
        if stderr:
            print(f"stderr: {stderr[:2000]}")
        return False

    print("Cargo check passed. Running tests...")
    _, stderr, code = run_command("cargo test")
    if code != 0:
        print("Rust tests failed.")
        if stderr:
            print(f"stderr: {stderr[:2000]}")
        return False

    print("Rust build and tests passed.")
    return True


def build_prompt(conclusion: str, logs: str) -> str:
    context = (
        "RUN CONTEXT:\n"
        f"- build_status: {conclusion}\n\n"
        "BUILD LOG (filtered):\n\n"
        f"{logs}\n"
    )
    return context + "\n\n" + PROMPT_TEXT


def call_jules(prompt: str, attempt: int = 1):
    """Create a Jules session, poll activities, and return a ChangeSet (or PR_CREATED/None)."""
    api_key = os.environ.get("JULES_API_KEY")
    repo = os.environ.get("GITHUB_REPOSITORY")

    if not api_key:
        print("Error: JULES_API_KEY not set.")
        sys.exit(1)

    print(f"\n--- Initializing Jules Session (Attempt {attempt}/{MAX_RETRIES}) ---")

    payload = {
        "prompt": prompt,
        "sourceContext": {
            "source": f"sources/github/{repo}",
            "githubRepoContext": {"startingBranch": "main"},
        },
        "automationMode": "AUTO_CREATE_PR",
    }

    print("Sending payload to Jules API:")
    print(json.dumps(payload, indent=2))

    try:
        resp = requests.post(
            f"{JULES_API_URL}/v1alpha/sessions",
            headers={"X-Goog-Api-Key": api_key},
            json=payload,
            timeout=60,
        )
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

    if resp.status_code != 200:
        print(f"Failed to create session: {resp.text}")
        return None

    session = resp.json()
    session_name = session["name"]
    print(f"Session created: {session_name}")

    max_polls = 180
    seen_ids = set()

    for i in range(max_polls):
        time.sleep(10)
        print(f"Polling activities... (Poll {i+1}/{max_polls})")

        try:
            r = requests.get(
                f"{JULES_API_URL}/v1alpha/{session_name}/activities",
                headers={"X-Goog-Api-Key": api_key},
                timeout=30,
            )
        except requests.exceptions.RequestException as e:
            print(f"Polling error: {e}")
            continue

        if r.status_code != 200:
            print(f"Error polling: {r.text}")
            continue

        activities = r.json().get("activities", [])
        activities.sort(key=lambda x: x.get("createTime", ""))

        latest_changeset = None

        for act in activities:
            act_id = act.get("id")
            if act_id in seen_ids:
                continue
            seen_ids.add(act_id)

            originator = act.get("originator", "UNKNOWN")
            print(f"\n--- New Activity ({originator}) ---")

            if "planGenerated" in act:
                steps = act["planGenerated"].get("plan", {}).get("steps", [])
                if steps:
                    print("Plan Generated:")
                    for step in steps:
                        print(f"  {step.get('index', '?')}. {step.get('title', '')}")

            if "progressUpdated" in act:
                prog = act["progressUpdated"]
                print(f"Status: {prog.get('title', '')}")
                if "description" in prog:
                    print(f"Details: {prog['description']}")

            if "artifacts" in act:
                for art in act["artifacts"]:
                    if "bashOutput" in art:
                        bo = art["bashOutput"]
                        print(f"Bash Command: {bo.get('command')}")
                        print(f"Output:\n{bo.get('output')}")
                    if "changeSet" in art:
                        print("Artifact: ChangeSet found.")
                        latest_changeset = art["changeSet"]
                    if "pullRequest" in art:
                        pr = art["pullRequest"]
                        print(f"Pull Request: {pr.get('title')} - {pr.get('url')}")
                        return "PR_CREATED"

            if "sessionCompleted" in act:
                print("Session Completed.")
                if latest_changeset:
                    return latest_changeset
                print("Session completed but no ChangeSet was produced.")
                return None

        if latest_changeset:
            return latest_changeset

    print("Timed out waiting for Jules to produce a ChangeSet.")
    return None


def main():
    conclusion, logs = get_run_info()
    prompt = build_prompt(conclusion, logs)

    print("\nPrompting Jules with:\n")
    print(prompt)
    print("\n")

    changeset = None
    for attempt in range(1, MAX_RETRIES + 1):
        changeset = call_jules(prompt, attempt)
        if changeset:
            break
        if attempt < MAX_RETRIES:
            print(f"\nJules didn't produce a changeset. Retrying in {RETRY_DELAY}s...")
            time.sleep(RETRY_DELAY)

    if not changeset:
        print("\nJules failed to produce a changeset after all retries.")
        sys.exit(0)

    if changeset == "PR_CREATED":
        print("\nJules created a PR directly. Nothing more to do.")
        sys.exit(0)

    patch = changeset.get("gitPatch", {}).get("unidiffPatch")
    msg = changeset.get("gitPatch", {}).get("suggestedCommitMessage", "Jules Improvement")

    if not patch:
        print("\nJules returned a ChangeSet but no unidiffPatch.")
        sys.exit(0)

    print("\n--- Applying Patch ---")
    print(f"Patch content:\n{patch}\n")

    with open("jules.patch", "w") as f:
        f.write(patch)

    run_command("git fetch origin main")
    run_command("git checkout -B main origin/main")

    _, err, code = run_command("git apply jules.patch")
    if code != 0:
        print(f"Failed to apply patch: {err}")
        print("Patch may be malformed.")
        sys.exit(0)

    run_command('git config user.name "Jules Bot"')
    run_command('git config user.email "jules-bot@google.com"')

    run_command("git add .")
    _, _, code = run_command("git diff --cached --quiet")
    if code == 0:
        print("\nNo changes to commit after applying patch.")
        sys.exit(0)

    was_passing_before = (conclusion == "success")

    if was_passing_before:
        print("\n--- Regression Check (build was passing before) ---")
        if not verify_rust_build():
            print("\nREGRESSION DETECTED: Build was passing, but Jules' changes broke it!")
            print("Reverting changes and aborting commit.")
            run_command("git checkout -- .")
            run_command("git clean -fd")
            sys.exit(0)
        print("Regression check passed: build still works.")
    else:
        print("\n--- Skipping regression check (build was already failing) ---")
        print("Jules' changes will be committed even if build still fails.")
        verify_rust_build()

    print("\n--- Committing and Creating PR ---")
    print(f"Commit message: {msg}")

    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    branch_name = f"jules/{timestamp}"

    run_command(f"git checkout -b {branch_name}", check=True)
    run_command(["git", "commit", "-m", msg], check=True)

    print(f"Pushing changes to branch {branch_name}...")
    run_command(f"git push origin {branch_name}", check=True)

    print("\n--- Creating Pull Request ---")
    github_token = os.environ.get("GITHUB_TOKEN")
    repo = os.environ.get("GITHUB_REPOSITORY")

    if not github_token:
        print("WARNING: GITHUB_TOKEN not set, cannot create PR.")
        sys.exit(0)

    if not repo:
        print("WARNING: GITHUB_REPOSITORY not set, cannot create PR.")
        sys.exit(0)

    pr_title = msg[:200] if len(msg) > 200 else msg
    pr_body = f"Automated improvement by Jules.\n\n**Summary:**\n{msg}"

    os.environ["GH_TOKEN"] = github_token

    gh_cmd = [
        "gh",
        "pr",
        "create",
        "--title",
        pr_title,
        "--body",
        pr_body,
        "--base",
        "main",
        "--head",
        branch_name,
    ]

    stdout, stderr, code = run_command(gh_cmd)
    if code == 0:
        print(f"PR created successfully via gh CLI: {stdout}")
        print("Done. PR creation attempted.")
        return

    print(f"gh CLI failed (code {code}): {stderr}")
    print("Falling back to GitHub API...")

    pr_payload = {"title": pr_title, "head": branch_name, "base": "main", "body": pr_body}

    try:
        pr_resp = requests.post(
            f"https://api.github.com/repos/{repo}/pulls",
            headers={
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json",
            },
            json=pr_payload,
            timeout=30,
        )

        print(f"API Response Status: {pr_resp.status_code}")
        if pr_resp.status_code == 201:
            pr_url = pr_resp.json().get("html_url")
            print(f"PR created successfully: {pr_url}")
        else:
            print(f"Failed to create PR: {pr_resp.status_code}")
            print(f"Response: {pr_resp.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error creating PR: {e}")

    print("Done. PR creation attempted.")


if __name__ == "__main__":
    main()
