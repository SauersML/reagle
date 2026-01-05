"""
Jules Optimizer Loop - Continuous improvement cycle for Rust code.

This script runs within the 'jules_loop.yml' workflow.
It analyzes the build log from the current run (or a provided log file),
decides on improvements, sends a request to the Jules API, applies the patch,
and pushes the changes.

REGRESSION PROTECTION:
- If the Rust build was PASSING before Jules' changes, we verify it still passes
  after applying the patch. If the build now fails (regression), we abort.
- If the Rust build was already FAILING, we allow commits even if still failing,
  as any progress is better than no progress.
"""
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


def strip_ansi(text):
    """Removes ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


def run_command(cmd, check=False):
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


def filter_noise(logs):
    """
    Remove noisy lines that don't help with debugging.
    These are mostly downloading/compiling dependency messages.
    """
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

    filtered_lines = []
    for line in logs.split('\n'):
        is_noise = any(pattern in line for pattern in noise_patterns)
        if not is_noise:
            filtered_lines.append(line)

    return '\n'.join(filtered_lines)


def get_run_info():
    """
    Retrieves status and logs.
    First checks for local environment variables provided by the workflow.
    """
    print("\n--- Getting Run Info ---")

    local_log_file = os.environ.get("LOCAL_LOG_FILE")
    local_status = os.environ.get("LOCAL_BUILD_STATUS")

    if local_log_file and local_status:
        print(f"Using local log file: {local_log_file} with status: {local_status}")
        try:
            with open(local_log_file, 'r') as f:
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


def verify_rust_build():
    """
    Run the Rust build and return True if it passes, False otherwise.
    """
    print("\n--- Verifying Rust Build ---")

    # Run cargo check first (faster), then cargo test
    stdout, stderr, code = run_command("cd Rust && cargo check --all-targets")

    if code == 0:
        print("Cargo check passed. Running tests...")
        stdout, stderr, code = run_command("cd Rust && cargo test")

        if code == 0:
            print("Rust build and tests passed.")
            return True
        else:
            print("Rust tests failed.")
            print(f"stderr: {stderr[:2000]}" if stderr else "")
            return False
    else:
        print("Rust build failed.")
        print(f"stderr: {stderr[:2000]}" if stderr else "")
        return False


def call_jules(prompt, attempt=1):
    """
    Interacts with Jules API to get a plan and changeset.
    Returns the changeset or None if Jules couldn't produce one.
    """
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
            "githubRepoContext": {"startingBranch": "main"}
        }
    }

    print("Sending payload to Jules API:")
    print(json.dumps(payload, indent=2))

    try:
        resp = requests.post(
            f"{JULES_API_URL}/v1alpha/sessions",
            headers={"X-Goog-Api-Key": api_key},
            json=payload,
            timeout=60
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

    max_polls = 180  # 30 minutes of polling
    seen_ids = set()

    for i in range(max_polls):
        time.sleep(10)
        print(f"Polling activities... (Poll {i+1}/{max_polls})")

        try:
            r = requests.get(
                f"{JULES_API_URL}/v1alpha/{session_name}/activities",
                headers={"X-Goog-Api-Key": api_key},
                timeout=30
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
                print("Plan Generated:")
                steps = act["planGenerated"].get("plan", {}).get("steps", [])
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

    # Common restrictions for all prompts
    version_restriction = (
        "\n\nNOTE:\n"
        "- You are encouraged to proactively search the web for Rust documentation.\n"
        "- DO NOT modify 'rust-toolchain.toml' if it exists - the Rust version may be pinned.\n"
        "- Focus ONLY on Rust/*.rs files and Rust/Cargo.toml for improvements.\n"
        "- This is a port of BEAGLE (genotype phasing/imputation) to Rust.\n"
        "- Reference the Java/ directory for the original implementation if needed.\n"
        "- Always try to improve something--commit and finish. No further instruction will be given.\n"
        "- Run 'cargo check' and 'cargo test' to verify your changes compile.\n"
    )

    if conclusion == "success":
        prompt = (
            "The Rust build passed successfully. "
            "Please find one thing to implement, improve, or optimize in the Rust code "
            "(specifically files in Rust/src/). You must successfully compile the code yourself. "
            "If the build fails, do not submit and keep working. "
            "This is a port of BEAGLE (a genotype phasing and imputation tool) from Java to Rust. "
            "You can: implement missing functionality by referencing Java/, optimize hot paths, "
            "add proper error handling, implement tests, or improve the data structures. "
            "Feel free to try big or multiple tasks. "
            "IMPORTANT: Ensure your changes compile with 'cargo check' and pass 'cargo test'. "
            "Do not break existing functionality."
            + version_restriction
        )
    else:
        prompt = (
            f"The Rust build failed. "
            f"Here are the logs from the run (ANSI colors stripped):\n\n{logs}\n\n"
            "Please analyze the logs and fix the errors in the Rust code. "
            "If the code does not compile, you can commit a small improvement even if not complete. "
            "You can search the web to find documentation for crates you're using. "
            "This is a port of BEAGLE (genotype phasing/imputation) from Java to Rust. "
            "Reference Java/ for the original implementation if helpful. "
            "You should check if your changes compile with 'cargo check'. "
            "However, if the code does not compile, improve what you can before submitting. "
            "It's okay if it still fails as long as it is in a better state. "
            "Feel free to fix multiple issues at once. You can do it!"
            + version_restriction
        )

    print(f"\nPrompting Jules with:\n{prompt}\n")

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

    out, err, code = run_command("git apply jules.patch")
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
        build_passes_now = verify_rust_build()

        if not build_passes_now:
            print("\n REGRESSION DETECTED: Build was passing, but Jules' changes broke it!")
            print("Reverting changes and aborting commit.")
            run_command("git checkout -- .")
            run_command("git clean -fd")
            sys.exit(0)
        else:
            print("Regression check passed: build still works.")
    else:
        print("\n--- Skipping regression check (build was already failing) ---")
        print("Jules' changes will be committed even if build still fails.")
        verify_rust_build()

    print("\n--- Committing and Pushing ---")
    print(f"Commit message: {msg}")

    run_command(['git', 'commit', '-m', msg], check=True)

    print("Pulling latest changes to avoid non-fast-forward...")
    run_command("git pull --rebase origin main", check=True)

    print("Pushing changes...")
    run_command("git push origin main", check=True)

    print("Push complete. Waiting for next CRON schedule.")


if __name__ == "__main__":
    main()
