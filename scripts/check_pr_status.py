
import subprocess
import json
import sys

import concurrent.futures
import re

import concurrent.futures
import re

def run_command(command, use_shell=False):
    try:
        # If use_shell is True, command should be a string, otherwise a list
        result = subprocess.run(command, check=True, capture_output=True, text=True, shell=use_shell)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        # Don't print error for grep finding nothing (exit code 1)
        if use_shell and e.returncode == 1:
            return ""
        # print(f"Error running command: {command}", file=sys.stderr)
        return None

def get_prs():
    # Get number, title, headRefName, and also url for convenience
    cmd = ["gh", "pr", "list", "--json", "number,title,headRefName,url"]
    output = run_command(cmd)
    if output:
        return json.loads(output)
    return []

def get_run_for_workflow(branch_name, workflow_name):
    cmd = [
        "gh", "run", "list", 
        "--workflow", workflow_name, 
        "--branch", branch_name, 
        "--limit", "1", 
        "--json", "databaseId,conclusion,status,url"
    ]
    output = run_command(cmd)
    if output:
        runs = json.loads(output)
        if runs:
            return runs[0]
    return None

def get_test_counts(run_id):
    # run_id must be a string or int
    # grep for "test result:"
    # Use shell=True to pipe
    cmd = f"gh run view {run_id} --log | grep 'test result:'"
    output = run_command(cmd, use_shell=True)
    
    total_passed = 0
    total_failed = 0
    
    if output:
        # Expected line format: ... test result: ok. 124 passed; 0 failed; ...
        # OR: ... test result: FAILED. 25 passed; 2 failed; ...
        # Regex to capture passed and failed counts
        # Look for pattern: result: \w+\. (\d+) passed; (\d+) failed
        matches = re.findall(r"result: \w+\. (\d+) passed; (\d+) failed", output)
        for p, f in matches:
            total_passed += int(p)
            total_failed += int(f)
            
    return total_passed, total_failed

def get_beagle_stats(run_id):
    """
    Parses the log to find the specific block for 'beagle_reference' tests
    and extracts pass/fail counts.
    Returns (passed, failed) or (0, 0) if not found.
    """
    # Fetch full log
    cmd = f"gh run view {run_id} --log"
    output = run_command(cmd, use_shell=True)
    
    if not output:
        return 0, 0
        
    lines = output.splitlines()
    found_beagle_start = False
    
    # We look for the line: Running .../deps/beagle_reference-...
    # The log has prefixes, so we check inclusion.
    
    for i, line in enumerate(lines):
        if "deps/beagle_reference" in line and "Running" in line:
            found_beagle_start = True
            # Now search forward for the next "test result:"
            # We assume the test result following this start line belongs to it.
            for j in range(i + 1, len(lines)):
                res_line = lines[j]
                if "test result:" in res_line:
                    # Parse this line
                    match = re.search(r"result: \w+\. (\d+) passed; (\d+) failed", res_line)
                    if match:
                        return int(match.group(1)), int(match.group(2))
                    else:
                        # Should match if "test result:" is present
                        return 0, 0
    
    return 0, 0

def get_seed_mean_diff(run_id):
    """
    Parses the log for seed accuracy lines like:
    'Seed 1: Java 98.57%, Rust 98.57%'
    Returns formatted mean diff string or 'N/A'.
    """
    cmd = f"gh run view {run_id} --log | grep -E \"Seed[[:space:]]+[0-9]+: Java\""
    output = run_command(cmd, use_shell=True)
    if not output:
        return "N/A"

    diffs = []
    pattern = re.compile(r"Seed\s+\d+:\s+Java\s+([0-9.]+)%\s*,\s*Rust\s+([0-9.]+)%")
    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            java = float(match.group(1))
            rust = float(match.group(2))
            diffs.append(rust - java)

    if not diffs:
        return "N/A"

    mean_diff = sum(diffs) / len(diffs)
    sign = "+" if mean_diff >= 0 else "-"
    return f"{sign}{abs(mean_diff):.2f}%"

def fetch_pr_info(pr):
    """Fetches workflow status for a single PR and returns the processed data."""
    number = pr['number']
    title = pr['title']
    branch = pr['headRefName']
    
    # 1. Imputation Quality Assessment
    imp_run = get_run_for_workflow(branch, "Imputation Quality Assessment")
    imp_disp = "N/A"
    
    if imp_run:
        status = imp_run.get('status', 'unknown')
        conclusion = imp_run.get('conclusion')
        
        if status == 'completed':
            if conclusion == 'success':
                imp_disp = "✅ PASS"
            elif conclusion == 'failure':
                imp_disp = "❌ FAIL"
            else:
                imp_disp = f"⚠️  {str(conclusion).upper()}"
        else:
            imp_disp = f"⏳ {status.upper()}"
    else:
        imp_disp = "❓ NO RUN"

    # 2. CI (Counts) & Beagle Reference
    ci_run = get_run_for_workflow(branch, "CI")
    ci_disp = "N/A"
    beagle_disp = "N/A"
    seed_disp = "N/A"
    
    if ci_run:
        status = ci_run.get('status', 'unknown')
        conclusion = ci_run.get('conclusion')
        run_id = ci_run.get('databaseId')
        
        # We try to get counts if it's completed (success or failure)
        if status == 'completed' and run_id:
             # Get total CI counts
             passed, failed = get_test_counts(run_id)
             if passed == 0 and failed == 0:
                 if conclusion == 'success':
                     ci_disp = "✅ 0/0 (?)"
                 elif conclusion == 'failure':
                     ci_disp = "❌ BUILD FAIL"
                 else:
                     ci_disp = f"{conclusion.upper()}"
             else:
                 ci_prefix = "✅" if conclusion == 'success' else "❌"
                 ci_disp = f"{ci_prefix} {passed}P / {failed}F"
                 
             # Get Beagle Reference Stats
             b_passed, b_failed = get_beagle_stats(run_id)
             b_total = b_passed + b_failed
             if b_total > 0:
                 percent = (b_passed / b_total) * 100
                 # Format: "89% (24/27)"
                 b_prefix = "✅" if b_failed == 0 else "❌"
                 beagle_disp = f"{b_prefix} {percent:.0f}% ({b_passed}/{b_total})"
             else:
                 # Check if CI failed (likely unit tests)
                 if conclusion == 'failure':
                     beagle_disp = "❌ Unit Fail"
                 else:
                     beagle_disp = "❓ Not Found"

             seed_disp = get_seed_mean_diff(run_id)
                 
        else:
            ci_disp = f"⏳ {status.upper()}"
            beagle_disp = "⏳"
            seed_disp = "⏳"
    else:
        ci_disp = "❓ NO RUN"
        beagle_disp = "-"
        seed_disp = "-"


    # Truncate branch name for display
    display_branch = (branch[:30] + '..') if len(branch) > 30 else branch
    
    return {
        'number': number,
        'imp_disp': imp_disp,
        'ci_disp': ci_disp,
        'beagle_disp': beagle_disp,
        'seed_disp': seed_disp,
        'display_branch': display_branch,
        'title': title
    }

def main():
    print("Fetching open PRs...")
    prs = get_prs()
    
    if not prs:
        print("No open PRs found.")
        return

    print(f"Found {len(prs)} open PRs. Fetching statuses concurrently...\n")
    
    # Header
    print(f"{'PR #':<6} | {'Imp. Qual.':<15} | {'CI Tests':<20} | {'Beagle Ref':<20} | {'Seed Δ':<8} | {'Branch':<33} | {'Title'}")
    print("-" * 165)

    # Use ThreadPoolExecutor to fetch statuses in parallel
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Map returns results in the order of the input iterable
        results = list(executor.map(fetch_pr_info, prs))

    for res in results:
        print(f"#{res['number']:<5} | {res['imp_disp']:<15} | {res['ci_disp']:<20} | {res['beagle_disp']:<20} | {res['seed_disp']:<8} | {res['display_branch']:<33} | {res['title']}")

if __name__ == "__main__":
    main()
