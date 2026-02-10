import subprocess
import re

def get_tests():
    output = subprocess.check_output(["cargo", "test", "--", "--list"], text=True)
    tests = []
    for line in output.splitlines():
        if ": test" in line:
            test_name = line.split(": test")[0].strip()
            tests.append(test_name)
    return tests

def run_test(test_name):
    try:
        # Running with a timeout to prevent hanging
        subprocess.check_call(["cargo", "test", test_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        return True
    except subprocess.CalledProcessError:
        return False
    except subprocess.TimeoutExpired:
        return "Timeout"

def main():
    tests = get_tests()
    print(f"Found {len(tests)} tests.")
    failed_tests = []
    timeout_tests = []
    
    for i, test in enumerate(tests):
        print(f"[{i+1}/{len(tests)}] Running {test}...", end="", flush=True)
        result = run_test(test)
        if result == True:
            print(" PASS")
        elif result == "Timeout":
            print(" TIMEOUT")
            timeout_tests.append(test)
        else:
            print(" FAIL")
            failed_tests.append(test)
            
    print("\n\nSummary:")
    print(f"Failed tests: {len(failed_tests)}")
    for t in failed_tests:
        print(f" - {t}")
        
    print(f"\nTimeout tests: {len(timeout_tests)}")
    for t in timeout_tests:
        print(f" - {t}")

if __name__ == "__main__":
    main()
