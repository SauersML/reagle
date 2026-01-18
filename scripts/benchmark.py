import os
import sys
import subprocess
import argparse

def run_cmd(cmd, shell=False):
    print(f"Running: {cmd}")
    if shell:
        subprocess.check_call(cmd, shell=True)
    else:
        subprocess.check_call(cmd)

def run_benchmark(person, file_path, format):
    # 1. Prepare Data
    print(f"=== Preparing data for {person} ({file_path}) ===")
    run_cmd(["python3", "scripts/prepare_data.py", "array", file_path, "target.vcf.gz"])
    run_cmd(["python3", "scripts/prepare_data.py", "reference", "ref.vcf.gz"])
    
    truth_dir = "data/kat_suricata" if person == "Kat" else "data/christopher_smith"
    run_cmd(["python3", "scripts/prepare_data.py", "truth", truth_dir, "truth.vcf.gz"])

    # 2. Run Reagle
    print("=== Running Reagle ===")
    run_cmd(["./target/release/reagle", "--ref", "ref.vcf.gz", "--gt", "target.vcf.gz", "--out", "reagle_out", "--chrom", "22"])

    # 3. Run Beagle
    print("=== Running Beagle ===")
    beagle_jar = "tests/fixtures/beagle_reference/beagle.27Feb25.75f.jar"
    run_cmd(["java", "-Xmx6g", "-jar", beagle_jar, "ref=ref.vcf.gz", "gt=target.vcf.gz", "out=beagle_out", "chrom=22", "nthreads=2"])

    # 4. Run Metrics
    print("=== Calculating Metrics ===")
    env = os.environ.copy()
    env["TEST_TRUTH_VCF"] = "truth.vcf.gz"
    
    # Reagle Metrics
    env["TEST_IMP_VCF"] = "reagle_out.vcf.gz"
    env["TEST_OUTPUT_JSON"] = "reagle_metrics.json"
    subprocess.check_call(["cargo", "test", "--test", "imputation_quality", "test_metrics_calculation_dummy", "--", "--nocapture"], env=env)
    
    # Beagle Metrics
    env["TEST_IMP_VCF"] = "beagle_out.vcf.gz"
    env["TEST_OUTPUT_JSON"] = "beagle_metrics.json"
    subprocess.check_call(["cargo", "test", "--test", "imputation_quality", "test_metrics_calculation_dummy", "--", "--nocapture"], env=env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--format", required=True)
    args = parser.parse_args()
    
    run_benchmark(args.person, args.file, args.format)
