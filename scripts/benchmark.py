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
    if not os.path.exists("target.vcf.gz"):
        print("ERROR: target.vcf.gz missing after prep!")
        run_cmd(["ls", "-l"])
    
    run_cmd(["./target/release/reagle", "--ref", "ref.vcf.gz", "--gt", "target.vcf.gz", "--out", "reagle_out", "--chrom", "22"])

    # 3. Run Beagle
    print("=== Running Beagle ===")
    beagle_jar = "tests/fixtures/beagle_reference/beagle.27Feb25.75f.jar"
    run_cmd(["java", "-Xmx6g", "-jar", beagle_jar, "ref=ref.vcf.gz", "gt=target.vcf.gz", "out=beagle_out", "chrom=22", "nthreads=2", "gp=true"])

    # 4. Run Metrics using the Python integration test
    print("=== Calculating Metrics ===")
    
    # Move outputs to expected locations for integration test
    import shutil
    os.makedirs("tests/data", exist_ok=True)
    shutil.copy("truth.vcf.gz", "tests/data/truth.vcf.gz")
    shutil.copy("reagle_out.vcf.gz", "tests/data/reagle_imputed.vcf.gz")
    shutil.copy("beagle_out.vcf.gz", "tests/data/beagle_imputed.vcf.gz")
    
    # Harmonize sample names - all files must have the same sample name for comparison
    # Create a sample name mapping file
    with open("sample_name.txt", "w") as f:
        f.write("SAMPLE\n")
    
    for vcf in ["tests/data/truth.vcf.gz", "tests/data/reagle_imputed.vcf.gz", "tests/data/beagle_imputed.vcf.gz"]:
        if os.path.exists(vcf):
            # Rename sample using bcftools reheader
            temp_vcf = vcf + ".tmp"
            run_cmd(f"bcftools reheader -s sample_name.txt {vcf} -o {temp_vcf}", shell=True)
            os.replace(temp_vcf, vcf)
            run_cmd(["bcftools", "index", "-f", vcf])
    
    os.remove("sample_name.txt")
    
    # Run metrics via integration test (it now auto-detects DS/GP vs GT-only format)
    run_cmd(["python3", "tests/integration_test.py", "metrics"])

if __name__ == "__main__":
    print("Benchmark Script v1.1")
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--format", required=True)
    args = parser.parse_args()
    
    run_benchmark(args.person, args.file, args.format)
