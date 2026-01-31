import os
import sys
import subprocess
import argparse
import shutil

BEAGLE_URL = "https://faculty.washington.edu/browning/beagle/beagle.27Feb25.75f.jar"
BEAGLE_JAR = "beagle.jar"


def run_cmd(cmd, shell=False):
    print(f"Running: {cmd}")
    if shell:
        subprocess.check_call(cmd, shell=True)
    else:
        subprocess.check_call(cmd)


def ensure_beagle():
    """Download Beagle JAR if not present."""
    if os.path.exists(BEAGLE_JAR):
        return BEAGLE_JAR
    print(f"Downloading Beagle from {BEAGLE_URL}...")
    if shutil.which("wget"):
        subprocess.check_call(["wget", "-q", BEAGLE_URL, "-O", BEAGLE_JAR])
    elif shutil.which("curl"):
        subprocess.check_call(["curl", "-fsSL", BEAGLE_URL, "-o", BEAGLE_JAR])
    else:
        raise RuntimeError("Neither wget nor curl found")
    return BEAGLE_JAR

def run_benchmark(person, file_path, format):
    # 1. Prepare Data
    print(f"=== Preparing data for {person} ({file_path}) ===")
    run_cmd(["python3", "scripts/prepare_data.py", "reference", "ref.vcf.gz"])
    run_cmd(["python3", "scripts/prepare_data.py", "array", file_path, "target.vcf.gz", "ref.vcf.gz"])

    truth_dir = "data/kat_suricata" if person == "Kat" else "data/christopher_smith"
    run_cmd(["python3", "scripts/prepare_data.py", "truth", truth_dir, "truth.vcf.gz", "ref.vcf.gz"])

    # 2. Run Reagle
    print("=== Running Reagle ===")
    if not os.path.exists("target.vcf.gz"):
        print("ERROR: target.vcf.gz missing after prep!")
        run_cmd(["ls", "-l"])
    
    run_cmd(["./target/release/reagle", "--ref", "ref.vcf.gz", "--gt", "target.vcf.gz", "--out", "reagle_out", "--chrom", "chr22"])

    # 3. Run Beagle
    print("=== Running Beagle ===")
    # Reference panels used for Beagle imputation must be fully phased and contain no missing data.
    # We create a derivative panel that excludes sites with missing calls and forces phased separators.
    # The original ref.vcf.gz is preserved for Reagle to ensure no loss of data in the primary pipeline.
    # NOTE: +setGT must run BEFORE filtering missing data to handle sparse panels correctly.
    # We first fill missing values with Reference (0) to ensure the panel is dense, then force phasing.
    run_cmd("bcftools +setGT ref.vcf.gz -Ou -- -t . -n 0 | bcftools +setGT - -Ou -- -t a -n p | bcftools view -Oz -o ref_beagle.vcf.gz", shell=True)
    run_cmd(["bcftools", "index", "-f", "ref_beagle.vcf.gz"])

    beagle_jar = ensure_beagle()
    run_cmd(["java", "-Xmx6g", "-jar", beagle_jar, "ref=ref_beagle.vcf.gz", "gt=target.vcf.gz", "out=beagle_out", "chrom=chr22", "nthreads=4", "gp=true"])
    
    # 4. Run Metrics using the Python integration test
    print("=== Calculating Metrics ===")
    
    # Move outputs to expected locations for integration test
    os.makedirs("tests/data", exist_ok=True)
    shutil.copy("truth.vcf.gz", "tests/data/truth.vcf.gz")
    shutil.copy("reagle_out.vcf.gz", "tests/data/reagle_imputed.vcf.gz")
    shutil.copy("beagle_out.vcf.gz", "tests/data/beagle_imputed.vcf.gz")
    shutil.copy("ref.vcf.gz", "tests/data/ref.vcf.gz")
    
    # Get sample counts from VCFs for proper reporting
    result = subprocess.run("bcftools query -l ref.vcf.gz", shell=True, capture_output=True, text=True)
    ref_samples = [s.strip() for s in result.stdout.strip().split('\n') if s.strip()]
    
    result = subprocess.run("bcftools query -l truth.vcf.gz", shell=True, capture_output=True, text=True)
    truth_samples = [s.strip() for s in result.stdout.strip().split('\n') if s.strip()]
    
    # Write sample list files for metrics stage
    with open("tests/data/train_samples.txt", "w") as f:
        f.write('\n'.join(ref_samples))
    with open("tests/data/test_samples.txt", "w") as f:
        f.write('\n'.join(truth_samples))
    
    print(f"Reference samples: {len(ref_samples)}")
    print(f"Test samples: {len(truth_samples)}")
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--format", required=True)
    args = parser.parse_args()
    
    run_benchmark(args.person, args.file, args.format)
