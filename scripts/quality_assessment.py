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


def print_tool_versions():
    print("\n=== Tool Versions ===")
    for cmd in [
        "bcftools --version | head -1",
        "tabix --version | head -1",
        "java -version 2>&1 | head -1",
        "python3 --version",
    ]:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            line = result.stdout.strip() or result.stderr.strip()
            print(f"{cmd.split()[0]}: {line if line else 'unknown'}")
        except Exception as e:
            print(f"{cmd.split()[0]}: Error - {e}")


def print_vcf_diagnostics(vcf_path, label):
    print(f"\n=== VCF Diagnostics: {label} ===")
    if not os.path.exists(vcf_path):
        print(f"Missing: {vcf_path}")
        return
    try:
        size_bytes = os.path.getsize(vcf_path)
        print(f"Path: {vcf_path}")
        print(f"Size: {size_bytes / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"Size: Error - {e}")
    try:
        rec = subprocess.run(
            f"bcftools index -n {vcf_path}",
            shell=True, capture_output=True, text=True
        ).stdout.strip()
        print(f"Records: {rec if rec else 'unknown'}")
    except Exception as e:
        print(f"Records: Error - {e}")
    try:
        hdr = subprocess.run(
            f"bcftools view -h {vcf_path} | head -20",
            shell=True, capture_output=True, text=True
        ).stdout.strip().split("\n")
        contigs = [l for l in hdr if l.startswith("##contig")]
        formats = [l for l in hdr if l.startswith("##FORMAT")]
        print(f"Contigs: {len(contigs)}")
        print(f"FORMAT fields: {len(formats)}")
    except Exception as e:
        print(f"Header: Error - {e}")
    try:
        first = subprocess.run(
            f"bcftools view -H {vcf_path} | head -1",
            shell=True, capture_output=True, text=True
        ).stdout.strip()
        last = subprocess.run(
            f"bcftools view -H {vcf_path} | tail -1",
            shell=True, capture_output=True, text=True
        ).stdout.strip()
        if first:
            f_parts = first.split("\t")
            if len(f_parts) >= 5:
                print(f"First: {f_parts[0]}:{f_parts[1]} {f_parts[3]}>{f_parts[4]}")
        if last:
            l_parts = last.split("\t")
            if len(l_parts) >= 5:
                print(f"Last:  {l_parts[0]}:{l_parts[1]} {l_parts[3]}>{l_parts[4]}")
    except Exception as e:
        print(f"Range: Error - {e}")


def prepare_reagle_reference_from_convert_genome(output_vcf="reagle_ref.vcf.gz"):
    """Uses convert_genome's panel output as Reagle reference and normalizes it to bgzipped VCF."""
    if os.path.exists(output_vcf) and (
        os.path.exists(output_vcf + ".csi") or os.path.exists(output_vcf + ".tbi")
    ):
        print(f"Reusing Reagle reference from cache: {output_vcf}")
        return output_vcf

    panel_candidates = [
        "convert_genome_array_out/panel.vcf.gz",
        "convert_genome_array_out/panel.vcf",
    ]
    panel_src = next((p for p in panel_candidates if os.path.exists(p)), None)
    if panel_src is None:
        raise RuntimeError(
            "convert_genome panel output not found in convert_genome_array_out/"
        )

    print(f"Using convert_genome panel for Reagle reference: {panel_src}")
    run_cmd(["bcftools", "view", panel_src, "-Oz", "-o", output_vcf])
    run_cmd(["bcftools", "index", "-f", output_vcf])
    return output_vcf


def prepare_beagle_reference():
    if os.path.exists("ref_beagle.vcf.gz") and (
        os.path.exists("ref_beagle.vcf.gz.csi") or os.path.exists("ref_beagle.vcf.gz.tbi")
    ):
        print("Reusing Beagle reference from cache: ref_beagle.vcf.gz")
        return

    print("Preparing Beagle reference panel (bcftools +setGT)...")
    prep_log = "ref_beagle_prep.log"
    prep_cmd = (
        "bcftools +setGT ref.vcf.gz -Ou -- -t . -n 0 "
        "| bcftools +setGT - -Ou -- -t a -n p "
        "| bcftools view -Oz -o ref_beagle.vcf.gz"
    )
    run_cmd(f"{prep_cmd} 2> {prep_log}", shell=True)
    if os.path.exists(prep_log):
        print("Beagle ref prep logs (bcftools):")
        with open(prep_log, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    print(f"  [bcftools] {line}")
                    if line.startswith("Filled"):
                        print("  [bcftools] Note: count is per marker×sample entries in ref panel, not target alleles")
        os.remove(prep_log)
    run_cmd(["bcftools", "index", "-f", "ref_beagle.vcf.gz"])


def run_benchmark(person, file_path):
    print_tool_versions()
    # 1. Prepare Data
    print(f"=== Preparing data for {person} ({file_path}) ===")
    run_cmd(["python3", "scripts/prepare_data.py", "reference", "ref.vcf.gz"])
    run_cmd(["python3", "scripts/prepare_data.py", "array", file_path, "target.vcf.gz", "ref.vcf.gz"])

    truth_dir = "data/kat_suricata" if person == "Kat" else "data/christopher_smith"
    run_cmd(["python3", "scripts/prepare_data.py", "truth", truth_dir, "truth.vcf.gz", "ref.vcf.gz"])
    reagle_ref = prepare_reagle_reference_from_convert_genome("reagle_ref.vcf.gz")

    print_vcf_diagnostics("ref.vcf.gz", "Reference Panel (Downloaded)")
    print_vcf_diagnostics(reagle_ref, "Reference Panel (convert_genome)")
    print_vcf_diagnostics("target.vcf.gz", "Target Array")
    print_vcf_diagnostics("truth.vcf.gz", "Truth WGS")

    # 2. Run Reagle
    print("=== Running Reagle ===")
    if not os.path.exists("target.vcf.gz"):
        print("ERROR: target.vcf.gz missing after prep!")
        run_cmd(["ls", "-l"])
    
    run_cmd(["./target/release/reagle", "--ref", reagle_ref, "--target", "target.vcf.gz", "--out", "reagle_out"])

    # 3. Run Beagle
    print("=== Running Beagle ===")
    # Reference panels used for Beagle imputation must be fully phased and contain no missing data.
    # We create a derivative panel that excludes sites with missing calls and forces phased separators.
    # The original ref.vcf.gz is preserved for Reagle to ensure no loss of data in the primary pipeline.
    # NOTE: +setGT must run BEFORE filtering missing data to handle sparse panels correctly.
    # We first fill missing values with Reference (0) to ensure the panel is dense, then force phasing.
    prepare_beagle_reference()
    print_vcf_diagnostics("ref_beagle.vcf.gz", "Reference Panel (Beagle)")

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

    print_vcf_diagnostics("tests/data/reagle_imputed.vcf.gz", "Reagle Imputed")
    print_vcf_diagnostics("tests/data/beagle_imputed.vcf.gz", "Beagle Imputed")
    
    # Run metrics via integration test (it now auto-detects DS/GP vs GT-only format)
    run_cmd(["python3", "tests/integration_test.py", "metrics"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", required=True)
    parser.add_argument("--file", required=True)
    # NOTE: --format removed; convert_genome auto-detects input format.
    args = parser.parse_args()
    
    run_benchmark(args.person, args.file)
