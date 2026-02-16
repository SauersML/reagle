import os
import sys
import subprocess
import argparse
import shutil
import tempfile
import urllib.request
from urllib.error import URLError, HTTPError

BEAGLE_JAR = "beagle.jar"
BEAGLE_URL = "https://faculty.washington.edu/browning/beagle/beagle.27Feb25.75f.jar"


def run_cmd(cmd, shell=False):
    print(f"Running: {cmd}")
    if shell:
        subprocess.check_call(cmd, shell=True)
    else:
        subprocess.check_call(cmd)


def _replace_vcf_and_index(tmp_vcf, out_vcf):
    os.replace(tmp_vcf, out_vcf)
    tmp_csi = tmp_vcf + ".csi"
    tmp_tbi = tmp_vcf + ".tbi"
    out_csi = out_vcf + ".csi"
    out_tbi = out_vcf + ".tbi"
    if os.path.exists(tmp_csi):
        os.replace(tmp_csi, out_csi)
        if os.path.exists(out_tbi):
            os.remove(out_tbi)
    elif os.path.exists(tmp_tbi):
        os.replace(tmp_tbi, out_tbi)
        if os.path.exists(out_csi):
            os.remove(out_csi)
    else:
        raise RuntimeError(f"Index missing for temporary VCF: {tmp_vcf}")


def _atomic_bgzip_copy(src_vcf, out_vcf):
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=".",
        prefix=f".{os.path.basename(out_vcf)}.",
        suffix=".tmp.vcf.gz",
    ) as tmp:
        tmp_vcf = tmp.name
    try:
        run_cmd(["bcftools", "view", src_vcf, "-Oz", "-o", tmp_vcf])
        run_cmd(["bcftools", "index", "-f", tmp_vcf])
        _replace_vcf_and_index(tmp_vcf, out_vcf)
    finally:
        for suffix in ("", ".csi", ".tbi"):
            p = tmp_vcf + suffix
            if os.path.exists(p):
                os.remove(p)


def _atomic_indexed_copy(src_vcf, out_vcf):
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=".",
        prefix=f".{os.path.basename(out_vcf)}.",
        suffix=".tmp.vcf.gz",
    ) as tmp:
        tmp_vcf = tmp.name
    try:
        shutil.copy2(src_vcf, tmp_vcf)
        if os.path.exists(src_vcf + ".csi"):
            shutil.copy2(src_vcf + ".csi", tmp_vcf + ".csi")
        elif os.path.exists(src_vcf + ".tbi"):
            shutil.copy2(src_vcf + ".tbi", tmp_vcf + ".tbi")
        else:
            run_cmd(["bcftools", "index", "-f", tmp_vcf])
        _replace_vcf_and_index(tmp_vcf, out_vcf)
    finally:
        for suffix in ("", ".csi", ".tbi"):
            p = tmp_vcf + suffix
            if os.path.exists(p):
                os.remove(p)


def _atomic_sorted_bgzip_indexed(vcf_path):
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=".",
        prefix=f".{os.path.basename(vcf_path)}.",
        suffix=".sorted.tmp.vcf.gz",
    ) as tmp:
        tmp_vcf = tmp.name
    try:
        run_cmd(["bcftools", "sort", "-Oz", "-o", tmp_vcf, vcf_path])
        run_cmd(["bcftools", "index", "-f", tmp_vcf])
        _replace_vcf_and_index(tmp_vcf, vcf_path)
    finally:
        for suffix in ("", ".csi", ".tbi"):
            p = tmp_vcf + suffix
            if os.path.exists(p):
                os.remove(p)


def _vcf_record_count(vcf_path):
    if vcf_path.endswith(".vcf") and not vcf_path.endswith(".vcf.gz"):
        result = subprocess.run(
            f"bcftools view -H {vcf_path} | wc -l",
            shell=True,
            check=True,
            capture_output=True,
            text=True,
        )
        count = (result.stdout or "").strip()
        return int(count) if count else 0

    result = subprocess.run(
        ["bcftools", "index", "-n", vcf_path],
        check=True,
        capture_output=True,
        text=True,
    )
    count = (result.stdout or "").strip()
    return int(count) if count else 0


def ensure_beagle():
    """Download Beagle JAR if not present."""
    if os.path.exists(BEAGLE_JAR) and os.path.getsize(BEAGLE_JAR) > 0:
        return BEAGLE_JAR
    download_errors = []
    print(f"Downloading Beagle from {BEAGLE_URL}...")
    for attempt in range(3):
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=".",
            prefix=f".{BEAGLE_JAR}.",
            suffix=".tmp",
        ) as tmp:
            tmp_jar = tmp.name
        try:
            if shutil.which("wget"):
                run_cmd(
                    [
                        "wget",
                        "-q",
                        "--tries=1",
                        "--timeout=30",
                        BEAGLE_URL,
                        "-O",
                        tmp_jar,
                    ]
                )
            elif shutil.which("curl"):
                run_cmd(
                    [
                        "curl",
                        "-fsSL",
                        "--retry",
                        "0",
                        "--connect-timeout",
                        "30",
                        BEAGLE_URL,
                        "-o",
                        tmp_jar,
                    ]
                )
            else:
                with urllib.request.urlopen(BEAGLE_URL, timeout=30) as resp, open(
                    tmp_jar, "wb"
                ) as out:
                    out.write(resp.read())
            if os.path.exists(tmp_jar) and os.path.getsize(tmp_jar) > 0:
                os.replace(tmp_jar, BEAGLE_JAR)
                return BEAGLE_JAR
            download_errors.append(f"attempt {attempt + 1}: downloaded empty file")
        except (subprocess.CalledProcessError, URLError, HTTPError, TimeoutError) as e:
            download_errors.append(f"attempt {attempt + 1}: {e}")
        finally:
            if os.path.exists(tmp_jar):
                os.remove(tmp_jar)
    raise RuntimeError("Failed to download Beagle jar from official URL: " + " | ".join(download_errors))


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


def print_disk_diagnostics(label):
    print(f"\n=== Disk Diagnostics: {label} ===")
    for cmd in [
        "df -h .",
        "du -sh .cache convert_genome_array_out convert_genome_truth_out tests/data 2>/dev/null || true",
    ]:
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            output = (result.stdout or "").strip()
            if output:
                print(output)
        except Exception as e:
            print(f"{cmd}: Error - {e}")


def prepare_reagle_reference_from_convert_genome(output_vcf="reagle_ref.vcf.gz"):
    """
    Materialize the Reagle reference panel as a bgzipped/indexed VCF.

    Context:
    - `scripts/prepare_data.py array ...` runs convert_genome and writes
      `convert_genome_array_out/panel.vcf(.gz)`.
    - Some convert_genome runs can emit an incomplete panel output that does
      not contain the full reference panel loci.
    - A severely undersized panel breaks imputation support; this function
      validates size and fails fast if the panel output is clearly broken.

    Operational note:
    - If `output_vcf` already exists, we reuse it (cache behavior).
    - If you need to force regeneration from fresh convert_genome outputs,
      remove `output_vcf*` and `convert_genome_array_out/` first.
    """
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

    ref_src = "ref.vcf.gz"
    if not os.path.exists(ref_src):
        raise RuntimeError("Missing ref.vcf.gz required for panel sanity check")

    panel_records = _vcf_record_count(panel_src)
    ref_records = _vcf_record_count(ref_src)

    if panel_records <= 0:
        raise RuntimeError(f"convert_genome panel is empty: {panel_src}")

    min_expected = int(ref_records * 0.5) if ref_records > 0 else 1
    if ref_records > 0 and panel_records < min_expected:
        raise RuntimeError(
            "convert_genome panel output appears incomplete/broken "
            f"({panel_records} records vs ref {ref_records}); "
            "refusing to run Reagle with this panel. "
            "Rebuild/update convert_genome and regenerate convert_genome_array_out/panel.vcf(.gz)."
        )

    print(
        f"Using convert_genome panel for Reagle reference: {panel_src} "
        f"({panel_records} records; ref={ref_records})"
    )
    if panel_src.endswith(".vcf.gz"):
        _atomic_indexed_copy(panel_src, output_vcf)
    else:
        _atomic_bgzip_copy(panel_src, output_vcf)
    return output_vcf


def prepare_beagle_reference():
    if os.path.exists("ref_beagle.vcf.gz") and (
        os.path.exists("ref_beagle.vcf.gz.csi") or os.path.exists("ref_beagle.vcf.gz.tbi")
    ):
        print("Reusing Beagle reference from cache: ref_beagle.vcf.gz")
        return

    print("Preparing Beagle reference panel (bcftools +setGT)...")
    prep_log = "ref_beagle_prep.log"
    tmp_out = "ref_beagle.vcf.gz.tmp"
    tmp_log = prep_log + ".tmp"
    prep_cmd = (
        "bcftools +setGT ref.vcf.gz -Ou -- -t . -n 0 "
        "| bcftools +setGT - -Ou -- -t a -n p "
        f"| bcftools view -Oz -o {tmp_out}"
    )
    run_cmd(f"{prep_cmd} 2> {tmp_log}", shell=True)
    run_cmd(["bcftools", "index", "-f", tmp_out])
    _replace_vcf_and_index(tmp_out, "ref_beagle.vcf.gz")
    if os.path.exists(tmp_log):
        os.replace(tmp_log, prep_log)
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


def run_benchmark(person, file_path):
    print_tool_versions()
    # 1. Prepare Data
    print(f"=== Preparing data for {person} ({file_path}) ===")
    if os.path.exists("ref.vcf.gz") and (
        os.path.exists("ref.vcf.gz.csi") or os.path.exists("ref.vcf.gz.tbi")
    ):
        print("Reusing reference panel from existing artifact: ref.vcf.gz")
    else:
        run_cmd(["python3", "scripts/prepare_data.py", "reference", "ref.vcf.gz"])
    # convert_genome array conversion writes:
    #   - target.vcf.gz (final target used by benchmark)
    #   - convert_genome_array_out/genotypes.vcf(.gz)
    #   - convert_genome_array_out/panel.vcf(.gz) [full rewritten panel artifact]
    # We keep these artifacts for diagnostics and compatibility checks.
    print_disk_diagnostics("before array conversion")
    run_cmd(["python3", "scripts/prepare_data.py", "array", file_path, "target.vcf.gz", "ref.vcf.gz"])
    print_disk_diagnostics("after array conversion")

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
            _atomic_sorted_bgzip_indexed(vcf)
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
