import os
import sys
import subprocess
import glob
import shutil


def _resolve_local_panel_path():
    candidates = [
        "ref.vcf.gz",
        os.path.join("tests", "data", "ref.vcf.gz"),
    ]

    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def _clean_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name in ("panel.vcf", "panel.vcf.gz", "genotypes.vcf", "genotypes.vcf.gz"):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            os.remove(path)

def _find_genotypes_vcf(output_dir):
    for name in ("genotypes.vcf", "genotypes.vcf.gz"):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            return path
    return None

def _update_panel_if_present(output_dir, panel_path):
    panel_candidates = [
        os.path.join(output_dir, "panel.vcf"),
        os.path.join(output_dir, "panel.vcf.gz"),
    ]
    panel_source = next((p for p in panel_candidates if os.path.exists(p)), None)
    if not panel_source:
        return False

    print(f"Updated panel detected at {panel_source}, updating {panel_path}...")
    subprocess.check_call(["bcftools", "view", panel_source, "-Oz", "-o", panel_path])
    subprocess.check_call(["bcftools", "index", "-f", panel_path])
    return True

def install_convert_genome():
    """Installs convert_genome using the official install script (pre-compiled binary)."""
    print("Installing convert_genome...")
    # Only install if not present
    if shutil.which("convert_genome"):
        print("convert_genome already installed.")
        return

    # Download and run the install script
    install_script_url = "https://raw.githubusercontent.com/SauersML/convert_genome/refs/heads/main/install.sh"
    subprocess.check_call(["bash", "-c", f"curl -fsSL {install_script_url} | bash"])

    # Add install location to PATH for this session
    home = os.path.expanduser("~")
    local_bin = os.path.join(home, ".local", "bin")
    if local_bin not in os.environ["PATH"]:
        os.environ["PATH"] = local_bin + os.pathsep + os.environ["PATH"]

def prepare_input_file(input_path):
    """
    Prepares the input file for conversion.
    - If split parts exist, combines them.
    - If zip file, extracts it.
    - Returns the path to the actual raw data file.
    """
    directory = os.path.dirname(input_path)
    basename = os.path.basename(input_path)
    
    # 1. Handle Split Files
    parts = sorted(glob.glob(os.path.join(directory, f"{basename}.part*")))
    if parts:
        combined_path = input_path.replace(".part-00", "") # Heuristic
        if not combined_path.endswith(".txt") and not combined_path.endswith(".csv"):
             combined_path = os.path.join(directory, "combined_input.txt")
             
        print(f"Detected split files. Combining {len(parts)} parts to {combined_path}...")
        # Fast shell concat
        subprocess.check_call(f"cat '{input_path}'.part* > '{combined_path}'", shell=True)
        return combined_path

    # 2. Handle Zip Files
    if input_path.endswith(".zip"):
        print(f"Detected zip file: {input_path}")
        extract_dir = os.path.join(directory, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        
        print("Unzipping...")
        subprocess.check_call(["unzip", "-o", input_path, "-d", extract_dir])
        
        # Find the data file inside
        # Look for txt or csv or large files
        candidates = []
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.endswith(".txt") or f.endswith(".csv") or f.endswith(".tsv"):
                    candidates.append(os.path.join(root, f))
        
        if not candidates:
            # Maybe it's a file without extension? Take the largest one.
            all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(extract_dir) for f in filenames]
            if all_files:
                largest = max(all_files, key=os.path.getsize)
                print(f"No obvious text file found, using largest file: {largest}")
                return largest
            else:
                raise ValueError("Zip file appeared empty.")
        
        # Heuristic: pick the largest text file
        target = max(candidates, key=os.path.getsize)
        print(f"Using extracted file: {target}")
        return target

    return input_path

def download_reference(output_vcf):
    """Downloads and prepares HGDP+1kG Chr22 reference (hg38)."""
    if os.path.exists(output_vcf) and os.path.exists(output_vcf + ".csi"):
        print("Reference already exists.")
        return

    print("Downloading HGDP+1kG reference panel (Chr22, hg38)...")
    # This URL is for gnomAD HGDP+1kG v3 (hg38)
    bcf_url = "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf"
    
    # Download BCF
    subprocess.check_call(["wget", "-q", bcf_url, "-O", "ref_raw.bcf"])
    
    # Convert BCF -> VCF.gz and Index
    print("Converting BCF to VCF.gz...")
    subprocess.check_call(f"bcftools view ref_raw.bcf -Oz -o {output_vcf}", shell=True)
    subprocess.check_call(["bcftools", "index", "-f", output_vcf])
    
    # Cleanup
    if os.path.exists("ref_raw.bcf"):
        os.remove("ref_raw.bcf")
    print(f"Reference prepared: {output_vcf}")

def prepare_truth(source, output_vcf):
    """
    Reconstructs and prepares Truth VCF (Chr22) from WGS data.
    
    IMPORTANT: Truth MUST come from WGS data (vcf.gz.part* files).
    Array data cannot be used as truth because it lacks HomRef genotypes.
    
    Source can be dir or person name.
    """
    if source.lower() == "kat":
        input_dir = "data/kat_suricata"
    elif source.lower() == "christopher":
        input_dir = "data/christopher_smith"
    else:
        input_dir = source

    print(f"Preparing Truth VCF from {input_dir}...")
    
    source_vcf = "truth_full.vcf.gz"
    
    # Check for split WGS VCF parts - this is REQUIRED for truth
    parts = sorted(glob.glob(os.path.join(input_dir, "*.vcf.gz.part*")))
    if parts:
        print(f"Found {len(parts)} split VCF parts, combining...")
        # Combine parts then compress properly through bcftools
        combined_raw = "truth_combined_raw.vcf.gz"
        subprocess.check_call(f"cat {os.path.join(input_dir, '*.vcf.gz.part*')} > {combined_raw}", shell=True)
        # Re-compress through bcftools to ensure valid BGZF
        subprocess.check_call(f"bcftools view {combined_raw} -Oz -o {source_vcf}", shell=True)
        os.remove(combined_raw)
    else:
        # Check for existing VCF.gz file (already combined)
        vcf_files = glob.glob(os.path.join(input_dir, "*.vcf.gz"))
        # Exclude any imputed or array-derived files
        wgs_vcfs = [f for f in vcf_files if "imputed" not in f.lower() and "array" not in f.lower()]
        
        if wgs_vcfs:
            best = max(wgs_vcfs, key=os.path.getsize)
            print(f"Found WGS VCF: {best}")
            # Re-compress through bcftools to ensure valid BGZF
            subprocess.check_call(f"bcftools view {best} -Oz -o {source_vcf}", shell=True)
        else:
            # NO WGS data found - this is an error condition
            print(f"\nERROR: No WGS truth data found in {input_dir}")
            print("Truth VCF requires WGS data (*.vcf.gz.part* or *.vcf.gz files).")
            print("Array data (text files) cannot be used as truth because they lack HomRef genotypes.")
            print("\nExpected file format: <sample_name>.vcf.gz.part-00, part-01, etc.")
            print("See data/README.md for instructions on adding WGS data.")
            sys.exit(1)

    # Index before filtering (required for filtering regions efficiently)
    print("Indexing Truth VCF...")
    subprocess.check_call(["bcftools", "index", "-t", source_vcf])

    panel_path = _resolve_local_panel_path()
    if not panel_path:
        raise RuntimeError(
            "HGDP+1KG panel VCF not found locally (expected one of: ref.vcf.gz, tests/data/ref.vcf.gz)."
        )

    install_convert_genome()

    ref_hg38_url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
    truth_output_dir = "convert_genome_truth_out"
    _clean_output_dir(truth_output_dir)
    truth_hg38_vcf = "truth_hg38.vcf.gz"

    cmd = [
        "convert_genome",
        source_vcf,
        ref_hg38_url,
        "--output-dir", truth_output_dir,
        "--assembly", "GRCh38",
        "--format", "vcf",
        "--standardize",
        "--panel", panel_path,
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    truth_raw_vcf = _find_genotypes_vcf(truth_output_dir)
    if not truth_raw_vcf:
        raise RuntimeError("convert_genome failed to produce genotypes.vcf")

    subprocess.check_call(["bcftools", "view", truth_raw_vcf, "-Oz", "-o", truth_hg38_vcf])
    subprocess.check_call(["bcftools", "index", "-f", truth_hg38_vcf])
    _update_panel_if_present(truth_output_dir, panel_path)

    # Rename chroms (22 -> chr22) to match reference panel which uses chr22 notation
    with open("chr_map.txt", "w") as f:
        f.write("22\tchr22\n")

    print("Filtering Truth to Chr22...")
    # Filter FIRST using index (regions 22 or chr22), then rename to chr22
    cmd = (
        f"bcftools view {truth_hg38_vcf} --regions 22,chr22 -Ou | "
        f"bcftools annotate --rename-chrs chr_map.txt -Oz -o {output_vcf}"
    )
    subprocess.check_call(cmd, shell=True)
    subprocess.check_call(["tabix", "-p", "vcf", output_vcf])
    
    if os.path.exists(source_vcf):
        os.remove(source_vcf)
    if os.path.exists(truth_hg38_vcf):
        os.remove(truth_hg38_vcf)
    if os.path.exists(truth_hg38_vcf + ".csi"):
        os.remove(truth_hg38_vcf + ".csi")
    if os.path.exists("chr_map.txt"):
        os.remove("chr_map.txt")
    print(f"Truth prepared: {output_vcf}")

def run_conversion(input_path, output_vcf):
    """Runs convert_genome to convert input to VCF (hg19) then lifts to hg38."""
    
    # Pre-process input (handle zip/split)
    raw_file = prepare_input_file(input_path)
    
    print(f"Converting {raw_file} to GRCh38 VCF...")

    ref_hg38_url = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"
    temp_output_dir = "convert_genome_array_out"
    _clean_output_dir(temp_output_dir)

    panel_path = _resolve_local_panel_path()
    if not panel_path:
        raise RuntimeError(
            "HGDP+1KG panel VCF not found locally (expected one of: ref.vcf.gz, tests/data/ref.vcf.gz)."
        )

    cmd = [
        "convert_genome",
        raw_file,
        ref_hg38_url,
        "--output-dir", temp_output_dir,
        "--assembly", "GRCh38",
        "--format", "vcf",
        "--standardize",
        "--panel", panel_path,
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

    temp_hg38_vcf = _find_genotypes_vcf(temp_output_dir)
    if not temp_hg38_vcf:
        raise RuntimeError("convert_genome failed to produce genotypes.vcf")

    print("Finalizing GRCh38 output...")
    print("Filtering invalid records (missing ALT but non-ref GT)...")
    filter_cmd = f"bcftools view {temp_hg38_vcf} -e 'ALT=\".\" && GT[*]=\"alt\"' -Oz -o {output_vcf}"
    subprocess.check_call(filter_cmd, shell=True)
    subprocess.check_call(["bcftools", "index", "-f", output_vcf])
    print("Conversion complete.")
    _update_panel_if_present(temp_output_dir, panel_path)

    # Cleanup extracted if it was temp
    if "extracted" in raw_file:
        shutil.rmtree(os.path.dirname(raw_file))
        
    print("Conversion preparation complete.")

if __name__ == "__main__":
    print("Prepare Data Script v1.2 (Clean Overwrite)")
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 prepare_data.py array <input_file> <output_vcf>")
        print("  python3 prepare_data.py reference <output_vcf>")
        print("  python3 prepare_data.py truth <person_or_dir> <output_vcf>")
        sys.exit(1)
        
    mode = sys.argv[1]
    
    if mode == "array":
        install_convert_genome()
        run_conversion(sys.argv[2], sys.argv[3])
        # Post-process array to Chr22 (redundant if run_conversion does it, but kept for safety if run_conversion changed)
        # Actually run_conversion now does everything including filtering to 22.
        # So we just verify.
        if not os.path.exists(sys.argv[3]):
             print(f"Error: {sys.argv[3]} was not created.")
             sys.exit(1)
        
    elif mode == "reference":
        download_reference(sys.argv[2])
        
    elif mode == "truth":
        prepare_truth(sys.argv[2], sys.argv[3])
    
    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
