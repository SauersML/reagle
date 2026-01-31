import os
import sys
import subprocess
import glob
import shutil
import shlex
from pathlib import Path

PANEL_BCF_URL = "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf"


def _repo_root():
    return Path(__file__).resolve().parent.parent


def _bump_nofile_limit(min_soft: int = 4096):
    try:
        import resource
    except Exception:
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:
        return
    target = max(soft, min_soft)
    if hard != resource.RLIM_INFINITY:
        target = min(target, hard)
    if target > soft:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        except Exception:
            pass

def _panel_cache_vcf():
    return _repo_root() / "tests" / "data" / "ref.vcf.gz"


def _download_file(url, dest):
    if shutil.which("wget"):
        subprocess.check_call(["wget", "-q", url, "-O", str(dest)])
        return
    if shutil.which("curl"):
        subprocess.check_call(["curl", "-fsSL", url, "-o", str(dest)])
        return
    raise RuntimeError("Neither wget nor curl found; cannot download reference panel.")


def _copy_or_link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _has_vcf_index(vcf_path: Path):
    return (vcf_path.with_suffix(vcf_path.suffix + ".csi")).exists() or (
        vcf_path.with_suffix(vcf_path.suffix + ".tbi")
    ).exists()


def _resolve_local_panel_path():
    candidates = []
    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / "ref.vcf.gz",
            cwd / "tests" / "data" / "ref.vcf.gz",
            cwd / "tests" / "fixtures" / "gnomad_hgdp" / "ref.vcf.gz",
            cwd / "microarray_profile" / "ref.vcf.gz",
        ]
    )

    root = _repo_root()
    if root != cwd:
        candidates.extend(
            [
                root / "ref.vcf.gz",
                root / "tests" / "data" / "ref.vcf.gz",
                root / "tests" / "fixtures" / "gnomad_hgdp" / "ref.vcf.gz",
                root / "microarray_profile" / "ref.vcf.gz",
            ]
        )

    candidates.append(_panel_cache_vcf())

    for path in candidates:
        if path and path.exists():
            return str(path)
    return None


def _ensure_reference_panel():
    panel_path = _resolve_local_panel_path()
    if panel_path:
        return panel_path

    if shutil.which("bcftools") is None:
        raise RuntimeError("bcftools not found on PATH (required to prepare HGDP+1KG panel).")

    cache_vcf = _panel_cache_vcf()
    cache_vcf.parent.mkdir(parents=True, exist_ok=True)
    if cache_vcf.exists():
        if _has_vcf_index(cache_vcf):
            return str(cache_vcf)
        subprocess.check_call(["bcftools", "index", "-f", str(cache_vcf)])
        return str(cache_vcf)

    print("HGDP+1KG panel not found locally; downloading to tests/data/ref.vcf.gz...")
    raw_bcf = cache_vcf.parent / "hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf"
    if not raw_bcf.exists():
        _download_file(PANEL_BCF_URL, raw_bcf)

    print("Converting cached BCF to VCF.gz...")
    subprocess.check_call(["bcftools", "view", str(raw_bcf), "-Oz", "-o", str(cache_vcf)])
    subprocess.check_call(["bcftools", "index", "-f", str(cache_vcf)])

    if raw_bcf.exists():
        raw_bcf.unlink()

    return str(cache_vcf)

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

def _get_vcf_chrom_name(vcf_path):
    """Returns the first chromosome name from a VCF/BCF file."""
    try:
        # Check first record's chromosome
        result = subprocess.run(
            ["bcftools", "query", "-f", "%CHROM\\n", vcf_path],
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')[0]

        # Fallback: check header (contig lines) if empty or query failed
        result = subprocess.run(
            ["bcftools", "view", "-h", vcf_path],
            capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if line.startswith("##contig=<ID="):
                # Parse ID from ##contig=<ID=chr22,length=...>
                start = line.find("ID=") + 3
                end = line.find(",", start)
                if end == -1: end = line.find(">", start)
                if start > 2 and end > start:
                    return line[start:end]
        return None
    except Exception as e:
        print(f"Warning: Failed to detect chromosome name from {vcf_path}: {e}")
        return None

def _clear_convert_genome_cache():
    """Removes any existing convert_genome binary and caches to force a fresh install."""
    # Remove known binary locations if present.
    binary_candidates = [
        shutil.which("convert_genome"),
        os.path.join(os.path.expanduser("~"), ".local", "bin", "convert_genome"),
        os.path.join(os.path.expanduser("~"), "bin", "convert_genome"),
        os.path.join(os.path.expanduser("~"), "bin", "convert_genome.exe"),
    ]
    for path in {p for p in binary_candidates if p}:
        if os.path.exists(path):
            print(f"Removing existing convert_genome binary: {path}")
            os.remove(path)

    # Remove common cache locations (best-effort).
    cache_dirs = [
        os.path.join(os.path.expanduser("~"), ".cache", "convert_genome"),
        os.path.join(os.path.expanduser("~"), "Library", "Caches", "convert_genome"),
        os.path.join(os.path.expanduser("~"), "Library", "Application Support", "convert_genome"),
    ]
    for path in cache_dirs:
        if os.path.isdir(path):
            print(f"Removing convert_genome cache: {path}")
            shutil.rmtree(path)

def install_convert_genome():
    """Installs convert_genome using the official install script (pre-compiled binary)."""
    print("Installing convert_genome (fresh install)...")
    _clear_convert_genome_cache()

    # Download and run the install script
    install_script_url = "https://raw.githubusercontent.com/SauersML/convert_genome/main/install.sh"
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
    if os.path.exists(output_vcf) and _has_vcf_index(Path(output_vcf)):
        print("Reference already exists.")
        return

    cached_panel = Path(_ensure_reference_panel())
    output_path = Path(output_vcf)
    _copy_or_link(cached_panel, output_path)

    if _has_vcf_index(cached_panel):
        for suffix in (".csi", ".tbi"):
            candidate = cached_panel.with_suffix(cached_panel.suffix + suffix)
            if candidate.exists():
                _copy_or_link(candidate, output_path.with_suffix(output_path.suffix + suffix))
    if not _has_vcf_index(output_path):
        subprocess.check_call(["bcftools", "index", "-f", str(output_path)])

    print(f"Reference prepared: {output_vcf}")

def prepare_truth(source, output_vcf):
    """
    Reconstructs and prepares Truth VCF (Chr22) from WGS data.
    
    IMPORTANT: Truth MUST come from WGS data (vcf.gz.part* files).
    Array data cannot be used as truth because it lacks HomRef genotypes.
    
    Source can be dir or person name.
    """
    if os.path.exists(output_vcf) and _has_vcf_index(Path(output_vcf)) and os.path.getsize(output_vcf) > 0:
        print(f"Truth already exists: {output_vcf}")
        return
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

    panel_path = _ensure_reference_panel()
    _bump_nofile_limit()

    install_convert_genome()
    _bump_nofile_limit()

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
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    subprocess.check_call(["bash", "-lc", f"ulimit -n 4096; {cmd_str}"])

    truth_raw_vcf = _find_genotypes_vcf(truth_output_dir)
    if not truth_raw_vcf:
        raise RuntimeError("convert_genome failed to produce genotypes.vcf")

    subprocess.check_call(["bcftools", "view", truth_raw_vcf, "-Oz", "-o", truth_hg38_vcf])
    subprocess.check_call(["bcftools", "index", "-f", truth_hg38_vcf])
    _update_panel_if_present(truth_output_dir, panel_path)

    # Detect reference panel chromosome style
    ref_chrom = _get_vcf_chrom_name(panel_path) or "chr22"
    print(f"Detected reference panel chromosome: {ref_chrom}")

    # Rename chroms to match reference panel
    # We map BOTH '22' and 'chr22' to the target style to be safe
    with open("chr_map.txt", "w") as f:
        f.write(f"22\t{ref_chrom}\n")
        f.write(f"chr22\t{ref_chrom}\n")

    print(f"Filtering Truth to {ref_chrom}...")
    # Filter FIRST using index (regions 22 or chr22), then rename to match ref
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

    panel_path = _ensure_reference_panel()

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
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    subprocess.check_call(["bash", "-lc", f"ulimit -n 4096; {cmd_str}"])

    temp_hg38_vcf = _find_genotypes_vcf(temp_output_dir)
    if not temp_hg38_vcf:
        raise RuntimeError("convert_genome failed to produce genotypes.vcf")

    print("Finalizing GRCh38 output...")
    print("Filtering invalid records (missing ALT but non-ref GT) and normalizing chromosomes...")

    # Detect reference panel chromosome style
    ref_chrom = _get_vcf_chrom_name(panel_path) or "chr22"
    print(f"Detected reference panel chromosome: {ref_chrom}")

    # Ensure chromosome naming convention matches reference for Beagle compatibility
    chr_map_path = "chr_map.txt"
    with open(chr_map_path, "w") as f:
        f.write(f"22\t{ref_chrom}\n")
        f.write(f"chr22\t{ref_chrom}\n")

    filter_cmd = (
        f"bcftools view {temp_hg38_vcf} -e 'ALT=\".\" && GT[*]=\"alt\"' -Ou | "
        f"bcftools annotate --rename-chrs {chr_map_path} -Oz -o {output_vcf}"
    )
    subprocess.check_call(filter_cmd, shell=True)
    subprocess.check_call(["bcftools", "index", "-f", output_vcf])

    if os.path.exists(chr_map_path):
        os.remove(chr_map_path)

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
