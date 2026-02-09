import os
import sys
import subprocess
import glob
import shutil
import shlex
import gzip
from pathlib import Path

PANEL_BCF_URL = "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf"
CHR22_FASTA_GZ_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr22.fa.gz"


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


def _download_file(url, dest):
    if shutil.which("wget"):
        subprocess.check_call(["wget", "-q", url, "-O", str(dest)])
        return
    if shutil.which("curl"):
        subprocess.check_call(["curl", "-fsSL", url, "-o", str(dest)])
        return
    raise RuntimeError("Neither wget nor curl found; cannot download reference panel.")


def _ensure_chr22_reference_fasta(local_gz: str = ".cache/reference/chr22.fa.gz",
                                  local_fa: str = ".cache/reference/chr22.fa") -> str:
    gz_path = Path(local_gz)
    fa_path = Path(local_fa)

    if fa_path.exists() and fa_path.stat().st_size > 0:
        print(f"Reusing cached chr22 FASTA: {fa_path}")
        return str(fa_path)

    if not gz_path.exists() or gz_path.stat().st_size == 0:
        gz_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading chr22 reference FASTA to {gz_path}...")
        _download_file(CHR22_FASTA_GZ_URL, gz_path)

    print(f"Decompressing chr22 FASTA to {fa_path}...")
    with gzip.open(gz_path, "rb") as src, open(fa_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return str(fa_path)


def _has_vcf_index(vcf_path: Path):
    return (vcf_path.with_suffix(vcf_path.suffix + ".csi")).exists() or (
        vcf_path.with_suffix(vcf_path.suffix + ".tbi")
    ).exists()


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


def _clear_convert_genome_cache():
    """Removes any existing convert_genome binary and caches to force a fresh install."""
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
    existing = shutil.which("convert_genome")
    if existing:
        try:
            subprocess.check_call(
                ["convert_genome", "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"convert_genome already available: {existing}")
            return
        except Exception:
            print("Existing convert_genome is not usable; reinstalling...")
            _clear_convert_genome_cache()
    else:
        print("convert_genome not found; installing...")

    install_script_url = "https://raw.githubusercontent.com/SauersML/convert_genome/main/install.sh"
    subprocess.check_call(["bash", "-c", f"curl -fsSL {install_script_url} | bash"])

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

    parts = sorted(glob.glob(os.path.join(directory, f"{basename}.part*")))
    if parts:
        combined_path = input_path.replace(".part-00", "")
        if not combined_path.endswith(".txt") and not combined_path.endswith(".csv"):
             combined_path = os.path.join(directory, "combined_input.txt")

        print(f"Detected split files. Combining {len(parts)} parts to {combined_path}...")
        subprocess.check_call(f"cat '{input_path}'.part* > '{combined_path}'", shell=True)
        return combined_path

    if input_path.endswith(".zip"):
        print(f"Detected zip file: {input_path}")
        extract_dir = os.path.join(directory, "extracted")
        os.makedirs(extract_dir, exist_ok=True)

        print("Unzipping...")
        subprocess.check_call(["unzip", "-o", input_path, "-d", extract_dir])

        candidates = []
        for root, _, files in os.walk(extract_dir):
            for f in files:
                if f.endswith(".txt") or f.endswith(".csv") or f.endswith(".tsv"):
                    candidates.append(os.path.join(root, f))

        if not candidates:
            all_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(extract_dir) for f in filenames]
            if all_files:
                largest = max(all_files, key=os.path.getsize)
                print(f"No obvious text file found, using largest file: {largest}")
                return largest
            else:
                raise ValueError("Zip file appeared empty.")

        target = max(candidates, key=os.path.getsize)
        print(f"Using extracted file: {target}")
        return target

    return input_path


def download_reference(output_vcf):
    """Downloads HGDP+1kG Chr22 reference panel to the specified path."""
    output_path = Path(output_vcf)

    if output_path.exists() and _has_vcf_index(output_path):
        print(f"Reference already exists: {output_vcf}")
        return

    if shutil.which("bcftools") is None:
        raise RuntimeError("bcftools not found on PATH (required to prepare HGDP+1KG panel).")

    print(f"Downloading HGDP+1KG panel to {output_vcf}...")
    raw_bcf = output_path.with_suffix(".bcf")

    _download_file(PANEL_BCF_URL, raw_bcf)

    print("Converting BCF to VCF.gz...")
    subprocess.check_call(["bcftools", "view", str(raw_bcf), "-Oz", "-o", str(output_path)])
    subprocess.check_call(["bcftools", "index", "-f", str(output_path)])

    if raw_bcf.exists():
        raw_bcf.unlink()

    print(f"Reference prepared: {output_vcf}")


def prepare_truth(source, output_vcf, panel_path):
    """
    Reconstructs and prepares Truth VCF (Chr22) from WGS data.

    IMPORTANT: Truth MUST come from WGS data (vcf.gz.part* files).
    Array data cannot be used as truth because it lacks HomRef genotypes.
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

    parts = sorted(glob.glob(os.path.join(input_dir, "*.vcf.gz.part*")))
    if parts:
        print(f"Found {len(parts)} split VCF parts, combining...")
        combined_raw = "truth_combined_raw.vcf.gz"
        subprocess.check_call(f"cat {os.path.join(input_dir, '*.vcf.gz.part*')} > {combined_raw}", shell=True)
        subprocess.check_call(f"bcftools view {combined_raw} -Oz -o {source_vcf}", shell=True)
        os.remove(combined_raw)
    else:
        vcf_files = glob.glob(os.path.join(input_dir, "*.vcf.gz"))
        wgs_vcfs = [f for f in vcf_files if "imputed" not in f.lower() and "array" not in f.lower()]

        if wgs_vcfs:
            best = max(wgs_vcfs, key=os.path.getsize)
            print(f"Found WGS VCF: {best}")
            subprocess.check_call(f"bcftools view {best} -Oz -o {source_vcf}", shell=True)
        else:
            print(f"\nERROR: No WGS truth data found in {input_dir}")
            print("Truth VCF requires WGS data (*.vcf.gz.part* or *.vcf.gz files).")
            print("Array data (text files) cannot be used as truth because they lack HomRef genotypes.")
            sys.exit(1)

    print("Indexing Truth VCF...")
    subprocess.check_call(["bcftools", "index", "-t", source_vcf])

    _bump_nofile_limit()
    install_convert_genome()
    _bump_nofile_limit()

    ref_hg38_fasta = _ensure_chr22_reference_fasta()
    truth_output_dir = "convert_genome_truth_out"
    _clean_output_dir(truth_output_dir)
    truth_hg38_vcf = "truth_hg38.vcf.gz"

    cmd = [
        "convert_genome",
        source_vcf,
        ref_hg38_fasta,
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

    with open("chr_map.txt", "w") as f:
        f.write("22\tchr22\n")

    print("Filtering Truth to Chr22...")
    cmd = (
        f"bcftools view {truth_hg38_vcf} --regions 22,chr22 -Ou | "
        f"bcftools annotate --rename-chrs chr_map.txt -Oz -o {output_vcf}"
    )
    subprocess.check_call(cmd, shell=True)
    subprocess.check_call(["tabix", "-p", "vcf", output_vcf])

    def _truth_header_ok(vcf_path):
        try:
            header = subprocess.check_output(
                ["bcftools", "view", "-h", vcf_path],
                text=True
            )
        except Exception:
            return False
        has_gt = "##FORMAT=<ID=GT" in header
        has_contig = "##contig=<ID=" in header
        return has_gt and has_contig

    if not _truth_header_ok(output_vcf):
        print("Truth header missing FORMAT/contig lines; rebuilding header...")
        header_txt = "truth_header.txt"
        subprocess.check_call(f"bcftools view -h {truth_hg38_vcf} > {header_txt}", shell=True)
        fixed = output_vcf + ".tmp"
        subprocess.check_call(["bcftools", "reheader", "-h", header_txt, output_vcf, "-o", fixed])
        os.replace(fixed, output_vcf)
        subprocess.check_call(["tabix", "-f", "-p", "vcf", output_vcf])
        if os.path.exists(header_txt):
            os.remove(header_txt)

    for f in [source_vcf, truth_hg38_vcf, truth_hg38_vcf + ".csi", "chr_map.txt"]:
        if os.path.exists(f):
            os.remove(f)
    print(f"Truth prepared: {output_vcf}")


def run_conversion(input_path, output_vcf, panel_path):
    """Runs convert_genome to convert input to hg38 VCF."""
    output_path = Path(output_vcf)
    if output_path.exists() and _has_vcf_index(output_path) and output_path.stat().st_size > 0:
        print(f"Array conversion already exists: {output_vcf}")
        return

    raw_file = prepare_input_file(input_path)

    print(f"Converting {raw_file} to GRCh38 VCF...")

    ref_hg38_fasta = _ensure_chr22_reference_fasta()
    temp_output_dir = "convert_genome_array_out"
    _clean_output_dir(temp_output_dir)

    cmd = [
        "convert_genome",
        raw_file,
        ref_hg38_fasta,
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
    print("Filtering invalid records (missing ALT but non-ref GT)...")
    filter_cmd = f"bcftools view {temp_hg38_vcf} -e 'ALT=\".\" && GT[*]=\"alt\"' -Oz -o {output_vcf}"
    subprocess.check_call(filter_cmd, shell=True)
    subprocess.check_call(["bcftools", "index", "-f", output_vcf])
    print("Conversion complete.")

    if "extracted" in raw_file:
        shutil.rmtree(os.path.dirname(raw_file))

    print("Conversion preparation complete.")


if __name__ == "__main__":
    print("Prepare Data Script v2.0")
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 prepare_data.py reference <output_vcf>")
        print("  python3 prepare_data.py array <input_file> <output_vcf> <panel_vcf>")
        print("  python3 prepare_data.py truth <person_or_dir> <output_vcf> <panel_vcf>")
        sys.exit(1)

    mode = sys.argv[1]

    if mode == "reference":
        download_reference(sys.argv[2])

    elif mode == "array":
        if len(sys.argv) < 5:
            print("Usage: python3 prepare_data.py array <input_file> <output_vcf> <panel_vcf>")
            sys.exit(1)
        install_convert_genome()
        run_conversion(sys.argv[2], sys.argv[3], sys.argv[4])
        if not os.path.exists(sys.argv[3]):
             print(f"Error: {sys.argv[3]} was not created.")
             sys.exit(1)

    elif mode == "truth":
        if len(sys.argv) < 5:
            print("Usage: python3 prepare_data.py truth <person_or_dir> <output_vcf> <panel_vcf>")
            sys.exit(1)
        prepare_truth(sys.argv[2], sys.argv[3], sys.argv[4])

    else:
        print(f"Unknown mode: {mode}")
        sys.exit(1)
