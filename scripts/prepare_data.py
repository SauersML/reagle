import os
import sys
import subprocess
import glob
import shutil
import shlex
import gzip
import tempfile
import json
import tarfile
import zipfile
import platform
import stat
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
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


def _replace_vcf_and_index(tmp_vcf: str, out_vcf: str):
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


def _atomic_bgzip_view(input_vcf: str, output_vcf: str, extra_args=None):
    args = list(extra_args or [])
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=".",
        prefix=f".{Path(output_vcf).name}.",
        suffix=".tmp.vcf.gz",
    ) as tmp:
        tmp_vcf = tmp.name
    try:
        subprocess.check_call(["bcftools", "view", input_vcf, *args, "-Oz", "-o", tmp_vcf])
        subprocess.check_call(["bcftools", "index", "-f", tmp_vcf])
        _replace_vcf_and_index(tmp_vcf, output_vcf)
    finally:
        for suffix in ("", ".csi", ".tbi"):
            p = tmp_vcf + suffix
            if os.path.exists(p):
                os.remove(p)


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
    fa_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(fa_path.parent),
        prefix=f".{fa_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        with gzip.open(gz_path, "rb") as src:
            shutil.copyfileobj(src, tmp)
    os.replace(tmp_path, fa_path)
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


def _compress_panel_if_needed(output_dir):
    panel_vcf = os.path.join(output_dir, "panel.vcf")
    panel_vcfgz = os.path.join(output_dir, "panel.vcf.gz")
    if not os.path.exists(panel_vcf):
        return
    print(f"Compressing large panel artifact: {panel_vcf} -> {panel_vcfgz}")
    subprocess.check_call(["bcftools", "view", panel_vcf, "-Oz", "-o", panel_vcfgz])
    subprocess.check_call(["bcftools", "index", "-f", panel_vcfgz])
    os.remove(panel_vcf)


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


def _github_json(url: str, timeout: int = 30):
    req = Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "reagle-prepare-data",
        },
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _platform_asset_tokens():
    sysname = platform.system().lower()
    machine = platform.machine().lower()
    os_tokens = []
    arch_tokens = []

    if sysname.startswith("linux"):
        os_tokens.extend(["linux"])
    elif sysname.startswith("darwin"):
        os_tokens.extend(["darwin", "mac", "macos", "osx"])
    elif sysname.startswith("windows"):
        os_tokens.extend(["windows", "win"])
    else:
        os_tokens.extend([sysname])

    if machine in ("x86_64", "amd64"):
        arch_tokens.extend(["x86_64", "amd64", "x64"])
    elif machine in ("aarch64", "arm64"):
        arch_tokens.extend(["aarch64", "arm64"])
    else:
        arch_tokens.extend([machine])

    return os_tokens, arch_tokens


def _select_convert_genome_asset(assets):
    os_tokens, arch_tokens = _platform_asset_tokens()
    scored = []
    for asset in assets:
        name = asset.get("name", "").lower()
        if not name:
            continue
        if not any(tok in name for tok in os_tokens):
            continue
        if not any(tok in name for tok in arch_tokens):
            continue
        if not (name.endswith(".tar.gz") or name.endswith(".tgz") or name.endswith(".zip")):
            continue
        score = 0
        if "convert_genome" in name:
            score += 4
        if name.endswith(".tar.gz") or name.endswith(".tgz"):
            score += 2
        if "musl" in name:
            score += 1
        scored.append((score, asset))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _download_to(url: str, dest: Path):
    req = Request(
        url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": "reagle-prepare-data",
        },
    )
    with urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def _find_convert_genome_in_tree(root: Path):
    candidates = []
    is_windows = platform.system().lower().startswith("windows")
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        base = path.name.lower()
        normalized = base[:-4] if base.endswith(".exe") else base
        if not (
            normalized == "convert_genome"
            or normalized.startswith("convert_genome-")
            or normalized.startswith("convert_genome_")
            or normalized.startswith("convert-genome-")
            or normalized.startswith("convert-genome_")
        ):
            continue
        # Prefer executable-looking files. On Windows, allow .exe-only naming.
        if is_windows:
            if base.endswith(".exe") or normalized.startswith("convert_genome"):
                candidates.append(path)
        else:
            mode = path.stat().st_mode
            if mode & stat.S_IXUSR:
                candidates.append(path)
            else:
                candidates.append(path)
    if candidates:
        def _rank(p: Path):
            base = p.name.lower()
            normalized = base[:-4] if base.endswith(".exe") else base
            exact = 0 if normalized == "convert_genome" else 1
            return (exact, len(p.parts), len(base))

        candidates.sort(key=_rank)
        return candidates[0]
    return None


def _install_convert_genome_from_release():
    release = _github_json("https://api.github.com/repos/SauersML/convert_genome/releases/latest")
    assets = release.get("assets", [])
    if not assets:
        raise RuntimeError("No release assets found for convert_genome.")
    asset = _select_convert_genome_asset(assets)
    if asset is None:
        names = ", ".join(a.get("name", "") for a in assets)
        raise RuntimeError(f"No matching convert_genome asset for this platform. Assets: {names}")

    asset_name = asset.get("name", "")
    asset_url = asset.get("browser_download_url", "")
    if not asset_name or not asset_url:
        raise RuntimeError("Selected convert_genome asset is missing name or download URL.")

    home = Path.home()
    install_dir = home / ".local" / "bin"
    install_dir.mkdir(parents=True, exist_ok=True)
    target_bin = install_dir / ("convert_genome.exe" if platform.system().lower().startswith("windows") else "convert_genome")

    with tempfile.TemporaryDirectory(prefix="convert_genome_install_") as td:
        tmpdir = Path(td)
        archive_path = tmpdir / asset_name
        print(f"Downloading convert_genome release asset: {asset_name}")
        _download_to(asset_url, archive_path)
        if archive_path.stat().st_size == 0:
            raise RuntimeError(f"Downloaded empty archive from {asset_url}")

        unpack_dir = tmpdir / "unpack"
        unpack_dir.mkdir(parents=True, exist_ok=True)

        lower_name = asset_name.lower()
        if lower_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(unpack_dir)
        else:
            if not tarfile.is_tarfile(archive_path):
                raise RuntimeError(
                    f"Downloaded asset is not a valid tar archive: {archive_path}"
                )
            with tarfile.open(archive_path, "r:*") as tf:
                tf.extractall(unpack_dir)

        extracted_bin = _find_convert_genome_in_tree(unpack_dir)
        if extracted_bin is None:
            raise RuntimeError(
                f"Could not find convert_genome binary after extracting {asset_name}"
            )

        shutil.copy2(extracted_bin, target_bin)
        mode = target_bin.stat().st_mode
        target_bin.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    os.environ["PATH"] = str(install_dir) + os.pathsep + os.environ["PATH"]
    return str(target_bin)


def install_convert_genome():
    """Installs convert_genome from GitHub release assets (no shell installer piping)."""
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

    try:
        installed = _install_convert_genome_from_release()
    except (URLError, HTTPError, OSError, RuntimeError, zipfile.BadZipFile, tarfile.TarError) as e:
        raise RuntimeError(f"Failed installing convert_genome from GitHub release assets: {e}") from e

    if shutil.which("convert_genome") is None:
        home = os.path.expanduser("~")
        local_bin = os.path.join(home, ".local", "bin")
        if local_bin not in os.environ["PATH"]:
            os.environ["PATH"] = local_bin + os.pathsep + os.environ["PATH"]

    subprocess.check_call(
        ["convert_genome", "--help"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"convert_genome installed: {installed}")


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
    _atomic_bgzip_view(str(raw_bcf), str(output_path))

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
    # Truth conversion only needs harmonized sample genotypes; it does not use
    # the rewritten panel artifact. Use direct-output mode to avoid materializing
    # convert_genome_truth_out/panel.vcf, which can consume tens of GB.
    truth_raw_vcf = "truth_hg38_raw.vcf"
    truth_hg38_vcf = "truth_hg38.vcf.gz"

    cmd = [
        "convert_genome",
        source_vcf,
        ref_hg38_fasta,
        truth_raw_vcf,
        "--assembly", "GRCh38",
        "--format", "vcf",
        "--standardize",
        "--panel", panel_path,
    ]

    print(f"Running: {' '.join(cmd)}")
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    subprocess.check_call(["bash", "-lc", f"ulimit -n 4096; {cmd_str}"])

    if not os.path.exists(truth_raw_vcf):
        raise RuntimeError("convert_genome failed to produce truth_hg38_raw.vcf")

    subprocess.check_call(["bcftools", "view", truth_raw_vcf, "-Oz", "-o", truth_hg38_vcf])
    subprocess.check_call(["bcftools", "index", "-f", truth_hg38_vcf])

    with open("chr_map.txt", "w") as f:
        f.write("22\tchr22\n")

    print("Filtering Truth to Chr22...")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=".",
        prefix=f".{Path(output_vcf).name}.",
        suffix=".tmp.vcf.gz",
    ) as tmp:
        tmp_truth = tmp.name
    try:
        cmd = (
            f"bcftools view {truth_hg38_vcf} --regions 22,chr22 -Ou | "
            f"bcftools annotate --rename-chrs chr_map.txt -Oz -o {tmp_truth}"
        )
        subprocess.check_call(cmd, shell=True)
        subprocess.check_call(["tabix", "-f", "-p", "vcf", tmp_truth])
        _replace_vcf_and_index(tmp_truth, output_vcf)
    finally:
        for suffix in ("", ".csi", ".tbi"):
            p = tmp_truth + suffix
            if os.path.exists(p):
                os.remove(p)

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

    for f in [source_vcf, truth_raw_vcf, truth_hg38_vcf, truth_hg38_vcf + ".csi", "chr_map.txt"]:
        if os.path.exists(f):
            os.remove(f)
    print(f"Truth prepared: {output_vcf}")


def run_conversion(input_path, output_vcf, panel_path):
    """
    Runs convert_genome on a microarray/raw consumer file and produces:

    1) output_vcf:
       The harmonized sample genotype VCF used as downstream target input.
       In this pipeline this is typically target.vcf.gz.

    2) convert_genome_array_out/genotypes.vcf(.gz):
       The direct converted per-sample genotypes emitted by convert_genome.
       We use this as the source for output_vcf after a final validity filter.

    3) convert_genome_array_out/panel.vcf(.gz):
       A rewritten panel produced by convert_genome to keep allele/site
       compatibility with genotypes.vcf.

       IMPORTANT:
       - This is expected to be a FULL panel rewrite (original panel streamed
         and rewritten with any required compatibility edits), not a tiny
         QA-only delta by design.
       - If this file appears suspiciously small, the likely causes are stale
         cache/artifacts or an incomplete prior run, not intended semantics.
       - We keep the file in convert_genome_array_out for diagnostics and for
         optional downstream use by other tools.
    """
    output_path = Path(output_vcf)
    if output_path.exists() and _has_vcf_index(output_path) and output_path.stat().st_size > 0:
        # Early return keeps conversion deterministic across repeated workflow
        # invocations, but also means convert_genome_array_out artifacts from a
        # previous run are reused as-is. When debugging panel anomalies, clear:
        #   - convert_genome_array_out/
        #   - output_vcf (+ index)
        # before rerunning.
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
        "--output-dir", temp_output_dir,
        "--assembly", "GRCh38",
        "--format", "vcf",
        "--standardize",
        "--panel", panel_path,
        "--ref", ref_hg38_fasta,
    ]

    print(f"Running: {' '.join(cmd)}")
    cmd_str = " ".join(shlex.quote(part) for part in cmd)
    subprocess.check_call(["bash", "-lc", f"ulimit -n 4096; {cmd_str}"])
    # After this command succeeds, convert_genome has written its outputs under
    # convert_genome_array_out/. We deliberately do not delete panel.vcf here
    # so quality assessment scripts can inspect or reuse it.
    _compress_panel_if_needed(temp_output_dir)

    temp_hg38_vcf = _find_genotypes_vcf(temp_output_dir)
    if not temp_hg38_vcf:
        raise RuntimeError("convert_genome failed to produce genotypes.vcf")

    print("Finalizing GRCh38 output...")
    # Safety filter for edge cases where ALT='.' but genotype encodes non-ref.
    # This protects downstream tools from invalid VCF records while keeping
    # the converted target focused on valid sites.
    print("Filtering invalid records (missing ALT but non-ref GT)...")
    _atomic_bgzip_view(temp_hg38_vcf, output_vcf, extra_args=["-e", 'ALT="." && GT[*]="alt"'])
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
