#!/usr/bin/env python3
"""
Integration test for Reagle (Beagle-inspired Rust implementation).

This test:
1. Downloads HGDP+1kG chr22 reference panel from gnomAD
2. Splits into reference (80%) and target (20%) panels
3. Downsamples target to GSAv3 array sites
4. Runs both Java Beagle and Reagle for imputation
5. Calculates imputation accuracy metrics (R², concordance, etc.)

IMPORTANT: Two dataset sizes available:
- prepare:         FULL chr22 (~830K markers) - integration testing
- prepare-profile: 5% of chr22 (~41K markers) - profiling/benchmarking

The 5% subset allows profiling to complete in ~4 hours.
The full dataset may cause single-threaded state allocation to stall (6+ hours).

Requirements:
- bcftools, tabix
- Java 11+ (for Beagle)
- Reagle binary (cargo build --release)

Usage:
  python integration_test.py              # Run all stages (FULL dataset)
  python integration_test.py prepare      # Download and prepare FULL chr22 VCFs
  python integration_test.py prepare-profile  # Prepare 5% subset for profiling
  python integration_test.py beagle       # Run Beagle imputation only
  python integration_test.py reagle       # Run Reagle imputation only
  python integration_test.py phasing-compare  # Compare phasing vs EagleImp/TRUTH baseline
  python integration_test.py metrics      # Calculate metrics only
"""

import os
import sys
import subprocess
import random
import gzip
import json
import shutil
from pathlib import Path
from collections import defaultdict
import math
import argparse


def run(cmd, check=True, capture=False):
    """Run a shell command."""
    print(f"CMD: {cmd}")
    sys.stdout.flush()
    if capture:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd, shell=True)
    if check and result.returncode != 0:
        if capture:
            print(f"STDERR: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result


def validate_vcf(path):
    """Return True if bcftools can read the VCF/BCF header."""
    if not Path(path).exists():
        return False
    result = subprocess.run(
        f"bcftools view -h {path}",
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False
    stderr = result.stderr or ""
    # Treat truncated BGZF warnings as invalid so cached partial files get rebuilt.
    if (
        "Failed to read BGZF block" in stderr
        or "No BGZF EOF marker" in stderr
        or "EOF marker is absent" in stderr
        or "input is probably truncated" in stderr
    ):
        return False
    return True


def has_vcf_records(path):
    """Return True if VCF/BCF has at least one record."""
    if not Path(path).exists():
        return False
    result = subprocess.run(
        f"bcftools view -H {path} | head -1",
        shell=True,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def has_index(path):
    return Path(str(path) + ".csi").exists() or Path(str(path) + ".tbi").exists()


def ensure_index(path, recreate_cmd=None):
    """Ensure a CSI/TBI index exists; optionally recreate file on failure."""
    if has_index(path):
        return True
    result = run(f"bcftools index -f {path}", check=False, capture=True)
    if result.returncode == 0:
        return True
    if recreate_cmd:
        Path(path).unlink(missing_ok=True)
        Path(str(path) + ".csi").unlink(missing_ok=True)
        Path(str(path) + ".tbi").unlink(missing_ok=True)
        run(recreate_cmd)
        run(f"bcftools index -f {path}")
        return True
    return False


def _open_maybe_gzip(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


def get_chrom_bounds(vcf_path, chrom):
    """Return (min_pos, max_pos) for chrom in VCF/VCF.GZ, or None if not found."""
    chrom_str = str(chrom)
    chrom_options = {chrom_str, f"chr{chrom_str}"}
    if chrom_str.startswith("chr"):
        chrom_options.add(chrom_str[3:])

    min_pos = None
    max_pos = None
    if str(vcf_path).endswith(".bcf"):
        return None

    try:
        with _open_maybe_gzip(vcf_path) as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) < 2:
                    continue
                if fields[0] not in chrom_options:
                    continue
                try:
                    pos = int(fields[1])
                except ValueError:
                    continue
                if min_pos is None or pos < min_pos:
                    min_pos = pos
                if max_pos is None or pos > max_pos:
                    max_pos = pos
    except OSError:
        return None

    if min_pos is None or max_pos is None:
        return None
    return (min_pos, max_pos)

def count_chrom_markers(vcf_path, chrom):
    """Return number of markers for chrom in VCF/VCF.GZ."""
    chrom_str = str(chrom)
    chrom_options = {chrom_str, f"chr{chrom_str}"}
    if chrom_str.startswith("chr"):
        chrom_options.add(chrom_str[3:])
    if str(vcf_path).endswith(".bcf"):
        return None
    n = 0
    try:
        with _open_maybe_gzip(vcf_path) as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) < 2:
                    continue
                if fields[0] in chrom_options:
                    n += 1
    except OSError:
        return None
    return n

def find_chrom_label(vcf_path, chrom):
    """Return the exact chromosome label found in the VCF for chrom."""
    chrom_str = str(chrom)
    chrom_options = {chrom_str, f"chr{chrom_str}"}
    if chrom_str.startswith("chr"):
        chrom_options.add(chrom_str[3:])
    if str(vcf_path).endswith(".bcf"):
        return None
    try:
        with _open_maybe_gzip(vcf_path) as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) < 2:
                    continue
                if fields[0] in chrom_options:
                    return fields[0]
    except OSError:
        return None
    return None

def compute_profile_region(vcf_path, chrom, fraction=0.05):
    """Return (region_str, min_pos, end_pos, chrom_label) for the first fraction of markers."""
    chrom_label = find_chrom_label(vcf_path, chrom) or f"chr{chrom}"
    total = count_chrom_markers(vcf_path, chrom)
    if not total:
        return None
    cutoff = max(1, int(total * fraction))
    chrom_str = str(chrom)
    chrom_options = {chrom_str, f"chr{chrom_str}"}
    if chrom_str.startswith("chr"):
        chrom_options.add(chrom_str[3:])
    if str(vcf_path).endswith(".bcf"):
        return None
    min_pos = None
    end_pos = None
    seen = 0
    try:
        with _open_maybe_gzip(vcf_path) as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) < 2:
                    continue
                if fields[0] not in chrom_options:
                    continue
                try:
                    pos = int(fields[1])
                except ValueError:
                    continue
                if min_pos is None:
                    min_pos = pos
                seen += 1
                if seen >= cutoff:
                    end_pos = pos
                    break
    except OSError:
        return None
    if min_pos is None or end_pos is None:
        return None
    return f"{chrom_label}:{min_pos}-{end_pos}", min_pos, end_pos, chrom_label


def resolve_region_arg(paths, chrom):
    """Use reference bounds for region selection."""
    ref_bounds = get_chrom_bounds(paths["ref_vcf"], chrom)
    if not ref_bounds:
        raise RuntimeError(f"Unable to determine reference bounds for chr{chrom}")
    return f"chr{chrom}:{ref_bounds[0]}-{ref_bounds[1]}"


def print_tool_help(label, cmd):
    try:
        result = run(f"{cmd} --help 2>&1 | head -5", capture=True, check=False)
        if result.stdout:
            print(f"{label} --help output:\n{result.stdout.strip()}")
    except Exception as e:
        print(f"Warning: {label} --help check failed: {e}")


def check_dependencies():
    """Check that required tools are installed."""
    deps = ["bcftools", "tabix", "java", "curl"]
    missing = []
    for dep in deps:
        try:
            subprocess.run(f"which {dep}", shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            missing.append(dep)

    if missing:
        print(f"Error: Missing dependencies: {', '.join(missing)}")
        sys.exit(1)

    print("All dependencies found.")


def find_executable(name, env_var=None):
    """Locate an executable using an env var override or PATH."""
    if env_var:
        override = os.environ.get(env_var)
        if override:
            path = Path(override)
            if path.exists():
                return path
    found = shutil.which(name)
    return Path(found) if found else None


def qref_path_for_ref(ref_path):
    """Return the expected .qref path for a reference VCF/BCF."""
    ref_path = Path(ref_path)
    suffixes = ref_path.suffixes
    base = ref_path
    if len(suffixes) >= 2 and suffixes[-2:] == [".vcf", ".gz"]:
        base = ref_path.with_suffix("").with_suffix("")
    elif suffixes and suffixes[-1] in [".vcf", ".bcf", ".gz"]:
        base = ref_path.with_suffix("")
    return Path(f"{base}.qref")


def ensure_eagleimp_qref(ref_vcf, eagleimp_bin):
    """Create a .qref reference if missing and return its path."""
    qref_path = qref_path_for_ref(ref_vcf)
    if qref_path.exists():
        return qref_path
    print(f"Creating Qref from {ref_vcf}...")
    run(f"{eagleimp_bin} --ref {ref_vcf} --makeQref")
    if qref_path.exists():
        return qref_path
    # EagleImp may drop extra suffixes (e.g., 22.ref.vcf.gz -> 22.qref)
    alt_name = ref_vcf.name.split(".", 1)[0] + ".qref"
    alt_path = ref_vcf.parent / alt_name
    if alt_path.exists():
        return alt_path
    raise RuntimeError(f"Expected Qref not found at {qref_path}")


def ensure_simple_genetic_map(ref_vcf, map_path, chrom="22"):
    """Create an EagleImp-compatible simple genetic map from VCF positions."""
    map_path = Path(map_path)
    if map_path.exists() and map_path.stat().st_size > 0:
        with open(map_path) as f:
            for line in f:
                header = line.strip()
                if not header:
                    continue
                header_ok = header in {
                    "chr position COMBINED_rate(cM/Mb) Genetic_Map(cM)",
                    "position COMBINED_rate(cM/Mb) Genetic_Map(cM)",
                }
                if header_ok:
                    return map_path
                break

    chrom_label = find_chrom_label(ref_vcf, chrom) or f"chr{chrom}"
    chrom_token = str(chrom).replace("chr", "")
    if chrom_token.upper() == "X":
        map_chr = "23"
    elif chrom_token.upper() == "Y":
        map_chr = "24"
    else:
        map_chr = chrom_token
    positions = []
    with _open_maybe_gzip(ref_vcf) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 2:
                continue
            if fields[0] != chrom_label:
                continue
            try:
                pos = int(fields[1])
            except ValueError:
                continue
            positions.append(pos)

    if not positions:
        raise RuntimeError(f"No positions found for {chrom_label} in {ref_vcf}")

    positions = sorted(set(positions))
    map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(map_path, "w") as f:
        # EagleImp requires one of:
        #   chr position COMBINED_rate(cM/Mb) Genetic_Map(cM)
        #   position COMBINED_rate(cM/Mb) Genetic_Map(cM)
        # If chromosome column is present, use numeric chromosome IDs (no "chr" prefix).
        f.write("chr position COMBINED_rate(cM/Mb) Genetic_Map(cM)\n")
        for pos in positions:
            cm = pos / 1_000_000.0
            f.write(f"{map_chr} {pos} 1.0 {cm:.6f}\n")

    return map_path


def ensure_plink_genetic_map(ref_vcf, map_path, chrom="22"):
    """Create a PLINK-style map file: chrom marker_id cM bp."""
    map_path = Path(map_path)
    if map_path.exists() and map_path.stat().st_size > 0:
        return map_path

    chrom_label = find_chrom_label(ref_vcf, chrom) or f"chr{chrom}"
    chrom_token = str(chrom).replace("chr", "")
    if chrom_token.upper() == "X":
        map_chr = "23"
    elif chrom_token.upper() == "Y":
        map_chr = "24"
    else:
        map_chr = chrom_token

    positions = []
    with _open_maybe_gzip(ref_vcf) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 2:
                continue
            if fields[0] != chrom_label:
                continue
            try:
                pos = int(fields[1])
            except ValueError:
                continue
            positions.append(pos)

    if not positions:
        raise RuntimeError(f"No positions found for {chrom_label} in {ref_vcf}")

    positions = sorted(set(positions))
    map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(map_path, "w") as f:
        for pos in positions:
            cm = pos / 1_000_000.0
            # Use chrom_label (matching the VCF, e.g. "chr22") rather than
            # the bare number (e.g. "22") so Beagle can look up its map.
            f.write(f"{chrom_label}\t.\t{cm:.6f}\t{pos}\n")

    return map_path


def ensure_position_genetic_map(ref_vcf, map_path, chrom="22"):
    """Create a position/rate/cM map usable by tools like IMPUTE5/GLIMPSE."""
    map_path = Path(map_path)
    if map_path.exists() and map_path.stat().st_size > 0:
        return map_path

    chrom_label = find_chrom_label(ref_vcf, chrom) or f"chr{chrom}"
    positions = []
    with _open_maybe_gzip(ref_vcf) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 2:
                continue
            if fields[0] != chrom_label:
                continue
            try:
                pos = int(fields[1])
            except ValueError:
                continue
            positions.append(pos)

    if not positions:
        raise RuntimeError(f"No positions found for {chrom_label} in {ref_vcf}")

    positions = sorted(set(positions))
    map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(map_path, "w") as f:
        f.write("position COMBINED_rate(cM/Mb) Genetic_Map(cM)\n")
        for pos in positions:
            cm = pos / 1_000_000.0
            f.write(f"{pos} 1.0 {cm:.6f}\n")

    return map_path


def tool_supports_flag(binary_path, flag):
    """Best-effort check whether a binary advertises a CLI flag."""
    try:
        result = subprocess.run(
            f"{binary_path} --help 2>&1",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        help_text = (result.stdout or "") + "\n" + (result.stderr or "")
        return flag in help_text
    except Exception:
        return False


def eagleimp_map_header_ok(map_path):
    map_path = Path(map_path)
    if not map_path.exists() or map_path.stat().st_size == 0:
        return False
    with open(map_path) as f:
        for line in f:
            header = line.strip()
            if not header:
                continue
            return header in {
                "chr position COMBINED_rate(cM/Mb) Genetic_Map(cM)",
                "position COMBINED_rate(cM/Mb) Genetic_Map(cM)",
            }
    return False


def find_eagleimp_phased_output(output_prefix, data_dir):
    """Find EagleImp phased output from common naming conventions."""
    candidates = [
        Path(str(output_prefix) + ".phased.vcf.gz"),
        Path(str(output_prefix) + ".phased.bcf"),
        Path(str(output_prefix) + ".vcf.gz"),
        Path(str(output_prefix) + ".bcf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in Path(data_dir).glob("*eagleimp*phased*.vcf.gz"):
        return candidate
    for candidate in Path(data_dir).glob("*eagleimp*phased*.bcf"):
        return candidate
    return None


def find_eagleimp_imputed_output(output_prefix, data_dir):
    """Find EagleImp imputed output from common naming conventions."""
    candidates = [
        Path(str(output_prefix) + ".imputed.vcf.gz"),
        Path(str(output_prefix) + ".imputed.bcf"),
        Path(str(output_prefix) + ".phased.vcf.gz"),
        Path(str(output_prefix) + ".phased.bcf"),
        Path(str(output_prefix) + ".vcf.gz"),
        Path(str(output_prefix) + ".bcf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for candidate in Path(data_dir).glob("*eagleimp*imput*.vcf.gz"):
        return candidate
    for candidate in Path(data_dir).glob("*eagleimp*imput*.bcf"):
        return candidate
    for candidate in Path(data_dir).glob("*eagleimp*phased*.vcf.gz"):
        return candidate
    for candidate in Path(data_dir).glob("*eagleimp*phased*.bcf"):
        return candidate
    return None


def download_if_missing(url, local_path):
    """Download a file if it doesn't exist."""
    if not os.path.exists(local_path):
        print(f"Downloading {url}...")
        run(f"curl -L -o {local_path} {url}")
    else:
        print(f"Using cached: {local_path}")


def load_gsa_sites(sites_file, chrom="22"):
    """
    Load GSA variant sites for a specific chromosome.
    Returns set of (chrom, pos) tuples.
    """
    sites = set()
    with open(sites_file) as f:
        for line in f:
            if line.startswith("CHROM") or line.startswith("#"):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                c = parts[0].replace("chr", "")
                if c == chrom:
                    pos = parts[1]
                    sites.add((f"chr{chrom}", int(pos)))
    print(f"Loaded {len(sites)} GSA sites for chr{chrom}")
    return sites


def split_samples(vcf_path, data_dir, test_fraction=0.2, seed=42):
    """Split samples into train (reference) and test sets."""
    random.seed(seed)

    # Get sample names
    result = run(f"bcftools query -l {vcf_path}", capture=True)
    samples = result.stdout.strip().split('\n')
    samples = [s for s in samples if s]  # Remove empty

    random.shuffle(samples)
    n_test = max(1, int(len(samples) * test_fraction))
    test_samples = samples[:n_test]
    train_samples = samples[n_test:]

    print(f"Total samples: {len(samples)}, Train: {len(train_samples)}, Test: {len(test_samples)}")

    # Write sample lists
    test_file = os.path.join(data_dir, "test_samples.txt")
    train_file = os.path.join(data_dir, "train_samples.txt")

    with open(test_file, 'w') as f:
        f.write('\n'.join(test_samples))
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_samples))

    return train_file, test_file, train_samples, test_samples


def create_regions_file(sites, output_path):
    """Create a regions file for bcftools from a set of sites."""
    sorted_sites = sorted(sites, key=lambda x: (x[0], x[1]))
    with open(output_path, 'w') as f:
        for chrom, pos in sorted_sites:
            f.write(f"{chrom}\t{pos}\n")
    return output_path


def run_beagle(ref_vcf, target_vcf, output_prefix, beagle_jar, nthreads=2, map_path=None):
    """Run Java Beagle for imputation."""
    map_arg = f" map={map_path}" if map_path else ""
    cmd = f"java -Xmx4g -jar {beagle_jar} ref={ref_vcf} gt={target_vcf} out={output_prefix} nthreads={nthreads} gp=true{map_arg}"
    try:
        run(cmd)
        output = f"{output_prefix}.vcf.gz"
        if os.path.exists(output):
            run(f"bcftools index -f {output}")
        return output
    except subprocess.CalledProcessError as e:
        print(f"Beagle failed: {e}")
        return None


def run_reagle(ref_vcf, target_vcf, output_prefix, reagle_bin, map_path=None):
    """Run Reagle for imputation."""
    output_vcf = f"{output_prefix}.vcf.gz"
    map_arg = f" --map {map_path}" if map_path else ""
    cmd = f"{reagle_bin} --ref {ref_vcf} --target {target_vcf} --out {output_prefix}{map_arg}"
    try:
        run(cmd)
        if os.path.exists(output_vcf):
            run(f"bcftools index -f {output_vcf}")
            return output_vcf
        else:
            print(f"Warning: Reagle output not found at {output_vcf}")
            # Check for uncompressed
            if os.path.exists(output_prefix):
                run(f"bgzip -f {output_prefix}")
                run(f"bcftools index -f {output_vcf}")
                return output_vcf
            return None
    except subprocess.CalledProcessError as e:
        print(f"Reagle failed: {e}")
        return None


def parse_genotype(gt_str):
    """Parse a VCF genotype string to allele tuple. Returns None for missing."""
    if not gt_str or gt_str == "." or "./." in gt_str or ".|." in gt_str:
        return None
    sep = '|' if '|' in gt_str else '/'
    parts = gt_str.split(sep)
    if len(parts) != 2:
        return None
    try:
        a1, a2 = int(parts[0]), int(parts[1])
        return (a1, a2)
    except ValueError:
        return None


def _parse_ds_field(ds_str):
    """Parse DS which may be a single float (biallelic) or comma-separated list (multiallelic).

    Returns a single float representing total non-reference dosage.
    """
    if ds_str is None or ds_str == ".":
        return None
    try:
        if "," in ds_str:
            parts = [p for p in ds_str.split(",") if p and p != "."]
            vals = [float(p) for p in parts]
            return float(sum(vals))
        return float(ds_str)
    except Exception:
        return None


def _gt_nonref_dosage(gt):
    """Total non-reference dosage for diploid GT (counts alleles != 0)."""
    if gt is None:
        return None
    a0, a1 = gt
    if a0 is None or a1 is None:
        return None
    if a0 == 255 or a1 == 255:
        return None
    return float((1 if a0 != 0 else 0) + (1 if a1 != 0 else 0))


def _gt_class_nonref(gt):
    """3-class genotype class in a non-ref sense (works for biallelic and multiallelic).

    0 = homref (0/0)
    1 = het (one ref, one non-ref)
    2 = hom-nonref (both alleles non-ref, possibly different)
    """
    if gt is None:
        return None
    a0, a1 = gt
    if a0 == 255 or a1 == 255:
        return None
    if a0 == 0 and a1 == 0:
        return 0
    if a0 != 0 and a1 != 0:
        return 2
    return 1


def _split_alleles(ref, alt_str):
    alts = []
    if alt_str and alt_str != ".":
        alts = str(alt_str).split(",")
    return [str(ref)] + [a for a in alts if a]


def _normalize_imputed_to_truth_multiallelic(t_ref, t_alt, i_ref, i_alt, i_gt, i_dos, i_gp):
    """Normalize imputed alleles into truth allele order for multi-allelic sites.

    We only handle allele *re-ordering* (same allele set). No strand/complement logic.
    We return a normalized GT and a normalized total non-ref dosage.
    """
    t_alleles = _split_alleles(t_ref, t_alt)
    i_alleles = _split_alleles(i_ref, i_alt)

    if set(t_alleles) != set(i_alleles):
        return False, i_gt, i_dos, i_gp

    # Map imputed allele index -> truth allele index by nucleotide string.
    imputed_to_truth = {}
    for i_idx, base in enumerate(i_alleles):
        try:
            t_idx = t_alleles.index(base)
        except ValueError:
            return False, i_gt, i_dos, i_gp
        imputed_to_truth[i_idx] = t_idx

    # Remap GT allele indices.
    if i_gt is not None:
        a0, a1 = i_gt
        if a0 in imputed_to_truth and a1 in imputed_to_truth:
            i_gt = (imputed_to_truth[a0], imputed_to_truth[a1])
        else:
            return False, i_gt, i_dos, i_gp

    # Normalize DS: for multiallelic, treat DS as total non-ref dosage (sum over ALTs).
    # If DS was a list, summing is already allele-order invariant.
    # If DS was scalar (some tools), keep as-is.
    # Also disable GP usage for multiallelic in our current harness.
    i_gp = None
    return True, i_gt, i_dos, i_gp


def _normalize_imputed_to_truth_alleles(t_ref, t_alt, i_ref, i_alt, i_gt, i_dos, i_gp):
    """Normalize imputed GT/DS/GP into the truth REF/ALT orientation for biallelic sites.

    This handles the common case where the same SNP is represented with swapped REF/ALT between VCFs.
    For swapped alleles, the allele index meaning flips: 0<->1.
    """
    # Only support biallelic swaps here.
    if "," in str(t_alt) or "," in str(i_alt):
        return False, i_gt, i_dos, i_gp

    if t_ref == i_ref and t_alt == i_alt:
        return False, i_gt, i_dos, i_gp

    swapped = t_ref == i_alt and t_alt == i_ref
    if not swapped:
        return False, i_gt, i_dos, i_gp

    # Flip hard-call genotype.
    if i_gt is not None:
        a0, a1 = i_gt
        if a0 in (0, 1) and a1 in (0, 1):
            i_gt = (1 - a0, 1 - a1)
        else:
            # Non-biallelic allele codes; give up on normalization.
            return False, i_gt, i_dos, i_gp

    # Invert dosage: DS is ALT dosage; after swapping ALT, DS' = 2 - DS.
    if i_dos is not None:
        try:
            i_dos = 2.0 - float(i_dos)
        except Exception:
            pass

    # Swap GP(0/0) <-> GP(1/1). Keep GP(0/1) the same.
    if i_gp is not None and len(i_gp) == 3:
        i_gp = (i_gp[2], i_gp[1], i_gp[0])

    return True, i_gt, i_dos, i_gp


def _stream_vcf_lines(cmd):
    """Stream VCF query output line by line to avoid loading all into memory."""
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in proc.stdout:
        line = line.strip()
        if line:
            yield line
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read()
        raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=stderr)


def _parse_truth_line(line, sample_indices):
    """Parse a truth VCF line into (key, sample_data_list, is_multiallelic)."""
    parts = line.split('\t')
    if len(parts) < 5:
        return None, None, False
    chrom, pos = parts[0], int(parts[1])
    ref, alt = parts[2], parts[3]
    key = (chrom, pos, ref, alt)
    is_multiallelic = ',' in alt
    gts = parts[4:]
    sample_data = []
    for idx in sample_indices:
        if idx < len(gts):
            gt_field = gts[idx].split(':')[0]
            gt = parse_genotype(gt_field)
            is_phased = '|' in gt_field
            sample_data.append(
                (
                    gt,
                    _gt_nonref_dosage(gt) if gt is not None else None,
                    is_phased,
                )
            )
        else:
            sample_data.append((None, None, False))
    return key, sample_data, is_multiallelic


def _parse_imputed_line(line, sample_indices, format_mode="full"):
    """Parse an imputed VCF line into (key, sample_data_list, is_multiallelic, missing_required).

    format_mode:
      - "full":   GT:DS:GP
      - "ds_only": GT:DS
      - "gp_only": GT:GP
      - "gt_only": GT
    """
    parts = line.split('\t')
    if len(parts) < 5:
        return None, None, False, False
    chrom, pos = parts[0], int(parts[1])
    ref, alt = parts[2], parts[3]
    key = (chrom, pos, ref, alt)
    is_multiallelic = ',' in alt
    sample_data_list = parts[4:]
    sample_data = []
    missing_required = False
    for idx in sample_indices:
        if idx < len(sample_data_list):
            data_str = sample_data_list[idx]
            fields = data_str.split(':')
            gt_field = fields[0]
            gt = parse_genotype(gt_field)
            is_phased = '|' in gt_field

            ds = None
            gp = None

            if format_mode == "full":
                # Expecting GT:DS:GP from bcftools query
                if len(fields) > 1 and fields[1] != '.':
                    ds = _parse_ds_field(fields[1])
                if len(fields) > 2 and fields[2] != '.':
                    try:
                        gp_parts = fields[2].split(',')
                        if len(gp_parts) == 3:
                            gp = (float(gp_parts[0]), float(gp_parts[1]), float(gp_parts[2]))
                    except:
                        pass
                if ds is None or gp is None:
                    missing_required = True
            elif format_mode == "ds_only":
                # Expecting GT:DS
                if len(fields) > 1 and fields[1] != '.':
                    ds = _parse_ds_field(fields[1])
            elif format_mode == "gp_only":
                # Expecting GT:GP
                if len(fields) > 1 and fields[1] != '.':
                    try:
                        gp_parts = fields[1].split(',')
                        if len(gp_parts) == 3:
                            gp = (float(gp_parts[0]), float(gp_parts[1]), float(gp_parts[2]))
                    except:
                        pass
            else:
                # GT-only: derive dosage from GT, GP unavailable.
                if gt is not None:
                    ds = _gt_nonref_dosage(gt)

            # If DS is missing but GP exists, derive DS from GP (biallelic).
            if ds is None and gp is not None and len(gp) == 3:
                ds = float(gp[1]) + 2.0 * float(gp[2])

            # If still no DS and GT exists, fallback to hard-call dosage.
            if ds is None and gt is not None:
                ds = _gt_nonref_dosage(gt)

            sample_data.append((gt, ds, is_phased, gp))
        else:
            sample_data.append((None, None, False, None))
    return key, sample_data, is_multiallelic, missing_required


def load_input_sites(input_vcf):
    if not input_vcf or not os.path.exists(input_vcf):
        return None
    # Use coordinate-only keys to avoid REF/ALT representation artifacts
    # (e.g., REF/ALT swaps or multiallelic ALT ordering differences).
    cmd = f"bcftools query -f '%CHROM\\t%POS\\n' {input_vcf}"
    sites = set()
    for line in _stream_vcf_lines(cmd):
        parts = line.split('\t')
        if len(parts) >= 2:
            chrom, pos = parts[0], int(parts[1])
            sites.add((chrom, pos))
    return sites


def get_vcf_samples(vcf_path):
    """Stream sample names from VCF header."""
    if not vcf_path or not os.path.exists(vcf_path):
        return []
    
    # 1. Check validity first
    check_cmd = f"bcftools view -h {vcf_path} 2>/dev/null | head -1"
    try:
        subprocess.run(check_cmd, shell=True, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print(f"Warning: VCF header check failed for {vcf_path}")
        return []

    # 2. Query samples (streaming)
    cmd = f"bcftools query -l {vcf_path}"
    samples = []
    try:
        for line in _stream_vcf_lines(cmd):
            s = line.strip()
            if s:
                samples.append(s)
    except Exception as e:
        print(f"Error reading samples from {vcf_path}: {e}")
        return []
        
    return samples


def calculate_metrics(truth_vcf, imputed_vcf, output_prefix, input_vcf=None, reference_vcf=None, require_ds_gp=True):
    """
    Calculate comprehensive imputation accuracy metrics.

    Memory-efficient streaming version using merge-join on sorted VCFs.

    Metrics:
    - Unphased genotype concordance (exact match ignoring phase: 0|1 == 1|0)
    - Allelic R² (correlation between true and imputed dosages)
    - Dosage Variance R² (correlation using variance-weighted approach)
    - Non-reference concordance (concordance for non-REF genotypes only)
    - IQS (Imputation Quality Score - chance-corrected concordance)
    - Switch Error Rate (phase switch errors for heterozygotes)
    - Confusion matrix (HomRef/Het/HomAlt)
    - Per-sample metrics
    - Per-MAF-bin metrics
    - INFO score approximation
    """
    import time
    start_time = time.time()
    diag_start = start_time

    if not imputed_vcf or not os.path.exists(imputed_vcf):
        print("Imputed VCF not found")
        return None

    print(f"\nCalculating metrics: {imputed_vcf} vs {truth_vcf}")

    # Get sample lists first (small memory footprint)
    # Get sample lists first (streaming to avoid memory issues with corrupt files)
    truth_samples = get_vcf_samples(truth_vcf)
    imputed_samples = get_vcf_samples(imputed_vcf)

    print(f"Truth samples: {len(truth_samples)}, Imputed samples: {len(imputed_samples)}")

    # Build sample index for common samples
    common_samples = set(truth_samples) & set(imputed_samples)
    if not common_samples:
        print("ERROR: No common samples between truth and imputed VCFs")
        return None
    common_samples_list = [s for s in truth_samples if s in common_samples]
    print(f"Common samples: {len(common_samples_list)}") 

    # Precompute sample index mappings for faster parsing
    truth_index = {s: i for i, s in enumerate(truth_samples)}
    imputed_index = {s: i for i, s in enumerate(imputed_samples)}
    truth_indices = [truth_index[s] for s in common_samples_list]
    imputed_indices = [imputed_index[s] for s in common_samples_list]
    n_common = len(common_samples_list)
    
    # ============================================================
    # DIAGNOSTIC OUTPUT - VCF Analysis
    # ============================================================
    print("\n" + "="*60)
    print("DIAGNOSTIC: VCF ANALYSIS")
    print("="*60)
    
    # Check VCF headers for build/format information
    print("\n📋 VCF Header Information:")
    diag_vcfs = [(truth_vcf, "TRUTH"), (imputed_vcf, "IMPUTED")]
    if reference_vcf:
        diag_vcfs.append((reference_vcf, "REFERENCE"))
    for vcf_path, label in diag_vcfs:
        try:
            header_result = subprocess.run(
                f"bcftools view -h {vcf_path} | head -30",
                shell=True, capture_output=True, text=True
            )
            header_lines = header_result.stdout.split('\n')
            
            # Extract key info
            contigs = [l for l in header_lines if l.startswith('##contig')]
            formats = [l for l in header_lines if l.startswith('##FORMAT')]
            
            print(f"\n  {label} VCF:")
            if contigs:
                print(f"    Contigs: {len(contigs)} found")
                # Show first contig for chr22
                chr22_contigs = [c for c in contigs if 'chr22' in c or 'ID=22,' in c]
                if chr22_contigs:
                    print(f"    Chr22 contig: {chr22_contigs[0][:100]}...")
            print(f"    FORMAT fields: {len(formats)}")
            has_gt = any('ID=GT' in f for f in formats)
            has_ds = any('ID=DS' in f for f in formats)
            has_gp = any('ID=GP' in f for f in formats)
            print(f"    Has GT: {has_gt}, DS: {has_ds}, GP: {has_gp}")
        except Exception as e:
            print(f"  {label}: Error reading header - {e}")

    # File-level diagnostics
    print("\n📦 VCF File Summary:")
    for vcf_path, label in diag_vcfs:
        try:
            size_bytes = os.path.getsize(vcf_path)
        except Exception:
            size_bytes = -1
        size_mb = (size_bytes / (1024 * 1024)) if size_bytes >= 0 else None
        size_str = f"{size_mb:.2f} MB" if size_mb is not None else "unknown"
        print(f"\n  {label}:")
        print(f"    Path: {vcf_path}")
        print(f"    Size: {size_str}")
        try:
            sample_result = subprocess.run(
                f"bcftools query -l {vcf_path}",
                shell=True, capture_output=True, text=True
            )
            samples = [s for s in sample_result.stdout.strip().split("\n") if s.strip()]
            print(f"    Samples: {len(samples)} {samples[:3] if samples else ''}")
        except Exception as e:
            print(f"    Samples: Error - {e}")
        try:
            count_result = subprocess.run(
                f"bcftools index -n {vcf_path}",
                shell=True, capture_output=True, text=True
            )
            count = count_result.stdout.strip()
            print(f"    Records: {count if count else 'unknown'}")
        except Exception as e:
            print(f"    Records: Error - {e}")
        try:
            first_result = subprocess.run(
                f"bcftools view -H {vcf_path} | head -1",
                shell=True, capture_output=True, text=True
            )
            last_result = subprocess.run(
                f"bcftools view -H {vcf_path} | tail -1",
                shell=True, capture_output=True, text=True
            )
            first_line = first_result.stdout.strip()
            last_line = last_result.stdout.strip()
            if first_line:
                f_parts = first_line.split("\t")
                if len(f_parts) >= 5:
                    print(f"    First: {f_parts[0]}:{f_parts[1]} {f_parts[3]}>{f_parts[4]}")
            if last_line:
                l_parts = last_line.split("\t")
                if len(l_parts) >= 5:
                    print(f"    Last:  {l_parts[0]}:{l_parts[1]} {l_parts[3]}>{l_parts[4]}")
        except Exception as e:
            print(f"    Range: Error - {e}")

    def _truncate_line(line, limit=2000):
        if len(line) <= limit:
            return line
        return line[:limit] + "...[truncated]"

    # Print example lines (raw records) from each file
    print("\n🧾 Example Records (first 3 lines):")
    for vcf_path, label in diag_vcfs:
        try:
            ex_result = subprocess.run(
                f"bcftools view -H {vcf_path} | head -3",
                shell=True, capture_output=True, text=True
            )
            lines = [l for l in ex_result.stdout.strip().split("\n") if l]
            print(f"\n  {label}:")
            if not lines:
                print("    (no records)")
            else:
                for line in lines:
                    print(f"    {_truncate_line(line)}")
        except Exception as e:
            print(f"  {label}: Error - {e}")
    
    # Sample first 10 sites from each VCF
    print("\n📍 Sample Sites (first 10):")
    for vcf_path, label in [(truth_vcf, "TRUTH"), (imputed_vcf, "IMPUTED")]:
        try:
            result = subprocess.run(
                f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT[\\t%GT]\\n' {vcf_path} | head -10",
                shell=True, capture_output=True, text=True
            )
            lines = [l for l in result.stdout.strip().split('\n') if l]
            print(f"\n  {label} (showing {len(lines)} sites):")
            for i, line in enumerate(lines, 1):
                parts = line.split('\t')
                if len(parts) >= 5:
                    chrom, pos, ref, alt, gt = parts[0], parts[1], parts[2], parts[3], parts[4]
                    print(f"    {i}. {chrom}:{pos} {ref}>{alt} GT={gt}")
        except Exception as e:
            print(f"  {label}: Error - {e}")
    
    # Check site overlap
    print("\n🔍 Site Overlap Analysis:")
    try:
        # Get first 100 sites from each
        truth_cmd = f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\n' {truth_vcf} | head -100"
        imputed_cmd = f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\n' {imputed_vcf} | head -100"
        
        truth_result = subprocess.run(truth_cmd, shell=True, capture_output=True, text=True)
        imputed_result = subprocess.run(imputed_cmd, shell=True, capture_output=True, text=True)
        
        truth_sites_sample = set(truth_result.stdout.strip().split('\n'))
        imputed_sites_sample = set(imputed_result.stdout.strip().split('\n'))
        
        overlap = truth_sites_sample & imputed_sites_sample
        print(f"  First 100 sites - Truth: {len(truth_sites_sample)}, Imputed: {len(imputed_sites_sample)}, Overlap: {len(overlap)}")
        
        if overlap:
            print(f"  ✓ Some sites match (good sign)")
        else:
            print(f"  ⚠️  NO OVERLAP in first 100 sites (coordinate mismatch?)")
            print(f"\n  Sample truth sites:")
            for site in list(truth_sites_sample)[:3]:
                print(f"    {site}")
            print(f"\n  Sample imputed sites:")
            for site in list(imputed_sites_sample)[:3]:
                print(f"    {site}")

        # Full overlap across all sites (exact CHROM/POS/REF/ALT)
        def _site_set_all(vcf_path):
            sites = set()
            cmd = f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\n' {vcf_path}"
            for line in _stream_vcf_lines(cmd):
                parts = line.split('\t')
                if len(parts) >= 4:
                    chrom, pos, ref, alt = parts[0], int(parts[1]), parts[2], parts[3]
                    sites.add((chrom, pos, ref, alt))
            return sites

        truth_all = _site_set_all(truth_vcf)
        imputed_all = _site_set_all(imputed_vcf)
        overlap_all = truth_all & imputed_all
        print(f"  Full overlap (exact CHROM/POS/REF/ALT): Truth={len(truth_all)}, Imputed={len(imputed_all)}, Overlap={len(overlap_all)}")
        if not overlap_all:
            print("  ⚠️  NO OVERLAP across all sites (exact match)")

        # Coordinate-only overlap (ignores REF/ALT) for multiple pairs
        def _pos_set(vcf_path, limit):
            cmd = f"bcftools query -f '%CHROM\\t%POS\\n' {vcf_path} | head -{limit}"
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return set([l for l in res.stdout.strip().split("\n") if l])

        pos_limit = 5000
        truth_pos = _pos_set(truth_vcf, pos_limit)
        imputed_pos = _pos_set(imputed_vcf, pos_limit)
        pos_overlap = truth_pos & imputed_pos
        print(f"  First {pos_limit} positions - Truth: {len(truth_pos)}, Imputed: {len(imputed_pos)}, Overlap: {len(pos_overlap)}")
        if reference_vcf:
            ref_pos = _pos_set(reference_vcf, pos_limit)
            print(f"  First {pos_limit} positions - Truth: {len(truth_pos)}, Ref: {len(ref_pos)}, Overlap: {len(truth_pos & ref_pos)}")
            print(f"  First {pos_limit} positions - Imputed: {len(imputed_pos)}, Ref: {len(ref_pos)}, Overlap: {len(imputed_pos & ref_pos)}")

        # Quick REF/ALT mismatch sample at shared positions
        if pos_overlap:
            truth_map = {}
            for line in truth_sites_sample:
                parts = line.split("\t")
                if len(parts) >= 4:
                    truth_map[f"{parts[0]}\t{parts[1]}"] = (parts[2], parts[3])
            imputed_map = {}
            for line in imputed_sites_sample:
                parts = line.split("\t")
                if len(parts) >= 4:
                    imputed_map[f"{parts[0]}\t{parts[1]}"] = (parts[2], parts[3])
            mismatch_examples = 0
            for key in list(pos_overlap)[:10]:
                t_vals = truth_map.get(key)
                i_vals = imputed_map.get(key)
                if t_vals and i_vals and t_vals != i_vals:
                    mismatch_examples += 1
                    print(f"  REF/ALT diff at {key}: truth {t_vals[0]}>{t_vals[1]} imputed {i_vals[0]}>{i_vals[1]}")
            if mismatch_examples == 0:
                print("  No REF/ALT diffs found in sampled shared positions")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "="*60)
    print("END DIAGNOSTIC OUTPUT")
    print("="*60 + "\n")
    diag_elapsed = time.time() - diag_start
    
    input_sites = load_input_sites(input_vcf)
    if input_sites is not None:
        print(f"Input genotyped sites: {len(input_sites)}")

    # Reference panel AF/MAF (required for meaningful MAF stratification)
    ref_key = None
    ref_afs = None
    ref_iter = None

    def _parse_ref_af_line(line):
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            return None, None
        chrom, pos_s, ref, alt, af_s = parts[0], parts[1], parts[2], parts[3], parts[4]
        try:
            pos = int(pos_s)
        except ValueError:
            return None, None

        afs = []
        if af_s and af_s != ".":
            for tok in af_s.split(","):
                if tok and tok != ".":
                    try:
                        afs.append(float(tok))
                    except ValueError:
                        continue
        # Key by (CHROM, POS) so we can still stratify even if REF/ALT representation differs.
        return (chrom, pos), (ref, alt, afs)

    def _maf_from_afs(afs):
        if not afs:
            return None
        alt_sum = sum(afs)
        ref_f = 1.0 - alt_sum
        all_freqs = [ref_f] + list(afs)
        maf = min(all_freqs)
        if maf < 0.0:
            maf = 0.0
        if maf > 0.5:
            maf = 0.5
        return maf

    if not reference_vcf or not os.path.exists(reference_vcf):
        raise RuntimeError("reference_vcf is required for MAF stratification but was not provided or does not exist")

    af_start = time.time()
    try:
        ref_cmd = (
            f"bcftools +fill-tags {reference_vcf} -Ou -- -t AF "
            f"| bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\t%INFO/AF\\n'"
        )
        ref_iter = _stream_vcf_lines(ref_cmd)
        try:
            first_line = next(ref_iter)
            ref_key, ref_afs = _parse_ref_af_line(first_line)
        except StopIteration:
            raise RuntimeError(f"Reference VCF AF stream was empty: {reference_vcf}")
    except Exception as e:
        raise RuntimeError(f"Could not initialize reference AF stream from {reference_vcf}: {e}")

    af_elapsed = time.time() - af_start
    # Calculate metrics
    unphased_concordant = 0  # Genotype match ignoring phase (0|1 == 1|0)
    total_compared = 0

    # Non-reference concordance (excludes 0/0 vs 0/0)
    nonref_concordant = 0
    nonref_total = 0

    # Online statistics for R² (Welford's algorithm) - avoids storing all dosages
    # We need: sum(t), sum(i), sum(t*i), sum(t^2), sum(i^2), count
    r2_stats = {"sum_t": 0.0, "sum_i": 0.0, "sum_ti": 0.0, "sum_tt": 0.0, "sum_ii": 0.0, "count": 0}

    # Confusion matrix: [true_class][imputed_class]
    # Classes: 0=HomRef, 1=Het, 2=HomAlt
    confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    # Per-sample tracking (arrays aligned to common_samples_list for speed)
    sample_concordant = [0] * n_common
    sample_total = [0] * n_common
    sample_sum_t = [0.0] * n_common
    sample_sum_i = [0.0] * n_common
    sample_sum_ti = [0.0] * n_common
    sample_sum_tt = [0.0] * n_common
    sample_sum_ii = [0.0] * n_common
    sample_switch_errors = [0] * n_common
    sample_switch_opportunities = [0] * n_common
    sample_sen_sum = [0.0] * n_common
    sample_sen_min = [1.0] * n_common
    sample_sen_max = [0.0] * n_common
    sample_sen_count = [0] * n_common
    sample_phase_concordant = [0] * n_common
    sample_phase_total = [0] * n_common
    sample_phase_flip = [None] * n_common

    # For IQS calculation: track per-site chance-corrected concordance
    # using empirical truth/imputed genotype marginals at each site.
    site_iqs_sum = 0.0
    site_iqs_count = 0
    site_iqs_weighted_sum = 0.0
    site_iqs_weighted_count = 0

    # For Hellinger score (requires GP field)
    hellinger_sum = 0.0
    hellinger_count = 0

    # Reference AF/MAF availability tracking (for MAF stratification)
    ref_af_missing = 0
    ref_af_sites = 0

    # For switch error rate
    switch_errors = 0
    switch_opportunities = 0
    switch_errors_input = 0
    switch_opportunities_input = 0
    phase_concordant = 0
    phase_total = 0

    # SEN (Scaled Euclidean Norm) score
    sen_sum = 0.0
    sen_count = 0

    # Missing genotype counters
    missing_truth = 0
    missing_imputed = 0
    missing_both = 0
    ref_alt_mismatch = 0
    ref_alt_swapped = 0
    ref_alt_mismatch_examples = 0
    ref_alt_swapped_examples = 0
    multiallelic_sites = 0
    multiallelic_reordered_ok = 0
    multiallelic_mismatch = 0

    # For N50 Phasing Block Length
    # sample -> list of block lengths (in bp)
    phase_blocks = defaultdict(list)
    # Per-sample phase tracking
    current_block_start = [-1] * n_common
    last_het_pos = [-1] * n_common

    # MAF bins for stratified analysis - FINER BINS for rare variants
    def get_maf_bin(maf):
        if maf < 0.001:
            return "ultra-rare (<0.1%)"
        elif maf < 0.005:
            return "very-rare (0.1-0.5%)"
        elif maf < 0.01:
            return "rare (0.5-1%)"
        elif maf < 0.05:
            return "low-freq (1-5%)"
        elif maf < 0.2:
            return "medium (5-20%)"
        else:
            return "common (>20%)"

    # MAF bins with online stats instead of storing lists
    maf_bins = defaultdict(lambda: {
        "unphased_concordant": 0, "total": 0,
        "sum_t": 0.0, "sum_i": 0.0, "sum_ti": 0.0, "sum_tt": 0.0, "sum_ii": 0.0,
        "iqs_sum": 0.0, "iqs_count": 0, "nonref_concordant": 0, "nonref_total": 0,
        "confusion": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "switch_errors": 0, "switch_opportunities": 0,
        "phase_concordant": 0, "phase_total": 0
    })

    # Mask-and-impute metrics (per-sample proxy)
    mask_rate = 0.02
    min_mask_gap = 50_000
    mask_seed = 1337
    last_mask_pos = {}
    masked_stats = {"sum_t": 0.0, "sum_i": 0.0, "sum_ti": 0.0, "sum_tt": 0.0, "sum_ii": 0.0, "count": 0}
    masked_concordant = 0
    masked_total = 0
    masked_nonref_concordant = 0
    masked_nonref_total = 0
    masked_brier_sum = 0.0
    masked_brier_n = 0
    ece_bins = [{"sum_conf": 0.0, "sum_acc": 0.0, "count": 0} for _ in range(10)]
    masked_maf_bins = defaultdict(lambda: {
        "sum_t": 0.0, "sum_i": 0.0, "sum_ti": 0.0, "sum_tt": 0.0, "sum_ii": 0.0, "count": 0,
        "nonref_concordant": 0, "nonref_total": 0
    })

    # === STREAMING SETUP ===
    print("Initializing streams...")

    def _vcf_site_count(path):
        try:
            res = subprocess.run(
                f"bcftools index -n {path}",
                shell=True, capture_output=True, text=True
            )
            if res.returncode == 0:
                return int(res.stdout.strip())
        except Exception:
            pass
        return None

    total_truth_sites = _vcf_site_count(truth_vcf)
    total_imputed_sites = _vcf_site_count(imputed_vcf)
    truth_only_sites = 0
    imputed_only_sites = 0
    
    # 1. Truth Stream
    truth_cmd = f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT[\\t%GT]\\n' {truth_vcf}"
    truth_iter = _stream_vcf_lines(truth_cmd)

    # 2. Imputed Stream
    # Check VCF header for DS/GP fields before querying
    # This handles both Beagle (with gp=true) and other tools that may not include DS/GP
    imputed_cmd_full = f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT[\\t%GT:%DS:%GP]\\n' {imputed_vcf}"
    
    # Check if DS and GP fields exist in VCF header
    has_ds = False
    has_gp = False
    try:
        header_result = subprocess.run(
            ["bcftools", "view", "-h", imputed_vcf],
            capture_output=True, text=True, timeout=30
        )
        if header_result.returncode == 0:
            header = header_result.stdout
            # Check for FORMAT=<ID=DS and FORMAT=<ID=GP in header
            has_ds = "ID=DS" in header
            has_gp = "ID=GP" in header
    except Exception as e:
        print(f"Warning: Could not check VCF header: {e}")

    # If header is missing fields, peek at the first data line's FORMAT column.
    if not (has_ds and has_gp):
        try:
            record_result = subprocess.run(
                ["bcftools", "view", "-H", "-n", "1", imputed_vcf],
                capture_output=True, text=True, timeout=30
            )
            if record_result.returncode == 0 and record_result.stdout.strip():
                first_line = record_result.stdout.strip().split('\n')[0]
                cols = first_line.split('\t')
                if len(cols) >= 9:
                    fmt_fields = set(cols[8].split(':'))
                    if "DS" in fmt_fields:
                        has_ds = True
                    if "GP" in fmt_fields:
                        has_gp = True
        except Exception as e:
            print(f"Warning: Could not inspect FORMAT fields from records: {e}")
    
    # Decide parsing mode based on header availability
    degraded_required = False
    if has_ds and has_gp:
        imputed_format_mode = "full"
        imputed_cmd = imputed_cmd_full
        print(f"Using GT:DS:GP format for {imputed_vcf}")
    elif has_ds and not has_gp:
        imputed_format_mode = "ds_only"
        imputed_cmd = f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT[\\t%GT:%DS]\\n' {imputed_vcf}"
        if require_ds_gp:
            print("WARNING: GP missing in header; proceeding with GT:DS only.")
            degraded_required = True
        else:
            print(f"Using GT:DS format for {imputed_vcf}")
    elif has_gp and not has_ds:
        imputed_format_mode = "gp_only"
        imputed_cmd = f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT[\\t%GT:%GP]\\n' {imputed_vcf}"
        if require_ds_gp:
            print("WARNING: DS missing in header; proceeding with GT:GP only.")
            degraded_required = True
        else:
            print(f"Using GT:GP format for {imputed_vcf}")
    else:
        imputed_format_mode = "gt_only"
        imputed_cmd = f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT[\\t%GT]\\n' {imputed_vcf}"
        if require_ds_gp:
            print("WARNING: DS/GP missing in header; proceeding with GT only.")
            degraded_required = True
        else:
            print(f"Using GT-only format for {imputed_vcf}")
    
    imputed_iter = _stream_vcf_lines(imputed_cmd)

    # Helper to get next parsed line
    def get_next_truth():
        try:
            line = next(truth_iter)
            return _parse_truth_line(line, truth_indices)
        except StopIteration:
            return None, None, False

    def get_next_imputed():
        try:
            line = next(imputed_iter)
            return _parse_imputed_line(line, imputed_indices, format_mode=imputed_format_mode)
        except StopIteration:
            return None, None, False, False

    # Initial fetch
    truth_key, truth_data, truth_multiallelic = get_next_truth()
    imp_key, imp_data, imp_multiallelic, imp_missing_required = get_next_imputed()

    # Track previous het for switch error calculation per sample
    prev_het = [None] * n_common  # (site, truth_gt, imputed_gt, maf_bin) per sample
    prev_het_input = [None] * n_common
    common_sites_count = 0
    skipped_missing_required = 0
    skipped_missing_required_matched = 0
    skipped_missing_required_imputed_only = 0
    last_pos = 0

    print("Streaming and comparing...")
    loop_start = time.time()
    
    # Merge-join loop
    def gt_class(gt):
        return _gt_class_nonref(gt)

    def mask_pick(chrom, pos, maf_bin):
        if input_sites is None:
            return False
        last_pos = last_mask_pos.get(maf_bin)
        if last_pos is not None and pos - last_pos < min_mask_gap:
            return False
        h = hash((chrom, pos, maf_bin, mask_seed)) & 0xFFFFFFFF
        if (h / 0xFFFFFFFF) < mask_rate:
            last_mask_pos[maf_bin] = pos
            return True
        return False

    while truth_key is not None and imp_key is not None:
        t_chrom, t_pos, t_ref, t_alt = truth_key
        i_chrom, i_pos, i_ref, i_alt = imp_key

        # Compare positions (Chrom then Pos)
        # Handle string chromosome comparison carefully if needed, 
        # but typical numeric/lexicographic sort holds for same reference.
        
        if t_chrom == i_chrom:
            if t_pos == i_pos:
                # MATCH! Process site
                if imp_missing_required:
                    skipped_missing_required += 1
                    skipped_missing_required_matched += 1
                    truth_key, truth_data, truth_multiallelic = get_next_truth()
                    imp_key, imp_data, imp_multiallelic, imp_missing_required = get_next_imputed()
                    continue
                site = truth_key
                last_pos = site[1]
                truth_site = truth_data
                imputed_site = imp_data
                swapped_site = False
                multiallelic_reordered = False
                if t_ref != i_ref or t_alt != i_alt:
                    # Try to handle biallelic REF/ALT swaps instead of discarding the site.
                    if ("," not in str(t_alt)) and ("," not in str(i_alt)) and (t_ref == i_alt and t_alt == i_ref):
                        swapped_site = True
                        ref_alt_swapped += 1
                        if ref_alt_swapped_examples < 5:
                            ref_alt_swapped_examples += 1
                            print(
                                f"[ALLELE SWAP] {t_chrom}:{t_pos} truth {t_ref}>{t_alt} imputed {i_ref}>{i_alt}"
                            )
                    else:
                        # If multi-allelic, try to allow allele re-ordering (same allele set).
                        t_is_multi = "," in str(t_alt)
                        i_is_multi = "," in str(i_alt)
                        if t_is_multi or i_is_multi:
                            t_alleles = _split_alleles(t_ref, t_alt)
                            i_alleles = _split_alleles(i_ref, i_alt)
                            if set(t_alleles) == set(i_alleles):
                                multiallelic_reordered = True
                            else:
                                multiallelic_mismatch += 1
                                if ref_alt_mismatch_examples < 5:
                                    ref_alt_mismatch_examples += 1
                                    print(
                                        f"[MULTIALLELIC MISMATCH] {t_chrom}:{t_pos} truth {t_ref}>{t_alt} imputed {i_ref}>{i_alt}"
                                    )
                                truth_key, truth_data, truth_multiallelic = get_next_truth()
                                imp_key, imp_data, imp_multiallelic, imp_missing_required = get_next_imputed()
                                continue
                        else:
                            ref_alt_mismatch += 1
                            if ref_alt_mismatch_examples < 5:
                                ref_alt_mismatch_examples += 1
                                print(
                                    f"[ALLELE MISMATCH] {t_chrom}:{t_pos} truth {t_ref}>{t_alt} imputed {i_ref}>{i_alt}"
                                )
                            truth_key, truth_data, truth_multiallelic = get_next_truth()
                            imp_key, imp_data, imp_multiallelic, imp_missing_required = get_next_imputed()
                            continue
                if truth_multiallelic or imp_multiallelic:
                    multiallelic_sites += 1
                common_sites_count += 1
                is_input_site = input_sites is not None and (t_chrom, t_pos) in input_sites

                # --- METRICS CALCULATION LOGIC (same as before) ---
                
                # Calculate AF/MAF for stratification from the reference panel.
                maf = None
                maf_bin = None
                site_key = (t_chrom, t_pos)
                while ref_key is not None and ref_key < site_key:
                    try:
                        ref_key, ref_afs = _parse_ref_af_line(next(ref_iter))
                    except StopIteration:
                        ref_key, ref_afs = None, None
                        break

                ref_af_sites += 1
                if ref_key == site_key and ref_afs is not None:
                    ref_ref, ref_alt, ref_af_list = ref_afs
                    maf = _maf_from_afs(ref_af_list)
                    if maf is not None:
                        maf_bin = get_maf_bin(maf)
                    else:
                        ref_af_missing += 1
                else:
                    ref_af_missing += 1
                site_concordant = 0
                site_total = 0
                # Site-level IQS accumulators (canonical probability-based kappa formulation):
                # - truth class counts W_i
                # - imputed class marginals Y_i (expected counts from GP; hard one-hot fallback)
                # - diagonal soft agreement mass for Po
                site_iqs_truth_counts = [0.0, 0.0, 0.0]
                site_iqs_imputed_marginals = [0.0, 0.0, 0.0]
                site_iqs_correct_mass = 0.0
                site_iqs_n = 0

                # Dosage calibration bins (predicted dosage -> mean truth dosage)
                # Bin over [0, 2] since diploid dosage is 0..2.
                ds_calib_bins = 20
                ds_calib_sum_pred = [0.0] * ds_calib_bins
                ds_calib_sum_truth = [0.0] * ds_calib_bins
                ds_calib_count = [0] * ds_calib_bins

                for sample_idx, _sample in enumerate(common_samples_list):
                    t_entry = truth_site[sample_idx] if truth_site is not None else None
                    i_entry = imputed_site[sample_idx] if imputed_site is not None else None
                    t_gt, t_dos, t_phased = (t_entry if t_entry else (None, None, False))
                    i_gt, i_dos, i_phased, i_gp = (i_entry if i_entry else (None, None, False, None))

                    if swapped_site:
                        _, i_gt, i_dos, i_gp = _normalize_imputed_to_truth_alleles(
                            t_ref, t_alt, i_ref, i_alt, i_gt, i_dos, i_gp
                        )

                    if multiallelic_reordered:
                        ok, i_gt, i_dos, i_gp = _normalize_imputed_to_truth_multiallelic(
                            t_ref, t_alt, i_ref, i_alt, i_gt, i_dos, i_gp
                        )
                        if ok:
                            multiallelic_reordered_ok += 1

                    if t_gt is None:
                        missing_truth += 1
                    if i_gt is None:
                        missing_imputed += 1
                    if t_gt is None and i_gt is None:
                        missing_both += 1

                    t_class = gt_class(t_gt)
                    i_class = gt_class(i_gt)

                    if t_class is None or i_dos is None:
                        continue

                    if i_class is None and i_gt is not None:
                        continue

                    # Hellinger score (biallelic only in this harness)
                    if i_gp is not None and t_class is not None and len(i_gp) == 3:
                        t_gp = (1.0, 0.0, 0.0) if t_class == 0 else ((0.0, 1.0, 0.0) if t_class == 1 else (0.0, 0.0, 1.0))
                        bc = sum(math.sqrt(t * i) for t, i in zip(t_gp, i_gp))
                        hellinger_dist = math.sqrt(max(0, 1 - bc))
                        hellinger_sum += hellinger_dist
                        hellinger_count += 1

                    # Dosage calibration bins
                    if i_dos is not None:
                        pred = max(0.0, min(2.0, float(i_dos)))
                        bin_idx = int((pred / 2.0) * ds_calib_bins)
                        if bin_idx >= ds_calib_bins:
                            bin_idx = ds_calib_bins - 1
                        ds_calib_sum_pred[bin_idx] += pred
                        ds_calib_sum_truth[bin_idx] += float(t_dos)
                        ds_calib_count[bin_idx] += 1

                    total_compared += 1
                    site_total += 1

                    # Online R² stats
                    r2_stats["sum_t"] += t_dos
                    r2_stats["sum_i"] += i_dos
                    r2_stats["sum_ti"] += t_dos * i_dos
                    r2_stats["sum_tt"] += t_dos * t_dos
                    r2_stats["sum_ii"] += i_dos * i_dos
                    r2_stats["count"] += 1

                    # SEN score
                    sen = 1.0 - ((t_dos - i_dos) ** 2) / 4.0
                    if sen < 0.0:
                        sen = 0.0
                    elif sen > 1.0:
                        sen = 1.0
                    sen_sum += sen
                    sen_count += 1

                    # MAF bin stats
                    if maf_bin is not None:
                        maf_bins[maf_bin]["sum_t"] += t_dos
                        maf_bins[maf_bin]["sum_i"] += i_dos
                        maf_bins[maf_bin]["sum_ti"] += t_dos * i_dos
                        maf_bins[maf_bin]["sum_tt"] += t_dos * t_dos
                        maf_bins[maf_bin]["sum_ii"] += i_dos * i_dos
                        maf_bins[maf_bin]["total"] += 1

                    # Sample stats
                    sample_total[sample_idx] += 1
                    sample_sum_t[sample_idx] += t_dos
                    sample_sum_i[sample_idx] += i_dos
                    sample_sum_ti[sample_idx] += t_dos * i_dos
                    sample_sum_tt[sample_idx] += t_dos * t_dos
                    sample_sum_ii[sample_idx] += i_dos * i_dos
                    sample_sen_sum[sample_idx] += sen
                    sample_sen_count[sample_idx] += 1
                    if sen < sample_sen_min[sample_idx]:
                        sample_sen_min[sample_idx] = sen
                    if sen > sample_sen_max[sample_idx]:
                        sample_sen_max[sample_idx] = sen

                    if i_class is not None:
                        confusion[t_class][i_class] += 1
                        if maf_bin is not None:
                            maf_bins[maf_bin]["confusion"][t_class][i_class] += 1

                    # Canonical IQS (kappa): use GP probabilities when available, otherwise hard-call one-hot.
                    iqs_probs = None
                    if i_gp is not None and len(i_gp) == 3:
                        try:
                            p0 = float(i_gp[0])
                            p1 = float(i_gp[1])
                            p2 = float(i_gp[2])
                            p_sum = p0 + p1 + p2
                            if p_sum > 0.0:
                                iqs_probs = (p0 / p_sum, p1 / p_sum, p2 / p_sum)
                        except Exception:
                            iqs_probs = None
                    elif i_class is not None:
                        iqs_probs = (1.0, 0.0, 0.0) if i_class == 0 else ((0.0, 1.0, 0.0) if i_class == 1 else (0.0, 0.0, 1.0))

                    if iqs_probs is not None:
                        site_iqs_n += 1
                        site_iqs_truth_counts[t_class] += 1.0
                        site_iqs_imputed_marginals[0] += iqs_probs[0]
                        site_iqs_imputed_marginals[1] += iqs_probs[1]
                        site_iqs_imputed_marginals[2] += iqs_probs[2]
                        site_iqs_correct_mass += iqs_probs[t_class]

                    # Concordance
                    if i_class is not None:
                        t_sorted = tuple(sorted(t_gt))
                        i_sorted = tuple(sorted(i_gt))
                        if t_sorted == i_sorted:
                            unphased_concordant += 1
                            site_concordant += 1
                            if maf_bin is not None:
                                maf_bins[maf_bin]["unphased_concordant"] += 1
                            sample_concordant[sample_idx] += 1

                    # Non-ref concordance
                    if t_class > 0 and i_class is not None:
                        nonref_total += 1
                        if maf_bin is not None:
                            maf_bins[maf_bin]["nonref_total"] += 1
                        if t_sorted == i_sorted:
                            nonref_concordant += 1
                            if maf_bin is not None:
                                maf_bins[maf_bin]["nonref_concordant"] += 1

                    # Switch errors
                    if t_class == 1 and i_class == 1 and t_phased and i_phased:
                        # Phase concordance should be invariant to global haplotype labeling.
                        if sample_phase_flip[sample_idx] is None:
                            sample_phase_flip[sample_idx] = (i_gt != t_gt)
                        flip = sample_phase_flip[sample_idx] or False
                        phase_match = (i_gt != t_gt) if flip else (i_gt == t_gt)
                        phase_total += 1
                        sample_phase_total[sample_idx] += 1
                        if phase_match:
                            phase_concordant += 1
                            sample_phase_concordant[sample_idx] += 1
                        if maf_bin is not None:
                            maf_bins[maf_bin]["phase_total"] += 1
                            if phase_match:
                                maf_bins[maf_bin]["phase_concordant"] += 1

                        sample_name = common_samples_list[sample_idx]
                        pos = site[1]
                        if current_block_start[sample_idx] < 0:
                            current_block_start[sample_idx] = pos

                        if prev_het[sample_idx] is not None:
                            prev_site, prev_t_gt, prev_i_gt, prev_maf_bin = prev_het[sample_idx]
                            # Phase consistency should be based on cis/trans (ref vs non-ref),
                            # not raw allele identity. This avoids false switches at multiallelic
                            # hets like 0|2 vs 0|1.
                            t_same_phase = ((t_gt[0] != 0) == (prev_t_gt[0] != 0))
                            i_same_phase = ((i_gt[0] != 0) == (prev_i_gt[0] != 0))

                            if t_same_phase != i_same_phase:
                                block_len = pos - current_block_start[sample_idx]
                                phase_blocks[sample_name].append(block_len)
                                current_block_start[sample_idx] = pos

                                switch_errors += 1
                                sample_switch_errors[sample_idx] += 1
                                if maf_bin is not None:
                                    maf_bins[maf_bin]["switch_errors"] += 1

                            switch_opportunities += 1
                            sample_switch_opportunities[sample_idx] += 1
                            if maf_bin is not None:
                                maf_bins[maf_bin]["switch_opportunities"] += 1
                        prev_het[sample_idx] = (site, t_gt, i_gt, maf_bin)
                        last_het_pos[sample_idx] = pos
                        if is_input_site:
                            if prev_het_input[sample_idx] is not None:
                                _, prev_t_gt, prev_i_gt, _ = prev_het_input[sample_idx]
                                t_same_phase = ((t_gt[0] != 0) == (prev_t_gt[0] != 0))
                                i_same_phase = ((i_gt[0] != 0) == (prev_i_gt[0] != 0))
                                if t_same_phase != i_same_phase:
                                    switch_errors_input += 1
                                switch_opportunities_input += 1
                            prev_het_input[sample_idx] = (site, t_gt, i_gt, maf_bin)

                    # Masked-snp metrics (proxy quality)
                    if maf_bin is not None and is_input_site and mask_pick(t_chrom, t_pos, maf_bin):
                        if i_class is not None:
                            masked_total += 1
                            masked_stats["sum_t"] += t_dos
                            masked_stats["sum_i"] += i_dos
                            masked_stats["sum_ti"] += t_dos * i_dos
                            masked_stats["sum_tt"] += t_dos * t_dos
                            masked_stats["sum_ii"] += i_dos * i_dos
                            masked_stats["count"] += 1
                            masked_maf_bins[maf_bin]["sum_t"] += t_dos
                            masked_maf_bins[maf_bin]["sum_i"] += i_dos
                            masked_maf_bins[maf_bin]["sum_ti"] += t_dos * i_dos
                            masked_maf_bins[maf_bin]["sum_tt"] += t_dos * t_dos
                            masked_maf_bins[maf_bin]["sum_ii"] += i_dos * i_dos
                            masked_maf_bins[maf_bin]["count"] += 1
                            if t_sorted == i_sorted:
                                masked_concordant += 1
                            if t_class > 0:
                                masked_nonref_total += 1
                                masked_maf_bins[maf_bin]["nonref_total"] += 1
                                if t_sorted == i_sorted:
                                    masked_nonref_concordant += 1
                                    masked_maf_bins[maf_bin]["nonref_concordant"] += 1
                        if i_gp is not None and i_class is not None:
                            y = [0.0, 0.0, 0.0]
                            y[t_class] = 1.0
                            brier = sum((p - yk) ** 2 for p, yk in zip(i_gp, y))
                            masked_brier_sum += brier
                            masked_brier_n += 1
                            conf = max(i_gp)
                            acc = 1.0 if i_class == t_class else 0.0
                            bin_idx = min(int(conf * len(ece_bins)), len(ece_bins) - 1)
                            ece_bins[bin_idx]["sum_conf"] += conf
                            ece_bins[bin_idx]["sum_acc"] += acc
                            ece_bins[bin_idx]["count"] += 1

                # IQS Calculation
                if site_iqs_n > 0:
                    observed_conc = site_iqs_correct_mass / site_iqs_n
                    expected_conc = 0.0
                    denom_n2 = float(site_iqs_n * site_iqs_n)
                    for cls in range(3):
                        expected_conc += (site_iqs_truth_counts[cls] * site_iqs_imputed_marginals[cls]) / denom_n2
                    if expected_conc < 1.0:
                        iqs = (observed_conc - expected_conc) / (1.0 - expected_conc)
                        site_iqs_sum += iqs
                        site_iqs_count += 1
                        site_iqs_weighted_sum += iqs * site_iqs_n
                        site_iqs_weighted_count += site_iqs_n
                        if maf_bin is not None:
                            maf_bins[maf_bin]["iqs_sum"] += iqs
                            maf_bins[maf_bin]["iqs_count"] += 1

                # Advance both
                truth_key, truth_data, truth_multiallelic = get_next_truth()
                imp_key, imp_data, imp_multiallelic, imp_missing_required = get_next_imputed()
            
            elif t_pos < i_pos:
                # Truth is behind, means site missing in imputation (or extra site in Truth)
                truth_only_sites += 1
                truth_key, truth_data, truth_multiallelic = get_next_truth()
            else:
                # Imputed is behind, means extra site in Imputation
                if imp_missing_required:
                    skipped_missing_required += 1
                    skipped_missing_required_imputed_only += 1
                else:
                    imputed_only_sites += 1
                imp_key, imp_data, imp_multiallelic, imp_missing_required = get_next_imputed()
        
        elif t_chrom < i_chrom:
             truth_only_sites += 1
             truth_key, truth_data, truth_multiallelic = get_next_truth()
        else:
             if imp_missing_required:
                 skipped_missing_required += 1
                 skipped_missing_required_imputed_only += 1
             else:
                 imputed_only_sites += 1
             imp_key, imp_data, imp_multiallelic, imp_missing_required = get_next_imputed()

    print(f"Common sites: {common_sites_count}")
    print(f"Skipped sites (missing DS/GP): {skipped_missing_required}")
    print(f"  - Matched sites skipped: {skipped_missing_required_matched}")
    print(f"  - Imputed-only sites skipped: {skipped_missing_required_imputed_only}")
    loop_elapsed = time.time() - loop_start

    # Close final phase blocks
    for i, start_pos in enumerate(current_block_start):
        if start_pos < 0:
            continue
        end_pos = last_het_pos[i]
        if end_pos >= start_pos:
            block_len = end_pos - start_pos
            phase_blocks[common_samples_list[i]].append(block_len)

    # Calculate overall metrics
    metrics = {}
    
    # Calculate N50 Phase Block Length
    all_lengths = []
    for lengths in phase_blocks.values():
        all_lengths.extend(lengths)

    def _quantile(sorted_vals, q):
        if not sorted_vals:
            return None
        if len(sorted_vals) == 1:
            return sorted_vals[0]
        pos = q * (len(sorted_vals) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return sorted_vals[lo]
        frac = pos - lo
        return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac

    if all_lengths:
        all_lengths.sort(reverse=True)
        total_len = sum(all_lengths)
        target = total_len / 2
        running_sum = 0
        n50 = 0
        for l in all_lengths:
            running_sum += l
            if running_sum >= target:
                n50 = l
                break
        metrics["n50_phase_block"] = n50
        lengths_sorted = sorted(all_lengths)
        metrics["phase_block_count"] = len(all_lengths)
        metrics["phase_block_len_mean"] = sum(all_lengths) / len(all_lengths)
        metrics["phase_block_len_min"] = min(all_lengths)
        metrics["phase_block_len_max"] = max(all_lengths)
        metrics["phase_block_len_median"] = _quantile(lengths_sorted, 0.5)
        metrics["phase_block_len_p10"] = _quantile(lengths_sorted, 0.10)
        metrics["phase_block_len_p90"] = _quantile(lengths_sorted, 0.90)
    else:
        metrics["n50_phase_block"] = 0.0
        metrics["phase_block_count"] = 0
        metrics["phase_block_len_mean"] = 0.0
        metrics["phase_block_len_min"] = 0.0
        metrics["phase_block_len_max"] = 0.0
        metrics["phase_block_len_median"] = 0.0
        metrics["phase_block_len_p10"] = 0.0
        metrics["phase_block_len_p90"] = 0.0

    # Per-sample phase block stats
    sample_block_counts = []
    sample_block_means = []
    sample_block_medians = []
    sample_block_mins = []
    sample_block_maxs = []
    sample_block_n50s = []
    for sample_name in common_samples_list:
        lengths = phase_blocks.get(sample_name, [])
        if not lengths:
            continue
        lengths_sorted = sorted(lengths)
        sample_block_counts.append(len(lengths))
        sample_block_means.append(sum(lengths) / len(lengths))
        sample_block_medians.append(_quantile(lengths_sorted, 0.5))
        sample_block_mins.append(min(lengths))
        sample_block_maxs.append(max(lengths))
        # N50 per sample
        lengths_desc = sorted(lengths, reverse=True)
        total_len = sum(lengths_desc)
        target = total_len / 2
        running_sum = 0
        n50 = 0
        for l in lengths_desc:
            running_sum += l
            if running_sum >= target:
                n50 = l
                break
        sample_block_n50s.append(n50)

    if sample_block_counts:
        metrics["sample_phase_block_count_mean"] = sum(sample_block_counts) / len(sample_block_counts)
        metrics["sample_phase_block_count_min"] = min(sample_block_counts)
        metrics["sample_phase_block_count_max"] = max(sample_block_counts)
    if sample_block_means:
        metrics["sample_phase_block_len_mean"] = sum(sample_block_means) / len(sample_block_means)
        metrics["sample_phase_block_len_min"] = min(sample_block_mins)
        metrics["sample_phase_block_len_max"] = max(sample_block_maxs)
    if sample_block_medians:
        metrics["sample_phase_block_len_median_mean"] = sum(sample_block_medians) / len(sample_block_medians)
    if sample_block_n50s:
        metrics["sample_phase_block_n50_mean"] = sum(sample_block_n50s) / len(sample_block_n50s)
        metrics["sample_phase_block_n50_min"] = min(sample_block_n50s)
        metrics["sample_phase_block_n50_max"] = max(sample_block_n50s)

    # Precision/Recall/F1 (Binary classification: Ref vs Non-Ref)
    # TP: Truth=Alt, Imputed=Alt
    # FP: Truth=Ref, Imputed=Alt
    # FN: Truth=Alt, Imputed=Ref
    # TN: Truth=Ref, Imputed=Ref
    
    tp = confusion[1][1] + confusion[1][2] + confusion[2][1] + confusion[2][2]
    fp = confusion[0][1] + confusion[0][2]
    fn = confusion[1][0] + confusion[2][0]
    
    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["fn"] = fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics["precision"] = precision
    metrics["recall"] = recall
    metrics["f1_score"] = f1
    metrics["unphased_concordant_count"] = unphased_concordant
    metrics["nonref_concordant_count"] = nonref_concordant
    metrics["missing_truth"] = missing_truth
    metrics["missing_imputed"] = missing_imputed
    metrics["missing_both"] = missing_both
    metrics["ref_alt_mismatch"] = ref_alt_mismatch
    metrics["ref_alt_swapped"] = ref_alt_swapped
    metrics["multiallelic_sites"] = multiallelic_sites

    if masked_total > 0:
        metrics["masked_total"] = masked_total
        metrics["masked_concordance"] = masked_concordant / masked_total
        if masked_nonref_total > 0:
            metrics["masked_nonref_concordance"] = masked_nonref_concordant / masked_nonref_total
            metrics["masked_nonref_total"] = masked_nonref_total
        if masked_stats["count"] > 1:
            n = masked_stats["count"]
            mean_t = masked_stats["sum_t"] / n
            mean_i = masked_stats["sum_i"] / n
            cov = masked_stats["sum_ti"] / n - mean_t * mean_i
            var_t = masked_stats["sum_tt"] / n - mean_t * mean_t
            var_i = masked_stats["sum_ii"] / n - mean_i * mean_i
            if var_t > 0 and var_i > 0:
                r = cov / math.sqrt(var_t * var_i)
                metrics["masked_r_squared"] = r ** 2
        if masked_brier_n > 0:
            metrics["masked_brier"] = masked_brier_sum / masked_brier_n
            total_ece = 0.0
            total_count = 0
            for b in ece_bins:
                if b["count"] > 0:
                    acc = b["sum_acc"] / b["count"]
                    conf = b["sum_conf"] / b["count"]
                    total_ece += abs(acc - conf) * b["count"]
                    total_count += b["count"]
            metrics["masked_ece"] = total_ece / total_count if total_count > 0 else None

        metrics["masked_by_maf"] = {}
        for maf_bin, data in sorted(masked_maf_bins.items()):
            if data["count"] > 1:
                n = data["count"]
                mean_t = data["sum_t"] / n
                mean_i = data["sum_i"] / n
                cov = data["sum_ti"] / n - mean_t * mean_i
                var_t = data["sum_tt"] / n - mean_t * mean_t
                var_i = data["sum_ii"] / n - mean_i * mean_i
                r2 = None
                if var_t > 0 and var_i > 0:
                    r = cov / math.sqrt(var_t * var_i)
                    r2 = r ** 2
                mb = {"r_squared": r2, "n": n}
                if data["nonref_total"] > 0:
                    mb["nonref_concordance"] = data["nonref_concordant"] / data["nonref_total"]
                    mb["nonref_total"] = data["nonref_total"]
                metrics["masked_by_maf"][maf_bin] = mb

    if total_compared > 0:
        metrics["unphased_concordance"] = unphased_concordant / total_compared
        metrics["total_genotypes"] = total_compared
        metrics["sites_compared"] = common_sites_count
        
        # Non-reference concordance
        if nonref_total > 0:
            metrics["nonref_concordance"] = nonref_concordant / nonref_total
            metrics["nonref_total"] = nonref_total
        
        # Switch error rate
        if switch_opportunities > 0:
            metrics["switch_error_rate"] = switch_errors / switch_opportunities
            metrics["switch_errors"] = switch_errors
            metrics["switch_opportunities"] = switch_opportunities
        if switch_opportunities_input > 0:
            metrics["switch_error_rate_input_sites"] = switch_errors_input / switch_opportunities_input
            metrics["switch_errors_input_sites"] = switch_errors_input
            metrics["switch_opportunities_input_sites"] = switch_opportunities_input
        if phase_total > 0:
            metrics["phase_concordance"] = phase_concordant / phase_total
            metrics["phase_concordant"] = phase_concordant
            metrics["phase_total"] = phase_total
            metrics["phase_flip_rate"] = 1.0 - metrics["phase_concordance"]
        
        # Confusion matrix
        metrics["confusion_matrix"] = confusion
        
        # Per-class accuracy
        for cls, name in [(0, "homref"), (1, "het"), (2, "homalt")]:
            row_total = sum(confusion[cls])
            if row_total > 0:
                metrics[f"{name}_accuracy"] = confusion[cls][cls] / row_total
                metrics[f"{name}_total"] = row_total

        # Calculate R² from online statistics
        n = r2_stats["count"]
        if n > 1:
            sum_t = r2_stats["sum_t"]
            sum_i = r2_stats["sum_i"]
            sum_ti = r2_stats["sum_ti"]
            sum_tt = r2_stats["sum_tt"]
            sum_ii = r2_stats["sum_ii"]

            mean_t = sum_t / n
            mean_i = sum_i / n

            # Cov = E[XY] - E[X]E[Y] = sum_ti/n - mean_t*mean_i
            # Var = E[X²] - E[X]² = sum_tt/n - mean_t²
            cov = sum_ti / n - mean_t * mean_i
            var_t = sum_tt / n - mean_t * mean_t
            var_i = sum_ii / n - mean_i * mean_i

            # Store sufficient statistics for exact global aggregation
            metrics["r2_stats"] = r2_stats

            if var_t > 0 and var_i > 0:
                r = cov / math.sqrt(var_t * var_i)
                metrics["r_squared"] = r ** 2
            else:
                metrics["r_squared"] = None
                
            # INFO score approximation (variance ratio)
            if var_t > 0:
                metrics["info_score_approx"] = var_i / var_t
        else:
            metrics["r_squared"] = None

        # Hellinger Score (if GP field was available)
    if hellinger_count > 0:
        metrics["hellinger_score"] = hellinger_sum / hellinger_count
        metrics["hellinger_n"] = hellinger_count

        # Calculate Rare Variant R² stats (MAF < 1%) from online stats
        rare_stats = {
            "sum_t": 0.0, "sum_i": 0.0, "sum_ti": 0.0,
            "sum_tt": 0.0, "sum_ii": 0.0, "count": 0
        }

        target_bins = ["ultra-rare (<0.1%)", "very-rare (0.1-0.5%)", "rare (0.5-1%)"]

        for bin_name in target_bins:
            if bin_name in maf_bins:
                b_data = maf_bins[bin_name]
                rare_stats["sum_t"] += b_data["sum_t"]
                rare_stats["sum_i"] += b_data["sum_i"]
                rare_stats["sum_ti"] += b_data["sum_ti"]
                rare_stats["sum_tt"] += b_data["sum_tt"]
                rare_stats["sum_ii"] += b_data["sum_ii"]
                rare_stats["count"] += b_data["total"]

        metrics["rare_r2_stats"] = rare_stats

        # Calculate overall IQS (mean across sites)
        if site_iqs_count > 0:
            metrics["iqs"] = site_iqs_sum / site_iqs_count
            metrics["iqs_weighted"] = (
                site_iqs_weighted_sum / site_iqs_weighted_count
                if site_iqs_weighted_count > 0 else None
            )
            metrics["iqs_median"] = None
        else:
            metrics["iqs"] = None
            metrics["iqs_weighted"] = None

        if sen_count > 0:
            metrics["sen_mean"] = sen_sum / sen_count
        else:
            metrics["sen_mean"] = None
            
        sample_concordances = []
        sample_r2s = []
        sample_sen_means = []
        sample_sen_mins = []
        sample_sen_maxs = []
        sample_phase_concordances = []

        for i in range(n_common):
            n = sample_total[i]
            if n > 0:
                sample_concordances.append(sample_concordant[i] / n)
            if n > 1:
                mean_t = sample_sum_t[i] / n
                mean_i = sample_sum_i[i] / n
                cov = sample_sum_ti[i] / n - mean_t * mean_i
                var_t = sample_sum_tt[i] / n - mean_t * mean_t
                var_i = sample_sum_ii[i] / n - mean_i * mean_i
                if var_t > 0 and var_i > 0:
                    r = cov / math.sqrt(var_t * var_i)
                    sample_r2s.append(r ** 2)
            if sample_sen_count[i] > 0:
                sample_sen_means.append(sample_sen_sum[i] / sample_sen_count[i])
                sample_sen_mins.append(sample_sen_min[i])
                sample_sen_maxs.append(sample_sen_max[i])
            if sample_phase_total[i] > 0:
                sample_phase_concordances.append(sample_phase_concordant[i] / sample_phase_total[i])
        
        if sample_concordances:
            metrics["sample_concordance_mean"] = sum(sample_concordances) / len(sample_concordances)
            metrics["sample_concordance_min"] = min(sample_concordances)
            metrics["sample_concordance_max"] = max(sample_concordances)
        if sample_r2s:
            metrics["sample_r2_mean"] = sum(sample_r2s) / len(sample_r2s)
            metrics["sample_r2_min"] = min(sample_r2s)
        if sample_sen_means:
            metrics["sample_sen_mean"] = sum(sample_sen_means) / len(sample_sen_means)
            metrics["sample_sen_median"] = None
            metrics["sample_sen_min"] = min(sample_sen_mins)
            metrics["sample_sen_max"] = max(sample_sen_maxs)
        if sample_phase_concordances:
            metrics["sample_phase_concordance_mean"] = sum(sample_phase_concordances) / len(sample_phase_concordances)
            metrics["sample_phase_concordance_min"] = min(sample_phase_concordances)
            metrics["sample_phase_concordance_max"] = max(sample_phase_concordances)

        # Per-MAF bin metrics
        metrics["by_maf"] = {}
        for maf_bin, data in sorted(maf_bins.items()):
            if data["total"] > 0:
                bin_metrics = {
                    "unphased_concordance": data["unphased_concordant"] / data["total"],
                    "n_genotypes": data["total"]
                }
                # Non-ref concordance per bin
                if data["nonref_total"] > 0:
                    bin_metrics["nonref_concordance"] = data["nonref_concordant"] / data["nonref_total"]
                
                # F1/Precision/Recall per bin
                b_conf = data["confusion"]
                b_tp = b_conf[1][1] + b_conf[1][2] + b_conf[2][1] + b_conf[2][2]
                b_fp = b_conf[0][1] + b_conf[0][2]
                b_fn = b_conf[1][0] + b_conf[2][0]
                
                b_prec = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
                b_rec = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
                b_f1 = 2 * b_prec * b_rec / (b_prec + b_rec) if (b_prec + b_rec) > 0 else 0.0
                
                bin_metrics["f1_score"] = b_f1
                bin_metrics["recall"] = b_rec
                
                # R² per bin (from online stats)
                n_bin = data["total"]
                if n_bin > 1:
                    mean_t = data["sum_t"] / n_bin
                    mean_i = data["sum_i"] / n_bin
                    cov = data["sum_ti"] / n_bin - mean_t * mean_i
                    var_t = data["sum_tt"] / n_bin - mean_t * mean_t
                    var_i = data["sum_ii"] / n_bin - mean_i * mean_i
                    if var_t > 0 and var_i > 0:
                        r = cov / math.sqrt(var_t * var_i)
                        bin_metrics["r_squared"] = r ** 2
                # IQS per bin
                if data["iqs_count"] > 0:
                    bin_metrics["iqs"] = data["iqs_sum"] / data["iqs_count"]
                # Switch error rate per bin
                if data["switch_opportunities"] > 0:
                    bin_metrics["switch_error_rate"] = data["switch_errors"] / data["switch_opportunities"]
                    bin_metrics["switch_errors"] = data["switch_errors"]
                    bin_metrics["switch_opportunities"] = data["switch_opportunities"]
                if data["phase_total"] > 0:
                    bin_metrics["phase_concordance"] = data["phase_concordant"] / data["phase_total"]
                    bin_metrics["phase_concordant"] = data["phase_concordant"]
                    bin_metrics["phase_total"] = data["phase_total"]

                # Sufficient stats for genome-wide MAF bin aggregation
                bin_metrics["agg_stats"] = {
                    "sum_t": data["sum_t"],
                    "sum_i": data["sum_i"],
                    "sum_ti": data["sum_ti"],
                    "sum_tt": data["sum_tt"],
                    "sum_ii": data["sum_ii"],
                    "count": data["total"],
                    "concordant": data["unphased_concordant"],
                    "nonref_concordant": data["nonref_concordant"],
                    "nonref_total": data["nonref_total"],
                    "switch_err": data["switch_errors"],
                    "switch_opp": data["switch_opportunities"],
                    "phase_concordant": data["phase_concordant"],
                    "phase_total": data["phase_total"],
                    "tp": b_tp, "fp": b_fp, "fn": b_fn
                }

                metrics["by_maf"][maf_bin] = bin_metrics
        
        # Per-sample switch error summary
        sample_switch_rates = []
        for i in range(n_common):
            if sample_switch_opportunities[i] > 0:
                sample_switch_rates.append(
                    sample_switch_errors[i] / sample_switch_opportunities[i]
                )
        if sample_switch_rates:
            metrics["sample_switch_error_mean"] = sum(sample_switch_rates) / len(sample_switch_rates)
            metrics["sample_switch_error_max"] = max(sample_switch_rates)
            metrics["sample_switch_error_min"] = min(sample_switch_rates)

    elapsed = time.time() - start_time
    metrics["calculation_time_sec"] = elapsed
    metrics["diagnostic_time_sec"] = diag_elapsed
    metrics["af_time_sec"] = af_elapsed
    metrics["metrics_loop_time_sec"] = loop_elapsed
    metrics["sites_truth_total"] = total_truth_sites
    metrics["sites_imputed_total"] = total_imputed_sites
    metrics["sites_truth_only"] = truth_only_sites
    metrics["sites_imputed_only"] = imputed_only_sites
    metrics["sites_common"] = common_sites_count
    metrics["sites_skipped_missing_required"] = skipped_missing_required
    metrics["sites_skipped_missing_required_matched"] = skipped_missing_required_matched
    metrics["sites_skipped_missing_required_imputed_only"] = skipped_missing_required_imputed_only
    metrics["degraded_missing_ds_gp"] = degraded_required
    metrics["ref_af_missing_sites"] = ref_af_missing
    metrics["ref_af_sites"] = ref_af_sites

    # Dosage calibration summary (predicted dosage bins)
    ds_calibration = []
    for i in range(ds_calib_bins):
        count = ds_calib_count[i]
        if count > 0:
            mean_pred = ds_calib_sum_pred[i] / count
            mean_truth = ds_calib_sum_truth[i] / count
        else:
            mean_pred = None
            mean_truth = None
        bin_lo = (i / ds_calib_bins) * 2.0
        bin_hi = ((i + 1) / ds_calib_bins) * 2.0
        ds_calibration.append({
            "bin": i,
            "bin_lo": bin_lo,
            "bin_hi": bin_hi,
            "mean_pred": mean_pred,
            "mean_truth": mean_truth,
            "count": count,
        })
    metrics["ds_calibration"] = ds_calibration

    # Save metrics to JSON for exact aggregation
    with open(output_prefix + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print results
    print("\n" + "=" * 60)
    print("IMPUTATION METRICS - COMPREHENSIVE ANALYSIS")
    print("=" * 60)

    if metrics:
        print(f"\n📊 OVERALL STATISTICS")
        sites = metrics.get('sites_compared')
        genotypes = metrics.get('total_genotypes')
        print(f"   Sites compared: {sites:,}" if sites is not None else "   Sites compared: N/A")
        print(f"   Genotypes compared: {genotypes:,}" if genotypes is not None else "   Genotypes compared: N/A")
        print(f"   Calculation time: {metrics.get('calculation_time_sec', 0):.1f}s")
        if metrics.get("diagnostic_time_sec") is not None:
            print(f"     Diagnostics:  {metrics.get('diagnostic_time_sec', 0):.1f}s")
        if metrics.get("af_time_sec") is not None:
            print(f"     AF extraction:{metrics.get('af_time_sec', 0):.1f}s")
        if metrics.get("metrics_loop_time_sec") is not None:
            print(f"     Metrics loop:{metrics.get('metrics_loop_time_sec', 0):.1f}s")
        
        print(f"\n🎯 ACCURACY METRICS")
        print(f"   Unphased concordance: {metrics.get('unphased_concordance', 0):.4f}")
        print(f"   Non-ref concordance:  {metrics.get('nonref_concordance', 0):.4f}" if metrics.get('nonref_concordance') else "   Non-ref concordance:  N/A")
        print(f"   F1 Score (Non-Ref):   {metrics.get('f1_score', 0):.4f}")
        print(f"   Precision / Recall:   {metrics.get('precision', 0):.4f} / {metrics.get('recall', 0):.4f}")
        print(f"   Overall R²:           {metrics.get('r_squared'):.4f}" if metrics.get('r_squared') else "   Overall R²:           N/A")
        print(f"   Overall IQS:          {metrics.get('iqs'):.4f}" if metrics.get('iqs') else "   Overall IQS:          N/A")
        print(f"   SEN (mean):           {metrics.get('sen_mean'):.4f}" if metrics.get('sen_mean') is not None else "   SEN (mean):           N/A")
        print(f"   Hellinger Score:      {metrics.get('hellinger_score'):.4f}" if metrics.get('hellinger_score') else "   Hellinger Score:      N/A (no GP)")
        print(f"   INFO score (approx):  {metrics.get('info_score_approx'):.4f}" if metrics.get('info_score_approx') else "   INFO score (approx):  N/A")
        
        if metrics.get('switch_error_rate') is not None:
            print(f"\n🔀 PHASING QUALITY")
            print(f"   Switch error rate:    {metrics.get('switch_error_rate'):.4f} ({metrics.get('switch_errors')}/{metrics.get('switch_opportunities')})")
            print(f"   N50 Phase Block:      {metrics.get('n50_phase_block'):.0f} bp")
            if metrics.get("phase_concordance") is not None:
                print(f"   Phase concordance:    {metrics.get('phase_concordance'):.4f}")
                print(f"   Phase flip rate:      {metrics.get('phase_flip_rate'):.4f}")
            if metrics.get('switch_error_rate_input_sites') is not None:
                print(f"   Switch error (input): {metrics.get('switch_error_rate_input_sites'):.4f} ({metrics.get('switch_errors_input_sites')}/{metrics.get('switch_opportunities_input_sites')})")
            if metrics.get("phase_block_len_mean") is not None:
                print(f"   Phase block length:   mean={metrics['phase_block_len_mean']:.0f} bp, median={metrics['phase_block_len_median']:.0f} bp")
                print(f"   Phase block length:   p10={metrics['phase_block_len_p10']:.0f} bp, p90={metrics['phase_block_len_p90']:.0f} bp")
                print(f"   Phase block count:    {metrics.get('phase_block_count', 0)}")
        
        print(f"\n📋 CONFUSION MATRIX (Truth vs Imputed)")
        print(f"   {'':12} {'HomRef':>10} {'Het':>10} {'HomAlt':>10}")
        labels = ['HomRef', 'Het', 'HomAlt']
        confusion_matrix = metrics.get('confusion_matrix', [[0,0,0],[0,0,0],[0,0,0]])
        for i, label in enumerate(labels):
            row = confusion_matrix[i]
            print(f"   {label:12} {row[0]:>10,} {row[1]:>10,} {row[2]:>10,}")
        
        # Add diagnostic warnings
        print(f"\n⚠️  DIAGNOSTIC WARNINGS")
        homref_count = confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[0][2]
        if homref_count == 0:
            print(f"   NOTE: Truth has ZERO HomRef genotypes (expected for variant-only VCFs)")
            print(f"   → Metrics are calculated ONLY at variant sites")
            print(f"   → Concordance may be inflated/deflated vs array-site evaluation")
        
        missing_truth = metrics.get('missing_truth', 0)
        missing_imputed = metrics.get('missing_imputed', 0)
        if missing_truth > 0 or missing_imputed > 0:
            print(f"   Missing genotypes: Truth={missing_truth:,}, Imputed={missing_imputed:,}")

        total_truth_sites = metrics.get('sites_truth_total')
        total_imputed_sites = metrics.get('sites_imputed_total')
        common_sites = metrics.get('sites_common', 0)
        if total_truth_sites:
            pct = (common_sites / total_truth_sites) * 100.0
            print(f"   Site overlap: {common_sites:,}/{total_truth_sites:,} truth sites ({pct:.2f}%)")
        if total_imputed_sites:
            pct = (common_sites / total_imputed_sites) * 100.0
            print(f"   Site overlap: {common_sites:,}/{total_imputed_sites:,} imputed sites ({pct:.2f}%)")
        
        skipped_missing_required = metrics.get('sites_skipped_missing_required', 0)
        if skipped_missing_required > 0:
            print(f"   Skipped sites (missing DS/GP): {skipped_missing_required:,}")
            skipped_matched = metrics.get('sites_skipped_missing_required_matched', 0)
            skipped_imputed_only = metrics.get('sites_skipped_missing_required_imputed_only', 0)
            print(f"     - Matched sites skipped: {skipped_matched:,}")
            print(f"     - Imputed-only sites skipped: {skipped_imputed_only:,}")
        
        ref_alt_mismatch = metrics.get('ref_alt_mismatch', 0)
        ref_alt_swapped = metrics.get('ref_alt_swapped', 0)
        if ref_alt_mismatch > 0:
            print(f"   REF/ALT mismatches: {ref_alt_mismatch:,} sites (coordinate/strand issue?)")
        if ref_alt_swapped > 0:
            print(f"   REF/ALT swaps normalized: {ref_alt_swapped:,} sites")

        ref_af_missing_sites = metrics.get('ref_af_missing_sites', 0)
        ref_af_sites = metrics.get('ref_af_sites', 0)
        if ref_af_missing_sites > 0 and ref_af_sites > 0:
            pct = (ref_af_missing_sites / ref_af_sites) * 100.0
            print(f"   Missing ref AF/MAF: {ref_af_missing_sites:,} / {ref_af_sites:,} sites ({pct:.2f}%)")
        
        multiallelic = metrics.get('multiallelic_sites', 0)
        if multiallelic > 0:
            print(f"   Multiallelic sites skipped: {multiallelic:,}")
        
        print(f"\n📊 PER-CLASS ACCURACY")
        for cls in ['homref', 'het', 'homalt']:
            acc = metrics.get(f'{cls}_accuracy')
            total = metrics.get(f'{cls}_total', 0)
            if acc is not None:
                print(f"   {cls.upper():12} {acc:.4f} (n={total:,})")
        
        print(f"\n👥 PER-SAMPLE STATISTICS")
        if metrics.get('sample_concordance_mean'):
            print(f"   Concordance: mean={metrics['sample_concordance_mean']:.4f}, min={metrics['sample_concordance_min']:.4f}, max={metrics['sample_concordance_max']:.4f}")
        if metrics.get('sample_r2_mean'):
            print(f"   R²:          mean={metrics['sample_r2_mean']:.4f}, min={metrics['sample_r2_min']:.4f}")
        if metrics.get('sample_sen_mean') is not None:
            print(f"   SEN:         mean={metrics['sample_sen_mean']:.4f}, min={metrics['sample_sen_min']:.4f}, max={metrics['sample_sen_max']:.4f}")
        if metrics.get('sample_switch_error_mean') is not None:
            print(f"   Switch Err:  mean={metrics['sample_switch_error_mean']:.4f}, min={metrics['sample_switch_error_min']:.4f}, max={metrics['sample_switch_error_max']:.4f}")
        if metrics.get('sample_phase_concordance_mean') is not None:
            print(f"   Phase conc:  mean={metrics['sample_phase_concordance_mean']:.4f}, min={metrics['sample_phase_concordance_min']:.4f}, max={metrics['sample_phase_concordance_max']:.4f}")
        if metrics.get('sample_phase_block_len_mean') is not None:
            print(f"   Block len:   mean={metrics['sample_phase_block_len_mean']:.0f} bp, min={metrics['sample_phase_block_len_min']:.0f}, max={metrics['sample_phase_block_len_max']:.0f}")
        if metrics.get('sample_phase_block_n50_mean') is not None:
            print(f"   Block N50:   mean={metrics['sample_phase_block_n50_mean']:.0f} bp, min={metrics['sample_phase_block_n50_min']:.0f}, max={metrics['sample_phase_block_n50_max']:.0f}")
        if metrics.get('sample_phase_block_count_mean') is not None:
            print(f"   Block cnt:   mean={metrics['sample_phase_block_count_mean']:.1f}, min={metrics['sample_phase_block_count_min']:.0f}, max={metrics['sample_phase_block_count_max']:.0f}")

        if "by_maf" in metrics:
            print(f"\n📈 BY MAF BIN (sorted by frequency)")
            print(f"   {'MAF Bin':<20} {'F1':>8} {'Conc':>8} {'R²':>8} {'SwitchErr':>10} {'PhaseConc':>10} {'N':>10}")
            print(f"   {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
            # Sort by actual frequency order
            bin_order = ["ultra-rare (<0.1%)", "very-rare (0.1-0.5%)", "rare (0.5-1%)", 
                        "low-freq (1-5%)", "medium (5-20%)", "common (>20%)"]
            for maf_bin in bin_order:
                if maf_bin in metrics["by_maf"]:
                    bin_metrics = metrics["by_maf"][maf_bin]
                    f1_str = f"{bin_metrics.get('f1_score', 0):.4f}"
                    conc = f"{bin_metrics['unphased_concordance']:.4f}"
                    r2_str = f"{bin_metrics.get('r_squared'):.4f}" if bin_metrics.get('r_squared') else "N/A"
                    switch_str = f"{bin_metrics.get('switch_error_rate'):.4f}" if bin_metrics.get('switch_error_rate') is not None else "N/A"
                    phase_str = f"{bin_metrics.get('phase_concordance'):.4f}" if bin_metrics.get('phase_concordance') is not None else "N/A"
                    print(f"   {maf_bin:<20} {f1_str:>8} {conc:>8} {r2_str:>8} {switch_str:>10} {phase_str:>10} {bin_metrics['n_genotypes']:>10,}")

        if metrics.get("masked_total"):
            print(f"\n🧪 MASKED-SNP METRICS (proxy)")
            print(f"   Masked total:         {metrics['masked_total']:,}")
            print(f"   Masked concordance:   {metrics.get('masked_concordance', 0):.4f}")
            if metrics.get("masked_nonref_concordance") is not None:
                print(f"   Masked non-ref conc:  {metrics.get('masked_nonref_concordance', 0):.4f}")
            if metrics.get("masked_r_squared") is not None:
                print(f"   Masked dosage R²:     {metrics.get('masked_r_squared', 0):.4f}")
            if metrics.get("masked_brier") is not None:
                print(f"   Masked Brier:         {metrics.get('masked_brier', 0):.4f}")
            if metrics.get("masked_ece") is not None:
                print(f"   Masked ECE:           {metrics.get('masked_ece', 0):.4f}")

    # Save detailed metrics to file
    metrics_file = f"{output_prefix}_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("IMPUTATION ACCURACY METRICS - DETAILED REPORT\n")
        f.write("=" * 60 + "\n\n")
        if metrics:
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Sites compared: {metrics.get('sites_compared', 'N/A')}\n")
            f.write(f"Genotypes compared: {metrics.get('total_genotypes', 'N/A')}\n")
            f.write(f"Calculation time: {metrics.get('calculation_time_sec', 0):.1f}s\n\n")
            
            f.write("ACCURACY METRICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"F1 Score (Non-Ref): {metrics.get('f1_score', 0):.6f}\n")
            f.write(f"Precision: {metrics.get('precision', 0):.6f}\n")
            f.write(f"Recall: {metrics.get('recall', 0):.6f}\n")
            f.write(f"Unphased concordance: {metrics.get('unphased_concordance', 0):.6f}\n")
            if metrics.get('nonref_concordance'):
                f.write(f"Non-ref concordance: {metrics['nonref_concordance']:.6f}\n")
            if metrics.get('r_squared'):
                f.write(f"Overall R²: {metrics['r_squared']:.6f}\n")
            if metrics.get('iqs'):
                f.write(f"Overall IQS: {metrics['iqs']:.6f}\n")
            if metrics.get('sen_mean') is not None:
                f.write(f"SEN (mean): {metrics['sen_mean']:.6f}\n")
            if metrics.get('info_score_approx'):
                f.write(f"INFO score (approx): {metrics['info_score_approx']:.6f}\n")
            if metrics.get('hellinger_score'):
                f.write(f"Hellinger Score: {metrics['hellinger_score']:.6f}\n")
            if metrics.get('switch_error_rate') is not None:
                f.write(f"Switch error rate: {metrics['switch_error_rate']:.6f}\n")
                f.write(f"N50 Phase Block: {metrics.get('n50_phase_block'):.0f} bp\n")
                if metrics.get('phase_concordance') is not None:
                    f.write(f"Phase concordance: {metrics['phase_concordance']:.6f}\n")
                    f.write(f"Phase flip rate: {metrics['phase_flip_rate']:.6f}\n")
                if metrics.get('phase_block_len_mean') is not None:
                    f.write(f"Phase block length mean: {metrics['phase_block_len_mean']:.0f} bp\n")
                    f.write(f"Phase block length median: {metrics['phase_block_len_median']:.0f} bp\n")
                    f.write(f"Phase block length p10: {metrics['phase_block_len_p10']:.0f} bp\n")
                    f.write(f"Phase block length p90: {metrics['phase_block_len_p90']:.0f} bp\n")
                    f.write(f"Phase block count: {metrics.get('phase_block_count', 0)}\n")
            if metrics.get('switch_error_rate_input_sites') is not None:
                f.write(f"Switch error (input sites): {metrics['switch_error_rate_input_sites']:.6f}\n")
            if metrics.get('sample_phase_concordance_mean') is not None:
                f.write(f"Per-sample phase concordance mean: {metrics['sample_phase_concordance_mean']:.6f}\n")
                f.write(f"Per-sample phase concordance min: {metrics['sample_phase_concordance_min']:.6f}\n")
                f.write(f"Per-sample phase concordance max: {metrics['sample_phase_concordance_max']:.6f}\n")
            if metrics.get('sample_phase_block_len_mean') is not None:
                f.write(f"Per-sample phase block mean: {metrics['sample_phase_block_len_mean']:.0f} bp\n")
                f.write(f"Per-sample phase block min: {metrics['sample_phase_block_len_min']:.0f} bp\n")
                f.write(f"Per-sample phase block max: {metrics['sample_phase_block_len_max']:.0f} bp\n")
            if metrics.get('sample_phase_block_n50_mean') is not None:
                f.write(f"Per-sample phase block N50 mean: {metrics['sample_phase_block_n50_mean']:.0f} bp\n")
                f.write(f"Per-sample phase block N50 min: {metrics['sample_phase_block_n50_min']:.0f} bp\n")
                f.write(f"Per-sample phase block N50 max: {metrics['sample_phase_block_n50_max']:.0f} bp\n")
            if metrics.get('sample_phase_block_count_mean') is not None:
                f.write(f"Per-sample phase block count mean: {metrics['sample_phase_block_count_mean']:.1f}\n")
                f.write(f"Per-sample phase block count min: {metrics['sample_phase_block_count_min']:.0f}\n")
                f.write(f"Per-sample phase block count max: {metrics['sample_phase_block_count_max']:.0f}\n")
            
            f.write("\nCONFUSION MATRIX\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'':12} {'HomRef':>10} {'Het':>10} {'HomAlt':>10}\n")
            for i, label in enumerate(['HomRef', 'Het', 'HomAlt']):
                row = metrics.get('confusion_matrix', [[0,0,0],[0,0,0],[0,0,0]])[i]
                f.write(f"{label:12} {row[0]:>10} {row[1]:>10} {row[2]:>10}\n")
            
            f.write("\nBY MAF BIN\n")
            f.write("-" * 40 + "\n")
            for maf_bin, bin_metrics in metrics.get("by_maf", {}).items():
                f.write(f"\n{maf_bin}:\n")
                f.write(f"  F1 Score: {bin_metrics.get('f1_score', 0):.6f}\n")
                f.write(f"  Recall: {bin_metrics.get('recall', 0):.6f}\n")
                f.write(f"  Concordance: {bin_metrics['unphased_concordance']:.6f}\n")
                if bin_metrics.get('nonref_concordance'):
                    f.write(f"  Non-ref concordance: {bin_metrics['nonref_concordance']:.6f}\n")
                if bin_metrics.get('r_squared'):
                    f.write(f"  R²: {bin_metrics['r_squared']:.6f}\n")
                if bin_metrics.get('iqs'):
                    f.write(f"  IQS: {bin_metrics['iqs']:.6f}\n")
                if bin_metrics.get('switch_error_rate') is not None:
                    f.write(f"  Switch error rate: {bin_metrics['switch_error_rate']:.6f}\n")
                if bin_metrics.get('phase_concordance') is not None:
                    f.write(f"  Phase concordance: {bin_metrics['phase_concordance']:.6f}\n")
                f.write(f"  N genotypes: {bin_metrics['n_genotypes']}\n")
    
    print(f"\n📄 Detailed metrics saved to: {metrics_file}")

    return metrics


def get_paths():
    """Get standard paths used across all stages."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = script_dir / "data"
    os.makedirs(data_dir, exist_ok=True)

    return {
        'script_dir': script_dir,
        'project_dir': project_dir,
        'data_dir': data_dir,
        'chr22_bcf': data_dir / "hgdp1kg_chr22.bcf",
        'chr22_vcf': data_dir / "hgdp1kg_chr22.vcf.gz",
        'gsa_file': data_dir / "GSAv2_hg38.tsv",
        'beagle_jar': data_dir / "beagle.jar",
        'reagle_bin': project_dir / "target" / "release" / "reagle",
        'ref_vcf': data_dir / "ref.vcf.gz",
        'truth_vcf': data_dir / "truth.vcf.gz",
        'input_vcf': data_dir / "input.vcf.gz",
        'gsa_regions': data_dir / "gsa_chr22.regions",
        'train_file': data_dir / "train_samples.txt",
        'test_file': data_dir / "test_samples.txt",
        'beagle_out': data_dir / "beagle_imputed",
        'reagle_out': data_dir / "reagle_imputed",
    }


def has_phased_genotypes(vcf_path, max_records=200):
    """Return True if any GT fields appear phased (contain '|')."""
    if not Path(vcf_path).exists():
        return False
    try:
        seen = 0
        with _open_maybe_gzip(vcf_path) as handle:
            for line in handle:
                if not line or line.startswith("#"):
                    continue
                fields = line.rstrip().split("\t")
                if len(fields) < 10:
                    continue
                for sample_field in fields[9:]:
                    gt = sample_field.split(":", 1)[0]
                    if "|" in gt:
                        return True
                seen += 1
                if seen >= max_records:
                    break
    except OSError:
        return False
    return False


def stage_prepare(keep_phased=False):
    """
    Download data and prepare reference/truth/input VCFs.

    This creates the FULL chr22 dataset (~830K markers) for integration testing.

    Dataset comparison:
    - stage_prepare():         100% of chr22 (~830K markers) - FULL integration test
    - stage_prepare_profile():   5% of chr22 (~41K markers)  - PROFILING subset

    The full dataset is used for production integration tests comparing against Beagle.
    WARNING: The full dataset may cause single-threaded state allocation to stall.

    Used by: .github/workflows/integration.yml (integration test jobs)
    NOT used by: .github/workflows/bench.yml (uses stage_prepare_profile instead)
    """
    print("=" * 60)
    print("STAGE: PREPARE - Download and prepare data (FULL chr22)")
    print("=" * 60)

    paths = get_paths()

    # Check dependencies
    check_dependencies()

    # Download HGDP+1kG chr22
    print("\n" + "=" * 60)
    print("Downloading HGDP+1kG chr22...")
    print("=" * 60)

    download_if_missing(
        "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf",
        str(paths['chr22_bcf'])
    )
    download_if_missing(
        "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf.csi",
        str(paths['chr22_bcf']) + ".csi"
    )

    # Convert BCF to VCF.gz for Java Beagle compatibility
    if not validate_vcf(paths['chr22_bcf']):
        print("ERROR: Cached BCF appears corrupted. Re-downloading...")
        paths['chr22_bcf'].unlink(missing_ok=True)
        Path(str(paths['chr22_bcf']) + ".csi").unlink(missing_ok=True)
        download_if_missing(
            "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf",
            str(paths['chr22_bcf'])
        )
        download_if_missing(
            "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf.csi",
            str(paths['chr22_bcf']) + ".csi"
        )

    if not validate_vcf(paths['chr22_vcf']) or not has_vcf_records(paths['chr22_vcf']):
        print("Converting BCF to VCF.gz...")
        paths['chr22_vcf'].unlink(missing_ok=True)
        Path(str(paths['chr22_vcf']) + ".csi").unlink(missing_ok=True)
        run(f"bcftools view {paths['chr22_bcf']} -O z -o {paths['chr22_vcf']}")
    ensure_index(paths['chr22_vcf'], recreate_cmd=f"bcftools view {paths['chr22_bcf']} -O z -o {paths['chr22_vcf']}")

    # Download GSA sites
    print("\n" + "=" * 60)
    print("Downloading GSA variant list...")
    print("=" * 60)

    download_if_missing(
        "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/GSAv2_hg38.tsv",
        str(paths['gsa_file'])
    )

    # Load GSA sites for chr22
    gsa_sites = load_gsa_sites(str(paths['gsa_file']), chrom="22")

    # Download Beagle
    download_if_missing(
        "https://faculty.washington.edu/browning/beagle/beagle.22Jul22.46e.jar",
        str(paths['beagle_jar'])
    )

    # Split samples
    print("\n" + "=" * 60)
    print("Splitting samples...")
    print("=" * 60)

    train_file, test_file, train_samples, test_samples = split_samples(
        str(paths['chr22_vcf']), str(paths['data_dir']), test_fraction=0.2, seed=42
    )

    # Create reference panel (train samples)
    if not validate_vcf(paths['ref_vcf']) or not has_vcf_records(paths['ref_vcf']):
        print("Creating reference panel...")
        paths['ref_vcf'].unlink(missing_ok=True)
        Path(str(paths['ref_vcf']) + ".csi").unlink(missing_ok=True)
        run(f"bcftools view -S {train_file} {paths['chr22_vcf']} -O z -o {paths['ref_vcf']}")
        run(f"bcftools index -f {paths['ref_vcf']}")
    if not has_index(paths['ref_vcf']):
        ensure_index(paths['ref_vcf'], recreate_cmd=f"bcftools view -S {train_file} {paths['chr22_vcf']} -O z -o {paths['ref_vcf']}")

    # Create truth (test samples, full density)
    if not validate_vcf(paths['truth_vcf']) or not has_vcf_records(paths['truth_vcf']):
        print("Creating truth VCF...")
        paths['truth_vcf'].unlink(missing_ok=True)
        Path(str(paths['truth_vcf']) + ".csi").unlink(missing_ok=True)
        run(f"bcftools view -S {test_file} {paths['chr22_vcf']} -O z -o {paths['truth_vcf']}")
        run(f"bcftools index -f {paths['truth_vcf']}")
    if not has_index(paths['truth_vcf']):
        ensure_index(paths['truth_vcf'], recreate_cmd=f"bcftools view -S {test_file} {paths['chr22_vcf']} -O z -o {paths['truth_vcf']}")

    # Create input (test samples, downsampled to GSA sites)
    # Default: unphase input so switch error rate measures true phasing accuracy.
    # Optional: keep phased input for manual runs.
    tmp_phased_path = paths['data_dir'] / "input_phased_tmp.vcf.gz"
    input_valid = validate_vcf(paths['input_vcf']) and has_vcf_records(paths['input_vcf'])
    input_is_phased = has_phased_genotypes(paths['input_vcf']) if input_valid else False
    needs_rebuild = not input_valid or (keep_phased and not input_is_phased) or (not keep_phased and input_is_phased)
    if needs_rebuild:
        create_regions_file(gsa_sites, str(paths['gsa_regions']))
        if keep_phased:
            print("Downsampling to GSA sites (keeping phasing)...")
            run(f"bcftools view -R {paths['gsa_regions']} {paths['truth_vcf']} -O z -o {paths['input_vcf']}")
        else:
            print("Downsampling to GSA sites and unphasing...")
            # Two-step process: downsample, then unphase
            tmp_phased = str(tmp_phased_path)
            run(f"bcftools view -R {paths['gsa_regions']} {paths['truth_vcf']} -O z -o {tmp_phased}")
            # Unphase: convert 0|1 to 0/1 using bcftools +setGT
            # The plugin sets genotypes to unphased while preserving allele values
            run(f"bcftools +setGT {tmp_phased} -O z -o {paths['input_vcf']} -- -t a -n u")
            # Clean up temp file
            os.remove(tmp_phased)
        run(f"bcftools index -f {paths['input_vcf']}")
    if not has_index(paths['input_vcf']):
        if keep_phased:
            ensure_index(
                paths['input_vcf'],
                recreate_cmd=f"bcftools view -R {paths['gsa_regions']} {paths['truth_vcf']} -O z -o {paths['input_vcf']}",
            )
        else:
            ensure_index(
                paths['input_vcf'],
                recreate_cmd=f"bcftools view -R {paths['gsa_regions']} {paths['truth_vcf']} -O z -o {tmp_phased_path} && bcftools +setGT {tmp_phased_path} -O z -o {paths['input_vcf']} -- -t a -n u",
            )
    if tmp_phased_path.exists():
        os.remove(tmp_phased_path)

    # Count variants
    n_truth = run(f"bcftools view -H {paths['truth_vcf']} | wc -l", capture=True).stdout.strip()
    n_input = run(f"bcftools view -H {paths['input_vcf']} | wc -l", capture=True).stdout.strip()
    print(f"\nTruth variants: {n_truth}")
    print(f"Input variants (GSA sites): {n_input}")
    print(f"Reference samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    print("\nPrepare stage completed successfully.")


def stage_prepare_profile(keep_phased=False):
    """
    Prepare reduced data (middle 5% of chr22 target markers) for profiling runs.

    This creates a SMALLER dataset (~41K markers) for fast profiling/benchmarking.

    Dataset comparison:
    - stage_prepare():         100% of chr22 (~830K markers) - FULL integration test
    - stage_prepare_profile():   5% of chr22 (~41K markers)  - PROFILING subset

    The 5% subset allows profiling runs to complete in ~4 hours on CI.
    The full 100% dataset causes single-threaded state allocation to stall for 6+ hours.

    Used by: .github/workflows/bench.yml (profiling jobs)
    NOT used by: .github/workflows/integration.yml (uses stage_prepare instead)
    """
    print("=" * 60)
    print("STAGE: PREPARE PROFILE - Middle 5% of chr22 (target markers)")
    print("=" * 60)

    paths = get_paths()

    # Check dependencies
    check_dependencies()

    # Download HGDP+1kG chr22
    print("\n" + "=" * 60)
    print("Downloading HGDP+1kG chr22...")
    print("=" * 60)

    download_if_missing(
        "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf",
        str(paths['chr22_bcf'])
    )
    download_if_missing(
        "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf.csi",
        str(paths['chr22_bcf']) + ".csi"
    )

    # Convert BCF to VCF.gz for Java Beagle compatibility
    if not validate_vcf(paths['chr22_bcf']):
        print("ERROR: Cached BCF appears corrupted. Re-downloading...")
        paths['chr22_bcf'].unlink(missing_ok=True)
        Path(str(paths['chr22_bcf']) + ".csi").unlink(missing_ok=True)
        download_if_missing(
            "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf",
            str(paths['chr22_bcf'])
        )
        download_if_missing(
            "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf.csi",
            str(paths['chr22_bcf']) + ".csi"
        )

    if not validate_vcf(paths['chr22_vcf']) or not has_vcf_records(paths['chr22_vcf']):
        print("Converting BCF to VCF.gz...")
        paths['chr22_vcf'].unlink(missing_ok=True)
        Path(str(paths['chr22_vcf']) + ".csi").unlink(missing_ok=True)
        run(f"bcftools view {paths['chr22_bcf']} -O z -o {paths['chr22_vcf']}")
    ensure_index(paths['chr22_vcf'], recreate_cmd=f"bcftools view {paths['chr22_bcf']} -O z -o {paths['chr22_vcf']}")

    # Download GSA sites first
    print("\n" + "=" * 60)
    print("Downloading GSA variant list...")
    print("=" * 60)

    download_if_missing(
        "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/GSAv2_hg38.tsv",
        str(paths['gsa_file'])
    )

    # Load GSA sites for chr22
    gsa_sites = load_gsa_sites(str(paths['gsa_file']), chrom="22")

    if not gsa_sites:
        raise RuntimeError("No GSA sites found for chr22")

    # Find chromosome label from VCF
    chrom_label = find_chrom_label(paths['chr22_vcf'], "22") or "chr22"

    # Compute region based on MIDDLE 5% of TARGET markers (GSA sites present in the VCF)
    gsa_present_positions = []
    with _open_maybe_gzip(paths['chr22_vcf']) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            try:
                pos = int(fields[1])
            except ValueError:
                continue
            if (fields[0], pos) in gsa_sites or (chrom_label, pos) in gsa_sites:
                gsa_present_positions.append(pos)

    gsa_present_positions = sorted(set(gsa_present_positions))
    if not gsa_present_positions:
        raise RuntimeError("No GSA sites found in chr22 VCF")

    total_markers = len(gsa_present_positions)
    window_size = max(1, int(total_markers * 0.05))
    start_idx = (total_markers - window_size) // 2
    end_idx = start_idx + window_size

    min_pos = gsa_present_positions[start_idx]
    end_pos = gsa_present_positions[end_idx - 1]

    region = f"{chrom_label}:{min_pos}-{end_pos}"

    print(f"Profiling region based on middle 5% of {total_markers} target markers:")
    print(f"  Region: {region} (positions {min_pos}..{end_pos})")
    print(f"  Target markers in region: {window_size} (indices {start_idx}-{end_idx-1})")

    # Create trimmed VCF covering this region
    trimmed_vcf = paths['data_dir'] / "hgdp1kg_chr22.profile5.vcf.gz"
    if not validate_vcf(trimmed_vcf) or not has_vcf_records(trimmed_vcf):
        print("Creating trimmed VCF for profiling...")
        trimmed_vcf.unlink(missing_ok=True)
        Path(str(trimmed_vcf) + ".csi").unlink(missing_ok=True)
        run(f"bcftools view -r {region} {paths['chr22_vcf']} -O z -o {trimmed_vcf}")
        run(f"bcftools index -f {trimmed_vcf}")
    if not has_index(trimmed_vcf):
        ensure_index(trimmed_vcf, recreate_cmd=f"bcftools view -r {region} {paths['chr22_vcf']} -O z -o {trimmed_vcf}")

    # Ensure profile outputs are regenerated even if cached full files exist
    for path in [paths['ref_vcf'], paths['truth_vcf'], paths['input_vcf']]:
        path.unlink(missing_ok=True)
        Path(str(path) + ".csi").unlink(missing_ok=True)
        Path(str(path) + ".tbi").unlink(missing_ok=True)

    # Filter GSA sites to this region and verify they exist in VCF
    filtered_sites = set()
    for chrom, pos in gsa_sites:
        if min_pos <= pos <= end_pos:
            filtered_sites.add((chrom_label, pos))

    # Intersect with markers present in trimmed VCF to guarantee overlap
    present_sites = set()
    with _open_maybe_gzip(trimmed_vcf) as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 2:
                continue
            try:
                pos = int(fields[1])
            except ValueError:
                continue
            present_sites.add((fields[0], pos))

    filtered_sites = {site for site in filtered_sites if site in present_sites}
    print(f"GSA sites in region (present in VCF): {len(filtered_sites)}")

    # Download Beagle
    download_if_missing(
        "https://faculty.washington.edu/browning/beagle/beagle.22Jul22.46e.jar",
        str(paths['beagle_jar'])
    )

    # Split samples using trimmed VCF
    print("\n" + "=" * 60)
    print("Splitting samples...")
    print("=" * 60)

    train_file, test_file, train_samples, test_samples = split_samples(
        str(trimmed_vcf), str(paths['data_dir']), test_fraction=0.2, seed=42
    )

    # Create reference panel (train samples)
    if not validate_vcf(paths['ref_vcf']) or not has_vcf_records(paths['ref_vcf']):
        print("Creating reference panel...")
        paths['ref_vcf'].unlink(missing_ok=True)
        Path(str(paths['ref_vcf']) + ".csi").unlink(missing_ok=True)
        run(f"bcftools view -S {train_file} {trimmed_vcf} -O z -o {paths['ref_vcf']}")
        run(f"bcftools index -f {paths['ref_vcf']}")
    if not has_index(paths['ref_vcf']):
        ensure_index(paths['ref_vcf'], recreate_cmd=f"bcftools view -S {train_file} {trimmed_vcf} -O z -o {paths['ref_vcf']}")

    # Create truth (test samples, full density)
    if not validate_vcf(paths['truth_vcf']) or not has_vcf_records(paths['truth_vcf']):
        print("Creating truth VCF...")
        paths['truth_vcf'].unlink(missing_ok=True)
        Path(str(paths['truth_vcf']) + ".csi").unlink(missing_ok=True)
        run(f"bcftools view -S {test_file} {trimmed_vcf} -O z -o {paths['truth_vcf']}")
        run(f"bcftools index -f {paths['truth_vcf']}")
    if not has_index(paths['truth_vcf']):
        ensure_index(paths['truth_vcf'], recreate_cmd=f"bcftools view -S {test_file} {trimmed_vcf} -O z -o {paths['truth_vcf']}")

    # Create input (test samples, downsampled to GSA sites)
    tmp_phased_path = paths['data_dir'] / "input_phased_tmp.vcf.gz"
    if not filtered_sites:
        raise RuntimeError("Profiling region contains no GSA sites present in VCF; increase subset size.")
    input_valid = validate_vcf(paths['input_vcf']) and has_vcf_records(paths['input_vcf'])
    input_is_phased = has_phased_genotypes(paths['input_vcf']) if input_valid else False
    needs_rebuild = not input_valid or (keep_phased and not input_is_phased) or (not keep_phased and input_is_phased)
    if needs_rebuild:
        create_regions_file(filtered_sites, str(paths['gsa_regions']))
        if keep_phased:
            print("Downsampling to GSA sites (keeping phasing)...")
            run(f"bcftools view -R {paths['gsa_regions']} {paths['truth_vcf']} -O z -o {paths['input_vcf']}")
        else:
            print("Downsampling to GSA sites and unphasing...")
            tmp_phased = str(tmp_phased_path)
            run(f"bcftools view -R {paths['gsa_regions']} {paths['truth_vcf']} -O z -o {tmp_phased}")
            run(f"bcftools +setGT {tmp_phased} -O z -o {paths['input_vcf']} -- -t a -n u")
            os.remove(tmp_phased)
        run(f"bcftools index -f {paths['input_vcf']}")
        if not has_vcf_records(paths['input_vcf']):
            raise RuntimeError("Profiling input VCF is empty after downsampling; adjust subset size.")
    if not has_index(paths['input_vcf']):
        if keep_phased:
            ensure_index(
                paths['input_vcf'],
                recreate_cmd=f"bcftools view -R {paths['gsa_regions']} {paths['truth_vcf']} -O z -o {paths['input_vcf']}",
            )
        else:
            ensure_index(
                paths['input_vcf'],
                recreate_cmd=f"bcftools view -R {paths['gsa_regions']} {paths['truth_vcf']} -O z -o {tmp_phased_path} && bcftools +setGT {tmp_phased_path} -O z -o {paths['input_vcf']} -- -t a -n u",
            )
    if tmp_phased_path.exists():
        os.remove(tmp_phased_path)

    # Count variants
    n_truth = run(f"bcftools view -H {paths['truth_vcf']} | wc -l", capture=True).stdout.strip()
    n_input = run(f"bcftools view -H {paths['input_vcf']} | wc -l", capture=True).stdout.strip()
    print(f"\nTruth variants: {n_truth}")
    print(f"Input variants (GSA sites): {n_input}")
    print(f"Reference samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    print("\nPrepare profile stage completed successfully.")


def stage_beagle():
    """Run Beagle imputation only."""
    print("=" * 60)
    print("STAGE: BEAGLE - Run Java Beagle imputation")
    print("=" * 60)

    paths = get_paths()

    # Verify required files exist
    for name in ['ref_vcf', 'input_vcf', 'beagle_jar']:
        if not paths[name].exists():
            print(f"ERROR: Required file not found: {paths[name]}")
            print("Run 'prepare' stage first.")
            sys.exit(1)

    print("\n--- Running Java Beagle ---")
    beagle_map = ensure_plink_genetic_map(
        paths["ref_vcf"],
        paths["data_dir"] / "chr22.plink.map",
        chrom="22"
    )
    beagle_vcf = run_beagle(
        str(paths['ref_vcf']),
        str(paths['input_vcf']),
        str(paths['beagle_out']),
        str(paths['beagle_jar']),
        nthreads=2,
        map_path=str(beagle_map),
    )

    if beagle_vcf and os.path.exists(beagle_vcf):
        print(f"\nBeagle output: {beagle_vcf}")
        print("Beagle stage completed successfully.")
    else:
        print("\nERROR: Beagle imputation failed!")
        sys.exit(1)


def stage_reagle():
    """Run Reagle imputation only."""
    print("=" * 60)
    print("STAGE: REAGLE - Run Reagle imputation")
    print("=" * 60)

    paths = get_paths()

    # Verify required files exist
    for name in ['ref_vcf', 'input_vcf']:
        if not paths[name].exists():
            print(f"ERROR: Required file not found: {paths[name]}")
            print("Run 'prepare' stage first.")
            sys.exit(1)

    if not paths['reagle_bin'].exists():
        print(f"ERROR: Reagle binary not found: {paths['reagle_bin']}")
        print("Build Reagle first with: cargo build --release")
        sys.exit(1)

    print("\n--- Running Reagle ---")
    reagle_map = ensure_plink_genetic_map(
        paths["ref_vcf"],
        paths["data_dir"] / "chr22.plink.map",
        chrom="22"
    )
    reagle_vcf = run_reagle(
        str(paths['ref_vcf']),
        str(paths['input_vcf']),
        str(paths['reagle_out']),
        str(paths['reagle_bin']),
        map_path=str(reagle_map),
    )

    if reagle_vcf and os.path.exists(reagle_vcf):
        print(f"\nReagle output: {reagle_vcf}")
        print("Reagle stage completed successfully.")
    else:
        print("\nERROR: Reagle imputation failed!")
        sys.exit(1)


def _prepare_phasing_input(paths):
    """Return an unphased input VCF path for phasing comparisons."""
    input_vcf = paths["input_vcf"]
    if has_phased_genotypes(input_vcf):
        print("Input VCF appears phased; creating an unphased copy for phasing comparison.")
        unphased_input = paths["data_dir"] / "input_unphased.vcf.gz"
        run(f"bcftools +setGT {input_vcf} -O z -o {unphased_input} -- -t a -n u")
        run(f"bcftools index -f {unphased_input}")
        return unphased_input
    return input_vcf


def _run_reagle_phasing(paths, input_vcf):
    """Run Reagle phasing-only and return phased VCF path."""
    if not paths["reagle_bin"].exists():
        print(f"ERROR: Reagle binary not found: {paths['reagle_bin']}")
        print("Build Reagle first with: cargo build --release")
        sys.exit(1)

    reagle_prefix = paths["data_dir"] / "reagle_phased"
    reagle_vcf = Path(str(reagle_prefix) + ".vcf.gz")
    if not reagle_vcf.exists():
        print("\n--- Running Reagle (phasing-only) ---")
        reagle_map = ensure_plink_genetic_map(
            paths["ref_vcf"],
            paths["data_dir"] / "chr22.plink.map",
            chrom="22"
        )
        run(
            f"{paths['reagle_bin']} --target {input_vcf} --ref {paths['ref_vcf']} "
            f"--map {reagle_map} --out {reagle_prefix}"
        )
    ensure_index(reagle_vcf)
    return reagle_vcf


def _run_eagleimp_phasing(paths, input_vcf):
    """Run EagleImp phasing-only and return phased VCF/BCF path."""
    eagleimp_bin = find_executable("eagleimp", env_var="EAGLEIMP_BIN")
    if not eagleimp_bin:
        print("ERROR: EagleImp binary not found (set EAGLEIMP_BIN or ensure in PATH).")
        sys.exit(1)

    gen_map_env = os.environ.get("EAGLEIMP_GENMAP")
    if gen_map_env:
        gen_map = Path(gen_map_env)
        if not eagleimp_map_header_ok(gen_map):
            print(
                f"WARNING: EAGLEIMP_GENMAP has incompatible header: {gen_map}. "
                "Falling back to generated simple map."
            )
            gen_map = ensure_simple_genetic_map(
                paths["ref_vcf"], paths["data_dir"] / "chr22.simple.map.txt", chrom="22"
            )
    else:
        gen_map = ensure_simple_genetic_map(paths["ref_vcf"], paths["data_dir"] / "chr22.simple.map.txt", chrom="22")

    # EagleImp expects reference filenames to start with the chromosome label.
    eagleimp_ref = paths["data_dir"] / "22.ref.vcf.gz"
    eagleimp_ref_csi = Path(str(eagleimp_ref) + ".csi")
    eagleimp_ref_tbi = Path(str(eagleimp_ref) + ".tbi")
    if not eagleimp_ref.exists():
        eagleimp_ref.symlink_to(paths["ref_vcf"])
    if not eagleimp_ref_csi.exists() and not eagleimp_ref_tbi.exists():
        src_index_csi = Path(str(paths["ref_vcf"]) + ".csi")
        src_index_tbi = Path(str(paths["ref_vcf"]) + ".tbi")
        if src_index_csi.exists():
            eagleimp_ref_csi.symlink_to(src_index_csi)
        elif src_index_tbi.exists():
            eagleimp_ref_tbi.symlink_to(src_index_tbi)

    qref_path = ensure_eagleimp_qref(eagleimp_ref, eagleimp_bin)

    eagleimp_prefix = paths["data_dir"] / "eagleimp_phased"
    eagleimp_vcf = find_eagleimp_phased_output(eagleimp_prefix, paths["data_dir"])
    if not eagleimp_vcf or not eagleimp_vcf.exists():
        print("\n--- Running EagleImp (phasing-only) ---")
        run(
            f"{eagleimp_bin} --geneticMap {gen_map} --ref {qref_path} --target {input_vcf} "
            f"--skipImputation --outputPhasedFile -o {eagleimp_prefix}"
        )
        eagleimp_vcf = find_eagleimp_phased_output(eagleimp_prefix, paths["data_dir"])

    if not eagleimp_vcf or not eagleimp_vcf.exists():
        print("ERROR: EagleImp phased output not found after run.")
        sys.exit(1)

    if str(eagleimp_vcf).endswith(".vcf.gz"):
        ensure_index(eagleimp_vcf)
    return eagleimp_vcf


def stage_phasing_reagle():
    """Run only Reagle phasing for phasing comparison pipeline."""
    print("=" * 60)
    print("STAGE: PHASING REAGLE - Reagle phasing-only")
    print("=" * 60)

    paths = get_paths()
    for name in ["ref_vcf", "input_vcf"]:
        if not paths[name].exists():
            print(f"ERROR: Required file not found: {paths[name]}")
            print("Run 'prepare' stage first.")
            sys.exit(1)

    input_vcf = _prepare_phasing_input(paths)
    reagle_vcf = _run_reagle_phasing(paths, input_vcf)
    print(f"\nReagle phased output: {reagle_vcf}")
    print("Phasing Reagle stage completed successfully.")


def stage_phasing_eagleimp():
    """Run only EagleImp phasing for phasing comparison pipeline."""
    print("=" * 60)
    print("STAGE: PHASING EAGLEIMP - EagleImp phasing-only")
    print("=" * 60)

    paths = get_paths()
    for name in ["ref_vcf", "input_vcf"]:
        if not paths[name].exists():
            print(f"ERROR: Required file not found: {paths[name]}")
            print("Run 'prepare' stage first.")
            sys.exit(1)

    input_vcf = _prepare_phasing_input(paths)
    eagleimp_vcf = _run_eagleimp_phasing(paths, input_vcf)
    print(f"\nEagleImp phased output: {eagleimp_vcf}")
    print("Phasing EagleImp stage completed successfully.")


def stage_phasing_metrics():
    """Calculate phasing metrics after phased outputs are produced."""
    print("=" * 60)
    print("STAGE: PHASING METRICS - Reagle vs EagleImp vs TRUTH")
    print("=" * 60)

    paths = get_paths()
    for name in ["ref_vcf", "truth_vcf", "input_vcf"]:
        if not paths[name].exists():
            print(f"ERROR: Required file not found: {paths[name]}")
            print("Run 'prepare' stage first.")
            sys.exit(1)

    input_vcf = _prepare_phasing_input(paths)
    reagle_vcf = paths["data_dir"] / "reagle_phased.vcf.gz"
    eagleimp_prefix = paths["data_dir"] / "eagleimp_phased"
    eagleimp_vcf = find_eagleimp_phased_output(eagleimp_prefix, paths["data_dir"])

    if not reagle_vcf.exists():
        print(f"ERROR: Reagle phased output not found: {reagle_vcf}")
        print("Run 'phasing-reagle' stage first.")
        sys.exit(1)
    ensure_index(reagle_vcf)

    if not eagleimp_vcf or not eagleimp_vcf.exists():
        print("ERROR: EagleImp phased output not found.")
        print("Run 'phasing-eagleimp' stage first.")
        sys.exit(1)
    if str(eagleimp_vcf).endswith(".vcf.gz"):
        ensure_index(eagleimp_vcf)

    print("\n" + "=" * 60)
    print("Calculating phasing accuracy metrics...")
    print("=" * 60)

    metrics = {}
    metrics["reagle"] = calculate_metrics(
        str(paths["truth_vcf"]),
        str(reagle_vcf),
        str(paths["data_dir"] / "reagle_phasing"),
        input_vcf=str(input_vcf),
        reference_vcf=str(paths["ref_vcf"]),
        require_ds_gp=False
    )
    metrics["eagleimp"] = calculate_metrics(
        str(paths["truth_vcf"]),
        str(eagleimp_vcf),
        str(paths["data_dir"] / "eagleimp_phasing"),
        input_vcf=str(input_vcf),
        reference_vcf=str(paths["ref_vcf"]),
        require_ds_gp=False
    )
    metrics["truth"] = calculate_metrics(
        str(paths["truth_vcf"]),
        str(paths["truth_vcf"]),
        str(paths["data_dir"] / "truth_phasing"),
        input_vcf=str(input_vcf),
        reference_vcf=str(paths["ref_vcf"]),
        require_ds_gp=False
    )

    print("\nPHASING SUMMARY")
    for name, data in metrics.items():
        if not data:
            print(f"{name.upper()}: FAILED/SKIPPED")
            continue
        ser = data.get("switch_error_rate", 0.0)
        n50 = data.get("n50_phase_block", 0.0)
        phase_conc = data.get("phase_concordance", 0.0)
        print(f"{name.upper()}: SER={ser:.4f} PhaseConc={phase_conc:.4f} N50={n50:.0f} bp")

    print("\nPhasing metrics stage completed successfully.")


def stage_phasing_compare():
    """Compare phasing accuracy between Reagle, EagleImp, and TRUTH baseline."""
    print("=" * 60)
    print("STAGE: PHASING COMPARE - Reagle vs EagleImp vs TRUTH")
    print("=" * 60)

    stage_phasing_reagle()
    stage_phasing_eagleimp()
    stage_phasing_metrics()
    print("\nPhasing comparison completed successfully.")


def stage_metrics():
    """Calculate and compare metrics for both tools."""
    print("=" * 60)
    print("STAGE: METRICS - Calculate accuracy metrics")
    print("=" * 60)

    paths = get_paths()

    # Verify truth exists
    if not paths['truth_vcf'].exists():
        print(f"ERROR: Truth VCF not found: {paths['truth_vcf']}")
        print("Run 'prepare' stage first.")
        sys.exit(1)

    # Check for imputed files
    results = {}
    beagle_vcf = str(paths['beagle_out']) + ".vcf.gz"
    reagle_vcf = str(paths['reagle_out']) + ".vcf.gz"

    if os.path.exists(beagle_vcf):
        results['beagle'] = beagle_vcf
    else:
        print(f"Warning: Beagle output not found: {beagle_vcf}")
        results['beagle'] = None

    if os.path.exists(reagle_vcf):
        results['reagle'] = reagle_vcf
    else:
        print(f"Warning: Reagle output not found: {reagle_vcf}")
        results['reagle'] = None

    if not any(results.values()):
        print("\nERROR: No imputed files found!")
        print("Run 'beagle' and/or 'reagle' stages first.")
        sys.exit(1)

    # Report chr/pos overlap across truth + imputed (ignore REF/ALT)
    def _pos_set_all(vcf_path):
        sites = set()
        cmd = f"bcftools query -f '%CHROM\\t%POS\\n' {vcf_path}"
        for line in _stream_vcf_lines(cmd):
            parts = line.split('\t')
            if len(parts) >= 2:
                chrom, pos = parts[0], parts[1]
                sites.add((chrom, pos))
        return sites

    truth_positions = _pos_set_all(str(paths['truth_vcf']))
    beagle_positions = _pos_set_all(results['beagle']) if results['beagle'] else set()
    reagle_positions = _pos_set_all(results['reagle']) if results['reagle'] else set()

    if truth_positions:
        if results['beagle']:
            missing_beagle = truth_positions - beagle_positions
            print(f"\n⚠️  Truth sites not present in BEAGLE (chr/pos only): {len(missing_beagle):,}")
        if results['reagle']:
            missing_reagle = truth_positions - reagle_positions
            print(f"⚠️  Truth sites not present in REAGLE (chr/pos only): {len(missing_reagle):,}")
        if results['beagle'] and results['reagle']:
            overlap_three = truth_positions & beagle_positions & reagle_positions
            print(f"✅ Intersection of TRUTH ∩ BEAGLE ∩ REAGLE (chr/pos only): {len(overlap_three):,}")

    # Calculate metrics
    print("\n" + "=" * 60)
    print("Calculating accuracy metrics...")
    print("=" * 60)

    all_metrics = {}
    degraded_any = False
    for name, vcf in results.items():
        print(f"\n{'=' * 50}")
        print(f"{name.upper()} RESULTS")
        print(f"{'=' * 50}")
        if vcf and os.path.exists(vcf):
            metrics = calculate_metrics(
                str(paths['truth_vcf']),
                vcf,
                str(paths['data_dir'] / f"{name}"),
                reference_vcf=str(paths['ref_vcf']) if paths['ref_vcf'].exists() else None
            )
            all_metrics[name] = metrics
            if metrics and metrics.get("degraded_missing_ds_gp"):
                degraded_any = True
        else:
            print(f"{name} output not found")
            all_metrics[name] = None

    # Load sample counts (fallback to VCF headers if files don't exist)
    n_train = 0
    n_test = 0
    if paths['train_file'].exists():
        with open(paths['train_file']) as f:
            n_train = len([l for l in f if l.strip()])
    elif paths['ref_vcf'].exists():
        # Fallback: count samples in reference VCF
        ref_samples = get_vcf_samples(str(paths['ref_vcf']))
        n_train = len(ref_samples)
        
    if paths['test_file'].exists():
        with open(paths['test_file']) as f:
            n_test = len([l for l in f if l.strip()])
    elif paths['truth_vcf'].exists():
        # Fallback: count samples in truth VCF
        truth_samples = get_vcf_samples(str(paths['truth_vcf']))
        n_test = len(truth_samples)

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Reference panel: {n_train} samples")
    print(f"Test panel: {n_test} samples")
    print()

    def _is_numeric_scalar(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    def _flatten_numeric_metrics(value, prefix=""):
        out = {}
        if isinstance(value, dict):
            for key in sorted(value.keys()):
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                out.update(_flatten_numeric_metrics(value[key], next_prefix))
        elif isinstance(value, list):
            for idx, item in enumerate(value):
                next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                out.update(_flatten_numeric_metrics(item, next_prefix))
        elif _is_numeric_scalar(value):
            val = float(value)
            if math.isfinite(val):
                out[prefix] = val
        return out

    def _fmt_metric_value(value):
        rounded = round(value)
        if abs(value - rounded) < 1e-12 and abs(rounded) >= 1000:
            return f"{int(rounded):,}"
        if abs(value) >= 1000:
            return f"{value:,.3f}"
        if abs(value) >= 1:
            return f"{value:.6f}"
        if value == 0:
            return "0"
        return f"{value:.6g}"

    for name, metrics in all_metrics.items():
        if metrics:
            print(f"{name.upper()}:")
            print(f"  Unphased concordance: {metrics.get('unphased_concordance', 0):.4f}")
            r2 = metrics.get('r_squared')
            print(f"  R²: {r2:.4f}" if r2 else "  R²: N/A")
            iqs = metrics.get('iqs')
            print(f"  IQS: {iqs:.4f}" if iqs else "  IQS: N/A")
        else:
            print(f"{name.upper()}: FAILED/SKIPPED")

    if all_metrics.get("reagle") and all_metrics.get("beagle"):
        reagle_flat = _flatten_numeric_metrics(all_metrics["reagle"])
        beagle_flat = _flatten_numeric_metrics(all_metrics["beagle"])
        shared = sorted(set(reagle_flat.keys()) & set(beagle_flat.keys()))

        print("\nREAGLE VS BEAGLE: ALL SHARED NUMERIC METRICS")
        print(
            f"  Shared metrics compared: {len(shared)} "
            f"(reagle={len(reagle_flat)}, beagle={len(beagle_flat)})"
        )
        for key in shared:
            reagle_val = reagle_flat[key]
            beagle_val = beagle_flat[key]
            delta = reagle_val - beagle_val
            print(
                f"  {key}: "
                f"reagle={_fmt_metric_value(reagle_val)} "
                f"beagle={_fmt_metric_value(beagle_val)} "
                f"delta={_fmt_metric_value(delta)}"
            )

    # Exit with appropriate code
    if not any(m for m in all_metrics.values()):
        print("\nERROR: All metrics calculations failed!")
        sys.exit(1)
    if degraded_any:
        print("\nERROR: Required DS/GP missing; metrics computed with degraded inputs.")
        sys.exit(1)

    print("\nMetrics stage completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Reagle Integration Test - HGDP+1kG Imputation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  prepare      Download data and prepare reference/truth/input VCFs
  prepare-profile  Prepare middle 5% of chr22 target markers for profiling
  beagle       Run Java Beagle imputation
  reagle       Run Reagle imputation
  phasing-reagle  Run Reagle phasing-only
  phasing-eagleimp  Run EagleImp phasing-only
  phasing-metrics  Calculate phasing metrics from phased outputs
  phasing-compare  Compare phasing accuracy (Reagle vs EagleImp vs TRUTH baseline)
  impute5      Run IMPUTE5 imputation
  minimac      Run Minimac4 imputation
  glimpse      Run GLIMPSE imputation
  metrics      Calculate and compare accuracy metrics
  all          Run all stages sequentially (default)
  
Full genome mode (for nightly CI):
  prepare-chr <N>    Prepare data for chromosome N
  impute-chr <N>     Run all imputations for chromosome N
  metrics-chr <N>    Calculate metrics for chromosome N
  summary            Aggregate metrics across all chromosomes

Examples:
  python integration_test.py                  # Run all stages (chr22 only)
  python integration_test.py prepare          # Just prepare data
  python integration_test.py prepare-profile  # Prepare profiling subset
  python integration_test.py prepare --keep-phased  # Prepare input without de-phasing
  python integration_test.py impute5          # Run IMPUTE5
  python integration_test.py prepare-chr 1    # Prepare chr1 for full genome
  python integration_test.py summary          # Aggregate all chromosome metrics
        """
    )
    parser.add_argument(
        'stage',
        nargs='?',
        default='all',
        choices=['all', 'prepare', 'prepare-profile', 'beagle', 'reagle', 'impute5', 'minimac', 
                 'glimpse', 'metrics', 'phasing-reagle', 'phasing-eagleimp', 'phasing-metrics',
                 'phasing-compare', 'prepare-chr', 'impute-chr', 
                 'metrics-chr', 'summary'],
        help='Stage to run (default: all)'
    )
    parser.add_argument(
        'chromosome',
        nargs='?',
        default='22',
        help='Chromosome number for -chr stages (default: 22)'
    )
    parser.add_argument(
        '--tools',
        default='beagle,reagle',
        help='Comma-separated list of tools to run (default: beagle,reagle)'
    )
    parser.add_argument(
        '--keep-phased',
        action='store_true',
        help='Keep phased input VCFs (skip de-phasing; default is to unphase input)'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Reagle Integration Test - HGDP+1kG Imputation Benchmark")
    print("=" * 60)

    if args.stage == 'prepare':
        stage_prepare(args.keep_phased)
    elif args.stage == 'prepare-profile':
        stage_prepare_profile(args.keep_phased)
    elif args.stage == 'beagle':
        stage_beagle()
    elif args.stage == 'reagle':
        stage_reagle()
    elif args.stage == 'phasing-reagle':
        stage_phasing_reagle()
    elif args.stage == 'phasing-eagleimp':
        stage_phasing_eagleimp()
    elif args.stage == 'phasing-metrics':
        stage_phasing_metrics()
    elif args.stage == 'phasing-compare':
        stage_phasing_compare()
    elif args.stage == 'impute5':
        stage_impute5()
    elif args.stage == 'minimac':
        stage_minimac()
    elif args.stage == 'glimpse':
        stage_glimpse()
    elif args.stage == 'metrics':
        stage_metrics()
    elif args.stage == 'prepare-chr':
        stage_prepare_chr(args.chromosome)
    elif args.stage == 'impute-chr':
        stage_impute_chr(args.chromosome, args.tools.split(','))
    elif args.stage == 'metrics-chr':
        stage_metrics_chr(args.chromosome)
    elif args.stage == 'summary':
        stage_summary()
    elif args.stage == 'all':
        # Run all stages sequentially
        stage_prepare(args.keep_phased)

        paths = get_paths()

        # Build Reagle if needed
        if not paths['reagle_bin'].exists():
            print("\nBuilding Reagle...")
            try:
                run(f"cd {paths['project_dir']} && cargo build --release")
            except:
                print("Warning: Failed to build Reagle")

        stage_beagle()

        if paths['reagle_bin'].exists():
            stage_reagle()
        else:
            print("\n--- Skipping Reagle (binary not available) ---")

        stage_metrics()

        print("\nIntegration test completed successfully.")


# =============================================================================
# Additional imputation tool stages
# =============================================================================

def stage_impute5():
    """Run IMPUTE5 imputation."""
    print("\n" + "=" * 60)
    print("STAGE: IMPUTE5 IMPUTATION")
    print("=" * 60)

    paths = get_paths()
    data_dir = paths['data_dir']
    
    # Download IMPUTE5 if not present
    impute5_bin = data_dir / "impute5"
    if not impute5_bin.exists():
        print("Downloading IMPUTE5...")
        run(f"curl -L -o {data_dir}/impute5.zip 'https://www.dropbox.com/sh/mwnceyhir8yze2j/AADbzP6QuAFPrj0Z9_I1RSmla?dl=1'")
        run(f"cd {data_dir} && unzip -q -o impute5.zip impute5_v1.2.0.zip && unzip -q -o impute5_v1.2.0.zip && mv impute5_v1.2.0/impute5_v1.2.0_static impute5 && chmod +x impute5")
    
    impute5_out = data_dir / "impute5_imputed.vcf.gz"
    if not impute5_out.exists():
        print("Running IMPUTE5...")
        try:
            # IMPUTE5 requires specific region format chr:start-end
            contig_len = get_contig_length(paths['ref_vcf'], "22")
            region_arg = f"chr22:1-{contig_len}" if contig_len else "chr22"
            
            run(f"{impute5_bin} --h {paths['ref_vcf']} --g {paths['input_vcf']} --r {region_arg} --buffer-region {region_arg} --o {impute5_out} --threads 4")
            run(f"bcftools index -f {impute5_out}")
        except Exception as e:
            print(f"IMPUTE5 failed: {e}")
    else:
        print(f"Using existing: {impute5_out}")
    print("IMPUTE5 stage completed.")


def stage_minimac():
    """Run Minimac4 imputation."""
    print("\n" + "=" * 60)
    print("STAGE: MINIMAC4 IMPUTATION")
    print("=" * 60)

    paths = get_paths()
    data_dir = paths['data_dir']
    
    # Download Minimac4 if not present
    minimac_bin = data_dir / "minimac4"
    if not minimac_bin.exists():
        print("Downloading Minimac4...")
        run(f"curl -L -o {data_dir}/minimac4.sh 'https://github.com/statgen/Minimac4/releases/download/v4.1.6/minimac4-4.1.6-Linux-x86_64.sh'")
        run(f"chmod +x {data_dir}/minimac4.sh")
        run(f"cd {data_dir} && ./minimac4.sh --prefix=. --skip-license --exclude-subdir")
        # The installer extracts to bin/minimac4
        if (data_dir / "bin" / "minimac4").exists():
             run(f"mv {data_dir}/bin/minimac4 {data_dir}/minimac4")
             run(f"rm -rf {data_dir}/bin {data_dir}/share {data_dir}/minimac4.sh")

    minimac_out = data_dir / "minimac_imputed.vcf.gz"
    if not minimac_out.exists():
        print("Running Minimac4...")
        try:
            contig_len = get_contig_length(paths['ref_vcf'], "22")
            region_arg = f"chr22:1-{contig_len}" if contig_len else "chr22"
            
            run(f"{minimac_bin} {paths['ref_vcf']} {paths['input_vcf']} --output {data_dir}/minimac_imputed.dose.vcf.gz --threads 4 --format GT,DS --region {region_arg}")
            # Minimac outputs to .dose.vcf.gz
            if (data_dir / "minimac_imputed.dose.vcf.gz").exists():
                run(f"mv {data_dir}/minimac_imputed.dose.vcf.gz {minimac_out}")
            run(f"bcftools index -f {minimac_out}")
        except Exception as e:
            print(f"Minimac4 failed: {e}")
    else:
        print(f"Using existing: {minimac_out}")
    print("Minimac4 stage completed.")


def stage_glimpse():
    """Run GLIMPSE imputation."""
    print("\n" + "=" * 60)
    print("STAGE: GLIMPSE IMPUTATION")
    print("=" * 60)

    paths = get_paths()
    data_dir = paths['data_dir']
    
    # Download GLIMPSE if not present
    glimpse_bin = data_dir / "glimpse_phase"
    if not glimpse_bin.exists():
        print("Downloading GLIMPSE...")
        run(f"curl -L -o {glimpse_bin} 'https://github.com/odelaneau/GLIMPSE/releases/download/v2.0.1/GLIMPSE2_phase_static'")
        run(f"chmod +x {glimpse_bin}")
    
    glimpse_out = data_dir / "glimpse_imputed.vcf.gz"
    if not glimpse_out.exists():
        print("Running GLIMPSE...")
        try:
            run(f"{glimpse_bin} --input-gl {paths['input_vcf']} --reference {paths['ref_vcf']} --input-region chr22 --output-region chr22 --output {data_dir}/glimpse_imputed.bcf --threads 4")
            run(f"bcftools view {data_dir}/glimpse_imputed.bcf -O z -o {glimpse_out}")
            run(f"bcftools index -f {glimpse_out}")
        except Exception as e:
            print(f"GLIMPSE failed: {e}")
    else:
        print(f"Using existing: {glimpse_out}")
    print("GLIMPSE stage completed.")


# =============================================================================
# Full genome (per-chromosome) stages
# =============================================================================

def get_chr_paths(chrom):
    """Get paths for a specific chromosome."""
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = script_dir / f"data_chr{chrom}"
    os.makedirs(data_dir, exist_ok=True)
    
    return {
        'data_dir': data_dir,
        'project_dir': project_dir,
        'ref_vcf': data_dir / "ref.vcf.gz",
        'truth_vcf': data_dir / "truth.vcf.gz",
        'input_vcf': data_dir / "input.vcf.gz",
        'reagle_bin': project_dir / "target" / "release" / "reagle",
    }


def get_contig_length(vcf_path, chrom):
    """Get the length of a chromosome from a VCF index."""
    try:
        # bcftools index -s returns: chrom length ...
        result = run(f"bcftools index -s {vcf_path}", capture=True)
        for line in result.stdout.strip().split('\n'):
            parts = line.split('\t')
            if parts[0] == f"chr{chrom}" or parts[0] == str(chrom):
                return int(parts[1])
        # Fallback if not found in index, try query (slower)
        # Or just return a safe large number if strictly needed, but better to fail
        print(f"Warning: Could not find length for chr{chrom} in {vcf_path} index")
        return None
    except Exception as e:
        print(f"Warning: Error getting contig length: {e}")
        return None


def stage_prepare_chr(chrom):
    """Prepare data for a specific chromosome."""
    print(f"\n{'=' * 60}")
    print(f"STAGE: PREPARE CHROMOSOME {chrom}")
    print("=" * 60)
    
    paths = get_chr_paths(chrom)
    data_dir = paths['data_dir']
    
    # Download HGDP+1kG data for this chromosome
    bcf_url = f"https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr{chrom}.filtered.SNV_INDEL.phased.shapeit5.bcf"
    local_bcf = data_dir / f"hgdp1kg_chr{chrom}.bcf"
    local_vcf = data_dir / f"hgdp1kg_chr{chrom}.vcf.gz"
    
    if not local_vcf.exists():
        print(f"Downloading chr{chrom}...")
        run(f"curl -L -o {local_bcf} '{bcf_url}'")
        run(f"curl -L -o {local_bcf}.csi '{bcf_url}.csi'")
        run(f"bcftools view {local_bcf} -O z -o {local_vcf}")
        run(f"bcftools index -f {local_vcf}")
    
    # Split samples (same logic as main prepare)
    train_file = data_dir / "train_samples.txt"
    test_file = data_dir / "test_samples.txt"
    
    if not train_file.exists():
        result = run(f"bcftools query -l {local_vcf}", capture=True)
        samples = result.stdout.strip().split('\n')
        random.seed(42)
        random.shuffle(samples)
        n_test = len(samples) // 5
        test_samples = samples[:n_test]
        train_samples = samples[n_test:]
        
        with open(train_file, 'w') as f:
            f.write('\n'.join(train_samples))
        with open(test_file, 'w') as f:
            f.write('\n'.join(test_samples))
    
    # Create ref, truth, input VCFs
    if not paths['ref_vcf'].exists():
        run(f"bcftools view -S {train_file} {local_vcf} -O z -o {paths['ref_vcf']}")
        run(f"bcftools index -f {paths['ref_vcf']}")
    
    if not paths['truth_vcf'].exists():
        run(f"bcftools view -S {test_file} {local_vcf} -O z -o {paths['truth_vcf']}")
        run(f"bcftools index -f {paths['truth_vcf']}")
    
    if not paths['input_vcf'].exists():
        # Download GSA sites and filter
        gsa_file = data_dir / "GSAv2_hg38.tsv"
        if not gsa_file.exists():
            run(f"curl -L -o {gsa_file} 'https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/GSAv2_hg38.tsv'")
        
        gsa_sites = load_gsa_sites(str(gsa_file), chrom=chrom)
        regions_file = data_dir / f"gsa_chr{chrom}.regions"
        create_regions_file(gsa_sites, str(regions_file))
        
        # Downsample to array sites (keep phased for fair comparison)
        run(f"bcftools view -R {regions_file} {paths['truth_vcf']} -O z -o {paths['input_vcf']}")
        run(f"bcftools index -f {paths['input_vcf']}")
    
    print(f"Chromosome {chrom} preparation complete.")


def stage_impute_chr(chrom, tools):
    """Run specified imputation tools for a chromosome."""
    print(f"\n{'=' * 60}")
    print(f"STAGE: IMPUTE CHROMOSOME {chrom}")
    print(f"Tools: {', '.join(tools)}")
    print("=" * 60)
    
    paths = get_chr_paths(chrom)
    
    for tool in tools:
        if tool == 'beagle':
            run_beagle_chr(chrom, paths)
        elif tool == 'reagle':
            run_reagle_chr(chrom, paths)
        elif tool == 'impute5':
            run_impute5_chr(chrom, paths)
        elif tool == 'minimac':
            run_minimac_chr(chrom, paths)
        elif tool == 'glimpse':
            run_glimpse_chr(chrom, paths)
        elif tool == 'eagleimp':
            run_eagleimp_chr(chrom, paths)


def run_beagle_chr(chrom, paths):
    """Run Beagle for a chromosome."""
    data_dir = paths['data_dir']
    beagle_jar = data_dir / "beagle.jar"
    out = data_dir / "beagle_imputed.vcf.gz"
    
    if not beagle_jar.exists():
        run(f"curl -L -o {beagle_jar} 'https://faculty.washington.edu/browning/beagle/beagle.22Jul22.46e.jar'")
    
    if not out.exists():
        map_path = ensure_plink_genetic_map(
            paths["ref_vcf"],
            data_dir / f"chr{chrom}.plink.map",
            chrom=str(chrom)
        )
        run(
            f"java -Xmx8g -jar {beagle_jar} ref={paths['ref_vcf']} gt={paths['input_vcf']} "
            f"map={map_path} out={data_dir}/beagle_imputed nthreads=4"
        )
        run(f"bcftools index -f {out}")


def run_reagle_chr(chrom, paths):
    """Run Reagle for a chromosome."""
    out = paths['data_dir'] / "reagle_imputed.vcf.gz"
    if paths['reagle_bin'].exists() and not out.exists():
        map_path = ensure_plink_genetic_map(
            paths["ref_vcf"],
            paths["data_dir"] / f"chr{chrom}.plink.map",
            chrom=str(chrom)
        )
        run(
            f"{paths['reagle_bin']} --ref {paths['ref_vcf']} --target {paths['input_vcf']} "
            f"--map {map_path} --out {paths['data_dir']}/reagle_imputed"
        )
        run(f"bcftools index -f {out}")


def run_impute5_chr(chrom, paths):
    """Run IMPUTE5 for a chromosome."""
    data_dir = paths['data_dir']
    
    # Check for binary in main data dir first, then chrom dir
    main_data_dir = paths['project_dir'] / "tests" / "data"
    impute5_bin = main_data_dir / "impute5"
    
    if not impute5_bin.exists():
        # Try finding in chrom dir or download
        impute5_bin = data_dir / "impute5"
        if not impute5_bin.exists():
            print(f"Downloading IMPUTE5 for chr{chrom}...")
            zip_path = data_dir / "impute5.zip"
            run(f"curl -L -o {zip_path} 'https://www.dropbox.com/sh/mwnceyhir8yze2j/AADbzP6QuAFPrj0Z9_I1RSmla?dl=1'")
            
            # Diagnostic: Check downloaded file type and size
            result = run(f"file {zip_path}", capture=True)
            print(f"IMPUTE5 download file type: {result.stdout.strip()}")
            result = run(f"ls -la {zip_path}", capture=True)
            print(f"IMPUTE5 download file size: {result.stdout.strip()}")
            
            if "Zip archive" not in run(f"file {zip_path}", capture=True).stdout:
                print("ERROR: Downloaded file is not a valid zip archive!")
                print("This may be a Dropbox redirect issue. Showing first 500 bytes:")
                run(f"head -c 500 {zip_path}", capture=True)
                raise RuntimeError("IMPUTE5 download failed - not a valid zip file")
            
            run(f"cd {data_dir} && unzip -q -o impute5.zip impute5_v1.2.0.zip && unzip -q -o impute5_v1.2.0.zip && mv impute5_v1.2.0/impute5_v1.2.0_static impute5 && chmod +x impute5")
            
            # Verify binary works
            print("Verifying IMPUTE5 binary...")
            try:
                result = run(f"{impute5_bin} --help 2>&1 | head -5", capture=True)
                print(f"IMPUTE5 --help output: {result.stdout[:200]}")
            except Exception as e:
                print(f"Warning: IMPUTE5 --help check failed: {e}")

    out = data_dir / "impute5_imputed.vcf.gz"
    if not out.exists():
        print(f"Running IMPUTE5 on chr{chrom}...")
        try:
            # IMPUTE5 requires an indexed reference and map file usually, but minimal example:
            # --h reference --g input --r region --o output
            # All tools now use phased input for fair comparison
            region_arg = resolve_region_arg(paths, chrom)
            print_tool_help("IMPUTE5", str(impute5_bin))
            print(f"IMPUTE5 region: {region_arg}")
            print(f"IMPUTE5 ref: {paths['ref_vcf']}")
            print(f"IMPUTE5 input (phased): {paths['input_vcf']}")
            impute5_map = ensure_position_genetic_map(
                paths["ref_vcf"],
                data_dir / f"chr{chrom}.impute5.map.txt",
                chrom=str(chrom),
            )
            run(
                f"{impute5_bin} --h {paths['ref_vcf']} --g {paths['input_vcf']} "
                f"--m {impute5_map} --r {region_arg} --buffer-region {region_arg} "
                f"--o {out} --threads 4"
            )
            run(f"bcftools index -f {out}")
        except Exception as e:
            print(f"IMPUTE5 failed on chr{chrom}: {e}")
    else:
        print(f"Using existing IMPUTE5 output for chr{chrom}")


def run_minimac_chr(chrom, paths):
    """Run Minimac4 for a chromosome."""
    data_dir = paths['data_dir']
    
    # Check for binary
    main_data_dir = paths['project_dir'] / "tests" / "data"
    minimac_bin = main_data_dir / "minimac4"
    
    if not minimac_bin.exists():
        minimac_bin = data_dir / "minimac4"
        if not minimac_bin.exists():
            print(f"Downloading Minimac4 for chr{chrom}...")
            sh_path = data_dir / "minimac4.sh"
            run(f"curl -L -o {sh_path} 'https://github.com/statgen/Minimac4/releases/download/v4.1.6/minimac4-4.1.6-Linux-x86_64.sh'")
            
            # Diagnostic: Check downloaded file type and size
            result = run(f"file {sh_path}", capture=True)
            print(f"Minimac4 download file type: {result.stdout.strip()}")
            result = run(f"ls -la {sh_path}", capture=True)
            print(f"Minimac4 download file size: {result.stdout.strip()}")
            
            # Verify it's a shell script (should contain "#!/bin/sh" or similar)
            result = run(f"head -c 100 {sh_path}", capture=True)
            if "#!/" not in result.stdout and "ELF" not in result.stdout:
                print(f"WARNING: Minimac4 installer may not be valid. First 100 bytes: {result.stdout}")
            
            run(f"chmod +x {sh_path}")
            run(f"cd {data_dir} && ./minimac4.sh --prefix=. --skip-license --exclude-subdir")
            if (data_dir / "bin" / "minimac4").exists():
                 run(f"mv {data_dir}/bin/minimac4 {data_dir}/minimac4")
                 run(f"rm -rf {data_dir}/bin {data_dir}/share {data_dir}/minimac4.sh")
            
            # Verify binary works
            print("Verifying Minimac4 binary...")
            try:
                result = run(f"{minimac_bin} --help 2>&1 | head -5", capture=True)
                print(f"Minimac4 --help output: {result.stdout[:200]}")
            except Exception as e:
                print(f"Warning: Minimac4 --help check failed: {e}")

    out = data_dir / "minimac_imputed.vcf.gz"
    if not out.exists():
        print(f"Running Minimac4 on chr{chrom}...")
        try:
            prefix = data_dir / "minimac_imputed"
            region_arg = resolve_region_arg(paths, chrom)
            
            # Minimac4 v4.x requires reference in MSAV format
            msav_ref = data_dir / "ref.msav"
            if not msav_ref.exists():
                print("Converting reference to MSAV format for Minimac4...")
                run(f"{minimac_bin} --compress-reference {paths['ref_vcf']} > {msav_ref}")
            
            print_tool_help("Minimac4", str(minimac_bin))
            print(f"Minimac4 region: {region_arg}")
            print(f"Minimac4 ref (msav): {msav_ref}")
            print(f"Minimac4 input: {paths['input_vcf']}")
            minimac_map_arg = ""
            minimac_map = ensure_position_genetic_map(
                paths["ref_vcf"],
                data_dir / f"chr{chrom}.minimac.map.txt",
                chrom=str(chrom),
            )
            if tool_supports_flag(str(minimac_bin), "--map"):
                minimac_map_arg = f" --map {minimac_map}"
            elif tool_supports_flag(str(minimac_bin), "--mapFile"):
                minimac_map_arg = f" --mapFile {minimac_map}"
            run(
                f"{minimac_bin} {msav_ref} {paths['input_vcf']} --output {prefix}.dose.vcf.gz "
                f"--threads 4 --format GT,DS --region {region_arg}{minimac_map_arg}"
            )
            
            # Helper to move output
            dose_out = data_dir / "minimac_imputed.dose.vcf.gz"
            if dose_out.exists():
                run(f"mv {dose_out} {out}")
            run(f"bcftools index -f {out}")
        except Exception as e:
            print(f"Minimac4 failed on chr{chrom}: {e}")
    else:
        print(f"Using existing Minimac4 output for chr{chrom}")


def run_glimpse_chr(chrom, paths):
    """Run GLIMPSE for a chromosome."""
    data_dir = paths['data_dir']
    
    # Check for binary
    main_data_dir = paths['project_dir'] / "tests" / "data"
    glimpse_bin = main_data_dir / "glimpse_phase"
    
    if not glimpse_bin.exists():
        glimpse_bin = data_dir / "glimpse_phase"
        if not glimpse_bin.exists():
            print(f"Downloading GLIMPSE for chr{chrom}...")
            run(f"curl -L -o {glimpse_bin} 'https://github.com/odelaneau/GLIMPSE/releases/download/v2.0.1/GLIMPSE2_phase_static'")
            run(f"chmod +x {glimpse_bin}")

    out = data_dir / "glimpse_imputed.vcf.gz"
    if not out.exists():
        print(f"Running GLIMPSE on chr{chrom}...")
        try:
            # GLIMPSE2_phase: --input-gl input.vcf --reference ref.vcf --input-region chr --output out.bcf
            bcf_out = data_dir / "glimpse_imputed.bcf"
            region_arg = resolve_region_arg(paths, chrom)
            print_tool_help("GLIMPSE2", str(glimpse_bin))
            print(f"GLIMPSE region: {region_arg}")
            print(f"GLIMPSE ref: {paths['ref_vcf']}")
            print(f"GLIMPSE input: {paths['input_vcf']}")
            glimpse_map = ensure_position_genetic_map(
                paths["ref_vcf"],
                data_dir / f"chr{chrom}.glimpse.map.txt",
                chrom=str(chrom),
            )
            run(
                f"{glimpse_bin} --input-gl {paths['input_vcf']} --reference {paths['ref_vcf']} "
                f"--map {glimpse_map} --input-region {region_arg} --output-region {region_arg} "
                f"--output {bcf_out} --threads 4"
            )
            run(f"bcftools view {bcf_out} -O z -o {out}")
            run(f"bcftools index -f {out}")
        except Exception as e:
            print(f"GLIMPSE failed on chr{chrom}: {e}")
    else:
        print(f"Using existing GLIMPSE output for chr{chrom}")


def run_eagleimp_chr(chrom, paths):
    """Run EagleImp imputation for a chromosome."""
    data_dir = paths['data_dir']
    out = data_dir / "eagleimp_imputed.vcf.gz"
    if out.exists():
        print(f"Using existing EagleImp output for chr{chrom}")
        return

    eagleimp_bin = find_executable("eagleimp", env_var="EAGLEIMP_BIN")
    if not eagleimp_bin:
        raise RuntimeError("EagleImp binary not found (set EAGLEIMP_BIN or ensure in PATH).")

    print(f"Running EagleImp on chr{chrom}...")
    try:
        print_tool_help("EagleImp", str(eagleimp_bin))
        map_path = ensure_simple_genetic_map(paths["ref_vcf"], data_dir / f"chr{chrom}.simple.map.txt", chrom=str(chrom))
        eagleimp_ref = data_dir / f"{chrom}.ref.vcf.gz"
        eagleimp_ref_csi = Path(str(eagleimp_ref) + ".csi")
        eagleimp_ref_tbi = Path(str(eagleimp_ref) + ".tbi")
        if not eagleimp_ref.exists():
            eagleimp_ref.symlink_to(paths["ref_vcf"])
        if not eagleimp_ref_csi.exists() and not eagleimp_ref_tbi.exists():
            src_index_csi = Path(str(paths["ref_vcf"]) + ".csi")
            src_index_tbi = Path(str(paths["ref_vcf"]) + ".tbi")
            if src_index_csi.exists():
                eagleimp_ref_csi.symlink_to(src_index_csi)
            elif src_index_tbi.exists():
                eagleimp_ref_tbi.symlink_to(src_index_tbi)
        qref_path = ensure_eagleimp_qref(eagleimp_ref, eagleimp_bin)
        eagleimp_prefix = data_dir / "eagleimp_imputed"
        run(
            f"{eagleimp_bin} --geneticMap {map_path} --ref {qref_path} "
            f"--target {paths['input_vcf']} --skipPhasing --imputeInfo gp "
            f"--outputPhasedFile -o {eagleimp_prefix}"
        )

        produced = find_eagleimp_imputed_output(eagleimp_prefix, data_dir)
        if not produced or not produced.exists():
            raise RuntimeError("EagleImp output not found after run")

        if str(produced).endswith(".bcf"):
            run(f"bcftools view {produced} -O z -o {out}")
        elif produced != out:
            run(f"cp {produced} {out}")

        run(f"bcftools index -f {out}")
    except Exception as e:
        print(f"EagleImp failed on chr{chrom}: {e}")


def stage_metrics_chr(chrom):
    """Calculate metrics for a specific chromosome."""
    print(f"\n{'=' * 60}")
    print(f"STAGE: METRICS CHROMOSOME {chrom}")
    print("=" * 60)
    
    paths = get_chr_paths(chrom)
    truth_vcf = str(paths['truth_vcf'])
    data_dir = paths['data_dir']
    
    tools = [
        ("beagle", "beagle_imputed.vcf.gz"),
        ("reagle", "reagle_imputed.vcf.gz"),
        ("impute5", "impute5_imputed.vcf.gz"),
        ("minimac", "minimac_imputed.vcf.gz"),
        ("glimpse", "glimpse_imputed.vcf.gz"),
        ("eagleimp", "eagleimp_imputed.vcf.gz"),
    ]
    
    degraded_any = False
    for prefix, filename in tools:
        imputed_path = data_dir / filename
        if imputed_path.exists():
            print(f"\n{'=' * 40}")
            print(f"Calculating metrics for {prefix.upper()}")
            print("=" * 40)
            try:
                calculate_metrics(
                    truth_vcf,
                    str(imputed_path),
                    str(data_dir / prefix),
                    input_vcf=str(paths['input_vcf']),
                    reference_vcf=str(paths['ref_vcf']) if paths['ref_vcf'].exists() else None
                )
                metrics_path = data_dir / f"{prefix}_metrics.json"
                if metrics_path.exists():
                    try:
                        with open(metrics_path) as f:
                            m = json.load(f)
                        if m.get("degraded_missing_ds_gp"):
                            degraded_any = True
                    except Exception:
                        pass
            except Exception as e:
                print(f"Error: {e}")
                degraded_any = True

    if degraded_any:
        print("ERROR: One or more tools missing DS/GP; metrics computed with degraded inputs.")
        sys.exit(1)


def stage_summary():
    """Aggregate metrics across all chromosomes and generate comprehensive report."""
    print("\n" + "=" * 60)
    print("STAGE: GENOME-WIDE SUMMARY")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    tools = ["beagle", "reagle", "impute5", "minimac", "glimpse", "eagleimp"]
    display_names = {
        "beagle": "Beagle 5.5",
        "reagle": "Reagle (Rust)",
        "impute5": "IMPUTE5",
        "minimac": "Minimac4",
        "glimpse": "GLIMPSE2",
        "eagleimp": "EagleImp"
    }

    final_metrics = []
    
    for tool in tools:
        print(f"\nProcessing {tool.upper()}...")
        
        # Aggregators
        total_sites_compared = 0
        total_time_sec = 0.0
        
        # Exact counts for rates
        agg_concordant = 0
        agg_genotypes = 0
        agg_nonref_concordant = 0
        agg_nonref_total = 0
        agg_switch_errors = 0
        agg_switch_opps = 0
        
        agg_tp = 0
        agg_fp = 0
        agg_fn = 0
        
        agg_n50_sum = 0.0
        agg_n50_count = 0
        
        # R2 sufficient stats
        r2_sum_t = 0.0
        r2_sum_i = 0.0
        r2_sum_ti = 0.0
        r2_sum_tt = 0.0
        r2_sum_ii = 0.0
        r2_n = 0
        
        # Rare R2 sufficient stats
        rare_sum_t = 0.0
        rare_sum_i = 0.0
        rare_sum_ti = 0.0
        rare_sum_tt = 0.0
        rare_sum_ii = 0.0
        rare_n = 0
        
        chromosomes_found = 0
        
        for chrom in range(1, 23):
            # Prefer JSON for exact stats
            json_file = script_dir / f"data_chr{chrom}" / f"{tool}_metrics.json"
            
            if json_file.exists():
                chromosomes_found += 1
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        
                    total_sites_compared += data.get("sites_compared", 0)
                    total_time_sec += data.get("calculation_time_sec", 0.0)
                    
                    # Concordance counts
                    n_genotypes = data.get("total_genotypes", 0)
                    agg_genotypes += n_genotypes
                    if "unphased_concordant_count" in data:
                        agg_concordant += data.get("unphased_concordant_count", 0)
                    else:
                        conc_rate = data.get("unphased_concordance", 0)
                        agg_concordant += int(conc_rate * n_genotypes)
                    
                    # Non-ref
                    nr_total = data.get("nonref_total", 0)
                    agg_nonref_total += nr_total
                    if "nonref_concordant_count" in data:
                        agg_nonref_concordant += data.get("nonref_concordant_count", 0)
                    else:
                        nr_rate = data.get("nonref_concordance", 0)
                        agg_nonref_concordant += int(nr_rate * nr_total)
                    
                    # F1/Prec/Recall
                    agg_tp += data.get("tp", 0)
                    agg_fp += data.get("fp", 0)
                    agg_fn += data.get("fn", 0)
                    
                    # N50
                    if "n50_phase_block" in data:
                        agg_n50_sum += data["n50_phase_block"]
                        agg_n50_count += 1
                    
                    # Switch Error
                    agg_switch_errors += data.get("switch_errors", 0)
                    agg_switch_opps += data.get("switch_opportunities", 0)
                    
                    # R2 stats
                    stats = data.get("r2_stats")
                    if stats:
                        r2_sum_t += stats["sum_t"]
                        r2_sum_i += stats["sum_i"]
                        r2_sum_ti += stats["sum_ti"]
                        r2_sum_tt += stats["sum_tt"]
                        r2_sum_ii += stats["sum_ii"]
                        r2_n += stats["count"]
                        
                    # Rare R2 stats
                    rstats = data.get("rare_r2_stats")
                    if rstats:
                        rare_sum_t += rstats["sum_t"]
                        rare_sum_i += rstats["sum_i"]
                        rare_sum_ti += rstats["sum_ti"]
                        rare_sum_tt += rstats["sum_tt"]
                        rare_sum_ii += rstats["sum_ii"]
                        rare_n += rstats["count"]
                        
                except Exception as e:
                    print(f"  Error reading chr{chrom} JSON: {e}")
            else:
                pass 
                # print(f"  chr{chrom}: MISSING")

        if chromosomes_found == 0:
            continue

        # Calculate exact global metrics
        global_conc = agg_concordant / agg_genotypes if agg_genotypes > 0 else 0.0
        global_nonref = agg_nonref_concordant / agg_nonref_total if agg_nonref_total > 0 else 0.0
        global_ser = agg_switch_errors / agg_switch_opps if agg_switch_opps > 0 else 0.0
        
        # Global F1/Prec/Recall
        global_prec = agg_tp / (agg_tp + agg_fp) if (agg_tp + agg_fp) > 0 else 0.0
        global_rec = agg_tp / (agg_tp + agg_fn) if (agg_tp + agg_fn) > 0 else 0.0
        global_f1 = 2 * global_prec * global_rec / (global_prec + global_rec) if (global_prec + global_rec) > 0 else 0.0
        
        # Mean N50 across chromosomes (simple average for summary)
        global_n50 = agg_n50_sum / agg_n50_count if agg_n50_count > 0 else 0.0
        
        # Calculate exact GLOBAL dosage R2
        global_r2 = 0.0
        if r2_n > 0:
            mean_t = r2_sum_t / r2_n
            mean_i = r2_sum_i / r2_n
            
            # Covariance * N
            cov_n = r2_sum_ti - (r2_sum_t * r2_sum_i / r2_n)
            # Variance * N
            var_t_n = r2_sum_tt - (r2_sum_t * r2_sum_t / r2_n)
            var_i_n = r2_sum_ii - (r2_sum_i * r2_sum_i / r2_n)
            
            if var_t_n > 0 and var_i_n > 0:
                r = cov_n / math.sqrt(var_t_n * var_i_n)
                global_r2 = r ** 2
                
        # Calculate exact GLOBAL Rare R2
        global_rare_r2 = 0.0
        if rare_n > 0:
            mean_t = rare_sum_t / rare_n
            mean_i = rare_sum_i / rare_n
            
            cov_n = rare_sum_ti - (rare_sum_t * rare_sum_i / rare_n)
            var_t_n = rare_sum_tt - (rare_sum_t * rare_sum_t / rare_n)
            var_i_n = rare_sum_ii - (rare_sum_i * rare_sum_i / rare_n)
            
            if var_t_n > 0 and var_i_n > 0:
                r = cov_n / math.sqrt(var_t_n * var_i_n)
                global_rare_r2 = r ** 2

        # Aggregate MAF bin stats genome-wide
        maf_bin_agg = {}
        bin_order = ["ultra-rare (<0.1%)", "very-rare (0.1-0.5%)", "rare (0.5-1%)",
                     "low-freq (1-5%)", "medium (5-20%)", "common (>20%)"]

        for chrom in range(1, 23):
            json_file = script_dir / f"data_chr{chrom}" / f"{tool}_metrics.json"
            if json_file.exists():
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    for maf_bin, bin_data in data.get("by_maf", {}).items():
                        agg = bin_data.get("agg_stats")
                        if agg:
                            if maf_bin not in maf_bin_agg:
                                maf_bin_agg[maf_bin] = {
                                    "sum_t": 0, "sum_i": 0, "sum_ti": 0,
                                    "sum_tt": 0, "sum_ii": 0, "count": 0,
                                    "concordant": 0, "nonref_concordant": 0,
                                    "nonref_total": 0, "switch_err": 0,
                                    "switch_opp": 0, "tp": 0, "fp": 0, "fn": 0
                                }
                            for k in maf_bin_agg[maf_bin]:
                                maf_bin_agg[maf_bin][k] += agg.get(k, 0)
                except:
                    pass

        # Calculate per-bin metrics
        maf_metrics = {}
        for maf_bin in bin_order:
            if maf_bin in maf_bin_agg:
                agg = maf_bin_agg[maf_bin]
                n = agg["count"]
                if n > 0:
                    conc = agg["concordant"] / n
                    # R²
                    mean_t = agg["sum_t"] / n
                    mean_i = agg["sum_i"] / n
                    cov_n = agg["sum_ti"] - (agg["sum_t"] * agg["sum_i"] / n)
                    var_t_n = agg["sum_tt"] - (agg["sum_t"] ** 2 / n)
                    var_i_n = agg["sum_ii"] - (agg["sum_i"] ** 2 / n)
                    r2 = 0.0
                    if var_t_n > 0 and var_i_n > 0:
                        r = cov_n / math.sqrt(var_t_n * var_i_n)
                        r2 = r ** 2
                    # F1
                    tp, fp, fn = agg["tp"], agg["fp"], agg["fn"]
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                    # SER
                    ser = agg["switch_err"] / agg["switch_opp"] if agg["switch_opp"] > 0 else 0.0

                    maf_metrics[maf_bin] = {"r2": r2, "conc": conc, "f1": f1, "ser": ser, "n": n}

        # Collect additional metrics from first chromosome with data
        confusion_matrix = None
        homref_acc = het_acc = homalt_acc = 0.0
        homref_total = het_total = homalt_total = 0
        iqs = info_score = sen_mean = 0.0
        sample_conc_mean = sample_conc_min = sample_conc_max = 0.0
        sample_r2_mean = sample_r2_min = 0.0
        sample_sen_mean = sample_sen_min = sample_sen_max = 0.0
        sample_ser_mean = sample_ser_min = sample_ser_max = 0.0
        masked_total = masked_conc = masked_nonref_conc = masked_r2 = 0

        # Get these from aggregate or first available chromosome
        for chrom in range(1, 23):
            json_file = script_dir / f"data_chr{chrom}" / f"{tool}_metrics.json"
            if json_file.exists():
                try:
                    with open(json_file) as f:
                        data = json.load(f)

                    # Aggregate confusion matrix
                    if confusion_matrix is None and "confusion_matrix" in data:
                        confusion_matrix = [[0]*3 for _ in range(3)]
                    if "confusion_matrix" in data:
                        for i in range(3):
                            for j in range(3):
                                confusion_matrix[i][j] += data["confusion_matrix"][i][j]

                    # Aggregate class accuracies
                    homref_total += data.get("homref_total", 0)
                    het_total += data.get("het_total", 0)
                    homalt_total += data.get("homalt_total", 0)

                    # Masked metrics
                    masked_total += data.get("masked_total", 0)

                except:
                    pass

        # Calculate accuracies from confusion matrix
        if confusion_matrix:
            homref_acc = confusion_matrix[0][0] / homref_total if homref_total > 0 else 0.0
            het_acc = confusion_matrix[1][1] / het_total if het_total > 0 else 0.0
            homalt_acc = confusion_matrix[2][2] / homalt_total if homalt_total > 0 else 0.0

        # Get sample stats and other metrics from first chromosome (these don't aggregate well)
        for chrom in range(1, 23):
            json_file = script_dir / f"data_chr{chrom}" / f"{tool}_metrics.json"
            if json_file.exists():
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    iqs = data.get("iqs", 0.0)
                    info_score = data.get("info_score_approx", 0.0)
                    sen_mean = data.get("sen_mean", 0.0)
                    sample_conc_mean = data.get("sample_concordance_mean", 0.0)
                    sample_conc_min = data.get("sample_concordance_min", 0.0)
                    sample_conc_max = data.get("sample_concordance_max", 0.0)
                    sample_r2_mean = data.get("sample_r2_mean", 0.0)
                    sample_r2_min = data.get("sample_r2_min", 0.0)
                    sample_sen_mean = data.get("sample_sen_mean", 0.0)
                    sample_sen_min = data.get("sample_sen_min", 0.0)
                    sample_sen_max = data.get("sample_sen_max", 0.0)
                    sample_ser_mean = data.get("sample_switch_error_mean", 0.0)
                    sample_ser_min = data.get("sample_switch_error_min", 0.0)
                    sample_ser_max = data.get("sample_switch_error_max", 0.0)
                    masked_conc = data.get("masked_concordance", 0.0)
                    masked_nonref_conc = data.get("masked_nonref_concordance", 0.0)
                    masked_r2 = data.get("masked_r_squared", 0.0)
                    break
                except:
                    pass

        final_metrics.append({
            'id': tool,
            'name': display_names[tool],
            'time': total_time_sec,
            'sites': total_sites_compared,
            'genotypes': agg_genotypes,
            'conc': global_conc,
            'nonref': global_nonref,
            'nonref_total': agg_nonref_total,
            'r2': global_r2,
            'rare_r2': global_rare_r2,
            'iqs': iqs,
            'info_score': info_score,
            'sen_mean': sen_mean,
            'ser': global_ser,
            'switch_errors': agg_switch_errors,
            'switch_opps': agg_switch_opps,
            'f1': global_f1,
            'prec': global_prec,
            'rec': global_rec,
            'tp': agg_tp,
            'fp': agg_fp,
            'fn': agg_fn,
            'n50': global_n50,
            'confusion_matrix': confusion_matrix,
            'homref_acc': homref_acc,
            'het_acc': het_acc,
            'homalt_acc': homalt_acc,
            'homref_total': homref_total,
            'het_total': het_total,
            'homalt_total': homalt_total,
            'sample_conc_mean': sample_conc_mean,
            'sample_conc_min': sample_conc_min,
            'sample_conc_max': sample_conc_max,
            'sample_r2_mean': sample_r2_mean,
            'sample_r2_min': sample_r2_min,
            'sample_sen_mean': sample_sen_mean,
            'sample_sen_min': sample_sen_min,
            'sample_sen_max': sample_sen_max,
            'sample_ser_mean': sample_ser_mean,
            'sample_ser_min': sample_ser_min,
            'sample_ser_max': sample_ser_max,
            'masked_total': masked_total,
            'masked_conc': masked_conc,
            'masked_nonref_conc': masked_nonref_conc,
            'masked_r2': masked_r2,
            'chromosomes': chromosomes_found,
            'maf_metrics': maf_metrics
        })
    
    if not final_metrics:
        print("No metrics found.")
        return

    # -- Sort by R2 (descending) --
    final_metrics.sort(key=lambda x: x['r2'], reverse=True)

    # Determine which chromosomes were tested
    chromosomes_tested = set()
    for chrom in range(1, 23):
        json_file = script_dir / f"data_chr{chrom}" / f"{tools[0]}_metrics.json"
        if json_file.exists():
            chromosomes_tested.add(chrom)

    # Format chromosome range description
    if len(chromosomes_tested) == 22:
        chr_desc = "All 22 autosomes"
    elif len(chromosomes_tested) == 1:
        chr_desc = f"Chromosome {list(chromosomes_tested)[0]}"
    else:
        chr_list = sorted(chromosomes_tested)
        chr_desc = f"Chromosomes {', '.join(map(str, chr_list))}"

    # -- Generate Markdown Report --
    md_lines = []
    md_lines.append("# 🧬 Imputation Benchmark Results")
    md_lines.append(f"**{chr_desc}**")
    md_lines.append(f"*Metrics aggregated exactly across all sites (Dosage R²).*")
    md_lines.append(f"")
    md_lines.append(f"**Test Setup:** All tools receive pre-phased genotype array data (GSA v2 sites) as input for fair comparison. Reference panel: HGDP+1kG phased haplotypes.")
    
    # Winner badges
    best_r2 = max(final_metrics, key=lambda x: x['r2'])
    best_time = min(final_metrics, key=lambda x: x['time'])
    best_f1 = max(final_metrics, key=lambda x: x['f1'])
    
    final_metrics_with_ser = [m for m in final_metrics if m['ser'] > 0]
    best_ser = min(final_metrics_with_ser, key=lambda x: x['ser']) if final_metrics_with_ser else None
    
    final_metrics_with_n50 = [m for m in final_metrics if m['n50'] > 0]
    best_n50 = max(final_metrics_with_n50, key=lambda x: x['n50']) if final_metrics_with_n50 else None

    # Best rare variant (key differentiator)
    best_rare = max(final_metrics, key=lambda x: x['rare_r2'])

    # Speedup calculation (vs Beagle as baseline)
    beagle_stats = next((x for x in final_metrics if x['id'] == 'beagle'), None)

    md_lines.append(f"\n### 🏆 Highlights")
    md_lines.append(f"- **Most Accurate (R²):** {best_r2['name']} ({best_r2['r2']:.4f})")
    md_lines.append(f"- **Best Rare Variants (R² <1%):** {best_rare['name']} ({best_rare['rare_r2']:.4f})")
    md_lines.append(f"- **Best F1 Score:** {best_f1['name']} ({best_f1['f1']:.4f})")
    md_lines.append(f"- **Fastest:** {best_time['name']} ({best_time['time']:.1f}s)")
    if beagle_stats:
        reagle_stats = next((x for x in final_metrics if x['id'] == 'reagle'), None)
        if reagle_stats and beagle_stats['time'] > 0:
            speedup = beagle_stats['time'] / reagle_stats['time']
            md_lines.append(f"- **Reagle Speedup:** {speedup:.1f}x faster than Beagle")
    if best_ser and best_ser['ser'] < 1.0:
        md_lines.append(f"- **Best Phasing (SER):** {best_ser['name']} ({best_ser['ser']:.4f})")
    if best_n50:
        md_lines.append(f"- **Longest Phase Blocks (N50):** {best_n50['name']} ({best_n50['n50']:.0f} bp)")
    
    md_lines.append(f"\n### 📊 Accuracy Metrics")
    md_lines.append("*Primary imputation quality metrics*\n")
    md_lines.append("| Tool | Dosage R² | Rare R² (<1%) | IQS | INFO Score | Concordance | Non-Ref Conc. | Precision | Recall | F1 Score | SEN (mean) |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m in final_metrics:
        # Comparison vs Reagle (if present)
        r2_diff = ""
        reagle_stats = next((x for x in final_metrics if x['id'] == 'reagle'), None)
        if reagle_stats and m['id'] != 'reagle':
            diff = m['r2'] - reagle_stats['r2']
            icon = "🔻" if diff < 0 else "🔺"
            r2_diff = f" ({icon}{abs(diff):.4f})"

        md_lines.append(f"| **{m['name']}** | {m['r2']:.4f}{r2_diff} | {m['rare_r2']:.4f} | {m['iqs']:.4f} | {m['info_score']:.4f} | {m['conc']:.4f} | {m['nonref']:.4f} | {m['prec']:.4f} | {m['rec']:.4f} | {m['f1']:.4f} | {m['sen_mean']:.4f} |")

    md_lines.append(f"\n### 🔀 Phasing Quality")
    md_lines.append("*Haplotype phasing accuracy metrics*\n")
    md_lines.append("| Tool | Switch Error Rate | Switch Errors | Switch Opportunities | N50 Phase Block (bp) |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for m in final_metrics:
        md_lines.append(f"| **{m['name']}** | {m['ser']:.4f} | {m['switch_errors']:,} | {m['switch_opps']:,} | {m['n50']:.0f} |")

    md_lines.append(f"\n### 📈 Per-Class Accuracy")
    md_lines.append("*Genotype calling accuracy by zygosity class*\n")
    md_lines.append("| Tool | HomRef Acc. | Het Acc. | HomAlt Acc. | HomRef N | Het N | HomAlt N |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m in final_metrics:
        md_lines.append(f"| **{m['name']}** | {m['homref_acc']:.4f} | {m['het_acc']:.4f} | {m['homalt_acc']:.4f} | {m['homref_total']:,} | {m['het_total']:,} | {m['homalt_total']:,} |")

    md_lines.append(f"\n### 👥 Per-Sample Statistics")
    md_lines.append("*Distribution of metrics across samples*\n")
    md_lines.append("| Tool | Conc. Mean | Conc. Min | Conc. Max | R² Mean | R² Min | SEN Mean | SEN Min | SEN Max | SER Mean | SER Min | SER Max |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m in final_metrics:
        md_lines.append(f"| **{m['name']}** | {m['sample_conc_mean']:.4f} | {m['sample_conc_min']:.4f} | {m['sample_conc_max']:.4f} | {m['sample_r2_mean']:.4f} | {m['sample_r2_min']:.4f} | {m['sample_sen_mean']:.4f} | {m['sample_sen_min']:.4f} | {m['sample_sen_max']:.4f} | {m['sample_ser_mean']:.4f} | {m['sample_ser_min']:.4f} | {m['sample_ser_max']:.4f} |")

    md_lines.append(f"\n### 📊 Overall Statistics")
    md_lines.append("*Dataset size and runtime*\n")
    md_lines.append("| Tool | Sites Compared | Genotypes | Time (s) | Speedup vs Beagle |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for m in final_metrics:
        speedup_str = "-"
        if beagle_stats and beagle_stats['time'] > 0:
            if m['id'] == 'beagle':
                speedup_str = "1.0x"
            else:
                speedup = beagle_stats['time'] / m['time'] if m['time'] > 0 else 0
                speedup_str = f"{speedup:.1f}x"
        md_lines.append(f"| **{m['name']}** | {m['sites']:,} | {m['genotypes']:,} | {m['time']:.1f} | {speedup_str} |")

    # Confusion matrices
    md_lines.append(f"\n### 📋 Confusion Matrices")
    md_lines.append("*Truth (rows) vs Imputed (columns): HomRef, Het, HomAlt*\n")

    for m in final_metrics:
        md_lines.append(f"\n**{m['name']}:**")
        if m['confusion_matrix']:
            cm = m['confusion_matrix']
            md_lines.append("```")
            md_lines.append("              HomRef        Het     HomAlt")
            md_lines.append(f"  HomRef  {cm[0][0]:>12,} {cm[0][1]:>10,} {cm[0][2]:>10,}")
            md_lines.append(f"  Het     {cm[1][0]:>12,} {cm[1][1]:>10,} {cm[1][2]:>10,}")
            md_lines.append(f"  HomAlt  {cm[2][0]:>12,} {cm[2][1]:>10,} {cm[2][2]:>10,}")
            md_lines.append("```")
        else:
            md_lines.append("*No confusion matrix data available*")

    # Masked SNP metrics
    md_lines.append(f"\n### 🧪 Masked-SNP Metrics")
    md_lines.append("*Quality assessment on held-out proxy variants*\n")
    md_lines.append("| Tool | Masked Total | Masked Concordance | Masked Non-Ref Conc. | Masked R² |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: |")

    for m in final_metrics:
        md_lines.append(f"| **{m['name']}** | {m['masked_total']} | {m['masked_conc']:.4f} | {m['masked_nonref_conc']:.4f} | {m['masked_r2']:.4f} |")

    # Raw counts table
    md_lines.append(f"\n### 🔢 Raw Counts")
    md_lines.append("*Underlying counts used to calculate metrics above*\n")
    md_lines.append("| Tool | True Positives | False Positives | False Negatives | Non-Ref Total | Switch Errors | Switch Opportunities |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m in final_metrics:
        md_lines.append(f"| **{m['name']}** | {m['tp']:,} | {m['fp']:,} | {m['fn']:,} | {m['nonref_total']:,} | {m['switch_errors']:,} | {m['switch_opps']:,} |")

    # MAF-stratified performance comparison table
    bin_order = ["ultra-rare (<0.1%)", "very-rare (0.1-0.5%)", "rare (0.5-1%)",
                 "low-freq (1-5%)", "medium (5-20%)", "common (>20%)"]
    bin_labels = {"ultra-rare (<0.1%)": "Ultra-rare (<0.1%)",
                  "very-rare (0.1-0.5%)": "Very-rare (0.1-0.5%)",
                  "rare (0.5-1%)": "Rare (0.5-1%)",
                  "low-freq (1-5%)": "Low-freq (1-5%)",
                  "medium (5-20%)": "Medium (5-20%)",
                  "common (>20%)": "Common (>20%)"}

    md_lines.append(f"\n### 📈 MAF-Stratified Performance (R²)")
    md_lines.append("*Dosage R² by Minor Allele Frequency bin - key metric for rare variant imputation quality*\n")

    # Header row
    header = "| MAF Bin |"
    sep = "| :--- |"
    for m in final_metrics:
        header += f" {m['name']} |"
        sep += " :---: |"
    md_lines.append(header)
    md_lines.append(sep)

    # Data rows
    for maf_bin in bin_order:
        row = f"| {bin_labels.get(maf_bin, maf_bin)} |"
        for m in final_metrics:
            maf_m = m.get('maf_metrics', {}).get(maf_bin)
            if maf_m:
                r2_val = maf_m['r2']
                row += f" {r2_val:.4f} |"
            else:
                row += " - |"
        md_lines.append(row)

    # Add N counts row
    row = "| **N genotypes** |"
    for m in final_metrics:
        total_n = sum(bm.get('n', 0) for bm in m.get('maf_metrics', {}).values())
        row += f" {total_n:,} |"
    md_lines.append(row)

    # Write to Summary file
    summary_file = script_dir / "genome_wide_summary.md"
    with open(summary_file, 'w') as f:
        f.write('\n'.join(md_lines))
        
    print(f"\nSummary written to: {summary_file}")
    
    # Print to console
    print('\n'.join(md_lines))
    
    # If running in GHA, append to job summary
    gha_summary = os.getenv('GITHUB_STEP_SUMMARY')
    if gha_summary:
        with open(gha_summary, 'a') as f:
            f.write('\n'.join(md_lines))


if __name__ == "__main__":
    main()
