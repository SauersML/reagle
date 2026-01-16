#!/usr/bin/env python3
"""
Integration test for Reagle (Rust Beagle implementation).

This test:
1. Downloads HGDP+1kG chr22 reference panel from gnomAD
2. Splits into reference (80%) and target (20%) panels
3. Downsamples target to GSAv3 array sites
4. Runs both Java Beagle and Reagle for imputation
5. Calculates imputation accuracy metrics (R², concordance, etc.)

Requirements:
- bcftools, tabix
- Java 11+ (for Beagle)
- Reagle binary (cargo build --release)

Usage:
  python integration_test.py              # Run all stages
  python integration_test.py prepare      # Download data and prepare VCFs
  python integration_test.py beagle       # Run Beagle imputation only
  python integration_test.py reagle       # Run Reagle imputation only
  python integration_test.py metrics      # Calculate metrics only
"""

import os
import sys
import subprocess
import random
import gzip
import json
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
    if "No BGZF EOF marker" in stderr or "Failed to read BGZF block" in stderr:
        return False
    return True


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


def resolve_region_arg(paths, chrom):
    """Prefer ref bounds; fall back to input bounds; then chrom only."""
    ref_bounds = get_chrom_bounds(paths["ref_vcf"], chrom)
    if ref_bounds:
        return f"chr{chrom}:{ref_bounds[0]}-{ref_bounds[1]}"
    input_bounds = get_chrom_bounds(paths["input_vcf"], chrom)
    if input_bounds:
        return f"chr{chrom}:{input_bounds[0]}-{input_bounds[1]}"
    return f"chr{chrom}"


def print_tool_help(label, cmd):
    try:
        result = run(f"{cmd} --help 2>&1 | head -5", capture=True, check=False)
        if result.stdout:
            print(f"{label} --help output:\n{result.stdout.strip()}")
    except Exception as e:
        print(f"Warning: {label} --help check failed: {e}")
    else:
        subprocess.check_call(cmd, shell=True)


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


def run_beagle(ref_vcf, target_vcf, output_prefix, beagle_jar, nthreads=2):
    """Run Java Beagle for imputation."""
    cmd = f"java -Xmx4g -jar {beagle_jar} ref={ref_vcf} gt={target_vcf} out={output_prefix} nthreads={nthreads} gp=true"
    try:
        run(cmd)
        output = f"{output_prefix}.vcf.gz"
        if os.path.exists(output):
            run(f"bcftools index -f {output}")
        return output
    except subprocess.CalledProcessError as e:
        print(f"Beagle failed: {e}")
        return None


def run_reagle(ref_vcf, target_vcf, output_prefix, reagle_bin):
    """Run Reagle for imputation."""
    output_vcf = f"{output_prefix}.vcf.gz"
    cmd = f"{reagle_bin} --ref {ref_vcf} --gt {target_vcf} --out {output_prefix} --gp"
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


def calculate_dosage(gt):
    """Calculate ALT dosage from genotype tuple."""
    if gt is None:
        return None
    return gt[0] + gt[1]


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


def _parse_truth_line(line, samples):
    """Parse a truth VCF line into (key, sample_data_dict)."""
    parts = line.split('\t')
    if len(parts) < 5:
        return None, None
    chrom, pos = parts[0], int(parts[1])
    key = (chrom, pos)
    gts = parts[4:]
    sample_data = {}
    for i, gt_str in enumerate(gts):
        if i < len(samples):
            gt_field = gt_str.split(':')[0]
            gt = parse_genotype(gt_field)
            is_phased = '|' in gt_field
            if gt is not None:
                sample_data[samples[i]] = (gt, calculate_dosage(gt), is_phased)
    return key, sample_data


def _parse_imputed_line(line, samples):
    """Parse an imputed VCF line into (key, sample_data_dict)."""
    parts = line.split('\t')
    if len(parts) < 5:
        return None, None
    chrom, pos = parts[0], int(parts[1])
    key = (chrom, pos)
    sample_data_list = parts[4:]
    sample_data = {}
    for i, data_str in enumerate(sample_data_list):
        if i < len(samples):
            fields = data_str.split(':')
            gt_field = fields[0]
            gt = parse_genotype(gt_field)
            is_phased = '|' in gt_field
            
            # Expecting GT:DS:GP from bcftools query
            ds = None
            gp = None
            
            # Parse DS (Estimated Dosage)
            if len(fields) > 1 and fields[1] != '.':
                try:
                    ds = float(fields[1])
                except ValueError:
                    pass
            
            # Parse GP (Genotype Probabilities)
            if len(fields) > 2 and fields[2] != '.':
                try:
                    gp_parts = fields[2].split(',')
                    if len(gp_parts) == 3:
                        gp = (float(gp_parts[0]), float(gp_parts[1]), float(gp_parts[2]))
                except:
                    pass
            
            # Fallback to hard-call dosage if DS missing
            if ds is None and gt is not None:
                ds = calculate_dosage(gt)

            if gt is not None:
                sample_data[samples[i]] = (gt, ds, is_phased, gp)
    return key, sample_data


def calculate_metrics(truth_vcf, imputed_vcf, output_prefix):
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

    if not imputed_vcf or not os.path.exists(imputed_vcf):
        print("Imputed VCF not found")
        return None

    print(f"\nCalculating metrics: {imputed_vcf} vs {truth_vcf}")

    # Get sample lists first (small memory footprint)
    samples_result = run(f"bcftools query -l {truth_vcf}", capture=True)
    truth_samples = samples_result.stdout.strip().split('\n')
    samples_result = run(f"bcftools query -l {imputed_vcf}", capture=True)
    imputed_samples = samples_result.stdout.strip().split('\n')

    print(f"Truth samples: {len(truth_samples)}, Imputed samples: {len(imputed_samples)}")

    # Build sample index for common samples
    common_samples = set(truth_samples) & set(imputed_samples)
    if not common_samples:
        print("ERROR: No common samples between truth and imputed VCFs")
        return None
    print(f"Common samples: {len(common_samples)}")

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

    # Per-sample tracking - use online stats instead of storing all dosages
    sample_metrics = defaultdict(lambda: {
        "concordant": 0, "total": 0,
        "sum_t": 0.0, "sum_i": 0.0, "sum_ti": 0.0, "sum_tt": 0.0, "sum_ii": 0.0,
        "switch_errors": 0, "switch_opportunities": 0
    })

    # For IQS calculation: track per-site concordance and expected concordance
    site_iqs_values = []

    # For Hellinger score (requires GP field)
    hellinger_sum = 0.0
    hellinger_count = 0

    # For switch error rate
    switch_errors = 0
    switch_opportunities = 0

    # For N50 Phasing Block Length
    # sample -> list of block lengths (in bp)
    phase_blocks = defaultdict(list)
    # sample -> start position of current block
    current_block_start = {}

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
        "iqs_values": [], "nonref_concordant": 0, "nonref_total": 0,
        "confusion": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "switch_errors": 0, "switch_opportunities": 0
    })

    # === STREAMING SETUP ===
    print("Initializing streams...")
    
    # 1. Truth Stream
    truth_cmd = f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT[\\t%GT]\\n' {truth_vcf}"
    truth_iter = _stream_vcf_lines(truth_cmd)

    # 2. Imputed Stream
    # Request GT:DS:GP. If failing, bcftools usually outputs '.' or we handle it in parser.
    # We use a broad request and rely on parser robustness.
    imputed_cmd = f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT[\\t%GT:%DS:%GP]\\n' {imputed_vcf}"
    imputed_iter = _stream_vcf_lines(imputed_cmd)

    # Helper to get next parsed line
    def get_next_truth():
        try:
            line = next(truth_iter)
            return _parse_truth_line(line, truth_samples)
        except StopIteration:
            return None, None

    def get_next_imputed():
        try:
            line = next(imputed_iter)
            return _parse_imputed_line(line, imputed_samples)
        except StopIteration:
            return None, None

    # Initial fetch
    truth_key, truth_data = get_next_truth()
    imp_key, imp_data = get_next_imputed()

    # Track previous het for switch error calculation per sample
    prev_het = {}  # sample -> (site, truth_gt, imputed_gt, maf_bin)
    common_sites_count = 0
    last_pos = 0

    print("Streaming and comparing...")
    
    # Merge-join loop
    while truth_key is not None and imp_key is not None:
        t_chrom, t_pos = truth_key
        i_chrom, i_pos = imp_key

        # Compare positions (Chrom then Pos)
        # Handle string chromosome comparison carefully if needed, 
        # but typical numeric/lexicographic sort holds for same reference.
        
        if t_chrom == i_chrom:
            if t_pos == i_pos:
                # MATCH! Process site
                common_sites_count += 1
                site = truth_key
                last_pos = site[1]
                truth_site = truth_data
                imputed_site = imp_data

                # --- METRICS CALCULATION LOGIC (same as before) ---
                
                # Calculate MAF from truth
                dosages_at_site = [v[1] for v in truth_site.values() if v[1] is not None]
                if dosages_at_site:
                    af = sum(dosages_at_site) / (2 * len(dosages_at_site))
                    maf = min(af, 1 - af)
                else:
                    maf = 0

                maf_bin = get_maf_bin(maf)
                site_concordant = 0
                site_total = 0

                for sample in truth_site:
                    if sample in imputed_site:
                        t_gt, t_dos, t_phased = truth_site[sample]
                        imp_values = imputed_site[sample]
                        i_gt, i_dos, i_phased = imp_values[0], imp_values[1], imp_values[2]
                        i_gp = imp_values[3] if len(imp_values) > 3 else None

                        if t_dos is not None and i_dos is not None:
                            # Hellinger score
                            if i_gp is not None:
                                t_gp = (1.0, 0.0, 0.0) if t_dos == 0 else ((0.0, 1.0, 0.0) if t_dos == 1 else (0.0, 0.0, 1.0))
                                bc = sum(math.sqrt(t * i) for t, i in zip(t_gp, i_gp))
                                hellinger_dist = math.sqrt(max(0, 1 - bc))
                                h_score = 1 - hellinger_dist
                                hellinger_sum += h_score
                                hellinger_count += 1
                            
                            total_compared += 1
                            site_total += 1

                            # Online R² stats
                            r2_stats["sum_t"] += t_dos
                            r2_stats["sum_i"] += i_dos
                            r2_stats["sum_ti"] += t_dos * i_dos
                            r2_stats["sum_tt"] += t_dos * t_dos
                            r2_stats["sum_ii"] += i_dos * i_dos
                            r2_stats["count"] += 1

                            # MAF bin stats
                            maf_bins[maf_bin]["sum_t"] += t_dos
                            maf_bins[maf_bin]["sum_i"] += i_dos
                            maf_bins[maf_bin]["sum_ti"] += t_dos * i_dos
                            maf_bins[maf_bin]["sum_tt"] += t_dos * t_dos
                            maf_bins[maf_bin]["sum_ii"] += i_dos * i_dos
                            maf_bins[maf_bin]["total"] += 1

                            # Sample stats
                            sample_metrics[sample]["total"] += 1
                            sample_metrics[sample]["sum_t"] += t_dos
                            sample_metrics[sample]["sum_i"] += i_dos
                            sample_metrics[sample]["sum_ti"] += t_dos * i_dos
                            sample_metrics[sample]["sum_tt"] += t_dos * t_dos
                            sample_metrics[sample]["sum_ii"] += i_dos * i_dos
                            
                            # Confusion matrix
                            t_class = 0 if t_dos == 0 else (2 if t_dos == 2 else 1)
                            i_class = 0 if i_dos == 0 else (2 if i_dos == 2 else 1)
                            confusion[t_class][i_class] += 1
                            maf_bins[maf_bin]["confusion"][t_class][i_class] += 1

                            # Concordance
                            t_sorted = tuple(sorted(t_gt))
                            i_sorted = tuple(sorted(i_gt))
                            if t_sorted == i_sorted:
                                unphased_concordant += 1
                                site_concordant += 1
                                maf_bins[maf_bin]["unphased_concordant"] += 1
                                sample_metrics[sample]["concordant"] += 1
                            
                            # Non-ref concordance
                            if t_dos > 0:
                                nonref_total += 1
                                maf_bins[maf_bin]["nonref_total"] += 1
                                if t_sorted == i_sorted:
                                    nonref_concordant += 1
                                    maf_bins[maf_bin]["nonref_concordant"] += 1
                            
                            # Switch errors
                            if t_dos == 1 and i_dos == 1 and t_phased and i_phased:
                                pos = site[1]
                                if sample not in current_block_start:
                                    current_block_start[sample] = pos

                                if sample in prev_het:
                                    prev_site, prev_t_gt, prev_i_gt, prev_maf_bin = prev_het[sample]
                                    t_same_phase = (t_gt[0] == prev_t_gt[0])
                                    i_same_phase = (i_gt[0] == prev_i_gt[0])
                                    
                                    if t_same_phase != i_same_phase:
                                        block_len = pos - current_block_start[sample]
                                        phase_blocks[sample].append(block_len)
                                        current_block_start[sample] = pos
                                        
                                        switch_errors += 1
                                        sample_metrics[sample]["switch_errors"] += 1
                                        maf_bins[maf_bin]["switch_errors"] += 1
                                    
                                    switch_opportunities += 1
                                    sample_metrics[sample]["switch_opportunities"] += 1
                                    maf_bins[maf_bin]["switch_opportunities"] += 1
                                prev_het[sample] = (site, t_gt, i_gt, maf_bin)

                # IQS Calculation
                if site_total > 0 and maf > 0 and maf < 1:
                    p = af
                    q = 1 - p
                    expected_conc = (q*q)**2 + (2*p*q)**2 + (p*p)**2
                    observed_conc = site_concordant / site_total
                    if expected_conc < 1.0:
                        iqs = (observed_conc - expected_conc) / (1.0 - expected_conc)
                        site_iqs_values.append(iqs)
                        maf_bins[maf_bin]["iqs_values"].append(iqs)

                # Advance both
                truth_key, truth_data = get_next_truth()
                imp_key, imp_data = get_next_imputed()
            
            elif t_pos < i_pos:
                # Truth is behind, means site missing in imputation (or extra site in Truth)
                truth_key, truth_data = get_next_truth()
            else:
                # Imputed is behind, means extra site in Imputation
                imp_key, imp_data = get_next_imputed()
        
        elif t_chrom < i_chrom:
             truth_key, truth_data = get_next_truth()
        else:
             imp_key, imp_data = get_next_imputed()

    print(f"Common sites: {common_sites_count}")

    # Close final phase blocks
    for sample, start_pos in current_block_start.items():
        block_len = last_pos - start_pos
        phase_blocks[sample].append(block_len)

    # Calculate overall metrics
    metrics = {}
    
    # Calculate N50 Phase Block Length
    all_lengths = []
    for lengths in phase_blocks.values():
        all_lengths.extend(lengths)
    
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
    else:
        metrics["n50_phase_block"] = 0.0

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
        if site_iqs_values:
            metrics["iqs"] = sum(site_iqs_values) / len(site_iqs_values)
            metrics["iqs_median"] = sorted(site_iqs_values)[len(site_iqs_values) // 2]
        else:
            metrics["iqs"] = None
            
        # Per-sample summary statistics (using online stats)
        sample_concordances = []
        sample_r2s = []
        for sample, data in sample_metrics.items():
            if data["total"] > 0:
                sample_concordances.append(data["concordant"] / data["total"])
            n = data["total"]
            if n > 1:
                mean_t = data["sum_t"] / n
                mean_i = data["sum_i"] / n
                cov = data["sum_ti"] / n - mean_t * mean_i
                var_t = data["sum_tt"] / n - mean_t * mean_t
                var_i = data["sum_ii"] / n - mean_i * mean_i
                if var_t > 0 and var_i > 0:
                    r = cov / math.sqrt(var_t * var_i)
                    sample_r2s.append(r ** 2)
        
        if sample_concordances:
            metrics["sample_concordance_mean"] = sum(sample_concordances) / len(sample_concordances)
            metrics["sample_concordance_min"] = min(sample_concordances)
            metrics["sample_concordance_max"] = max(sample_concordances)
        if sample_r2s:
            metrics["sample_r2_mean"] = sum(sample_r2s) / len(sample_r2s)
            metrics["sample_r2_min"] = min(sample_r2s)

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
                if data["iqs_values"]:
                    bin_metrics["iqs"] = sum(data["iqs_values"]) / len(data["iqs_values"])
                # Switch error rate per bin
                if data["switch_opportunities"] > 0:
                    bin_metrics["switch_error_rate"] = data["switch_errors"] / data["switch_opportunities"]
                    bin_metrics["switch_errors"] = data["switch_errors"]
                    bin_metrics["switch_opportunities"] = data["switch_opportunities"]

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
                    "tp": b_tp, "fp": b_fp, "fn": b_fn
                }

                metrics["by_maf"][maf_bin] = bin_metrics
        
        # Per-sample switch error summary
        sample_switch_rates = []
        for sample, data in sample_metrics.items():
            if data["switch_opportunities"] > 0:
                sample_switch_rates.append(data["switch_errors"] / data["switch_opportunities"])
        if sample_switch_rates:
            metrics["sample_switch_error_mean"] = sum(sample_switch_rates) / len(sample_switch_rates)
            metrics["sample_switch_error_max"] = max(sample_switch_rates)
            metrics["sample_switch_error_max"] = max(sample_switch_rates)
            metrics["sample_switch_error_min"] = min(sample_switch_rates)

    elapsed = time.time() - start_time
    metrics["calculation_time_sec"] = elapsed

    # Save metrics to JSON for exact aggregation
    with open(output_prefix + "_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Print results
    print("\n" + "=" * 60)
    print("IMPUTATION METRICS - COMPREHENSIVE ANALYSIS")
    print("=" * 60)

    if metrics:
        print(f"\n📊 OVERALL STATISTICS")
        print(f"   Sites compared: {metrics.get('sites_compared', 'N/A'):,}")
        print(f"   Genotypes compared: {metrics.get('total_genotypes', 'N/A'):,}")
        print(f"   Calculation time: {metrics.get('calculation_time_sec', 0):.1f}s")
        
        print(f"\n🎯 ACCURACY METRICS")
        print(f"   Unphased concordance: {metrics.get('unphased_concordance', 0):.4f}")
        print(f"   Non-ref concordance:  {metrics.get('nonref_concordance', 0):.4f}" if metrics.get('nonref_concordance') else "   Non-ref concordance:  N/A")
        print(f"   F1 Score (Non-Ref):   {metrics.get('f1_score', 0):.4f}")
        print(f"   Precision / Recall:   {metrics.get('precision', 0):.4f} / {metrics.get('recall', 0):.4f}")
        print(f"   Overall R²:           {metrics.get('r_squared'):.4f}" if metrics.get('r_squared') else "   Overall R²:           N/A")
        print(f"   Overall IQS:          {metrics.get('iqs'):.4f}" if metrics.get('iqs') else "   Overall IQS:          N/A")
        print(f"   Hellinger Score:      {metrics.get('hellinger_score'):.4f}" if metrics.get('hellinger_score') else "   Hellinger Score:      N/A (no GP)")
        print(f"   INFO score (approx):  {metrics.get('info_score_approx'):.4f}" if metrics.get('info_score_approx') else "   INFO score (approx):  N/A")
        
        if metrics.get('switch_error_rate') is not None:
            print(f"\n🔀 PHASING QUALITY")
            print(f"   Switch error rate:    {metrics.get('switch_error_rate'):.4f} ({metrics.get('switch_errors')}/{metrics.get('switch_opportunities')})")
            print(f"   N50 Phase Block:      {metrics.get('n50_phase_block'):.0f} bp")
        
        print(f"\n📋 CONFUSION MATRIX (Truth vs Imputed)")
        print(f"   {'':12} {'HomRef':>10} {'Het':>10} {'HomAlt':>10}")
        labels = ['HomRef', 'Het', 'HomAlt']
        for i, label in enumerate(labels):
            row = metrics.get('confusion_matrix', [[0,0,0],[0,0,0],[0,0,0]])[i]
            print(f"   {label:12} {row[0]:>10,} {row[1]:>10,} {row[2]:>10,}")
        
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
        if metrics.get('sample_switch_error_mean') is not None:
            print(f"   Switch Err:  mean={metrics['sample_switch_error_mean']:.4f}, min={metrics['sample_switch_error_min']:.4f}, max={metrics['sample_switch_error_max']:.4f}")

        if "by_maf" in metrics:
            print(f"\n📈 BY MAF BIN (sorted by frequency)")
            print(f"   {'MAF Bin':<20} {'F1':>8} {'Conc':>8} {'R²':>8} {'SwitchErr':>10} {'N':>10}")
            print(f"   {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
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
                    print(f"   {maf_bin:<20} {f1_str:>8} {conc:>8} {r2_str:>8} {switch_str:>10} {bin_metrics['n_genotypes']:>10,}")

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
            if metrics.get('info_score_approx'):
                f.write(f"INFO score (approx): {metrics['info_score_approx']:.6f}\n")
            if metrics.get('hellinger_score'):
                f.write(f"Hellinger Score: {metrics['hellinger_score']:.6f}\n")
            if metrics.get('switch_error_rate') is not None:
                f.write(f"Switch error rate: {metrics['switch_error_rate']:.6f}\n")
                f.write(f"N50 Phase Block: {metrics.get('n50_phase_block'):.0f} bp\n")
            
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


def stage_prepare():
    """Download data and prepare reference/truth/input VCFs."""
    print("=" * 60)
    print("STAGE: PREPARE - Download and prepare data")
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

    if not validate_vcf(paths['chr22_vcf']):
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
    if not validate_vcf(paths['ref_vcf']):
        print("Creating reference panel...")
        paths['ref_vcf'].unlink(missing_ok=True)
        Path(str(paths['ref_vcf']) + ".csi").unlink(missing_ok=True)
        run(f"bcftools view -S {train_file} {paths['chr22_vcf']} -O z -o {paths['ref_vcf']}")
    ensure_index(paths['ref_vcf'], recreate_cmd=f"bcftools view -S {train_file} {paths['chr22_vcf']} -O z -o {paths['ref_vcf']}")

    # Create truth (test samples, full density)
    if not validate_vcf(paths['truth_vcf']):
        print("Creating truth VCF...")
        paths['truth_vcf'].unlink(missing_ok=True)
        Path(str(paths['truth_vcf']) + ".csi").unlink(missing_ok=True)
        run(f"bcftools view -S {test_file} {paths['chr22_vcf']} -O z -o {paths['truth_vcf']}")
    ensure_index(paths['truth_vcf'], recreate_cmd=f"bcftools view -S {test_file} {paths['chr22_vcf']} -O z -o {paths['truth_vcf']}")

    # Create input (test samples, downsampled to GSA sites, UNPHASED)
    # We unphase the input so switch error rate measures TRUE phasing accuracy
    tmp_phased_path = paths['data_dir'] / "input_phased_tmp.vcf.gz"
    if not validate_vcf(paths['input_vcf']):
        print("Downsampling to GSA sites and unphasing...")
        create_regions_file(gsa_sites, str(paths['gsa_regions']))
        # Two-step process: downsample, then unphase
        tmp_phased = str(tmp_phased_path)
        run(f"bcftools view -R {paths['gsa_regions']} {paths['truth_vcf']} -O z -o {tmp_phased}")
        # Unphase: convert 0|1 to 0/1 using bcftools +setGT
        # The plugin sets genotypes to unphased while preserving allele values
        run(f"bcftools +setGT {tmp_phased} -O z -o {paths['input_vcf']} -- -t a -n u")
        # Clean up temp file
        os.remove(tmp_phased)
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
    beagle_vcf = run_beagle(
        str(paths['ref_vcf']),
        str(paths['input_vcf']),
        str(paths['beagle_out']),
        str(paths['beagle_jar']),
        nthreads=2
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
    reagle_vcf = run_reagle(
        str(paths['ref_vcf']),
        str(paths['input_vcf']),
        str(paths['reagle_out']),
        str(paths['reagle_bin'])
    )

    if reagle_vcf and os.path.exists(reagle_vcf):
        print(f"\nReagle output: {reagle_vcf}")
        print("Reagle stage completed successfully.")
    else:
        print("\nERROR: Reagle imputation failed!")
        sys.exit(1)


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

    # Calculate metrics
    print("\n" + "=" * 60)
    print("Calculating accuracy metrics...")
    print("=" * 60)

    all_metrics = {}
    for name, vcf in results.items():
        print(f"\n{'=' * 50}")
        print(f"{name.upper()} RESULTS")
        print(f"{'=' * 50}")
        if vcf and os.path.exists(vcf):
            metrics = calculate_metrics(
                str(paths['truth_vcf']),
                vcf,
                str(paths['data_dir'] / f"{name}")
            )
            all_metrics[name] = metrics
        else:
            print(f"{name} output not found")
            all_metrics[name] = None

    # Load sample counts
    n_train = 0
    n_test = 0
    if paths['train_file'].exists():
        with open(paths['train_file']) as f:
            n_train = len([l for l in f if l.strip()])
    if paths['test_file'].exists():
        with open(paths['test_file']) as f:
            n_test = len([l for l in f if l.strip()])

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Reference panel: {n_train} samples")
    print(f"Test panel: {n_test} samples")
    print()

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

    # Exit with appropriate code
    if not any(m for m in all_metrics.values()):
        print("\nERROR: All metrics calculations failed!")
        sys.exit(1)

    print("\nMetrics stage completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Reagle Integration Test - HGDP+1kG Imputation Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  prepare      Download data and prepare reference/truth/input VCFs
  beagle       Run Java Beagle imputation
  reagle       Run Reagle imputation
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
  python integration_test.py impute5          # Run IMPUTE5
  python integration_test.py prepare-chr 1    # Prepare chr1 for full genome
  python integration_test.py summary          # Aggregate all chromosome metrics
        """
    )
    parser.add_argument(
        'stage',
        nargs='?',
        default='all',
        choices=['all', 'prepare', 'beagle', 'reagle', 'impute5', 'minimac', 
                 'glimpse', 'metrics', 'prepare-chr', 'impute-chr', 
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

    args = parser.parse_args()

    print("=" * 60)
    print("Reagle Integration Test - HGDP+1kG Imputation Benchmark")
    print("=" * 60)

    if args.stage == 'prepare':
        stage_prepare()
    elif args.stage == 'beagle':
        stage_beagle()
    elif args.stage == 'reagle':
        stage_reagle()
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
        stage_prepare()

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
        
        # Downsample and unphase
        tmp_phased = data_dir / "input_phased_tmp.vcf.gz"
        run(f"bcftools view -R {regions_file} {paths['truth_vcf']} -O z -o {tmp_phased}")
        run(f"bcftools +setGT {tmp_phased} -O z -o {paths['input_vcf']} -- -t a -n u")
        run(f"bcftools index -f {paths['input_vcf']}")
        os.remove(tmp_phased)
    
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


def run_beagle_chr(chrom, paths):
    """Run Beagle for a chromosome."""
    data_dir = paths['data_dir']
    beagle_jar = data_dir / "beagle.jar"
    out = data_dir / "beagle_imputed.vcf.gz"
    
    if not beagle_jar.exists():
        run(f"curl -L -o {beagle_jar} 'https://faculty.washington.edu/browning/beagle/beagle.22Jul22.46e.jar'")
    
    if not out.exists():
        run(f"java -Xmx8g -jar {beagle_jar} ref={paths['ref_vcf']} gt={paths['input_vcf']} out={data_dir}/beagle_imputed nthreads=4")
        run(f"bcftools index -f {out}")


def run_reagle_chr(chrom, paths):
    """Run Reagle for a chromosome."""
    out = paths['data_dir'] / "reagle_imputed.vcf.gz"
    if paths['reagle_bin'].exists() and not out.exists():
        run(f"{paths['reagle_bin']} --ref {paths['ref_vcf']} --gt {paths['input_vcf']} --out {paths['data_dir']}/reagle_imputed")
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
            region_arg = resolve_region_arg(paths, chrom)
            print_tool_help("IMPUTE5", str(impute5_bin))
            print(f"IMPUTE5 region: {region_arg}")
            print(f"IMPUTE5 ref: {paths['ref_vcf']}")
            print(f"IMPUTE5 input: {paths['input_vcf']}")
            run(f"{impute5_bin} --h {paths['ref_vcf']} --g {paths['input_vcf']} --r {region_arg} --buffer-region {region_arg} --o {out} --threads 4")
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
            # Minimac4: --refHaps ref.vcf --haps input.vcf --prefix out --region chr
            region_arg = resolve_region_arg(paths, chrom)
            print_tool_help("Minimac4", str(minimac_bin))
            print(f"Minimac4 region: {region_arg}")
            print(f"Minimac4 ref: {paths['ref_vcf']}")
            print(f"Minimac4 input: {paths['input_vcf']}")
            run(f"{minimac_bin} {paths['ref_vcf']} {paths['input_vcf']} --output {prefix}.dose.vcf.gz --threads 4 --format GT,DS --region {region_arg}")
            
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
            run(f"{glimpse_bin} --input-gl {paths['input_vcf']} --reference {paths['ref_vcf']} --input-region {region_arg} --output-region {region_arg} --output {bcf_out} --threads 4")
            run(f"bcftools view {bcf_out} -O z -o {out}")
            run(f"bcftools index -f {out}")
        except Exception as e:
            print(f"GLIMPSE failed on chr{chrom}: {e}")
    else:
        print(f"Using existing GLIMPSE output for chr{chrom}")


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
    ]
    
    for prefix, filename in tools:
        imputed_path = data_dir / filename
        if imputed_path.exists():
            print(f"\n{'=' * 40}")
            print(f"Calculating metrics for {prefix.upper()}")
            print("=" * 40)
            try:
                calculate_metrics(truth_vcf, str(imputed_path), str(data_dir / prefix))
            except Exception as e:
                print(f"Error: {e}")


def stage_summary():
    """Aggregate metrics across all chromosomes and generate comprehensive report."""
    print("\n" + "=" * 60)
    print("STAGE: GENOME-WIDE SUMMARY")
    print("=" * 60)
    
    script_dir = Path(__file__).parent
    tools = ["beagle", "reagle", "impute5", "minimac", "glimpse"]
    display_names = {
        "beagle": "Beagle 5.4",
        "reagle": "Reagle (Rust)",
        "impute5": "IMPUTE5",
        "minimac": "Minimac4",
        "glimpse": "GLIMPSE2"
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
                    conc_rate = data.get("unphased_concordance", 0)
                    agg_genotypes += n_genotypes
                    agg_concordant += int(conc_rate * n_genotypes) # Reconstruct count
                    
                    # Non-ref
                    nr_rate = data.get("nonref_concordance", 0)
                    nr_total = data.get("nonref_total", 0)
                    agg_nonref_total += nr_total
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
            json_file = script_dir / f"{tool}_chr{chrom}_metrics.json"
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

        final_metrics.append({
            'id': tool,
            'name': display_names[tool],
            'time': total_time_sec,
            'conc': global_conc,
            'nonref': global_nonref,
            'r2': global_r2,
            'rare_r2': global_rare_r2,
            'ser': global_ser,
            'f1': global_f1,
            'prec': global_prec,
            'rec': global_rec,
            'n50': global_n50,
            'chromosomes': chromosomes_found,
            'maf_metrics': maf_metrics
        })
    
    if not final_metrics:
        print("No metrics found.")
        return

    # -- Sort by R2 (descending) --
    final_metrics.sort(key=lambda x: x['r2'], reverse=True)
    
    # -- Generate Markdown Report --
    md_lines = []
    md_lines.append("# 🧬 Imputation Benchmark Results")
    md_lines.append(f"**Genome-wide comparison (All 22 autosomes)**")
    md_lines.append(f"*Metrics aggregated exactly across all sites (Dosage R²).*")
    
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
    
    md_lines.append(f"\n### 📊 Comprehensive Comparison")
    md_lines.append("| Tool | R² | Rare R² (<1%) | F1 Score | Non-Ref Conc. | Switch Error | N50 (bp) | Time (s) | Speedup |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for m in final_metrics:
        # Comparison vs Reagle (if present)
        r2_diff = ""
        reagle_stats = next((x for x in final_metrics if x['id'] == 'reagle'), None)
        if reagle_stats and m['id'] != 'reagle':
            diff = m['r2'] - reagle_stats['r2']
            icon = "🔻" if diff < 0 else "🔺"
            r2_diff = f" ({icon}{abs(diff):.4f})"

        # Speedup vs Beagle
        speedup_str = "-"
        if beagle_stats and beagle_stats['time'] > 0:
            if m['id'] == 'beagle':
                speedup_str = "1.0x"
            else:
                speedup = beagle_stats['time'] / m['time'] if m['time'] > 0 else 0
                speedup_str = f"{speedup:.1f}x"

        md_lines.append(f"| **{m['name']}** | {m['r2']:.4f}{r2_diff} | {m['rare_r2']:.4f} | {m['f1']:.4f} | {m['nonref']:.4f} | {m['ser']:.4f} | {m['n50']:.0f} | {m['time']:.1f} | {speedup_str} |")

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
