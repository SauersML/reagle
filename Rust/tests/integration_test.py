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
        if check and result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            raise subprocess.CalledProcessError(result.returncode, cmd)
        return result
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
    cmd = f"java -Xmx4g -jar {beagle_jar} ref={ref_vcf} gt={target_vcf} out={output_prefix} nthreads={nthreads}"
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
    cmd = f"{reagle_bin} --ref {ref_vcf} --gt {target_vcf} --out {output_prefix}"
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


def calculate_metrics(truth_vcf, imputed_vcf, output_prefix):
    """
    Calculate comprehensive imputation accuracy metrics.

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

    # Load truth genotypes - store both genotype tuple and dosage
    truth_gts = {}  # (chrom, pos) -> {sample: (gt_tuple, dosage, is_phased)}
    result = run(f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT[\\t%GT]\\n' {truth_vcf}", capture=True)
    samples_result = run(f"bcftools query -l {truth_vcf}", capture=True)
    truth_samples = samples_result.stdout.strip().split('\n')

    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 5:
            continue
        chrom, pos, ref, alt = parts[0], parts[1], parts[2], parts[3]
        key = (chrom, int(pos))
        gts = parts[4:]
        truth_gts[key] = {}
        for i, gt_str in enumerate(gts):
            if i < len(truth_samples):
                gt_field = gt_str.split(':')[0]  # Handle GT:other fields
                gt = parse_genotype(gt_field)
                is_phased = '|' in gt_field
                if gt is not None:
                    truth_gts[key][truth_samples[i]] = (gt, calculate_dosage(gt), is_phased)

    # Load imputed genotypes - store both genotype tuple and dosage
    imputed_gts = {}
    result = run(f"bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT[\\t%GT]\\n' {imputed_vcf}", capture=True)
    samples_result = run(f"bcftools query -l {imputed_vcf}", capture=True)
    imputed_samples = samples_result.stdout.strip().split('\n')

    for line in result.stdout.strip().split('\n'):
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) < 5:
            continue
        chrom, pos = parts[0], parts[1]
        key = (chrom, int(pos))
        gts = parts[4:]
        imputed_gts[key] = {}
        for i, gt_str in enumerate(gts):
            if i < len(imputed_samples):
                gt_field = gt_str.split(':')[0]
                gt = parse_genotype(gt_field)
                is_phased = '|' in gt_field
                if gt is not None:
                    imputed_gts[key][imputed_samples[i]] = (gt, calculate_dosage(gt), is_phased)

    # Calculate metrics
    unphased_concordant = 0  # Genotype match ignoring phase (0|1 == 1|0)
    total_compared = 0
    
    # Non-reference concordance (excludes 0/0 vs 0/0)
    nonref_concordant = 0
    nonref_total = 0

    truth_dosages = []
    imputed_dosages = []

    # Confusion matrix: [true_class][imputed_class]
    # Classes: 0=HomRef, 1=Het, 2=HomAlt
    confusion = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    # Per-sample tracking
    sample_metrics = defaultdict(lambda: {
        "concordant": 0, "total": 0, "truth_dos": [], "imp_dos": [],
        "switch_errors": 0, "switch_opportunities": 0
    })

    # For IQS calculation: track per-site concordance and expected concordance
    site_iqs_values = []
    
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
    
    maf_bins = defaultdict(lambda: {
        "unphased_concordant": 0, "total": 0, "truth": [], "imputed": [],
        "iqs_values": [], "nonref_concordant": 0, "nonref_total": 0,
        "confusion": [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        "switch_errors": 0, "switch_opportunities": 0
    })

    common_sites = set(truth_gts.keys()) & set(imputed_gts.keys())
    print(f"Common sites: {len(common_sites)}")
    
    # Sort sites for switch error calculation
    sorted_sites = sorted(common_sites, key=lambda x: (x[0], x[1]))
    
    # Track previous het for switch error calculation per sample
    prev_het = {}  # sample -> (site, truth_gt, imputed_gt, maf_bin)

    for site in sorted_sites:
        truth_site = truth_gts[site]
        imputed_site = imputed_gts.get(site, {})

        # Calculate MAF from truth
        dosages_at_site = [v[1] for v in truth_site.values() if v[1] is not None]
        if dosages_at_site:
            af = sum(dosages_at_site) / (2 * len(dosages_at_site))
            maf = min(af, 1 - af)
        else:
            maf = 0

        # Determine MAF bin using finer categories
        maf_bin = get_maf_bin(maf)

        # Track per-site concordance for IQS
        site_concordant = 0
        site_total = 0

        for sample in truth_site:
            if sample in imputed_site:
                t_gt, t_dos, t_phased = truth_site[sample]
                i_gt, i_dos, i_phased = imputed_site[sample]

                if t_dos is not None and i_dos is not None:
                    total_compared += 1
                    site_total += 1
                    truth_dosages.append(t_dos)
                    imputed_dosages.append(i_dos)

                    maf_bins[maf_bin]["truth"].append(t_dos)
                    maf_bins[maf_bin]["imputed"].append(i_dos)
                    maf_bins[maf_bin]["total"] += 1
                    
                    # Per-sample tracking
                    sample_metrics[sample]["total"] += 1
                    sample_metrics[sample]["truth_dos"].append(t_dos)
                    sample_metrics[sample]["imp_dos"].append(i_dos)
                    
                    # Classify genotypes for confusion matrix
                    t_class = 0 if t_dos == 0 else (2 if t_dos == 2 else 1)
                    i_class = 0 if i_dos == 0 else (2 if i_dos == 2 else 1)
                    confusion[t_class][i_class] += 1
                    maf_bins[maf_bin]["confusion"][t_class][i_class] += 1

                    # Unphased concordance: sort alleles so 0|1 == 1|0
                    t_sorted = tuple(sorted(t_gt))
                    i_sorted = tuple(sorted(i_gt))
                    if t_sorted == i_sorted:
                        unphased_concordant += 1
                        site_concordant += 1
                        maf_bins[maf_bin]["unphased_concordant"] += 1
                        sample_metrics[sample]["concordant"] += 1
                    
                    # Non-reference concordance
                    if t_dos > 0:  # Truth is not HomRef
                        nonref_total += 1
                        maf_bins[maf_bin]["nonref_total"] += 1
                        if t_sorted == i_sorted:
                            nonref_concordant += 1
                            maf_bins[maf_bin]["nonref_concordant"] += 1
                    
                    # Switch error detection (for hets only, when both are phased)
                    if t_dos == 1 and i_dos == 1 and t_phased and i_phased:
                        pos = site[1]
                        if sample not in current_block_start:
                            current_block_start[sample] = pos

                        if sample in prev_het:
                            prev_site, prev_t_gt, prev_i_gt, prev_maf_bin = prev_het[sample]
                            # Determine if there's a switch by comparing phase relationships
                            # If truth maintains same phase but imputed flips, it's a switch
                            t_same_phase = (t_gt[0] == prev_t_gt[0])  # First allele same as prev first
                            i_same_phase = (i_gt[0] == prev_i_gt[0])
                            
                            if t_same_phase != i_same_phase:
                                # Switch error! End current block
                                block_len = pos - current_block_start[sample]
                                phase_blocks[sample].append(block_len)
                                # Start new block at current position
                                current_block_start[sample] = pos
                                
                                switch_errors += 1
                                sample_metrics[sample]["switch_errors"] += 1
                                maf_bins[maf_bin]["switch_errors"] += 1
                            
                            switch_opportunities += 1
                            sample_metrics[sample]["switch_opportunities"] += 1
                            maf_bins[maf_bin]["switch_opportunities"] += 1
                        prev_het[sample] = (site, t_gt, i_gt, maf_bin)

        # Calculate IQS for this site
        # IQS = (observed - expected) / (1 - expected)
        # Expected concordance under HWE: P(0/0)^2 + P(0/1)^2 + P(1/1)^2
        # With p = alt allele freq: (1-p)^4 + 4p^2(1-p)^2 + p^4
        if site_total > 0 and maf > 0 and maf < 1:
            p = af  # Use AF not MAF for expected calculation
            q = 1 - p
            # HWE genotype frequencies
            p_00 = q * q
            p_01 = 2 * p * q
            p_11 = p * p
            # Expected concordance by chance
            expected_conc = p_00 * p_00 + p_01 * p_01 + p_11 * p_11
            observed_conc = site_concordant / site_total

            if expected_conc < 1.0:  # Avoid division by zero
                iqs = (observed_conc - expected_conc) / (1.0 - expected_conc)
                site_iqs_values.append(iqs)
                maf_bins[maf_bin]["iqs_values"].append(iqs)
    
    # Close final phase blocks
    last_pos = sorted_sites[-1][1] if sorted_sites else 0
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
        metrics["sites_compared"] = len(common_sites)
        
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

        # Calculate R²
        if len(truth_dosages) > 1:
            mean_t = sum(truth_dosages) / len(truth_dosages)
            mean_i = sum(imputed_dosages) / len(imputed_dosages)

            cov = sum((t - mean_t) * (i - mean_i) for t, i in zip(truth_dosages, imputed_dosages))
            var_t = sum((t - mean_t) ** 2 for t in truth_dosages)
            var_i = sum((i - mean_i) ** 2 for i in imputed_dosages)

            # Store sufficient statistics for exact global aggregation
            metrics["r2_stats"] = {
                "sum_t": sum(truth_dosages),
                "sum_i": sum(imputed_dosages),
                "sum_ti": sum(t * i for t, i in zip(truth_dosages, imputed_dosages)),
                "sum_tt": sum(t * t for t in truth_dosages),
                "sum_ii": sum(i * i for i in imputed_dosages),
                "count": len(truth_dosages)
            }

            if var_t > 0 and var_i > 0:
                r = cov / math.sqrt(var_t * var_i)
                metrics["r_squared"] = r ** 2
            else:
                metrics["r_squared"] = None
                
            # INFO score approximation (variance ratio)
            # INFO ≈ 1 - (mean imputed variance) / (expected variance under HWE)
            # For hard calls: var_expected = 2*p*q, var_observed = 0
            # This is a rough approximation
            if var_t > 0:
                metrics["info_score_approx"] = var_i / var_t
        else:
            metrics["r_squared"] = None

        # Calculate Rare Variant R² stats (MAF < 1%)
        rare_stats = {
            "sum_t": 0.0, "sum_i": 0.0, "sum_ti": 0.0,
            "sum_tt": 0.0, "sum_ii": 0.0, "count": 0
        }
        
        # We need to iterate again or collect during the main loop?
        # The main loop collected `truth_dosages` and `imputed_dosages` but mixed all MAFs.
        # However, we have `maf_bins`.
        # "ultra-rare (<0.1%)", "very-rare (0.1-0.5%)", "rare (0.5-1%)" are the ones we want.
        target_bins = ["ultra-rare (<0.1%)", "very-rare (0.1-0.5%)", "rare (0.5-1%)"]
        
        for bin_name in target_bins:
            if bin_name in maf_bins:
                b_data = maf_bins[bin_name]
                if "truth" in b_data:
                    for t, i in zip(b_data["truth"], b_data["imputed"]):
                        rare_stats["sum_t"] += t
                        rare_stats["sum_i"] += i
                        rare_stats["sum_ti"] += t * i
                        rare_stats["sum_tt"] += t * t
                        rare_stats["sum_ii"] += i * i
                        rare_stats["count"] += 1
                        
        metrics["rare_r2_stats"] = rare_stats

        # Calculate overall IQS (mean across sites)
        if site_iqs_values:
            metrics["iqs"] = sum(site_iqs_values) / len(site_iqs_values)
            metrics["iqs_median"] = sorted(site_iqs_values)[len(site_iqs_values) // 2]
        else:
            metrics["iqs"] = None
            
        # Per-sample summary statistics
        sample_concordances = []
        sample_r2s = []
        for sample, data in sample_metrics.items():
            if data["total"] > 0:
                sample_concordances.append(data["concordant"] / data["total"])
            if len(data["truth_dos"]) > 1:
                mean_t = sum(data["truth_dos"]) / len(data["truth_dos"])
                mean_i = sum(data["imp_dos"]) / len(data["imp_dos"])
                cov = sum((t - mean_t) * (i - mean_i) for t, i in zip(data["truth_dos"], data["imp_dos"]))
                var_t = sum((t - mean_t) ** 2 for t in data["truth_dos"])
                var_i = sum((i - mean_i) ** 2 for i in data["imp_dos"])
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
                
                # R² per bin
                if len(data["truth"]) > 1:
                    mean_t = sum(data["truth"]) / len(data["truth"])
                    mean_i = sum(data["imputed"]) / len(data["imputed"])
                    cov = sum((t - mean_t) * (i - mean_i)
                             for t, i in zip(data["truth"], data["imputed"]))
                    var_t = sum((t - mean_t) ** 2 for t in data["truth"])
                    var_i = sum((i - mean_i) ** 2 for i in data["imputed"])
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
                    "sum_t": sum(data["truth"]),
                    "sum_i": sum(data["imputed"]),
                    "sum_ti": sum(t * i for t, i in zip(data["truth"], data["imputed"])),
                    "sum_tt": sum(t * t for t in data["truth"]),
                    "sum_ii": sum(i * i for i in data["imputed"]),
                    "count": len(data["truth"]),
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
    if not paths['chr22_vcf'].exists():
        print("Converting BCF to VCF.gz...")
        run(f"bcftools view {paths['chr22_bcf']} -O z -o {paths['chr22_vcf']}")
        run(f"bcftools index -f {paths['chr22_vcf']}")

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
    if not paths['ref_vcf'].exists():
        print("Creating reference panel...")
        run(f"bcftools view -S {train_file} {paths['chr22_vcf']} -O z -o {paths['ref_vcf']}")
        run(f"bcftools index -f {paths['ref_vcf']}")

    # Create truth (test samples, full density)
    if not paths['truth_vcf'].exists():
        print("Creating truth VCF...")
        run(f"bcftools view -S {test_file} {paths['chr22_vcf']} -O z -o {paths['truth_vcf']}")
        run(f"bcftools index -f {paths['truth_vcf']}")

    # Create input (test samples, downsampled to GSA sites, UNPHASED)
    # We unphase the input so switch error rate measures TRUE phasing accuracy
    if not paths['input_vcf'].exists():
        print("Downsampling to GSA sites and unphasing...")
        create_regions_file(gsa_sites, str(paths['gsa_regions']))
        # Two-step process: downsample, then unphase
        tmp_phased = str(paths['data_dir'] / "input_phased_tmp.vcf.gz")
        run(f"bcftools view -R {paths['gsa_regions']} {paths['truth_vcf']} -O z -o {tmp_phased}")
        # Unphase: convert 0|1 to 0/1 using bcftools +setGT
        # The plugin sets genotypes to unphased while preserving allele values
        run(f"bcftools +setGT {tmp_phased} -O z -o {paths['input_vcf']} -- -t a -n u")
        run(f"bcftools index -f {paths['input_vcf']}")
        # Clean up temp file
        os.remove(tmp_phased)

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
        run(f"curl -L -o {data_dir}/impute5.zip 'https://www.dropbox.com/s/raw/mwmgzjx5vvmbuaz/impute5_v1.2.0_static.zip'")
        run(f"cd {data_dir} && unzip -q impute5.zip && mv impute5_v1.2.0_static/impute5_1.2.0_static impute5 && chmod +x impute5")
    
    impute5_out = data_dir / "impute5_imputed.vcf.gz"
    if not impute5_out.exists():
        print("Running IMPUTE5...")
        try:
            run(f"{impute5_bin} --h {paths['ref_vcf']} --g {paths['input_vcf']} --r chr22 --o {impute5_out} --threads 4")
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
        run(f"curl -L -o {data_dir}/minimac4.tar.gz 'https://github.com/statgen/Minimac4/releases/download/v4.1.6/minimac4-4.1.6-Linux-x86_64.tar.gz'")
        run(f"cd {data_dir} && tar -xzf minimac4.tar.gz && mv minimac4-4.1.6-Linux-x86_64/bin/minimac4 . && chmod +x minimac4")
    
    minimac_out = data_dir / "minimac_imputed.vcf.gz"
    if not minimac_out.exists():
        print("Running Minimac4...")
        try:
            run(f"{minimac_bin} --refHaps {paths['ref_vcf']} --haps {paths['input_vcf']} --prefix {data_dir}/minimac_imputed --cpus 4 --format GT,DS")
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
            run(f"{glimpse_bin} --input {paths['input_vcf']} --reference {paths['ref_vcf']} --output {data_dir}/glimpse_imputed.bcf --threads 4")
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
            run(f"curl -L -o {data_dir}/impute5.zip 'https://www.dropbox.com/s/raw/mwmgzjx5vvmbuaz/impute5_v1.2.0_static.zip'")
            run(f"cd {data_dir} && unzip -q impute5.zip && mv impute5_v1.2.0_static/impute5_1.2.0_static impute5 && chmod +x impute5")

    out = data_dir / "impute5_imputed.vcf.gz"
    if not out.exists():
        print(f"Running IMPUTE5 on chr{chrom}...")
        try:
            # IMPUTE5 requires an indexed reference and map file usually, but minimal example:
            # --h reference --g input --r region --o output
            run(f"{impute5_bin} --h {paths['ref_vcf']} --g {paths['input_vcf']} --r chr{chrom} --o {out} --threads 4")
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
            run(f"curl -L -o {data_dir}/minimac4.tar.gz 'https://github.com/statgen/Minimac4/releases/download/v4.1.6/minimac4-4.1.6-Linux-x86_64.tar.gz'")
            run(f"cd {data_dir} && tar -xzf minimac4.tar.gz && mv minimac4-4.1.6-Linux-x86_64/bin/minimac4 . && chmod +x minimac4")

    out = data_dir / "minimac_imputed.vcf.gz"
    if not out.exists():
        print(f"Running Minimac4 on chr{chrom}...")
        try:
            prefix = data_dir / "minimac_imputed"
            # Minimac4: --refHaps ref.vcf --haps input.vcf --prefix out
            run(f"{minimac_bin} --refHaps {paths['ref_vcf']} --haps {paths['input_vcf']} --prefix {prefix} --cpus 4 --format GT,DS")
            
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
            # GLIMPSE2_phase: --input input.vcf --reference ref.vcf --output out.bcf
            bcf_out = data_dir / "glimpse_imputed.bcf"
            run(f"{glimpse_bin} --input {paths['input_vcf']} --reference {paths['ref_vcf']} --output {bcf_out} --threads 4")
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

    md_lines.append(f"\n### 🏆 Highlights")
    md_lines.append(f"- **Most Accurate (R²):** {best_r2['name']} ({best_r2['r2']:.4f})")
    md_lines.append(f"- **Best F1 Score:** {best_f1['name']} ({best_f1['f1']:.4f})")
    md_lines.append(f"- **Fastest:** {best_time['name']} ({best_time['time']:.1f}s)")
    if best_ser and best_ser['ser'] < 1.0:
        md_lines.append(f"- **Best Phasing (SER):** {best_ser['name']} ({best_ser['ser']:.4f})")
    if best_n50:
        md_lines.append(f"- **Longest Phase Blocks (N50):** {best_n50['name']} ({best_n50['n50']:.0f} bp)")
    
    md_lines.append(f"\n### 📊 Comprehensive Comparison")
    md_lines.append("| Tool | R² | Rare R² (<1%) | F1 Score | Non-Ref Conc. | Switch Error | N50 Block (bp) | Runtime (s) | Chrs |")
    md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for m in final_metrics:
        # Comparison vs Reagle (if present)
        r2_diff = ""
        reagle_stats = next((x for x in final_metrics if x['id'] == 'reagle'), None)
        if reagle_stats and m['id'] != 'reagle':
            diff = m['r2'] - reagle_stats['r2']
            icon = "🔻" if diff < 0 else "🔺"
            r2_diff = f" ({icon}{abs(diff):.4f})"
            
        md_lines.append(f"| **{m['name']}** | {m['r2']:.4f}{r2_diff} | {m['rare_r2']:.4f} | {m['f1']:.4f} | {m['nonref']:.4f} | {m['ser']:.4f} | {m['n50']:.0f} | {m['time']:.1f} | {m['chromosomes']} |")

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

