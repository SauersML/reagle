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
    sample_metrics = defaultdict(lambda: {"concordant": 0, "total": 0, "truth_dos": [], "imp_dos": []})

    # For IQS calculation: track per-site concordance and expected concordance
    site_iqs_values = []
    
    # For switch error rate
    switch_errors = 0
    switch_opportunities = 0

    # MAF bins for stratified analysis
    maf_bins = defaultdict(lambda: {
        "unphased_concordant": 0, "total": 0, "truth": [], "imputed": [],
        "iqs_values": [], "nonref_concordant": 0, "nonref_total": 0,
        "confusion": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    })

    common_sites = set(truth_gts.keys()) & set(imputed_gts.keys())
    print(f"Common sites: {len(common_sites)}")
    
    # Sort sites for switch error calculation
    sorted_sites = sorted(common_sites, key=lambda x: (x[0], x[1]))
    
    # Track previous het for switch error calculation per sample
    prev_het = {}  # sample -> (site, truth_gt, imputed_gt)

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

        # Determine MAF bin
        if maf < 0.01:
            maf_bin = "rare (<1%)"
        elif maf < 0.05:
            maf_bin = "low (1-5%)"
        elif maf < 0.2:
            maf_bin = "medium (5-20%)"
        else:
            maf_bin = "common (>20%)"

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
                        if sample in prev_het:
                            prev_site, prev_t_gt, prev_i_gt = prev_het[sample]
                            # Determine if there's a switch by comparing phase relationships
                            # If truth maintains same phase but imputed flips, it's a switch
                            t_same_phase = (t_gt[0] == prev_t_gt[0])  # First allele same as prev first
                            i_same_phase = (i_gt[0] == prev_i_gt[0])
                            if t_same_phase != i_same_phase:
                                switch_errors += 1
                            switch_opportunities += 1
                        prev_het[sample] = (site, t_gt, i_gt)

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

    # Calculate overall metrics
    metrics = {}

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
                metrics["by_maf"][maf_bin] = bin_metrics

    elapsed = time.time() - start_time
    metrics["calculation_time_sec"] = elapsed

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
        print(f"   Overall R²:           {metrics.get('r_squared'):.4f}" if metrics.get('r_squared') else "   Overall R²:           N/A")
        print(f"   Overall IQS:          {metrics.get('iqs'):.4f}" if metrics.get('iqs') else "   Overall IQS:          N/A")
        print(f"   INFO score (approx):  {metrics.get('info_score_approx'):.4f}" if metrics.get('info_score_approx') else "   INFO score (approx):  N/A")
        
        if metrics.get('switch_error_rate') is not None:
            print(f"\n🔀 PHASING QUALITY")
            print(f"   Switch error rate:    {metrics.get('switch_error_rate'):.4f} ({metrics.get('switch_errors')}/{metrics.get('switch_opportunities')})")
        
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

        if "by_maf" in metrics:
            print(f"\n📈 BY MAF BIN")
            print(f"   {'MAF Bin':<15} {'Conc':>8} {'NonRef':>8} {'R²':>8} {'IQS':>8} {'N':>10}")
            print(f"   {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
            for maf_bin, bin_metrics in metrics["by_maf"].items():
                conc = f"{bin_metrics['unphased_concordance']:.4f}"
                nonref = f"{bin_metrics.get('nonref_concordance', 0):.4f}" if bin_metrics.get('nonref_concordance') else "N/A"
                r2_str = f"{bin_metrics.get('r_squared'):.4f}" if bin_metrics.get('r_squared') else "N/A"
                iqs_str = f"{bin_metrics.get('iqs'):.4f}" if bin_metrics.get('iqs') else "N/A"
                print(f"   {maf_bin:<15} {conc:>8} {nonref:>8} {r2_str:>8} {iqs_str:>8} {bin_metrics['n_genotypes']:>10,}")

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
  prepare   Download data and prepare reference/truth/input VCFs
  beagle    Run Java Beagle imputation (requires prepare)
  reagle    Run Reagle imputation (requires prepare + build)
  metrics   Calculate and compare accuracy metrics (requires imputation)
  all       Run all stages sequentially (default)

Examples:
  python integration_test.py              # Run all stages
  python integration_test.py prepare      # Just prepare data
  python integration_test.py beagle       # Just run Beagle
  python integration_test.py metrics      # Just calculate metrics
        """
    )
    parser.add_argument(
        'stage',
        nargs='?',
        default='all',
        choices=['all', 'prepare', 'beagle', 'reagle', 'metrics'],
        help='Stage to run (default: all)'
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
    elif args.stage == 'metrics':
        stage_metrics()
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


if __name__ == "__main__":
    main()
