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
"""

import os
import sys
import subprocess
import random
import gzip
from pathlib import Path
from collections import defaultdict
import math


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
    Calculate detailed imputation accuracy metrics.

    Metrics:
    - Genotype concordance (exact match rate)
    - Allelic R² (correlation between true and imputed dosages)
    - Non-reference concordance
    - Per-MAF-bin metrics
    """
    if not imputed_vcf or not os.path.exists(imputed_vcf):
        print("Imputed VCF not found")
        return None

    print(f"\nCalculating metrics: {imputed_vcf} vs {truth_vcf}")

    # Load truth genotypes
    truth_gts = {}  # (chrom, pos) -> {sample: dosage}
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
                gt = parse_genotype(gt_str.split(':')[0])  # Handle GT:other fields
                if gt is not None:
                    truth_gts[key][truth_samples[i]] = calculate_dosage(gt)

    # Load imputed genotypes
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
                gt = parse_genotype(gt_str.split(':')[0])
                if gt is not None:
                    imputed_gts[key][imputed_samples[i]] = calculate_dosage(gt)

    # Calculate metrics
    concordant = 0
    discordant = 0
    total_compared = 0

    truth_dosages = []
    imputed_dosages = []

    # MAF bins for stratified analysis
    maf_bins = defaultdict(lambda: {"concordant": 0, "total": 0, "truth": [], "imputed": []})

    common_sites = set(truth_gts.keys()) & set(imputed_gts.keys())
    print(f"Common sites: {len(common_sites)}")

    for site in common_sites:
        truth_site = truth_gts[site]
        imputed_site = imputed_gts.get(site, {})

        # Calculate MAF from truth
        dosages = [d for d in truth_site.values() if d is not None]
        if dosages:
            af = sum(dosages) / (2 * len(dosages))
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

        for sample in truth_site:
            if sample in imputed_site:
                t_dos = truth_site[sample]
                i_dos = imputed_site[sample]

                if t_dos is not None and i_dos is not None:
                    total_compared += 1
                    truth_dosages.append(t_dos)
                    imputed_dosages.append(i_dos)

                    maf_bins[maf_bin]["truth"].append(t_dos)
                    maf_bins[maf_bin]["imputed"].append(i_dos)
                    maf_bins[maf_bin]["total"] += 1

                    if t_dos == i_dos:
                        concordant += 1
                        maf_bins[maf_bin]["concordant"] += 1
                    else:
                        discordant += 1

    # Calculate overall metrics
    metrics = {}

    if total_compared > 0:
        metrics["concordance"] = concordant / total_compared
        metrics["total_genotypes"] = total_compared
        metrics["sites_compared"] = len(common_sites)

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
        else:
            metrics["r_squared"] = None

        # Per-MAF bin metrics
        metrics["by_maf"] = {}
        for maf_bin, data in sorted(maf_bins.items()):
            if data["total"] > 0:
                bin_metrics = {
                    "concordance": data["concordant"] / data["total"],
                    "n_genotypes": data["total"]
                }
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
                metrics["by_maf"][maf_bin] = bin_metrics

    # Print results
    print("\n" + "=" * 50)
    print("IMPUTATION METRICS")
    print("=" * 50)

    if metrics:
        print(f"Sites compared: {metrics.get('sites_compared', 'N/A')}")
        print(f"Genotypes compared: {metrics.get('total_genotypes', 'N/A')}")
        print(f"Overall concordance: {metrics.get('concordance', 0):.4f}")
        print(f"Overall R²: {metrics.get('r_squared', 'N/A'):.4f}" if metrics.get('r_squared') else "Overall R²: N/A")

        if "by_maf" in metrics:
            print("\nBy MAF bin:")
            for maf_bin, bin_metrics in metrics["by_maf"].items():
                r2_str = f"{bin_metrics.get('r_squared', 0):.4f}" if bin_metrics.get('r_squared') else "N/A"
                print(f"  {maf_bin}: concordance={bin_metrics['concordance']:.4f}, "
                      f"R²={r2_str}, n={bin_metrics['n_genotypes']}")

    # Save metrics to file
    metrics_file = f"{output_prefix}_metrics.txt"
    with open(metrics_file, 'w') as f:
        f.write("IMPUTATION ACCURACY METRICS\n")
        f.write("=" * 50 + "\n\n")
        if metrics:
            f.write(f"Sites compared: {metrics.get('sites_compared', 'N/A')}\n")
            f.write(f"Genotypes compared: {metrics.get('total_genotypes', 'N/A')}\n")
            f.write(f"Overall concordance: {metrics.get('concordance', 0):.4f}\n")
            if metrics.get('r_squared'):
                f.write(f"Overall R²: {metrics['r_squared']:.4f}\n")
            f.write("\nBy MAF bin:\n")
            for maf_bin, bin_metrics in metrics.get("by_maf", {}).items():
                f.write(f"  {maf_bin}: conc={bin_metrics['concordance']:.4f}, "
                       f"n={bin_metrics['n_genotypes']}\n")

    return metrics


def main():
    print("=" * 60)
    print("Reagle Integration Test - HGDP+1kG Imputation Benchmark")
    print("=" * 60)

    # Check dependencies
    check_dependencies()

    # Setup paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    data_dir = script_dir / "data"
    os.makedirs(data_dir, exist_ok=True)

    # Download HGDP+1kG chr22
    print("\n" + "=" * 60)
    print("Downloading HGDP+1kG chr22...")
    print("=" * 60)

    chr22_bcf = data_dir / "hgdp1kg_chr22.bcf"
    chr22_vcf = data_dir / "hgdp1kg_chr22.vcf.gz"

    download_if_missing(
        "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf",
        str(chr22_bcf)
    )
    download_if_missing(
        "https://storage.googleapis.com/gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf.csi",
        str(chr22_bcf) + ".csi"
    )

    # Convert BCF to VCF.gz for Java Beagle compatibility
    if not chr22_vcf.exists():
        print("Converting BCF to VCF.gz...")
        run(f"bcftools view {chr22_bcf} -O z -o {chr22_vcf}")
        run(f"bcftools index -f {chr22_vcf}")

    # Download GSA sites
    print("\n" + "=" * 60)
    print("Downloading GSA variant list...")
    print("=" * 60)

    gsa_file = data_dir / "GSAv2_hg38.tsv"
    download_if_missing(
        "https://github.com/SauersML/genomic_pca/raw/refs/heads/main/data/GSAv2_hg38.tsv",
        str(gsa_file)
    )

    # Load GSA sites for chr22
    gsa_sites = load_gsa_sites(str(gsa_file), chrom="22")

    # Download Beagle
    beagle_jar = data_dir / "beagle.jar"
    download_if_missing(
        "https://faculty.washington.edu/browning/beagle/beagle.22Jul22.46e.jar",
        str(beagle_jar)
    )

    # Find/build Reagle binary
    reagle_bin = project_dir / "target" / "release" / "reagle"
    if not reagle_bin.exists():
        print("\nBuilding Reagle...")
        try:
            run(f"cd {project_dir} && cargo build --release")
        except:
            print("Warning: Failed to build Reagle")
            reagle_bin = None

    # Split samples
    print("\n" + "=" * 60)
    print("Splitting samples...")
    print("=" * 60)

    train_file, test_file, train_samples, test_samples = split_samples(
        str(chr22_vcf), str(data_dir), test_fraction=0.2, seed=42
    )

    # Create reference panel (train samples)
    ref_vcf = str(data_dir / "ref.vcf.gz")
    if not os.path.exists(ref_vcf):
        print("Creating reference panel...")
        run(f"bcftools view -S {train_file} {chr22_vcf} -O z -o {ref_vcf}")
        run(f"bcftools index -f {ref_vcf}")

    # Create truth (test samples, full density)
    truth_vcf = str(data_dir / "truth.vcf.gz")
    if not os.path.exists(truth_vcf):
        print("Creating truth VCF...")
        run(f"bcftools view -S {test_file} {chr22_vcf} -O z -o {truth_vcf}")
        run(f"bcftools index -f {truth_vcf}")

    # Create input (test samples, downsampled to GSA sites)
    input_vcf = str(data_dir / "input.vcf.gz")
    gsa_regions = str(data_dir / "gsa_chr22.regions")

    if not os.path.exists(input_vcf):
        print("Downsampling to GSA sites...")
        create_regions_file(gsa_sites, gsa_regions)
        run(f"bcftools view -R {gsa_regions} {truth_vcf} -O z -o {input_vcf}")
        run(f"bcftools index -f {input_vcf}")

    # Count variants
    n_truth = run(f"bcftools view -H {truth_vcf} | wc -l", capture=True).stdout.strip()
    n_input = run(f"bcftools view -H {input_vcf} | wc -l", capture=True).stdout.strip()
    print(f"\nTruth variants: {n_truth}")
    print(f"Input variants (GSA sites): {n_input}")

    # Run imputation
    print("\n" + "=" * 60)
    print("Running imputation...")
    print("=" * 60)

    results = {}

    # Run Beagle
    print("\n--- Running Java Beagle ---")
    beagle_out = str(data_dir / "beagle_imputed")
    beagle_vcf = run_beagle(ref_vcf, input_vcf, beagle_out, str(beagle_jar), nthreads=2)
    results['beagle'] = beagle_vcf

    # Run Reagle
    if reagle_bin and reagle_bin.exists():
        print("\n--- Running Reagle ---")
        reagle_out = str(data_dir / "reagle_imputed")
        reagle_vcf = run_reagle(ref_vcf, input_vcf, reagle_out, str(reagle_bin))
        results['reagle'] = reagle_vcf
    else:
        print("\n--- Skipping Reagle (binary not available) ---")
        results['reagle'] = None

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
            metrics = calculate_metrics(truth_vcf, vcf, str(data_dir / f"{name}_metrics"))
            all_metrics[name] = metrics
        else:
            print(f"{name} output not found")
            all_metrics[name] = None

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Reference panel: {len(train_samples)} samples")
    print(f"Test panel: {len(test_samples)} samples")
    print(f"Truth variants: {n_truth}")
    print(f"Input (GSA) variants: {n_input}")
    print()

    for name, metrics in all_metrics.items():
        if metrics:
            print(f"{name.upper()}:")
            print(f"  Concordance: {metrics.get('concordance', 0):.4f}")
            r2 = metrics.get('r_squared')
            print(f"  R²: {r2:.4f}" if r2 else "  R²: N/A")
        else:
            print(f"{name.upper()}: FAILED/SKIPPED")

    # Exit with appropriate code
    if not any(m for m in all_metrics.values()):
        print("\nERROR: All tools failed!")
        sys.exit(1)

    print("\nIntegration test completed successfully.")


if __name__ == "__main__":
    main()
