# Imputation Accuracy Root Cause Analysis

## Problem
Rust imputation DR2 for imputed markers is significantly lower than Java Beagle:
- Rust mean DR2: 0.1542
- Java mean DR2: 0.1998

At specific markers (e.g., position 20066665), Java produces dosage=1.0 for Sample 0
while Rust produces dosage≈0.0001 for ALL samples.

## Root Cause
The Rust HMM uses hard phased genotype calls without considering genotype likelihoods.

### Example
At marker 20066422 (flanking the failing imputed marker):
- Sample 0 genotype: `0/0` (REF/REF)
- Sample 0 dosage: `0.150` (15% probability of ALT)
- Sample 0 GL: `-0.48,-0.48,-0.48` (UNIFORM = uninformative)

The hard call `0/0` is essentially a random guess when GLs are uniform.

### The Chain of Failure
1. **Phasing**: Converts `0/0` to `0|0` without considering GL uncertainty
2. **IBS/State Selection**: Uses hard `0|0` call
3. **Mismatch Counting**: Counts hard mismatches between target `0|0` and reference haplotypes
4. **HMM Emission**: Reference haplotypes carrying ALT at 20066422 get penalized
5. **Result**: ALT-carrying reference haplotypes (which also carry ALT at 20066665 due to LD)
   get low posterior probability
6. **Imputed Dosage**: Near-zero for ALL samples because no high-probability states carry ALT

### Why Java Works
Java Beagle likely uses genotype likelihoods to weight HMM emissions:
- Uniform GL → No penalty for mismatch
- Confident GL → Normal mismatch penalty

This allows uncertain positions to not incorrectly bias the HMM against certain reference haplotypes.

## Proposed Fix

### Option 1: Full GL-Based Emissions (Recommended)
1. Parse GL values from VCF during reading
2. Store GL confidence per marker per sample
3. Modify `compute_cluster_mismatches` to:
   - For confident markers: use hard mismatch count
   - For uncertain markers: use soft emission weight based on GL

### Option 2: Skip Uncertain Markers
1. Detect markers with uniform/near-uniform GLs
2. Skip these markers in mismatch calculation
3. Simple but loses information

### Option 3: Use Dosage for Soft Emissions
1. Parse DS field from VCF
2. Use dosage to weight mismatches:
   - If DS≈0 or DS≈2: hard emission
   - If DS≈1: soft emission

## Files to Modify
- `src/io/vcf.rs`: Parse GL or DS fields
- `src/data/storage/matrix.rs`: Store GL/confidence data
- `src/pipelines/imputation.rs`: Use soft emissions in `compute_cluster_mismatches`

## Test Case
Position 20066665 provides a clear test:
- After fix, Sample 0 should get dosage close to 1.0 (matching Java)
- AF should be ~0.05 (matching reference panel)
