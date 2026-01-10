# Imputation Accuracy Investigation

## Current State
- Rust imputed DR2: 0.1542
- Java imputed DR2: 0.1998
- Gap: 0.046 (23% relative difference)

## Root Cause Analysis

The core issue is that **rare variant imputation fails** when flanking genotyped markers have uncertain genotype calls.

### Example Case (Position 20066665)
- Java: Sample 0 DS = 1.0 (correct)
- Rust: Sample 0 DS = 0.0001 (near zero)
- AF in reference panel: ~5%

### Chain of Failure
1. Flanking marker (pos 20066422) has Sample 0 with:
   - GT = 0/0 (hard call)
   - DS = 0.150 (15% ALT probability)
   - GL = -0.48,-0.48,-0.48 (uniform = uninformative)

2. Due to LD, reference haplotypes carrying ALT at 20066422 also carry ALT at 20066665

3. HMM emission penalizes reference haplotypes that don't match target's 0/0 call

4. Result: ALT-carrying haplotypes get near-zero posterior probability

5. Imputed dosage at 20066665: near-zero for ALL samples

## Approaches Tested

### 1. Skip Low-Confidence Markers
- Skip markers with GL confidence <= 128 (uniform GL)
- Result: WORSE (0.1426 vs 0.1542)
- Reason: Losing information from confident samples at those markers

### 2. Don't Count Mismatches for Uncertain Markers
- Still count as observation, but don't add to mismatch count
- Result: WORSE (0.1426 vs 0.1542)
- Reason: Effectively same as approach 1

### 3. Scale Error Rate by Uncertainty
- Increase p_mismatch based on fraction of uncertain markers
- Result: MINIMAL EFFECT
- Reason: Error rate is calculated via Li-Stephens formula, not default

### 4. Increase Error Rate Globally
- Changed p_mismatch from 0.0001 to 0.1
- Result: NO CHANGE (error rate overridden by Li-Stephens)

## Key Insights

1. **The error rate used is ~0.0002** (from Li-Stephens formula with 382 haplotypes)

2. **Match/mismatch ratio is ~5000:1** - very aggressive penalty for mismatches

3. **Simple confidence filtering hurts more than helps** because it removes
   information from confident samples that happen to share markers with
   uncertain samples

4. **The fix needs to be PER-SAMPLE, PER-MARKER** - treating uncertain
   observations as providing no discrimination power rather than wrong
   discrimination

## Additional Approaches Tested

### Soft Mismatch Counting (scaled integers)
- Attempted to use fractional mismatch counts based on confidence
- Result: MUCH WORSE (0.1276) because the emission formula expects integer values
- The scaling breaks the emission probability calculation

### Error Rate Scaling (Li-Stephens)
- Attempted to increase Li-Stephens p_mismatch by 5x, 50x
- Result: No improvement or WORSE
- Higher error rate doesn't help because it affects ALL mismatches equally

## Potential Solutions (Not Yet Implemented)

### Option 1: Float-Based Emission Model
Change the emission calculation to use floats instead of integers:
- Store confidence-weighted observation counts as floats
- Compute soft emissions: P(emit) = (1-p_err)^n_match * p_err^n_mismatch
  where n_match and n_mismatch can be fractional

### Option 2: Parse DS Field for Soft Emissions
Instead of using hard GT calls, use the DS field from target VCF:
- DS=0 means high confidence REF/REF → full mismatch penalty for ALT states
- DS=1 means uncertain → 50% mismatch probability for both states
- DS=2 means high confidence ALT/ALT → full mismatch penalty for REF states

### Option 3: Separate HMM Runs
Run separate HMM for confident vs uncertain markers:
- Confident markers: use normal binary emission
- Uncertain markers: skip or use uniform emission
- Combine posteriors with weighting

### Option 4: Match Java Implementation More Closely
If we can access Java source or documentation, understand exactly how Java
handles GL uncertainty in imputation. The difference must be documented
somewhere.

## Files Involved
- `src/pipelines/imputation.rs`: HMM forward-backward and mismatch counting
- `src/model/parameters.rs`: Error rate and recombination calculations
- `src/io/vcf.rs`: GL parsing and confidence computation
- `src/data/storage/matrix.rs`: Confidence storage

## Summary

The gap between Rust (0.1542) and Java (0.1998) imputed DR2 persists. The root
cause is well-understood: uncertain genotype calls at flanking markers bias
the HMM against rare-variant-carrying haplotypes. However, simple fixes make
accuracy worse because they affect all samples equally. A proper fix requires
per-sample, per-marker soft emissions, which requires more significant
refactoring of the mismatch counting and emission calculation code.
