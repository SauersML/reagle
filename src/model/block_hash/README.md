# Block-Hash Clade HMM

This module implements the **Adaptive Full-Panel Block-Hash Clade HMM** to fix the critical HMM state continuity bug in Reagle's imputation and phasing pipelines.

## The Bug: "Index Scrambling"

The previous implementation (`build_pbwt_hap_indices_for_batch`) selected different neighbor sets at each cluster independently. This caused "index scrambling," where the same state index (e.g., State 0) referred to different reference haplotypes across adjacent marker clusters.

This broke the **Li-Stephens assumption of haplotype continuity**, where staying in the "same state" implies copying from the "same haplotype".

## The Solution: Block-Hash HMM

Instead of dynamic neighbor selection per marker, we perform **Full-Panel Block Hashing**:

1.  **Block Compression**: Compress the **ENTIRE** reference panel into unique haplotype patterns per window.
2.  **Pattern Tracking**: Track probability per unique pattern (plus a reservoir for rare patterns), while retaining `GlobalId`s for consistent mapping.
3.  **Transition Bridging**: Use a `TransitionBridge` to correctly map probability mass between windows (Pattern ID $A$ in Window 1 $\to$ Pattern ID $B$ in Window 2) based on the shared `GlobalId`.

## Architecture

The architecture is designed for high performance and zero-cost parallelization.

### Immutable/Mutable Separation

*   **`CompressedBlock` (Immutable)**:
    *   Built once per window.
    *   Contains reference panel compression, pattern counts, and `GlobalId` mappings.
    *   Wrapped in `Arc` and shared across all target samples (zero-cost sharing).

*   **`BlockHmmWorkspace` (Mutable)**:
    *   Allocated per sample (thread-local or pooled).
    *   Contains Forward/Backward probability buffers, emission buffers, and checkpoints.
    *   Prevents re-compression overhead for every sample.

*   **`ReferenceMap` (Immutable)**:
    *   Pre-computed container of all blocks and transitions.
    *   Builds all `CompressedBlock`s and `TransitionBridge`s once at startup.
    *   Enables efficient parallel imputation.

### Type Safety: `GlobalId` vs `PatternId`

To prevent index confusion, we use distinct newtypes:

*   **`GlobalId`**: Identifies a physical haplotype in the reference panel (0..N). Stable across the entire chromosome.
*   **`PatternId`**: Identifies a unique compressed pattern within a specific window. Local to that window.

### Multiallelic Safety

The implementation uses exact allele matching instead of a biallelic 0/1 assumption:
*   **Match**: `1.0 - error_rate` (regardless of allele value)
*   **Mismatch**: `error_rate` (works for alleles {0, 1, 2, 3, ...})

## Performance

*   **Memory**: <15GB for 1KG+HGDP (~8k haplotypes).
*   **Compression**: ~5-10x reduction (8k haplotypes $\to$ 800-1600 unique patterns per window).
*   **Parallelization**: Zero-cost reference sharing via `Arc`.
*   **SIMD**: Leverages AVX-512 kernels.
