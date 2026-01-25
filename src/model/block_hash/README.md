# Block-Hash Clade HMM

This module implements the **Adaptive Full-Panel Block-Hash Clade HMM**, a high-performance solution to fix the critical HMM state continuity bug in Reagle's imputation and phasing pipelines.

## The Problem: "Index Scrambling"

In previous implementations (similar to standard PBWT methods), the algorithm selected different neighbor sets at each marker cluster dynamically. While efficient, this caused "index scrambling" where the same state index (e.g., `State[5]`) referred to different physical reference haplotypes across adjacent marker clusters.

This broke the **Li-Stephens model assumption**, which relies on tracking specific haplotypes to model recombination accurately. The result was reduced accuracy in phasing and imputation, particularly for rare variants and complex regions.

## The Solution

Instead of dynamic neighbor selection that changes haplotype-to-state mapping on the fly, we implemented a **Full-Panel Compression** approach:

1. **Global Reference**: We compress the **ENTIRE** reference panel into unique patterns per window.
2. **Deterministic States**: We track probability per pattern (plus a "reservoir" for rare patterns), retaining **Global IDs** for correct mapping.
3. **Transition Bridges**: We use a sparse transition matrix (`TransitionBridge`) to correctly map probability mass between windows, respecting the physical identity of haplotypes.

## Architecture

### Immutable/Mutable Separation

To enable efficient parallelization, we strictly separate immutable reference data from mutable state:

- **`CompressedBlock` (Immutable)**: 
  - Contains reference panel compression, pattern counts, and global ID mappings.
  - Built once and `Arc`-wrapped for zero-cost sharing across threads.
  
- **`BlockHmmWorkspace` (Mutable)**:
  - Per-sample workspace (thread-local or pooled).
  - Contains forward/backward buffers, emission scratch space, and checkpoints.
  - Prevents re-compression overhead during processing.

- **`ReferenceMap` (Immutable)**:
  - Pre-computed container of blocks and transitions.
  - Enables efficient parallel imputation across thousands of samples.

### Key Performance Characteristics

- **Memory Efficient**: <15GB for 1KG+HGDP (~8k haplotypes).
- **High Compression**: Achieves ~5-10x compression (8k haps → 800-1600 unique patterns per window).
- **Parallelization**: Zero-cost reference sharing via `Arc`.
- **SIMD Optimized**: Leverages AVX-512 kernels from existing `HmmUpdater`.

## Multiallelic Safety

The implementation uses exact allele matching instead of a biallelic 0/1 assumption. This ensures correctness for multiallelic sites:
- **Match**: `1.0 - error_rate` (regardless of allele value).
- **Mismatch**: `error_rate` (works for alleles {0, 1, 2, 3, ...}).
