# Block-Hash Clade HMM

This module implements the **Adaptive Full-Panel Block-Hash Clade HMM** to fix the critical HMM state continuity bug in Reagle's imputation and phasing pipelines.

## The Bug: Index Scrambling

The previous implementation (`build_pbwt_hap_indices_for_batch`) selected different neighbor sets at each cluster. This caused "index scrambling," where the same state index (e.g., `State[5]`) referred to different reference haplotypes across adjacent marker clusters. This fundamentally broke the Li-Stephens model assumption of haplotype continuity, where staying in state $i$ implies copying from the same physical haplotype.

## The Solution: Compressed Patterns

Instead of dynamic neighbor selection, we implemented a full-panel compression approach:

1.  **Full Panel Compression:** We compress the *entire* reference panel into unique haplotype patterns per window.
2.  **Pattern Tracking:** We track probability mass for each unique pattern (plus a "reservoir" for rare patterns), while maintaining mappings to Global IDs for continuity.
3.  **Transition Bridges:** We use a `TransitionBridge` to correctly map probability mass from patterns in window $t$ to patterns in window $t+1$.

## Architecture

### Immutable/Mutable Separation

To support efficient parallelization, we enforce a strict separation between immutable reference data and mutable workspace memory:

*   **`CompressedBlock` (Immutable):** Built once and wrapped in an `Arc`. It contains the reference panel compression, pattern counts, and global ID mappings. It allows zero-cost sharing across threads.
*   **`BlockHmmWorkspace` (Mutable):** A per-sample workspace (thread-local or pooled) that contains forward/backward probability buffers, emission caches, and checkpoints. This prevents re-allocation overhead.
*   **`ReferenceMap` (Immutable):** A pre-computed container of all `CompressedBlock`s and `TransitionBridge`s. It enables efficient parallel imputation across thousands of samples.

### Core Components

*   **`types.rs`:** Defines `GlobalId` and `PatternId` newtypes to strictly prevent index confusion.
*   **`compressed_block.rs`:** Stores the immutable reference data with pattern mappings.
*   **`workspace.rs`:** Manages mutable per-sample HMM state buffers.
*   **`transition.rs`:** Implements a CSR sparse format for deterministic, cache-friendly state transitions.
*   **`compression.rs`:** logic to build `CompressedBlock` from a `GenotypeMatrix` using dictionary compression.
*   **`hmm.rs`:** Implements forward/backward passes, reusing existing SIMD kernels.

## Multiallelic Safety

The implementation uses exact allele matching rather than assuming biallelic (0/1) sites:
*   **Match:** `1.0 - error_rate` (regardless of allele value)
*   **Mismatch:** `error_rate`

This ensures correctness for multiallelic sites (alleles {0, 1, 2, 3, ...}).

## Performance Characteristics

*   **Memory:** <15GB for 1KG+HGDP datasets (~8k haplotypes).
*   **Compression:** Achieves ~5-10x compression (8k haplotypes → 800-1600 unique patterns per window).
*   **Parallelization:** Zero-cost reference sharing via `Arc`.
*   **SIMD:** Leverages AVX-512 kernels from the existing `HmmUpdater`.
