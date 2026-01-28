# Block-Hash Clade HMM

This module implements the **Adaptive Full-Panel Block-Hash Clade HMM** to fix the critical HMM state continuity bug in Reagle's imputation and phasing pipelines.

## The Bug: Index Scrambling

The previous implementation (`build_pbwt_hap_indices_for_batch`) selected different neighbor sets at each cluster (PBWT checkpoint). This caused "index scrambling" where the same state index (e.g., state 0) referred to different reference haplotypes across marker clusters. This breaks the **Li-Stephens** assumption of haplotype continuity, where a state index is assumed to represent the same haplotype (or clade) across the chromosome.

## The Solution

Instead of dynamic neighbor selection that changes the set of haplotypes, we:
1.  **Compress the ENTIRE reference panel** into unique patterns per window.
2.  **Track probability per pattern** (plus a reservoir for rare variants), while retaining global IDs for mapping/bridging.
3.  Use **TransitionBridge** to correctly map probability between windows.

## Architecture

### Immutable/Mutable Separation (Critical for Parallelization)

*   **CompressedBlock (immutable)**: Built once, `Arc`-wrapped, shared across all samples.
    *   Contains reference panel compression, pattern counts, and global ID mappings.
    *   Enables zero-cost sharing across threads.

*   **BlockHmmWorkspace (mutable)**: Per-sample workspace, thread-local or pooled.
    *   Contains forward/backward buffers, emissions, and checkpoints.
    *   Prevents re-compression overhead.

*   **ReferenceMap (immutable)**: Pre-computed container of blocks and transitions.
    *   Builds all `CompressedBlock`s and `TransitionBridge`s once.
    *   Enables efficient parallel imputation across samples.

### Core Components

*   `types.rs`: Type-safe `GlobalId` and `PatternId` newtypes to prevent index confusion.
*   `compressed_block.rs`: Immutable reference data with pattern mappings.
*   `workspace.rs`: Mutable per-sample HMM state buffers.
*   `transition.rs`: CSR (Compressed Sparse Row) sparse format for deterministic, cache-friendly transitions.
*   `compression.rs`: Builds `CompressedBlock` from `GenotypeMatrix` using `DictionaryColumn`.
*   `hmm.rs`: Forward/backward passes reusing existing `HmmUpdater` SIMD kernels.

### Multiallelic Safety

Uses exact allele matching instead of biallelic 0/1 assumption:
*   **Match**: `1.0 - error_rate` (regardless of allele value).
*   **Mismatch**: `error_rate` (works for alleles {0, 1, 2, 3, ...}).

### Performance Characteristics

*   **Memory**: <15GB for 1KG+HGDP (~8k haplotypes).
*   **Compression**: ~5-10x (8k haps → 800-1600 unique patterns per window).
*   **Parallelization**: Zero-cost reference sharing via `Arc`.
*   **SIMD**: Leverages AVX-512 kernels from existing `HmmUpdater`.
