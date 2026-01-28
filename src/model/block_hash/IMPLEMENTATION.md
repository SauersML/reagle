# Block-Hash HMM Implementation Details

This document details the implementation of the Block-Hash HMM, focusing on the core components and the specialized HMM kernel.

## Core Components

### 1. CompressedBlock (Immutable)

The `CompressedBlock` struct represents a window of markers where reference haplotypes are compressed into unique patterns.

*   **Patterns**: Unique haplotype sequences within the window.
*   **Pattern Counts**: The number of reference haplotypes that map to each pattern. This is crucial for the weighted HMM updates.
*   **Global ID Mapping**: Maps global haplotype IDs to local pattern IDs, allowing tracking of specific haplotypes if needed.
*   **Allele Storage**: Stores alleles for each pattern efficiently.
*   **Reservoir**: A special "reservoir" state collects rare haplotypes that don't fit into the main patterns (if a limit is set), ensuring full coverage of the reference panel.

### 2. BlockHmmWorkspace (Mutable)

The `BlockHmmWorkspace` holds the mutable state for a single sample's HMM run.

*   **Forward/Backward Buffers**: `fwd` and `bwd` arrays store the probabilities for each pattern.
*   **History Buffers**: Stores checkpoints of the forward pass to enable the backward pass to reconstruct state without re-running the full forward pass.
*   **Emissions**: Temporary buffer for emission probabilities.
*   **Reservoir Probabilities**: `reservoir_prob_fwd` and `reservoir_prob_bwd` track the probability mass in the reservoir state.

### 3. WeightedHmmUpdater (SIMD Kernel)

The HMM update logic is implemented in `WeightedHmmUpdater` (in `weighted_kernel.rs`). Unlike the standard Li-Stephens kernel where all reference haplotypes are equally likely transitions (uniform prior), the Block-Hash HMM weights transitions by the **pattern cardinality**.

#### Update Logic

The forward update for pattern `i` is:

```
F_new[i] = Emissions[i] * ( (1 - r) * F_old[i] + r * (Count[i] / N_ref) * Sum_F_old )
```

Where:
*   `r` is the recombination rate.
*   `Count[i]` is the number of haplotypes matching pattern `i`.
*   `N_ref` is the total number of reference haplotypes.

This ensures that a pattern representing 100 haplotypes receives 100x more "recombination mass" than a pattern representing 1 haplotype, preserving the correct Li-Stephens probability density.

#### SIMD Optimization

The kernel uses `wide::f32x8` (AVX/AVX2/AVX-512) to process 8 patterns at a time. The weighting factors (`Count[i] / N_ref`) are pre-calculated or applied efficiently within the vectorized loop.

### 4. TransitionBridge

Transitions between windows are handled by `TransitionBridge` (in `transition.rs`). It uses a Compressed Sparse Row (CSR) format to map probability mass from patterns in Window `t` to patterns in Window `t+1`.

*   **Determinism**: The CSR format ensures that the mapping is deterministic and independent of thread scheduling.
*   **Efficiency**: Only non-zero transitions are stored/processed.

## HMM Flow

1.  **Forward Pass**:
    *   Iterate through markers in a block.
    *   Compute emission probabilities (soft alleles supported).
    *   Update `fwd` probabilities using `WeightedHmmUpdater`.
    *   Handle reservoir state updates.
    *   Normalize and store checkpoints.

2.  **Backward Pass & Posterior Emission**:
    *   Iterate backwards through markers.
    *   Retrieve `fwd` probabilities from history/checkpoints.
    *   Compute emission probabilities.
    *   Compute posterior probabilities (`fwd * bwd`).
    *   Update `bwd` probabilities.
    *   Aggregate posteriors into alleles (handling multiallelic sites).

## Multiallelic Support

The implementation explicitly handles multiallelic sites.
*   **Emissions**: Computed based on exact allele match/mismatch, not just 0/1.
*   **Posteriors**: Aggregated for all observed alleles.
