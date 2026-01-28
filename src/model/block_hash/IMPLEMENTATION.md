# Implementation Details: Block-Hash HMM

This document details the internal data structures and algorithms of the Block-Hash HMM module.

## Core Data Structures

### 1. Index Types (`types.rs`)

We use `newtype` wrappers to enforce type safety and prevent index confusion.

*   **`GlobalId(u32)`**:
    *   Represents a specific haplotype index in the original reference panel.
    *   Range: `0` to `n_ref_haplotypes - 1`.
    *   Invariant: Stable across all windows. `GlobalId(k)` always refers to the $k$-th haplotype in the VCF.

*   **`PatternId(u32)`**:
    *   Represents a unique haplotype pattern *within a specific window*.
    *   Range: `0` to `n_patterns - 1`.
    *   **Sentinel**: `PatternId::RESERVOIR` (`u32::MAX`) represents all rare haplotypes that were truncated into the reservoir.

### 2. Compressed Reference (`compressed_block.rs`)

The `CompressedBlock` struct stores the reference data for a single window in a compressed format.

```rust
pub struct CompressedBlock {
    // Genomic range
    pub start_marker: usize,
    pub end_marker: usize,

    // Mapping: Global Hap -> Local Pattern
    // Crucial for resolving the "Index Scrambling" bug.
    pub hap_to_state: Vec<PatternId>,

    // Pattern metadata
    pub pattern_counts: Vec<f32>,     // Number of haplotypes per pattern
    pub pattern_globals: Vec<GlobalId>, // Reverse mapping (for debug/export)

    // Reservoir (Truncated Haplotypes)
    pub reservoir_count: u32,
    pub reservoir_freqs: Vec<f32>,    // Allele frequencies of reservoir haps

    // Fast Emission Data
    pub unpacked_alleles: Vec<u8>,    // Flat buffer: [pattern * width + marker]
}
```

### 3. Mutable Workspace (`workspace.rs`)

The `BlockHmmWorkspace` holds the per-sample mutable state required for the Forward-Backward algorithm.

```rust
pub struct BlockHmmWorkspace {
    // Probability Buffers (SIMD-aligned)
    pub fwd: AVec<f32, ConstAlign<32>>,
    pub bwd: AVec<f32, ConstAlign<32>>,

    // Reservoir State
    pub reservoir_prob_fwd: f32,
    pub reservoir_prob_bwd: f32,

    // Checkpoints for long-range phasing
    pub checkpoints: Vec<(Vec<f32>, f32)>,
}
```

## Algorithms

### 1. Compression (Dictionary Column)

The compression algorithm reduces the reference panel into unique patterns.

1.  **Windowing**: The chromosome is divided into fixed-size windows (e.g., 64 markers).
2.  **Hashing**: Within each window, every reference haplotype is hashed.
3.  **Deduplication**: Identical haplotypes are grouped into a single `PatternId`.
4.  **Truncation**: Patterns with count < `min_count` (rare variants) are merged into the **Reservoir**.
5.  **Mapping**: A mapping `hap_to_state[GlobalId] -> PatternId` is stored to enable transitioning between windows.

### 2. Transition Bridge (CSR)

To transition between Window $W_i$ and Window $W_{i+1}$, we cannot simply map `PatternId(k)` $\to$ `PatternId(k)` because the patterns change.

We use a **Transition Bridge** derived from the shared `GlobalId`s:

1.  Construct a sparse matrix where $M_{uv}$ is the number of haplotypes that are in Pattern $u$ in Window $i$ AND Pattern $v$ in Window $i+1$.
2.  This is stored in **Compressed Sparse Row (CSR)** format for efficient matrix-vector multiplication.
3.  **Update Rule**:
    $$ P(v | \text{Window } i+1) = \sum_{u} P(u | \text{Window } i) \times \frac{M_{uv}}{\text{count}(u)} $$

This ensures that probability mass flows correctly along the physical haplotypes.

### 3. Forward-Backward Algorithm

The HMM uses a standard Forward-Backward algorithm but optimized for the block structure:

*   **Intra-Block**: Standard Li-Stephens updates using the `unpacked_alleles` for emissions.
*   **Inter-Block**: Apply the `TransitionBridge` to map probabilities to the new set of patterns.
*   **Reservoir Handling**: The reservoir is treated as a single "average" state. Its emission probability is calculated based on the allele frequencies of the haplotypes it contains (`reservoir_freqs`).

## Memory Layout & Alignment

*   **SIMD**: `AVec` with 32-byte alignment is used for probability buffers to enable AVX-512 operations.
*   **Flat Buffers**: `unpacked_alleles` uses a flat `Vec<u8>` layout `[pattern * width + marker]` to avoid pointer chasing and maximize cache locality during the innermost emission loop.
