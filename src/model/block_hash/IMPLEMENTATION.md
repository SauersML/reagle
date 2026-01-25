# Implementation Details

This document outlines the technical implementation details of the Block-Hash HMM module.

## Core Components

### 1. Type Safety (`types.rs`)

To prevent "index scrambling" bugs, we use distinct newtypes for different index spaces:

- **`GlobalId`**: Identifies a specific haplotype in the reference panel (0..N). Stable across the entire chromosome.
- **`PatternId`**: Identifies a unique haplotype pattern within a specific window (0..K). Local to the window.
- **`PatternId::RESERVOIR`**: A sentinel value indicating a haplotype that falls into the "reservoir" (truncated set of rare patterns).

The compiler enforces that these types cannot be mixed, preventing accidental indexing errors.

### 2. Transition Bridge (`transition.rs`)

The `TransitionBridge` is the key mechanism for fixing the state continuity bug. It provides a deterministic, sparse transition matrix between windows.

- **CSR Format**: Uses Compressed Sparse Row (CSR) format for cache-friendly probability transfer.
- **Determinism**: Transitions are sorted to ensure bit-exact reproducibility across runs.
- **Mass Conservation**: Strictly enforces probability mass conservation (sum = 1.0) during transitions.
- **Reservoir Handling**: Explicitly handles transitions to/from the reservoir state, distributing mass proportionally based on pattern cardinality.

### 3. HMM Kernel (`hmm.rs`)

The HMM kernel integrates with Reagle's existing SIMD-optimized infrastructure.

- **Reuse**: Reuses `WeightedHmmUpdater` kernels instead of rewriting scalar loops.
- **Soft Probabilities**: Supports "soft" allele probabilities (Posteriors) for input, allowing uncertainty propagation.
- **Forward/Backward**: Implements full Forward-Backward algorithm with numerical stability checks.
- **Normalization**: Constant-time normalization to prevent underflow.

### 4. Compressed Block (`compressed_block.rs`)

The `CompressedBlock` represents a window of the reference panel.

- **Bit-Packing**: Genotypes are bit-packed for memory efficiency.
- **Pattern Deduplication**: Identical haplotypes within the window are collapsed into a single pattern.
- **Reservoir Logic**: Rare patterns (below a frequency threshold) can be grouped into a "reservoir" to save compute, while maintaining correct aggregate probability flow.

### 5. Workspace (`workspace.rs`)

The `BlockHmmWorkspace` manages mutable state during HMM execution.

- **Scratch Buffers**: Pre-allocated vectors for forward/backward tables to avoid allocation in the inner loop.
- **Emissions Buffer**: Dedicated buffer for emission probabilities.
- **Checkpoints**: Support for checkpointing state for efficient re-computation or debugging.

## Multiallelic Support

The implementation treats alleles as `u8` values and performs exact matching. This allows native support for multiallelic variants without converting them to multiple biallelic markers.

- **Match**: `P(obs | hidden) = 1 - e`
- **Mismatch**: `P(obs | hidden) = e / (n_alleles - 1)`

## Parallelization Strategy

The design heavily favors data parallelism:
- Reference structures (`ReferenceMap`, `CompressedBlock`, `TransitionBridge`) are immutable and shared via `Arc`.
- Each thread processes a distinct sample (or batch of samples) using its own `BlockHmmWorkspace`.
- `rayon` is used for parallel construction of the reference map and parallel processing of samples.
