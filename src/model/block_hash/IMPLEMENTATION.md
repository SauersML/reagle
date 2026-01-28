# Block-Hash HMM Implementation Details

## Architecture Overview

### Immutable/Mutable Separation

A critical design choice for parallelization is the separation of immutable reference data from mutable per-sample state:

- **CompressedBlock (immutable)**: Built once, Arc-wrapped, and shared across all samples.
  - Contains reference panel compression, pattern counts, and global ID mappings.
  - Allows zero-cost sharing across threads.

- **BlockHmmWorkspace (mutable)**: Per-sample workspace, thread-local or pooled.
  - Contains forward/backward buffers, emissions, and checkpoints.
  - Prevents re-compression overhead during processing.

- **ReferenceMap (immutable)**: Pre-computed container of blocks and transitions.
  - Builds all `CompressedBlock`s and `TransitionBridge`s once.
  - Enables efficient parallel imputation across samples.

## Transition Bridge: CSR-based Probability Transfer

The `TransitionBridge` (implemented in `transition.rs`) provides the KEY fix for the index scrambling bug using an efficient sparse matrix representation (CSR format).

### The Insight

`DictionaryColumn` already provides the Global ID → Pattern ID mapping via `hap_to_pattern()`. By zipping these mappings between adjacent windows, we can build a deterministic sparse transition matrix that correctly routes probability mass.

### Sparse Transition Matrix

The transition matrix is stored in CSR (Compressed Sparse Row) format, which is deterministic and cache-friendly compared to hash map approaches.

Fields:
- `sources`: Source pattern IDs (sorted for deterministic iteration).
- `destinations`: Destination pattern IDs (parallel to sources).
- `weights`: Transition weights (parallel to sources/destinations).
- `transpose_rows` / `transpose_cols`: Backward transition indices (transposed).
- `reservoir_to_pattern_ids`: Transitions from the reservoir to specific patterns.
- `pattern_to_reservoir_ids`: Transitions from specific patterns to the reservoir.

This structure allows for efficient Forward and Backward passes by iterating over linear arrays, minimizing cache misses and branching.
