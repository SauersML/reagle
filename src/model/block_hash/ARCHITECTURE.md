# Block-Hash HMM Architecture (v2.0)

## Critical Architectural Fix

Based on code review, the original implementation had a **critical flaw** that prevented parallel processing:

### The Problem

**MicroWindow conflated immutable and mutable data**:
```rust
// WRONG: Mixes reference data with per-sample HMM state
struct MicroWindow {
    storage: Arc<DictionaryColumn>,  // Immutable (reference panel)
    fwd_probs: Vec<f32>,             // Mutable (per-sample state)
    bwd_probs: Vec<f32>,             // Mutable (per-sample state)
}
```

**Consequences**:
- Cannot share reference data across threads
- For 1,000 samples, must either:
  - Re-compress reference 1,000 times → **CPU waste**
  - Clone MicroWindow 1,000 times → **RAM waste**
- Prevents efficient parallel processing

### The Solution

**Separate immutable from mutable**:

```rust
// Immutable: Built ONCE, Arc-wrapped, shared across all samples
struct CompressedBlock {
    storage: Arc<DictionaryColumn>,
    pattern_counts: Vec<f32>,
    pattern_to_globals: Vec<Vec<GlobalId>>,
    // No mutable HMM state!
}

// Mutable: Per-sample workspace (thread-local or pooled)
struct BlockHmmWorkspace {
    fwd: Vec<f32>,
    bwd: Vec<f32>,
    emissions: Vec<f32>,
    checkpoints: Vec<(Vec<f32>, f32)>,  // For forward-backward combination
}

// Container: Pre-computed map built once
struct ReferenceMap {
    blocks: Vec<Arc<CompressedBlock>>,  // Share across threads (zero-cost)
    bridges: Vec<Arc<TransitionBridge>>,  // Pre-computed once
}
```

## New Architecture

```
┌─────────────────────────────────────────────────────────┐
│ ReferenceMap (Arc-wrapped, built once per chromosome)  │
├─────────────────────────────────────────────────────────┤
│  blocks: Vec<Arc<CompressedBlock>>                      │
│  bridges: Vec<Arc<TransitionBridge>>                    │
│  window_size: 32                                         │
│  max_states: 4096                                        │
└─────────────────────────────────────────────────────────┘
         ↓ Arc::clone (zero-cost)
   ┌───────────────────────────┐
   │  Per-Sample Processing    │
   │  (Parallel / Thread-Local)│
   ├───────────────────────────┤
   │  workspace: BlockHmmWorkspace  │
   │    ├─ fwd: Vec<f32>       │
   │    ├─ bwd: Vec<f32>       │
   │    └─ checkpoints         │
   └───────────────────────────┘
```

## API Changes

### Old API (Single-threaded)
```rust
// WRONG: Cannot parallelize
let mut windows = build_all_windows(&ref_data, 32, 4096);

for sample in samples {
    forward_pass_all_windows(&mut windows, &sample.genotypes, 0.001, 0.0001);
    backward_pass_all_windows(&mut windows, &sample.genotypes, 0.001, 0.0001);
    // Extract results...
}
```

### New API (Parallel-ready)
```rust
// Build reference map ONCE
let ref_map = ReferenceMap::build(&ref_data, 32, 4096, 0.0001);

// Process samples in parallel
samples.par_iter().map(|sample| {
    // Get thread-local workspace
    let mut ws = ref_map.create_workspace();

    // Run HMM (ref_map is Arc-cloned, zero-cost)
    let dosages = ref_map.impute_sample(&sample.genotypes, 0.001, &mut ws);

    dosages
}).collect()
```

## Missing Output Generation

The original implementation lacked **dosage emission logic**:

### Forward-Backward Combination Problem

To compute dosage at marker M, you need:
- `fwd[M]` = Forward probability at M
- `bwd[M]` = Backward probability at M
- `dosage[M]` = Σ (fwd[i] × bwd[i] × allele[i])

But the old code overwrote buffers:
```rust
// WRONG: Loses intermediate forward states
forward_pass();   // fwd[0..N] computed
backward_pass();  // OVERWRITES fwd[], can't combine!
```

### Solution: Checkpointing

```rust
// Forward pass with checkpointing
for block_idx in 0..n_blocks {
    ws.save_checkpoint(block_idx);  // Save fwd state at block start
    forward_within_block(...);
    apply_transition(...);
}

// Backward pass + emission
for block_idx in (0..n_blocks).rev() {
    ws.restore_checkpoint(block_idx);  // Recover fwd state
    let dosages = backward_and_emit_block(
        fwd: &ws.fwd,      // From checkpoint
        bwd: &mut ws.bwd,  // Current backward
    );
}
```

## Transition Pre-computation

The old code rebuilt transitions for every sample:

```rust
// WRONG: Recomputes transition matrix 1000x for 1000 samples
for sample in samples {
    for win_idx in 0..n_windows {
        let bridge = TransitionBridge::build(  // WASTEFUL!
            &windows[win_idx],
            &windows[win_idx + 1],
            recomb_rate
        );
        bridge.apply(...);
    }
}
```

New code pre-computes once:

```rust
// Build transitions ONCE (depends only on reference panel)
let bridges: Vec<Arc<TransitionBridge>> = (0..n_windows-1)
    .map(|i| Arc::new(TransitionBridge::build(&blocks[i], &blocks[i+1], r)))
    .collect();

// Reuse for all samples (Arc-cloned, zero-cost)
for sample in samples {
    for i in 0..bridges.len() {
        bridges[i].apply(...);  // Just matrix multiply, no rebuild
    }
}
```

## Implementation Status

### ✅ Completed
- `compressed_block.rs` - Immutable reference data structure
- `workspace.rs` - Mutable per-sample state with checkpointing
- `reference_map.rs` - Container for parallel processing
- `compression.rs` - Updated to build CompressedBlock

### ⚠️ In Progress (Does Not Compile)
- `hmm.rs` - Needs workspace-based functions:
  - `forward_within_block(block, ws)`
  - `backward_and_emit_block(block, ws) -> Vec<f32>`
- `transition.rs` - Needs workspace-based methods:
  - `apply_forward(block_a, block_b, ws)`
  - `apply_backward(block_a, block_b, ws)`

### 📋 Todo
- Update `imputation_streaming.rs` to use ReferenceMap
- Integration tests with parallel processing
- Performance benchmarks (expected ~10-30x speedup)
- Memory profiling

## Performance Benefits

| Metric | Old (MicroWindow) | New (ReferenceMap) |
|--------|-------------------|-------------------|
| Reference compression | N times (per sample) | Once (shared) |
| Transition building | N times (per sample) | Once (pre-computed) |
| Memory per sample | Full MicroWindow clone | Small workspace |
| Parallelization | Impossible | Thread-safe (Arc) |
| Expected speedup | Baseline | ~10-30x |

## Migration Path

1. **Phase 1**: Complete workspace-based HMM functions
2. **Phase 2**: Test with single sample (validate correctness)
3. **Phase 3**: Wire into pipeline with parallel processing
4. **Phase 4**: Performance benchmarking and optimization
5. **Phase 5**: Remove old MicroWindow API

## References

- Original architecture: `IMPLEMENTATION.md`
- Code review: User feedback on architectural flaws
- Li-Stephens HMM: Stephens & Scheet (2005)
- Parallel imputation: Similar to Minimac4 architecture
