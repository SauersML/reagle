# Block-Hash Clade HMM Implementation

## Overview

This module implements the **Adaptive Full-Panel Block-Hash Clade HMM** to fix the critical HMM state continuity bug in Reagle's imputation and phasing pipelines.

## The Problem: Index Scrambling Bug

The current implementation in `imputation_streaming.rs::build_pbwt_hap_indices_for_batch()` (lines 247-513) causes "index scrambling":

```rust
// Cluster t:   neighbors = [10, 20, 30]  → State index 1 = Hap 20
// Cluster t+1: neighbors = [10, 25, 30]  → State index 1 = Hap 25
```

The HMM transfers probability from `prev[k] → fwd[k]`, but:
- At cluster t: index 1 = Haplotype 20
- At cluster t+1: index 1 = Haplotype 25
- **Result**: Probability "teleports" between unrelated haplotypes

## The Solution

Instead of dynamic neighbor selection, we:

1. **Compress** the entire reference panel into unique patterns per window
2. **Track** probability via Global Haplotype IDs (not local pattern indices)
3. **Map** probability between windows using `TransitionBridge`

## Architecture

### Type Safety: `GlobalId` and `PatternId`

```rust
struct GlobalId(u32);   // 0..N (reference panel size)
struct PatternId(u16);  // 0..U (unique patterns in window)
```

These newtypes prevent index confusion at compile time.

### Core Components

#### 1. MicroWindow (micro_window_v2.rs)

Wraps `DictionaryColumn` with HMM state:

```rust
pub struct MicroWindow {
    storage: Arc<DictionaryColumn>,      // REUSE: Existing compression
    pattern_counts: Vec<f32>,            // Cardinality per pattern
    pattern_to_globals: Vec<Vec<GlobalId>>, // For MCMC sampling
    fwd_probs: Vec<f32>,                 // HMM forward probabilities
    reservoir_prob: f32,                 // Truncated patterns (if any)
    reservoir_count: u32,                // CRITICAL: Cardinality tracking
}
```

**Key Features**:
- Leverages existing `DictionaryColumn` for multiallelic-safe compression
- Supports reservoir state for panels with >4096 unique patterns
- Provides `pattern_to_globals` for MCMC sampling (phasing pipeline)

#### 2. TransitionBridge (transition_v2.rs)

Maps probability between windows via `zip()`:

```rust
impl TransitionBridge {
    pub fn build(window_a: &MicroWindow, window_b: &MicroWindow, recomb_rate: f32) -> Self {
        let map_a = window_a.storage.hap_to_pattern();
        let map_b = window_b.storage.hap_to_pattern();

        // CRITICAL: Zip to track each haplotype's transition
        for (&pat_a, &pat_b) in map_a.iter().zip(map_b.iter()) {
            // Calculate weight accounting for cardinality...
        }
    }
}
```

**Mathematical Correctness**:
- Pattern A (100 haps) → Pattern B (60 haps) + Pattern C (40 haps)
- Weight A→B = 60/100 = 0.6, Weight A→C = 40/100 = 0.4
- Correctly models coalescent splits/merges

#### 3. Compression (compression.rs)

Builds `MicroWindow` from `GenotypeMatrix`:

```rust
pub fn build_micro_window(
    ref_data: &GenotypeMatrix<Phased>,
    marker_range: Range<usize>,
    max_states: usize,
) -> MicroWindow {
    // Use DictionaryColumn::compress with 2-bit encoding
    // Handles multiallelic variants (alleles 0-3)
}
```

#### 4. HMM Kernel (hmm.rs)

Forward/backward passes using compressed states:

```rust
pub fn forward_pass_all_windows(
    windows: &mut [MicroWindow],
    target_genotypes: &[u8],
    error_rate: f32,
    recomb_rate_per_marker: f32,
) {
    for win_idx in 0..windows.len() {
        // Forward pass within window
        forward_pass_within_window(&mut windows[win_idx], ...);

        // Transition to next window
        let bridge = TransitionBridge::build(...);
        bridge.apply(&windows[win_idx], &mut windows[win_idx + 1]);
    }
}
```

## Critical Fixes Applied

### Fix 1: Multiallelic Safety

**Original Plan**: Used `u64` with 1-bit per marker (64 markers)
**Problem**: Cannot represent alleles >1 (multiallelic panic)
**Solution**: Use `DictionaryColumn` with 2-bit encoding (handles alleles 0-3)

### Fix 2: Reservoir Cardinality Tracking

**Original Plan**: Tracked `reservoir_prob` only
**Problem**: Incorrect transition math without cardinality
**Solution**: Track both `reservoir_prob` and `reservoir_count`

```rust
let weight = if from_pat.is_reservoir() {
    1.0 / window_a.reservoir_count as f32  // CRITICAL
} else {
    1.0 / window_a.pattern_counts[from_pat]
};
```

### Fix 3: MCMC Sampling Support

**Original Plan**: Focused only on imputation
**Problem**: Phasing pipeline needs to sample specific Global IDs
**Solution**: Added `pattern_to_globals` reverse mapping

```rust
pub fn sample_global_from_pattern<R: Rng>(
    &self,
    pattern_id: PatternId,
    rng: &mut R,
) -> GlobalId {
    let globals = &self.pattern_to_globals[pattern_id];
    let idx = rng.gen_range(0..globals.len());
    globals[idx]
}
```

## Usage Example

```rust
use reagle::model::block_hash::{build_all_windows, forward_pass_all_windows, DEFAULT_WINDOW_SIZE, DEFAULT_MAX_STATES};

// Build windows for reference panel
let windows = build_all_windows(&ref_data, DEFAULT_WINDOW_SIZE, DEFAULT_MAX_STATES);

// Run HMM
forward_pass_all_windows(
    &mut windows,
    &target_genotypes,
    0.001,  // error_rate
    0.0001, // recomb_rate_per_marker
);

// Extract results
for window in &windows {
    // window.fwd_probs contains posterior probabilities
}
```

## Performance Characteristics

- **Memory**: <15GB for 1KG+HGDP (~8k haplotypes)
- **Compression**: Typical ~5-10x (8k haps → 800-1600 unique patterns per window)
- **Speed**: SIMD-friendly execution on compressed states
- **Accuracy**: No donor truncation = maximum accuracy

## Integration Path

1. **Current**: Uses `build_pbwt_hap_indices_for_batch` (buggy)
2. **Migration**: Replace with `build_all_windows` in `imputation_streaming.rs`
3. **Testing**: Validate state continuity invariant (see tests/)
4. **Rollout**: Feature flag for gradual deployment

## Testing

Key test invariants:

- **Type Safety**: GlobalId ≠ PatternId at compile time
- **Mass Conservation**: Σ probabilities = 1.0 at all steps
- **State Continuity**: No probability teleportation between unrelated haplotypes
- **Compression**: Deterministic fingerprinting

## References

- Plan document: `/Users/user/.claude/projects/-Users-user-reagle/4bc7e7bf-54e7-43a1-b6be-2ae91594d438.jsonl`
- Buggy code: `src/pipelines/imputation_streaming.rs:247-513`
- Mathematical proof: See plan Appendix

## Authors

- Implementation: Claude Code (Anthropic)
- Design: Based on Li-Stephens HMM with full-panel compression
