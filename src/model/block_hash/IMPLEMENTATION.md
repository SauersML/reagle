# Implementation Details

## Data Structures

### Identifiers
To prevent index confusion, the implementation distinguishes between:
- `GlobalId`: Index of a haplotype in the original reference panel (0..N).
- `PatternId`: Index of a unique pattern in a compressed block (0..K).
- `Reservoir`: A special state collecting rare patterns to handle "long tail" haplotypes efficiently.

### Compression
The reference panel is divided into fixed-size windows (blocks). Within each block, identical haplotypes are grouped into patterns.
A `TransitionBridge` is pre-calculated to map patterns from one block to patterns in the next block.

## Algorithms

### Li-Stephens Model
The HMM implements the Li-Stephens model, which accounts for:
- **Recombination**: Probability of switching from one haplotype to another.
- **Mutation/Error**: Probability of the observed genotype differing from the template haplotype.

### Forward-Backward
The standard Forward-Backward algorithm is adapted:
1. **Forward Pass**: Calculates probabilities of being in each state given the observed data up to that point.
2. **Backward Pass**: Calculates probabilities of observing the future data given the current state.
3. **Posterior Decoding**: Combines forward and backward probabilities to infer missing genotypes.

### Weighted Transitions
Transitions are weighted by the number of haplotypes represented by each pattern (pattern cardinality).
This ensures that the probabilistic model accurately reflects the population frequencies.

## Optimizations

- **SIMD**: The backward pass uses manual SIMD vectorization (via the `wide` crate) to process multiple states in parallel.
- **Memory Pooling**: `BlockHmmWorkspace` reuses buffers to minimize allocation overhead.
