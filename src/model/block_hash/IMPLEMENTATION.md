# Block-Hash HMM Implementation Details

## Data Structures

### GlobalId vs PatternId

- **GlobalId**: Represents the original haplotype index in the reference panel ($0$ to $N-1$).
- **PatternId**: Represents the unique haplotype pattern index within a compressed block ($0$ to $K-1$, where $K \ll N$).

### Compression

Reference haplotypes are compressed into blocks. Within each block, identical haplotypes are grouped into patterns. This reduces the state space for emission calculations from $N$ to $K$.

### Transitions

Transitions are computed using a specialized bridge structure that accounts for recombination between blocks. The transition probability includes:
- **Switch**: Transition to a different haplotype due to recombination.
- **Stay**: Remain on the same haplotype.

## Algorithm

The HMM uses a modified Forward-Backward algorithm:

1.  **Forward Pass**: Computes probabilities of being in each state at each marker, moving left to right.
2.  **Backward Pass**: Computes probabilities moving right to left.
3.  **Posterior Decoding**: Combines forward and backward probabilities to estimate genotype probabilities.

## Optimizations

- **SIMD**: Vectorized operations for probability updates.
- **Reservoir Sampling**: Used for efficient state selection in large panels.
- **Checkpointing**: Reduces memory usage by storing state only at block boundaries.
