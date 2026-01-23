# Block-Hash Implementation Details

## Data Structures

### CompressedBlock

Stores the compressed representation of reference haplotypes.

- `dictionary`: A list of unique haplotype patterns in the block.
- `map`: A mapping from global haplotype ID to the dictionary index (PatternId).

### TransitionBridge

Calculates transition probabilities between blocks. It uses the `WeightedKernel` to distribute probability mass based on genetic distance.

## Algorithms

### Compression

Haplotypes are grouped into blocks based on a fixed window size. Within each block, unique allele sequences are identified and stored in the dictionary.

### Forward Pass

Calculates the probability of observing the target haplotype given the reference haplotypes up to the current position.

### Backward Pass

Calculates the probability of observing the target haplotype given the reference haplotypes from the end of the chromosome back to the current position.
