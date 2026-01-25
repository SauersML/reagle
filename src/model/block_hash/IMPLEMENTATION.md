# Implementation Details

## Data Structures

- `ReferenceMap`: The top-level structure containing the sequence of compressed blocks.
- `CompressedBlock`: Represents a window of markers. Stores unique haplotype patterns and their counts.
- `PatternId`: Local identifier for a haplotype pattern within a block.
- `GlobalId`: Global identifier for a reference haplotype.

## Algorithm

1. **Forward Pass**:
   - Calculates forward probabilities for each pattern in each block.
   - Handles transitions between blocks using the Li-Stephens model components (recombination and mutation).

2. **Backward Pass**:
   - Calculates backward probabilities.

3. **Posterior Decoding**:
   - Combines forward and backward probabilities to estimate allele dosages and genotype probabilities.
