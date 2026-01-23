# Block-Hash HMM Implementation Details

## Data Structures

### CompressedBlock

The `CompressedBlock` struct is the core data structure for storing compressed haplotype data. It maintains a mapping from the global haplotype ID ($N$) to the local pattern ID ($K$). This is achieved using a `hap_to_state` vector.

### TransitionBridge

The `TransitionBridge` is responsible for calculating transition probabilities. It implements the Li-Stephens model, but operates directly on the global haplotype IDs. This avoids the overhead of mapping back and forth between global IDs and pattern IDs during the transition calculation, which is the most computationally intensive part of the HMM.

## Algorithms

### Forward-Backward Algorithm

The forward-backward algorithm is implemented using SIMD instructions where possible to accelerate the calculation of posterior probabilities. The `HmmUpdater` struct provides the `fwd_update` and `bwd_update` methods for this purpose.

### Dictionary Compression

Dictionary compression is used to reduce the memory footprint of the haplotype data. Blocks of markers are compressed into unique patterns, and only the unique patterns are stored. The `CompressedBlock` struct manages this compression and provides efficient access to the allele data.
