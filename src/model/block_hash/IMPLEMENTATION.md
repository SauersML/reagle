# Implementation Details

## GlobalId vs PatternId

The Block-Hash HMM implementation distinguishes between `GlobalId` and `PatternId` to prevent 'index scrambling' bugs.

- `GlobalId`: A stable 0..N index representing the global haplotype index across the chromosome.
- `PatternId`: A local 0..K compressed index representing the unique haplotype pattern within a specific window.

This distinction ensures that we don't accidentally mix up global haplotype indices with local compressed pattern indices, which could lead to incorrect state transitions and imputation errors.
