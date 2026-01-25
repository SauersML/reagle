# Block-Hash HMM

This module implements a block-hashing Hidden Markov Model (HMM) for efficient genotype imputation and phasing.
It is based on the algorithms used in the BEAGLE software, optimized for performance in Rust.

## Overview

The Block-Hash HMM compresses the reference panel into blocks of unique haplotypes (patterns).
Transitions are calculated between patterns rather than individual haplotypes, significantly
reducing the computational complexity from O(N) to O(K) where N is the number of reference haplotypes
and K is the number of unique patterns in a window.

## Key Features

- **Compression**: Reference haplotypes are compressed into blocks.
- **Pattern-based HMM**: Calculations are performed on unique patterns.
- **SIMD Acceleration**: Critical paths are optimized using SIMD instructions.
- **Streaming**: Supports streaming processing for large datasets.
