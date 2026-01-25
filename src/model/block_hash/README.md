# Block-Hash HMM

This module implements a block-hashing HMM for efficient genotype imputation.

## Overview

The block-hash HMM compresses the reference panel into local blocks of identical haplotypes. This allows for significant speedups in HMM calculations by processing blocks of haplotypes together rather than individually.

## Key Components

- `Workspace`: Manages the memory and state for the HMM.
- `Transition`: Handles the transition probabilities between states.
- `Emission`: Calculates the emission probabilities.
