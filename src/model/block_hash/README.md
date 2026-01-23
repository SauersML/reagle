# Block Hash HMM Model

This module implements the Block-Hash Hidden Markov Model (HMM) used for genotype imputation and phasing.

## Overview

The Block-Hash HMM is an efficient implementation of the Li and Stephens model, optimized for large reference panels. It uses a pattern-based compression scheme (dictionary compression) and a hash-based transition matrix to achieve high performance.

## Key Components

- **CompressedBlock**: Represents a block of markers compressed using dictionary coding.
- **TransitionBridge**: Handles transitions between blocks.
- **WeightedKernel**: Implements the HMM forward/backward kernel.
- **ReferenceMap**: Manages the reference haplotype data.

## Usage

See the `IMPLEMENTATION.md` file for detailed implementation notes.
