# Block-Hash HMM

This directory contains the implementation of the Block-Hash Hidden Markov Model used for efficient genotype imputation and phasing.

## Overview

The Block-Hash HMM optimizes the Li-Stephens model by using a compressed representation of reference haplotypes. Instead of calculating transition probabilities for every haplotype pair, it groups identical haplotype segments into "blocks" and calculates transitions between these blocks.

## Key Components

- `compressed_block.rs`: Data structures for compressed haplotype blocks.
- `compression.rs`: Logic for compressing reference haplotypes into blocks.
- `hmm.rs`: The HMM implementation, including forward and backward passes.
- `reference_map.rs`: Manages the mapping of reference haplotypes to the HMM states.
- `transition.rs`: Handles transition probability calculations.
- `weighted_kernel.rs`: Implements the weighted kernel for probability distribution.

## Usage

This module is used internally by the imputation pipeline to perform efficient inference on large reference panels.
