# Block-Hash HMM

This module implements the core HMM logic for imputation using the block-hash compression scheme.

## Overview

The block-hash HMM improves performance by compressing the reference panel into local haplotype blocks.
States are represented by unique haplotype patterns within each block, significantly reducing the state space compared to the full reference panel.
