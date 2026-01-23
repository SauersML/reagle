# Block-Hash HMM

This module implements a specialized HMM for genotype imputation using a block-hashing approach for efficiency.

## Overview

The Block-Hash HMM implementation tracks probability states by **Global Haplotype ID ($N$)** instead of the compressed Patterns ($K$). This design choice simplifies the transition logic and allows for more direct haplotype tracking.

## Core Components

### CompressedBlock

`CompressedBlock` is responsible for storing the haplotype data in a compressed format. It maps global haplotype IDs ($N$) to their corresponding patterns ($K$) via `hap_to_state`. This mapping is crucial for efficient emission probability lookup.

### TransitionBridge

`TransitionBridge` handles the transition probabilities between markers. It performs a global Li-Stephens update directly on the global haplotype IDs, without needing to map back and forth to patterns during the transition step. This approach maintains the full haplotype context while still benefiting from the compression in the emission step.
