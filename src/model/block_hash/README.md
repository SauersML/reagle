# Block-Hash Clade HMM

This module implements the Adaptive Full-Panel Block-Hash Clade HMM to fix
the critical HMM state continuity bug in Reagle's imputation and phasing pipelines.

## The Bug

The current implementation (`build_pbwt_hap_indices_for_batch`) selects different
neighbor sets at each cluster, causing "index scrambling" where the same state index
refers to different reference haplotypes across marker clusters. This breaks the
Li-Stephens assumption of haplotype continuity.

## The Solution

Instead of dynamic neighbor selection, we:
1. Compress the ENTIRE reference panel into unique patterns per window
2. Track probability per pattern (plus reservoir), while retaining global IDs for mapping/bridging
3. Use TransitionBridge to correctly map probability between windows
