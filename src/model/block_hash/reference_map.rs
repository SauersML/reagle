//! # Reference Map: Pre-computed Immutable Data Structure
//!
//! This module defines the ReferenceMap which holds:
//! - All CompressedBlocks (immutable, Arc-wrapped)
//! - All TransitionBridges (pre-computed, Arc-wrapped)
//!
//! Built ONCE for the entire reference panel, then shared across all samples.

use super::compressed_block::CompressedBlock;
use super::transition::TransitionBridge;
use super::workspace::BlockHmmWorkspace;
use crate::pipelines::imputation::AllelePosteriors;
use std::sync::Arc;

/// Pre-computed reference map for block-hash HMM
///
/// Contains all immutable reference data and pre-computed transitions.
/// Built once, shared across all target samples via Arc.
#[derive(Debug)]
pub struct ReferenceMap {
    /// Compressed blocks (immutable, Arc-wrapped)
    pub blocks: Vec<Arc<CompressedBlock>>,

    /// Pre-computed transition bridges between consecutive blocks
    /// bridges[i] maps from blocks[i] to blocks[i+1]
    pub bridges: Vec<Arc<TransitionBridge>>,

    /// Window size in markers
    pub window_size: usize,

    /// Actual maximum states observed in any block
    pub max_observed_states: usize,
}

impl ReferenceMap {
    /// Build a ReferenceMap from pre-compressed blocks and boundary rates.
    ///
    /// # Arguments
    /// * `blocks` - Compressed blocks in window order.
    /// * `boundary_rates` - Recombination rates between consecutive blocks.
    /// * `window_size` - Block size used for workspace sizing.
    pub fn build_from_blocks(
        blocks: Vec<Arc<CompressedBlock>>,
        boundary_rates: &[f32],
        window_size: usize,
    ) -> Arc<Self> {
        assert_eq!(
            boundary_rates.len(),
            blocks.len().saturating_sub(1),
            "boundary_rates must have len = blocks.len() - 1"
        );

        let mut bridges = Vec::with_capacity(blocks.len().saturating_sub(1));
        for i in 0..blocks.len().saturating_sub(1) {
            let bridge = TransitionBridge::build(&blocks[i], &blocks[i + 1], boundary_rates[i]);
            bridges.push(Arc::new(bridge));
        }

        let max_observed_states = blocks
            .iter()
            .map(|b| b.n_patterns())
            .max()
            .unwrap_or(0);

        Arc::new(Self {
            blocks,
            bridges,
            window_size,
            max_observed_states,
        })
    }

    /// Allocate a workspace for this reference map
    ///
    /// Uses the ACTUAL observed max patterns, not the theoretical config limit.
    /// This prevents over-allocation and ensures safety even if max_states=0 (no limit).
    pub fn create_workspace(&self) -> BlockHmmWorkspace {
        BlockHmmWorkspace::new(self.max_observed_patterns(), self.blocks.len(), self.window_size)
    }

    /// Calculate the actual maximum number of states required by any block
    pub fn max_observed_patterns(&self) -> usize {
        self.max_observed_states
    }

    /// Run forward pass up to a specific marker index
    ///
    /// Restores the checkpoint of the containing block and advances to the target marker.
    /// Used for extracting HMM state for priors handoff.
    pub fn forward_to_marker(
        &self,
        target_genotypes: &[u8],
        error_rate: f32,
        ws: &mut BlockHmmWorkspace,
        marker_idx: usize,
    ) {
        // Find block containing marker_idx
        let block_idx = self.blocks.partition_point(|b| b.end_marker <= marker_idx);
        
        if block_idx < self.blocks.len() {
            let block = &self.blocks[block_idx];
            
            // Restore checkpoint
            ws.restore_checkpoint(block_idx, block.n_patterns());
            
            // Extract genotypes for this block
            let block_genotypes =
                &target_genotypes[block.start_marker..block.end_marker.min(target_genotypes.len())];
            
            let local_marker = marker_idx - block.start_marker;
            
            // Run forward partial
            super::hmm::forward_to_marker_in_block(
                block,
                block_genotypes,
                error_rate,
                ws,
                local_marker
            );
        }
    }

    /// Run forward pass with checkpointing
    ///
    /// Saves forward state at the start of each block for later combination.
    pub fn forward_pass(
        &self,
        target_genotypes: &[u8],
        error_rate: f32,
        ws: &mut BlockHmmWorkspace,
    ) {
        for (block_idx, block) in self.blocks.iter().enumerate() {
            // Save checkpoint at start of this block
            ws.save_checkpoint(block_idx, block.n_patterns());

            // Extract target genotypes for this block
            let block_genotypes =
                &target_genotypes[block.start_marker..block.end_marker.min(target_genotypes.len())];

            // Run forward within block
            super::hmm::forward_within_block(block, block_genotypes, error_rate, ws);

            // Apply transition to next block
            if block_idx < self.bridges.len() {
                self.bridges[block_idx].apply_forward(block, &self.blocks[block_idx + 1], ws);
            }
        }
    }


    ///
    /// Combines saved forward state with backward probabilities to compute posteriors.
    pub fn backward_and_emit_posteriors(
        &self,
        target_genotypes: &[u8],
        error_rate: f32,
        ws: &mut BlockHmmWorkspace,
    ) -> Vec<AllelePosteriors> {
        // Pre-allocate posteriors with default values to enable direct slicing
        let mut posteriors = vec![AllelePosteriors::Biallelic(0.0); target_genotypes.len()];

        // Initialize backward to neutral likelihood (1.0)
        // Standard HMM backward pass: beta_T(i) = 1.0 for all states
        if let Some(last_block) = self.blocks.last() {
             let n_patterns = last_block.n_patterns();
             ws.bwd[..n_patterns].fill(1.0);

             // Initialize reservoir to 1.0 (neutral likelihood)
             // Do NOT weight by reservoir_count - backward variables are likelihoods, not mass
             if last_block.reservoir_count > 0 {
                 ws.reservoir_prob_bwd = 1.0;
             } else {
                 ws.reservoir_prob_bwd = 0.0;
             }
        }

        // Backward pass in reverse order
        for block_idx in (0..self.blocks.len()).rev() {
            let block = &self.blocks[block_idx];

            let start = block.start_marker;
            let end = block.end_marker.min(target_genotypes.len());
            
            // Extract target genotypes for this block
            let block_genotypes = &target_genotypes[start..end];

            // Restore forward checkpoint for this block
            ws.restore_checkpoint(block_idx, block.n_patterns());

            // Run backward and emit posteriors directly into the pre-allocated slice
            // Slice range in `posteriors` is `start..end`
            let output_slice = &mut posteriors[start..end];
            
            super::hmm::backward_and_emit_block(
                block,
                block_genotypes,
                error_rate,
                ws,
                output_slice,
            );

            // Apply inverse transition to previous block
            if block_idx > 0 {
                self.bridges[block_idx - 1].apply_backward(
                    &self.blocks[block_idx - 1],
                    block,
                    ws,
                );
            }
        }

        posteriors
    }



}
