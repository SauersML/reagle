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
use crate::data::storage::matrix::GenotypeMatrix;
use crate::data::storage::phase_state::Phased;
use std::sync::Arc;

/// Pre-computed reference map for block-hash HMM
///
/// Contains all immutable reference data and pre-computed transitions.
/// Built once, shared across all target samples via Arc.
pub struct ReferenceMap {
    /// Compressed blocks (immutable, Arc-wrapped)
    pub blocks: Vec<Arc<CompressedBlock>>,

    /// Pre-computed transition bridges between consecutive blocks
    /// bridges[i] maps from blocks[i] to blocks[i+1]
    pub bridges: Vec<Arc<TransitionBridge>>,

    /// Window size in markers
    pub window_size: usize,

    /// Maximum states per block
    pub max_states: usize,
}

impl ReferenceMap {
    /// Build a ReferenceMap from a reference panel
    ///
    /// # Arguments
    /// * `ref_data` - Reference panel genotype matrix
    /// * `window_size` - Size of each window in markers
    /// * `max_states` - Maximum unique patterns per window
    /// * `recomb_rates` - Recombination rates between markers [marker_idx].
    ///                    Rate at i is between marker i and i+1.
    ///
    /// # Returns
    /// Arc-wrapped ReferenceMap ready for parallel sample processing
    pub fn build(
        ref_data: &GenotypeMatrix<Phased>,
        window_size: usize,
        max_states: usize,
        recomb_rates: &[f32],
    ) -> Arc<Self> {
        let n_markers = ref_data.n_markers();
        let n_windows = (n_markers + window_size - 1) / window_size;

        assert_eq!(recomb_rates.len(), n_markers, "Recombination rates must length must match n_markers");

        // Build all compressed blocks
        let mut blocks = Vec::with_capacity(n_windows);

        for win_idx in 0..n_windows {
            let start = win_idx * window_size;
            let end = (start + window_size).min(n_markers);
            
            let block_rates = &recomb_rates[start..end];

            // Build compressed block (uses existing compression module)
            let block = super::compression::build_compressed_block(
                ref_data,
                start..end,
                max_states,
                block_rates,
            );

            blocks.push(Arc::new(block));
        }

        // Pre-compute all transition bridges
        let mut bridges = Vec::with_capacity(n_windows.saturating_sub(1));

        for i in 0..n_windows.saturating_sub(1) {
            // Rate between block i and i+1
            // Block i ends at `blocks[i].end_marker`.
            // The last marker in block i is `end_marker - 1`.
            // The rate at `end_marker - 1` defines transition to `end_marker`.
            let boundary_rate = recomb_rates[blocks[i].end_marker - 1];
            
            let bridge = TransitionBridge::build(&blocks[i], &blocks[i + 1], boundary_rate);
            bridges.push(Arc::new(bridge));
        }

        Arc::new(Self {
            blocks,
            bridges,
            window_size,
            max_states,
        })
    }

    /// Allocate a workspace for this reference map
    pub fn create_workspace(&self) -> BlockHmmWorkspace {
        BlockHmmWorkspace::new(self.max_states, self.blocks.len(), self.window_size)
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

    /// Run backward pass and emit dosages
    ///
    /// Combines saved forward state with backward probabilities to compute dosages.
    pub fn backward_and_emit_dosages(
        &self,
        target_genotypes: &[u8],
        error_rate: f32,
        ws: &mut BlockHmmWorkspace,
    ) -> Vec<f32> {
        let mut dosages = Vec::with_capacity(target_genotypes.len());

        // Initialize backward uniform
        if let Some(last_block) = self.blocks.last() {
            let n_patterns = last_block.n_patterns();
            let uniform = 1.0 / n_patterns as f32;
            ws.bwd[..n_patterns].fill(uniform);
            ws.reservoir_prob_bwd = 0.0;
        }

        // Backward pass in reverse order
        for block_idx in (0..self.blocks.len()).rev() {
            let block = &self.blocks[block_idx];

            // Extract target genotypes for this block
            let block_genotypes =
                &target_genotypes[block.start_marker..block.end_marker.min(target_genotypes.len())];

            // Restore forward checkpoint for this block
            ws.restore_checkpoint(block_idx, block.n_patterns());

            // Run backward and emit dosages for this block
            let block_dosages = super::hmm::backward_and_emit_block(
                block,
                block_genotypes,
                error_rate,
                ws,
            );

            // Prepend to dosages (we're going backwards)
            dosages.splice(0..0, block_dosages);

            // Apply inverse transition to previous block
            if block_idx > 0 {
                self.bridges[block_idx - 1].apply_backward(
                    &self.blocks[block_idx - 1],
                    block,
                    ws,
                );
            }
        }

        dosages
    }


}
