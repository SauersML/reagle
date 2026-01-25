//! # Reference Map: Pre-computed Immutable Data Structure
//!
//! This module defines the ReferenceMap which holds:
//! - All CompressedBlocks (immutable, Arc-wrapped)
//! - All TransitionBridges (pre-computed, Arc-wrapped)
//!
//! Built ONCE for the entire reference panel, then shared across all samples.

use super::compressed_block::CompressedBlock;
use super::hmm::{TargetAlleleProbs, TargetAlleleProbsView};
use super::transition::TransitionBridge;
use super::workspace::BlockHmmWorkspace;
use crate::pipelines::imputation::AllelePosteriors;
use rayon::prelude::*;
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

    /// Recombination rates between consecutive blocks (len = blocks.len()-1)
    pub boundary_rates: Vec<f32>,
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

        // Build bridges in parallel - each bridge only depends on consecutive blocks
        let bridges: Vec<Arc<TransitionBridge>> = (0..blocks.len().saturating_sub(1))
            .into_par_iter()
            .map(|i| {
                let bridge = TransitionBridge::build(&blocks[i], &blocks[i + 1], boundary_rates[i]);
                Arc::new(bridge)
            })
            .collect();

        let max_observed_states = blocks.iter().map(|b| b.n_patterns()).max().unwrap_or(0);

        Arc::new(Self {
            blocks,
            bridges,
            window_size,
            max_observed_states,
            boundary_rates: boundary_rates.to_vec(),
        })
    }

    /// Allocate a workspace for this reference map
    ///
    /// Uses the ACTUAL observed max patterns, not the theoretical config limit.
    /// This prevents over-allocation and ensures safety even if max_states=0 (no limit).
    pub fn create_workspace(&self) -> BlockHmmWorkspace {
        BlockHmmWorkspace::new(
            self.max_observed_patterns(),
            self.blocks.len(),
            self.window_size,
        )
    }

    /// Calculate the actual maximum number of states required by any block
    pub fn max_observed_patterns(&self) -> usize {
        self.max_observed_states
    }

    /// Run forward pass up to a specific marker using soft allele probabilities
    pub fn forward_to_marker_probs(
        &self,
        target_probs: &TargetAlleleProbs,
        error_rate: f32,
        ws: &mut BlockHmmWorkspace,
        marker_idx: usize,
        initial_recomb_rate: f32,
    ) {
        let block_idx = self.blocks.partition_point(|b| b.end_marker <= marker_idx);

        if block_idx < self.blocks.len() {
            let block = &self.blocks[block_idx];
            ws.restore_checkpoint(block_idx, block.n_patterns());

            let local_marker = marker_idx - block.start_marker;
            let view = TargetAlleleProbsView::new(target_probs, block.start_marker);
            super::hmm::forward_to_marker_in_block_probs(
                block,
                &view,
                error_rate,
                ws,
                local_marker,
                if block_idx == 0 {
                    initial_recomb_rate
                } else {
                    0.0
                },
            );
        }
    }

    /// Run forward pass with checkpointing using soft allele probabilities
    pub fn forward_pass_probs(
        &self,
        target_probs: &TargetAlleleProbs,
        error_rate: f32,
        ws: &mut BlockHmmWorkspace,
        initial_recomb_rate: f32,
    ) {
        for (block_idx, block) in self.blocks.iter().enumerate() {
            ws.save_checkpoint(block_idx, block.n_patterns());

            let view = TargetAlleleProbsView::new(target_probs, block.start_marker);
            super::hmm::forward_within_block_probs(
                block,
                &view,
                error_rate,
                ws,
                if block_idx == 0 {
                    initial_recomb_rate
                } else {
                    0.0
                },
            );

            if block_idx < self.bridges.len() {
                self.bridges[block_idx].apply_forward(block, &self.blocks[block_idx + 1], ws);
            }
        }
    }

    /// Combine forward state and backward probabilities to compute posteriors (soft inputs).
    pub fn backward_and_emit_posteriors_probs(
        &self,
        target_probs: &TargetAlleleProbs,
        error_rate: f32,
        ws: &mut BlockHmmWorkspace,
        initial_recomb_rate: f32,
    ) -> Vec<AllelePosteriors> {
        let mut posteriors = vec![AllelePosteriors::Biallelic(0.0); target_probs.n_markers()];

        if let Some(last_block) = self.blocks.last() {
            let n_patterns = last_block.n_patterns();
            ws.bwd[..n_patterns].fill(1.0);
            if last_block.reservoir_count > 0 {
                ws.reservoir_prob_bwd = 1.0;
            } else {
                ws.reservoir_prob_bwd = 0.0;
            }
        }

        for block_idx in (0..self.blocks.len()).rev() {
            let block = &self.blocks[block_idx];

            let start = block.start_marker;
            let end = block.end_marker.min(target_probs.n_markers());

            ws.restore_checkpoint(block_idx, block.n_patterns());

            let view = TargetAlleleProbsView::new(target_probs, block.start_marker);
            let output_slice = &mut posteriors[start..end];
            super::hmm::backward_and_emit_block_probs(
                block,
                &view,
                error_rate,
                ws,
                output_slice,
                if block_idx == 0 {
                    initial_recomb_rate
                } else {
                    0.0
                },
            );

            if block_idx > 0 {
                self.bridges[block_idx - 1].apply_backward(&self.blocks[block_idx - 1], block, ws);
            }
        }

        posteriors
    }
}
