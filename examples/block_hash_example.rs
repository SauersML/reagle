//! # Block-Hash Clade HMM Example
//!
//! Demonstrates usage of the block-hash HMM module for imputation.
//!
//! This example shows how to:
//! 1. Build compressed MicroWindows from a reference panel
//! 2. Run forward/backward HMM passes
//! 3. Extract posterior probabilities
//!
//! Run with: cargo run --example block_hash_example

use reagle::model::block_hash::{DEFAULT_WINDOW_SIZE, DEFAULT_MAX_STATES};

fn main() {
    println!("Block-Hash Clade HMM Example");
    println!("============================\n");

    println!("This example demonstrates the block-hash HMM API:");
    println!("1. build_all_windows() - Compress reference panel into MicroWindows");
    println!("2. forward_pass_all_windows() - Run HMM forward pass with SIMD");
    println!("3. backward_pass_all_windows() - Run HMM backward pass");
    println!();

    println!("Default configuration:");
    println!("  Window size: {} markers", DEFAULT_WINDOW_SIZE);
    println!("  Max states: {} unique patterns", DEFAULT_MAX_STATES);
    println!();

    println!("Key features:");
    println!("  - Type-safe GlobalId and PatternId (prevents index confusion)");
    println!("  - CSR sparse matrix transitions (deterministic, cache-friendly)");
    println!("  - AVX-512 SIMD kernel reuse (HmmUpdater integration)");
    println!("  - Multiallelic variant support (via DictionaryColumn)");
    println!("  - Reservoir state for large panels (>4096 unique patterns)");
    println!();

    println!("Integration example:");
    println!("```rust");
    println!("// Load reference panel (GenotypeMatrix<Phased>)");
    println!("let ref_data = load_reference_panel(...);");
    println!();
    println!("// Build compressed windows");
    println!("let mut windows = build_all_windows(");
    println!("    &ref_data,");
    println!("    DEFAULT_WINDOW_SIZE,  // 32 markers per window");
    println!("    DEFAULT_MAX_STATES,   // 4096 max unique patterns");
    println!(");");
    println!();
    println!("// Run forward pass");
    println!("forward_pass_all_windows(");
    println!("    &mut windows,");
    println!("    &target_genotypes,  // Target sample genotypes");
    println!("    0.001,              // Error rate");
    println!("    0.0001,             // Recombination rate per marker");
    println!(");");
    println!();
    println!("// Run backward pass");
    println!("backward_pass_all_windows(");
    println!("    &mut windows,");
    println!("    &target_genotypes,");
    println!("    0.001,");
    println!("    0.0001,");
    println!(");");
    println!();
    println!("// Extract posterior probabilities");
    println!("for window in &windows {{");
    println!("    for (pattern_id, &prob) in window.fwd_probs.iter().enumerate() {{");
    println!("        println!(\"Pattern {{}} probability: {{}}\", pattern_id, prob);");
    println!("    }}");
    println!("}}");
    println!("```");
    println!();

    println!("For full integration, replace build_pbwt_hap_indices_for_batch()");
    println!("in src/pipelines/imputation_streaming.rs with this implementation.");
}
