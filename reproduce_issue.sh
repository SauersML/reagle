#!/bin/bash
export RUST_BACKTRACE=1
cargo test --test reference_comparison test_imputation_vs_ground_truth -- --nocapture
