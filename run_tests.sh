#!/bin/bash

tests=(
"test_missing_confidence_is_not_full_by_default"
"test_pl_priors_respect_ref_alt_swap_mapping"
"test_gt_should_match_gp_argmax_for_missing_marker"
"test_hardcall_should_match_posterior_mode_for_missing_marker"
"test_posteriors_should_reflect_balanced_ref_for_missing_marker"
"test_stage1_gating_should_anchor_rare_marker_phase"
"test_streaming_overlap_should_not_shift_genotyped_markers"
"test_priors_use_recent_context_across_window_boundary"
"test_gt_matches_hap_ap_argmax_not_gp_argmax"
"test_gp_equals_ap_convolution"
"test_phase_states_capacity_not_destabilizing_phasing"
"test_two_x_neighbors_not_causing_random_phase_flips"
"test_stage2_overlap_priors_use_start_not_end"
"test_low_confidence_vs_missing_emissions_equivalence"
"test_phasing_should_vary_under_ambiguous_signal_across_seeds"
"test_state_selection_preserves_rare_haplotype_linkage"
"test_stage2_rare_marker_phase_should_not_be_seed_locked"
"test_boundary_handoff_should_preserve_unique_haplotype_signal"
"test_boundary_handoff_should_match_single_window_confidence"
"test_stage2_rare_marker_phase_should_vary_across_multiple_seeds"
"test_low_confidence_penalty_accumulates_in_region"
"test_hardcall_emissions_block_ref_override_when_no_pl"
"test_phase_state_capacity_should_not_change_output_on_simple_ld"
"test_uniform_recomb_shift_should_not_overweight_rare_pattern"
"test_pl_het_signal_not_erased_by_zero_maf_prior"
"test_pbwt_backward_span_contributes"
"test_ibs2_missing_not_universally_matching"
)

for test in "${tests[@]}"; do
    echo "Running $test..."
    cargo test --test hypothesis_tests $test > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        echo "FAILED: $test"
        # Run again with output to see error
        cargo test --test hypothesis_tests $test
    else
        echo "PASSED: $test"
    fi
done
