sed -i 's/if weight <= 0.0 {/if weight < 0.0 {/g' src/pipelines/imputation_streaming.rs
sed -i 's/if freq <= 0.0 {/if freq < 0.0 {/g' src/pipelines/imputation_streaming.rs
sed -i 's/let abyss_top = select_top_k(&window_scores\[i\], abyss_rank_cutoff);/let abyss_top = select_top_k_allow_zero(\&window_scores\[i\], abyss_rank_cutoff);/g' src/pipelines/imputation_streaming.rs
sed -i 's/if window_rank_hits\[i\]\[h\] == 0 || !score.is_finite() || score <= 0.0 {/if window_rank_hits\[i\]\[h\] == 0 || !score.is_finite() || score < 0.0 {/g' src/pipelines/imputation_streaming.rs
sed -i 's/if !score.is_finite() || score <= 0.0 {/if !score.is_finite() || score < 0.0 {/g' src/pipelines/imputation_streaming.rs
sed -i 's/if score <= 0.0 || !score.is_finite() {/if score < 0.0 || !score.is_finite() {/g' src/pipelines/imputation_streaming.rs
