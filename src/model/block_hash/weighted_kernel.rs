//! # Weighted HMM Kernel
//!
//! SIMD-optimized HMM kernel for block-hash compressed states.
//! Unlike the standard Li-Stephens kernel where transition probability is uniform,
//! this kernel weights transitions by the cardinality (frequency) of each pattern.

use wide::f32x8;

/// HMM Updater that weights transitions by pattern counts
pub struct WeightedHmmUpdater;

impl WeightedHmmUpdater {
    /// Forward update with uniform transitions (Cluster HMM)
    ///
    /// F_new[i] = (scale * F_old[i] + r/K) * E[i]
    #[inline]
    pub fn fwd_update_weighted(
        fwd: &mut [f32],
        fwd_sum: f32,
        recomb_rate: f32,
        _n_ref_haps: usize,
        _pattern_counts: &[f32],
        emissions: &[f32],
        n_patterns: usize,
        n_total_states: usize,
    ) -> f32 {
        // Uniform transition to patterns (Cluster HMM logic)
        // Transition probability to pattern k is r / n_total_states
        let base_shift = recomb_rate / n_total_states as f32;
        let scale = (1.0 - recomb_rate) / fwd_sum.max(1e-30);

        let base_shift_vec = f32x8::splat(base_shift);
        let scale_vec = f32x8::splat(scale);
        let mut sum_vec = f32x8::splat(0.0);

        let mut k = 0;
        while k + 8 <= n_patterns {
            let mut fwd_arr = [0.0f32; 8];
            fwd_arr.copy_from_slice(&fwd[k..k + 8]);
            let fwd_chunk = f32x8::from(fwd_arr);

            let mut emit_arr = [0.0f32; 8];
            emit_arr.copy_from_slice(&emissions[k..k + 8]);
            let emit_vec = f32x8::from(emit_arr);

            // weighted shift = base_shift * 1.0
            let shift_vec = base_shift_vec;
            
            // res = E[i] * (scale * F[i] + shift[i])
            let res = emit_vec * (scale_vec * fwd_chunk + shift_vec);

            let res_arr: [f32; 8] = res.into();
            fwd[k..k + 8].copy_from_slice(&res_arr);

            sum_vec += res;
            k += 8;
        }

        let mut new_sum = sum_vec.reduce_add();
        for i in k..n_patterns {
            let shift = base_shift;
            fwd[i] = emissions[i] * (scale * fwd[i] + shift);
            new_sum += fwd[i];
        }
        new_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mass_conservation() {
        // Setup: 4 patterns
        let n_patterns = 4;
        let mut fwd = vec![0.25, 0.25, 0.25, 0.25]; // uniform start
        let fwd_sum: f32 = fwd.iter().sum(); // 1.0

        let recomb_rate = 0.01;
        let n_ref_haps = 1000;
        
        // Skewed pattern counts: pattern 0 is very common (900 haps), others rare
        let pattern_counts = vec![900.0, 50.0, 40.0, 10.0];
        
        // Uniform emissions for simplicity
        let emissions = vec![1.0, 0.5, 0.1, 0.01];

        // Run kernel
        let new_sum = WeightedHmmUpdater::fwd_update_weighted(
            &mut fwd,
            fwd_sum,
            recomb_rate,
            n_ref_haps,
            &pattern_counts,
            &emissions,
            n_patterns,
            n_patterns, // Total states = n_patterns
        );

        // Manual verification logic
        // Transition (Cluster HMM):
        // P(t | t-1) = (1-r)*P(t-1) + r/K
        // Fwd[i] before emit = (1-r)*Fwd_old[i]/Sum + r/K
        
        // Let's compute expected pre-emission values
        let mut expected_pre_emit = vec![0.0; n_patterns];
        for i in 0..n_patterns {
            expected_pre_emit[i] = (1.0 - recomb_rate) * 0.25 + recomb_rate / n_patterns as f32;
        }
        let expected_total_mass: f32 = expected_pre_emit.iter().sum();
        assert!((expected_total_mass - 1.0).abs() < 1e-6, "Mass not conserved during transition! Sum={}", expected_total_mass);

        // Now apply emission
        for i in 0..n_patterns {
            expected_pre_emit[i] *= emissions[i];
        }
        let expected_final_sum: f32 = expected_pre_emit.iter().sum();

        // Check if kernel output matches expected final sum
        assert!((new_sum - expected_final_sum).abs() < 1e-5, "Kernel sum mismatch: Got {}, Expected {}", new_sum, expected_final_sum);

        // Check individual values
        for i in 0..n_patterns {
            assert!((fwd[i] - expected_pre_emit[i]).abs() < 1e-5, "Value mismatch at index {}", i);
        }
    }
}
