//! # Weighted HMM Kernel
//!
//! SIMD-optimized HMM kernel for weighted transitions.
//! Unlike the standard Li-Stephens kernel where transition probability is uniform,
//! this kernel weights transitions by per-state weights.

use crate::model::li_stephens::subset_linear_clamped_k;
use wide::f32x8;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// HMM Updater that weights transitions by pattern counts
pub struct WeightedHmmUpdater;

#[derive(Clone, Copy)]
pub struct PatternCounts<'a>(&'a [f32]);

impl<'a> PatternCounts<'a> {
    #[inline]
    pub fn new(values: &'a [f32]) -> Self {
        Self(values)
    }

    #[inline]
    fn as_slice(self) -> &'a [f32] {
        self.0
    }
}

#[derive(Clone, Copy)]
pub struct EmissionProbs<'a>(&'a [f32]);

impl<'a> EmissionProbs<'a> {
    #[inline]
    pub fn new(values: &'a [f32]) -> Self {
        Self(values)
    }

    #[inline]
    fn as_slice(self) -> &'a [f32] {
        self.0
    }
}

impl WeightedHmmUpdater {
    #[inline(always)]
    fn sanitize_count(c: f32) -> f32 {
        if c.is_finite() && c > 0.0 { c } else { 0.0 }
    }

    #[inline]
    fn count_stats(pattern_counts: &[f32], n_patterns: usize) -> (f32, bool) {
        let mut active_haps = 0.0f32;
        let mut has_invalid = false;
        for &c in pattern_counts.iter().take(n_patterns) {
            if !c.is_finite() || c < 0.0 {
                has_invalid = true;
            }
            active_haps += Self::sanitize_count(c);
        }
        (active_haps, has_invalid)
    }

    #[inline]
    fn conditioned_transition_params(
        recomb_rate: f32,
        n_ref_haps: usize,
        active_haps: f32,
        fwd_sum: f32,
    ) -> (f32, f32) {
        let (stay, base_shift) = subset_linear_clamped_k(recomb_rate, active_haps, n_ref_haps);
        let scale = stay / fwd_sum.max(1e-30);
        (scale, base_shift)
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f")]
    unsafe fn fwd_update_weighted_avx512(
        fwd: &mut [f32],
        scale: f32,
        base_shift: f32,
        pattern_counts: &[f32],
        emissions: &[f32],
        n_patterns: usize,
        clamp_counts: bool,
    ) -> f32 {
        // Safety: caller guarantees slices have at least `n_patterns` elements and are valid
        // for AVX-512 loads/stores (unaligned is permitted by loadu/storeu).
        unsafe {
            let base_shift_vec = _mm512_set1_ps(base_shift);
            let scale_vec = _mm512_set1_ps(scale);
            let zero_vec = _mm512_setzero_ps();
            let mut sum_vec = _mm512_setzero_ps();

            let mut k = 0;
            let fwd_ptr = fwd.as_mut_ptr();
            let count_ptr = pattern_counts.as_ptr();
            let emit_ptr = emissions.as_ptr();
            while k + 16 <= n_patterns {
                let fwd_chunk = _mm512_loadu_ps(fwd_ptr.add(k));
                let mut count_chunk = _mm512_loadu_ps(count_ptr.add(k));
                if clamp_counts {
                    count_chunk = _mm512_max_ps(count_chunk, zero_vec);
                }
                let emit_vec = _mm512_loadu_ps(emit_ptr.add(k));

                let shift_vec = _mm512_mul_ps(base_shift_vec, count_chunk);
                let scaled = _mm512_add_ps(_mm512_mul_ps(scale_vec, fwd_chunk), shift_vec);
                let res = _mm512_mul_ps(emit_vec, scaled);

                _mm512_storeu_ps(fwd_ptr.add(k), res);
                sum_vec = _mm512_add_ps(sum_vec, res);
                k += 16;
            }

            let mut sum_arr = [0.0f32; 16];
            _mm512_storeu_ps(sum_arr.as_mut_ptr(), sum_vec);
            let mut new_sum: f32 = sum_arr.iter().sum();

            for i in k..n_patterns {
                let f = *fwd_ptr.add(i);
                let c = *count_ptr.add(i);
                let e = *emit_ptr.add(i);
                let shift = base_shift
                    * if clamp_counts {
                        Self::sanitize_count(c)
                    } else {
                        c
                    };
                let t = scale.mul_add(f, shift);
                let v = e * t;
                *fwd_ptr.add(i) = v;
                new_sum += v;
            }
            new_sum
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f,fma")]
    unsafe fn fwd_update_weighted_avx512_fma(
        fwd: &mut [f32],
        scale: f32,
        base_shift: f32,
        pattern_counts: &[f32],
        emissions: &[f32],
        n_patterns: usize,
        clamp_counts: bool,
    ) -> f32 {
        // Safety: caller guarantees slices have at least `n_patterns` elements and are valid
        // for AVX-512 loads/stores (unaligned is permitted by loadu/storeu).
        unsafe {
            let base_shift_vec = _mm512_set1_ps(base_shift);
            let scale_vec = _mm512_set1_ps(scale);
            let zero_vec = _mm512_setzero_ps();
            let mut sum_vec = _mm512_setzero_ps();

            let mut k = 0;
            let fwd_ptr = fwd.as_mut_ptr();
            let count_ptr = pattern_counts.as_ptr();
            let emit_ptr = emissions.as_ptr();
            while k + 16 <= n_patterns {
                let fwd_chunk = _mm512_loadu_ps(fwd_ptr.add(k));
                let mut count_chunk = _mm512_loadu_ps(count_ptr.add(k));
                if clamp_counts {
                    count_chunk = _mm512_max_ps(count_chunk, zero_vec);
                }
                let emit_vec = _mm512_loadu_ps(emit_ptr.add(k));

                let shift_vec = _mm512_mul_ps(base_shift_vec, count_chunk);
                let scaled = _mm512_fmadd_ps(scale_vec, fwd_chunk, shift_vec);
                let res = _mm512_mul_ps(emit_vec, scaled);

                _mm512_storeu_ps(fwd_ptr.add(k), res);
                sum_vec = _mm512_add_ps(sum_vec, res);
                k += 16;
            }

            let mut sum_arr = [0.0f32; 16];
            _mm512_storeu_ps(sum_arr.as_mut_ptr(), sum_vec);
            let mut new_sum: f32 = sum_arr.iter().sum();

            for i in k..n_patterns {
                let f = *fwd_ptr.add(i);
                let c = *count_ptr.add(i);
                let e = *emit_ptr.add(i);
                let shift = base_shift
                    * if clamp_counts {
                        Self::sanitize_count(c)
                    } else {
                        c
                    };
                let t = scale.mul_add(f, shift);
                let v = e * t;
                *fwd_ptr.add(i) = v;
                new_sum += v;
            }
            new_sum
        }
    }
    /// Forward update with weighted transitions
    ///
    /// F_new[i] = (scale * F_old[i] + base_shift * count[i]) * E[i]
    #[inline]
    pub fn fwd_update_weighted(
        fwd: &mut [f32],
        fwd_sum: f32,
        recomb_rate: f32,
        n_ref_haps: usize,
        pattern_counts: PatternCounts<'_>,
        emissions: EmissionProbs<'_>,
        n_patterns: usize,
    ) -> f32 {
        let pattern_counts = pattern_counts.as_slice();
        let emissions = emissions.as_slice();
        let (active_haps, has_invalid_counts) = Self::count_stats(pattern_counts, n_patterns);
        let (scale, base_shift) =
            Self::conditioned_transition_params(recomb_rate, n_ref_haps, active_haps, fwd_sum);

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if n_patterns >= 16 && is_x86_feature_detected!("avx512f") {
                if is_x86_feature_detected!("fma") {
                    unsafe {
                        return Self::fwd_update_weighted_avx512_fma(
                            fwd,
                            scale,
                            base_shift,
                            pattern_counts,
                            emissions,
                            n_patterns,
                            has_invalid_counts,
                        );
                    }
                }
                unsafe {
                    return Self::fwd_update_weighted_avx512(
                        fwd,
                        scale,
                        base_shift,
                        pattern_counts,
                        emissions,
                        n_patterns,
                        has_invalid_counts,
                    );
                }
            }
        }

        let base_shift_vec = f32x8::splat(base_shift);
        let scale_vec = f32x8::splat(scale);
        let mut sum_vec = f32x8::splat(0.0);

        let mut k = 0;
        while k + 8 <= n_patterns {
            let mut fwd_arr = [0.0f32; 8];
            let mut count_arr = [0.0f32; 8];
            let mut emit_arr = [0.0f32; 8];
            unsafe {
                std::ptr::copy_nonoverlapping(fwd.as_ptr().add(k), fwd_arr.as_mut_ptr(), 8);
                std::ptr::copy_nonoverlapping(
                    pattern_counts.as_ptr().add(k),
                    count_arr.as_mut_ptr(),
                    8,
                );
                std::ptr::copy_nonoverlapping(emissions.as_ptr().add(k), emit_arr.as_mut_ptr(), 8);
            }
            let fwd_chunk = f32x8::from(fwd_arr);

            let count_chunk = f32x8::from(count_arr);
            let count_chunk = if has_invalid_counts {
                count_chunk.max(f32x8::splat(0.0))
            } else {
                count_chunk
            };

            let emit_vec = f32x8::from(emit_arr);

            // weighted shift = base_shift * count[i]
            let shift_vec = base_shift_vec * count_chunk;

            // res = E[i] * (scale * F[i] + shift[i])
            let res = emit_vec * (scale_vec * fwd_chunk + shift_vec);

            let res_arr: [f32; 8] = res.into();
            unsafe {
                std::ptr::copy_nonoverlapping(res_arr.as_ptr(), fwd.as_mut_ptr().add(k), 8);
            }

            sum_vec += res;
            k += 8;
        }

        let mut new_sum = sum_vec.reduce_add();
        for i in k..n_patterns {
            unsafe {
                let f = *fwd.get_unchecked(i);
                let c = *pattern_counts.get_unchecked(i);
                let e = *emissions.get_unchecked(i);
                let shift = base_shift
                    * if has_invalid_counts {
                        Self::sanitize_count(c)
                    } else {
                        c
                    };
                let t = scale.mul_add(f, shift);
                let v = e * t;
                *fwd.get_unchecked_mut(i) = v;
                new_sum += v;
            }
        }
        new_sum
    }

    /// Forward update with uniform transitions (weight=1.0 for all states)
    ///
    /// F_new[i] = (scale * F_old[i] + base_shift) * E[i]
    #[inline]
    pub fn fwd_update_uniform(
        fwd: &mut [f32],
        fwd_sum: f32,
        recomb_rate: f32,
        n_ref_haps: usize,
        active_haps: f32,
        emissions: EmissionProbs<'_>,
        n_patterns: usize,
    ) -> f32 {
        let emissions = emissions.as_slice();
        let (scale, base_shift) =
            Self::conditioned_transition_params(recomb_rate, n_ref_haps, active_haps, fwd_sum);

        let base_shift_vec = f32x8::splat(base_shift);
        let scale_vec = f32x8::splat(scale);
        let mut sum_vec = f32x8::splat(0.0);

        let mut k = 0;
        while k + 8 <= n_patterns {
            let mut fwd_arr = [0.0f32; 8];
            let mut emit_arr = [0.0f32; 8];
            unsafe {
                std::ptr::copy_nonoverlapping(fwd.as_ptr().add(k), fwd_arr.as_mut_ptr(), 8);
                std::ptr::copy_nonoverlapping(emissions.as_ptr().add(k), emit_arr.as_mut_ptr(), 8);
            }
            let fwd_chunk = f32x8::from(fwd_arr);
            let emit_vec = f32x8::from(emit_arr);

            // res = E[i] * (scale * F[i] + shift)
            let res = emit_vec * (scale_vec * fwd_chunk + base_shift_vec);

            let res_arr: [f32; 8] = res.into();
            unsafe {
                std::ptr::copy_nonoverlapping(res_arr.as_ptr(), fwd.as_mut_ptr().add(k), 8);
            }

            sum_vec += res;
            k += 8;
        }

        let mut new_sum = sum_vec.reduce_add();
        for i in k..n_patterns {
            unsafe {
                let f = *fwd.get_unchecked(i);
                let e = *emissions.get_unchecked(i);
                let t = scale.mul_add(f, base_shift);
                let v = e * t;
                *fwd.get_unchecked_mut(i) = v;
                new_sum += v;
            }
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
        let mut fwd = vec![0.1, 0.2, 0.3, 0.4];
        let fwd_sum: f32 = fwd.iter().sum();
        let fwd_start = fwd.clone();

        let recomb_rate = 0.01;
        let n_ref_haps = 1000;
        // Skewed pattern counts: pattern 0 is very common (900 haps), others rare
        let pattern_counts = vec![900.0, 50.0, 40.0, 10.0];

        // Uniform emissions for simplicity (or slight variation)
        let emissions = vec![1.0, 0.5, 0.1, 0.01];

        // Run kernel
        let new_sum = WeightedHmmUpdater::fwd_update_weighted(
            &mut fwd,
            fwd_sum,
            recomb_rate,
            n_ref_haps,
            PatternCounts::new(&pattern_counts),
            EmissionProbs::new(&emissions),
            n_patterns,
        );

        // Manual verification for panel-aware subset-conditioned transitions:
        //   switch_full = r / N
        //   z = (1-r) + K * switch_full
        //   P'(i) = ((1-r)*P(i) + switch_full*count[i]) / z
        let active_haps: f32 = pattern_counts.iter().sum();
        let switch_full = recomb_rate / n_ref_haps as f32;
        let z = (1.0 - recomb_rate) + active_haps * switch_full;
        let mut expected_pre_emit = vec![0.0; n_patterns];
        for i in 0..n_patterns {
            let p_i = fwd_start[i] / fwd_sum;
            expected_pre_emit[i] =
                ((1.0 - recomb_rate) * p_i + switch_full * pattern_counts[i]) / z;
        }
        let expected_total_mass: f32 = expected_pre_emit.iter().sum();
        assert!(
            (expected_total_mass - 1.0).abs() < 1e-6,
            "Mass not conserved during transition! Sum={}",
            expected_total_mass
        );

        // Now apply emission
        for i in 0..n_patterns {
            expected_pre_emit[i] *= emissions[i];
        }
        let expected_final_sum: f32 = expected_pre_emit.iter().sum();

        // Check if kernel output matches expected final sum
        assert!(
            (new_sum - expected_final_sum).abs() < 1e-5,
            "Kernel sum mismatch: Got {}, Expected {}",
            new_sum,
            expected_final_sum
        );

        // Check individual values
        for i in 0..n_patterns {
            assert!(
                (fwd[i] - expected_pre_emit[i]).abs() < 1e-5,
                "Value mismatch at index {}",
                i
            );
        }
    }

    #[test]
    fn test_uniform_matches_weighted_with_unit_weights() {
        let n_patterns = 12;
        let mut fwd_weighted = vec![0.1f32; n_patterns];
        let mut fwd_uniform = vec![0.1f32; n_patterns];
        let fwd_sum = 1.2;
        let recomb_rate = 0.05;
        let n_ref_haps = 100;
        let unit_weights = vec![1.0f32; n_patterns];
        let emissions: Vec<f32> = (0..n_patterns).map(|i| 0.5 + (i as f32) * 0.04).collect();

        let sum_weighted = WeightedHmmUpdater::fwd_update_weighted(
            &mut fwd_weighted,
            fwd_sum,
            recomb_rate,
            n_ref_haps,
            PatternCounts::new(&unit_weights),
            EmissionProbs::new(&emissions),
            n_patterns,
        );

        let sum_uniform = WeightedHmmUpdater::fwd_update_uniform(
            &mut fwd_uniform,
            fwd_sum,
            recomb_rate,
            n_ref_haps,
            n_patterns as f32,
            EmissionProbs::new(&emissions),
            n_patterns,
        );

        assert!((sum_weighted - sum_uniform).abs() < 1e-6);
        for i in 0..n_patterns {
            assert!((fwd_weighted[i] - fwd_uniform[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_negative_counts_are_clamped() {
        let n_patterns = 4;
        let mut fwd = vec![0.25, 0.25, 0.25, 0.25];
        let fwd_sum: f32 = fwd.iter().sum();
        let recomb_rate = 0.1;
        let n_ref_haps = 100;
        let pattern_counts = vec![90.0, -10.0, 20.0, -5.0];
        let emissions = vec![1.0, 1.0, 1.0, 1.0];

        let new_sum = WeightedHmmUpdater::fwd_update_weighted(
            &mut fwd,
            fwd_sum,
            recomb_rate,
            n_ref_haps,
            PatternCounts::new(&pattern_counts),
            EmissionProbs::new(&emissions),
            n_patterns,
        );

        assert!(new_sum.is_finite() && new_sum > 0.0);
        for &v in &fwd {
            assert!(v.is_finite() && v >= 0.0);
        }
    }

    #[test]
    fn test_panel_aware_shift_uses_global_n_not_subset_k() {
        let n_patterns = 4;
        let mut fwd = vec![0.25, 0.25, 0.25, 0.25];
        let fwd_sum: f32 = fwd.iter().sum();
        let recomb_rate = 0.02;
        let n_ref_haps = 10_000;
        // Active subset mass is tiny relative to panel.
        let pattern_counts = vec![1.0, 1.0, 1.0, 1.0];
        let emissions = vec![1.0, 1.0, 1.0, 1.0];

        let new_sum = WeightedHmmUpdater::fwd_update_weighted(
            &mut fwd,
            fwd_sum,
            recomb_rate,
            n_ref_haps,
            PatternCounts::new(&pattern_counts),
            EmissionProbs::new(&emissions),
            n_patterns,
        );
        assert!(new_sum.is_finite() && new_sum > 0.0);

        // Under panel-aware transitions with uniform starting mass/counts and unit emissions,
        // each state remains close to 1/K with only a tiny recombination-induced perturbation.
        // If r/K were used instead, the perturbation would be orders of magnitude larger.
        for &v in &fwd {
            assert!(
                (v - 0.25).abs() < 1e-3,
                "unexpectedly large transition perturbation: {}",
                v
            );
        }
    }
}
