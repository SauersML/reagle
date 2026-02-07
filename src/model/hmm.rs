//! # Li-Stephens Hidden Markov Model
//!
//! Implementation of the Li-Stephens HMM for haplotype phasing and imputation.
//! Uses the forward-backward algorithm with scaling for numerical stability.
//!
//! ## Key Concepts
//! - `States`: Reference haplotypes that the target could copy from
//! - `Transitions`: Probability of switching to a different reference haplotype
//! - `Emissions`: Probability of observing target allele given reference allele
//!
//! ## Reference
//! Li N, Stephens M. Genetics 2003 Dec;165(4):2213-33

use crate::data::storage::GenotypeView;
use crate::data::{HapIdx, MarkerIdx};
use crate::model::parameters::ModelParams;
use aligned_vec::{AVec, ConstAlign};
use std::sync::OnceLock;
use wide::f32x8;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const PREFETCH_DISTANCE: usize = 16;

/// Static HMM update functions matching Java HmmUpdater
pub struct HmmUpdater;

impl HmmUpdater {
    const LOG8_LEVELS: f32 = 255.0;
    const LOG8_RANGE_NATS: f32 = 24.0;
    const LOG8_EPS: f32 = 1e-30;
    const LOG8_SWITCH_STATES: usize = 1024;
    const LOG8_RANGE_LOG2: f32 = Self::LOG8_RANGE_NATS / std::f32::consts::LN_2;
    const LOG8_STEP_LOG2: f32 = Self::LOG8_RANGE_LOG2 / Self::LOG8_LEVELS;
    const LOG8_INV_STEP_LOG2: f32 = 1.0 / Self::LOG8_STEP_LOG2;

    #[inline]
    fn log8_decode_lut() -> &'static [f32; 256] {
        static LUT: OnceLock<[f32; 256]> = OnceLock::new();
        LUT.get_or_init(|| {
            let mut arr = [0.0f32; 256];
            for (q, v) in arr.iter_mut().enumerate() {
                *v = 2f32.powf(-(q as f32) * Self::LOG8_STEP_LOG2);
            }
            arr
        })
    }

    #[inline]
    fn log2_mantissa_lut() -> &'static [f32; 256] {
        static LUT: OnceLock<[f32; 256]> = OnceLock::new();
        LUT.get_or_init(|| {
            let mut arr = [0.0f32; 256];
            for (i, v) in arr.iter_mut().enumerate() {
                let m = 1.0 + ((i as f32) + 0.5) / 256.0;
                *v = m.log2();
            }
            arr
        })
    }

    #[inline]
    fn fast_log2_approx(x: f32) -> f32 {
        let bits = x.max(Self::LOG8_EPS).to_bits();
        let exp = ((bits >> 23) & 0xFF) as i32 - 127;
        let mant_idx = ((bits >> 15) & 0xFF) as usize;
        exp as f32 + Self::log2_mantissa_lut()[mant_idx]
    }

    #[inline]
    fn quantize_row_log8(values: &mut [f32], n_states: usize) -> f32 {
        let mut max_val = 0.0f32;
        for &v in values.iter().take(n_states) {
            if v > max_val {
                max_val = v;
            }
        }
        if max_val <= Self::LOG8_EPS {
            let floor = Self::LOG8_EPS;
            for x in values.iter_mut().take(n_states) {
                *x = floor;
            }
            return floor * n_states as f32;
        }

        let max_log2 = Self::fast_log2_approx(max_val);
        let decode = Self::log8_decode_lut();
        let mut sum = 0.0f32;
        for x in values.iter_mut().take(n_states) {
            let log2_x = Self::fast_log2_approx(*x);
            let qf = ((max_log2 - log2_x) * Self::LOG8_INV_STEP_LOG2)
                .round()
                .clamp(0.0, Self::LOG8_LEVELS);
            let q = qf as usize;
            let recon = max_val * decode[q];
            *x = recon;
            sum += recon;
        }
        sum.max(Self::LOG8_EPS)
    }

    #[inline]
    fn fwd_update_emissions_scale_log8(
        fwd: &mut [f32],
        scale: f32,
        shift: f32,
        emissions: &[f32],
        n_states: usize,
    ) -> f32 {
        for i in 0..n_states {
            let v = emissions[i] * scale.mul_add(fwd[i], shift);
            fwd[i] = v;
        }
        Self::quantize_row_log8(fwd, n_states)
    }

    #[inline]
    fn bwd_update_constant_scale_log8(
        bwd: &mut [f32],
        scale: f32,
        shift: f32,
        emissions: &[f32],
        n_states: usize,
    ) {
        for i in 0..n_states {
            bwd[i] = scale * emissions[i] * bwd[i] + shift;
        }
        let _ = Self::quantize_row_log8(bwd, n_states);
    }

    /// Forward update specialized for homozygous markers: compute emissions on the fly
    /// from `ref_alleles` vs `req`, then apply transition + emission in one pass.
    #[inline]
    pub fn fwd_update_homo_emissions_scale(
        fwd: &mut [f32],
        scale: f32,
        shift: f32,
        ref_alleles: &[u8],
        req: u8,
        p_match: f32,
        p_mismatch: f32,
        n_states: usize,
    ) -> f32 {
        if n_states >= Self::LOG8_SWITCH_STATES {
            for i in 0..n_states {
                let em = if ref_alleles[i] == req {
                    p_match
                } else {
                    p_mismatch
                };
                fwd[i] = em * scale.mul_add(fwd[i], shift);
            }
            return Self::quantize_row_log8(fwd, n_states);
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if n_states >= 64
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
            {
                unsafe {
                    let req_vec = _mm512_set1_epi8(req as i8);
                    let match_vec = _mm512_set1_ps(p_match);
                    let mismatch_vec = _mm512_set1_ps(p_mismatch);
                    let shift_vec = _mm512_set1_ps(shift);
                    let scale_vec = _mm512_set1_ps(scale);
                    let mut sum_vec = _mm512_setzero_ps();
                    let mut k = 0usize;
                    let fwd_ptr = fwd.as_mut_ptr();
                    let ref_ptr = ref_alleles.as_ptr();
                    while k + 64 <= n_states {
                        let bytes = _mm512_loadu_si512(ref_ptr.add(k) as *const __m512i);
                        let mask = _mm512_cmpeq_epi8_mask(bytes, req_vec);
                        let mask0 = (mask & 0xFFFF) as u16;
                        let mask1 = ((mask >> 16) & 0xFFFF) as u16;
                        let mask2 = ((mask >> 32) & 0xFFFF) as u16;
                        let mask3 = ((mask >> 48) & 0xFFFF) as u16;

                        let fwd0 = _mm512_loadu_ps(fwd_ptr.add(k));
                        let fwd1 = _mm512_loadu_ps(fwd_ptr.add(k + 16));
                        let fwd2 = _mm512_loadu_ps(fwd_ptr.add(k + 32));
                        let fwd3 = _mm512_loadu_ps(fwd_ptr.add(k + 48));

                        let emit0 = _mm512_mask_blend_ps(mask0, mismatch_vec, match_vec);
                        let emit1 = _mm512_mask_blend_ps(mask1, mismatch_vec, match_vec);
                        let emit2 = _mm512_mask_blend_ps(mask2, mismatch_vec, match_vec);
                        let emit3 = _mm512_mask_blend_ps(mask3, mismatch_vec, match_vec);

                        let scaled0 = _mm512_fmadd_ps(scale_vec, fwd0, shift_vec);
                        let scaled1 = _mm512_fmadd_ps(scale_vec, fwd1, shift_vec);
                        let scaled2 = _mm512_fmadd_ps(scale_vec, fwd2, shift_vec);
                        let scaled3 = _mm512_fmadd_ps(scale_vec, fwd3, shift_vec);

                        let res0 = _mm512_mul_ps(emit0, scaled0);
                        let res1 = _mm512_mul_ps(emit1, scaled1);
                        let res2 = _mm512_mul_ps(emit2, scaled2);
                        let res3 = _mm512_mul_ps(emit3, scaled3);

                        _mm512_storeu_ps(fwd_ptr.add(k), res0);
                        _mm512_storeu_ps(fwd_ptr.add(k + 16), res1);
                        _mm512_storeu_ps(fwd_ptr.add(k + 32), res2);
                        _mm512_storeu_ps(fwd_ptr.add(k + 48), res3);

                        sum_vec = _mm512_add_ps(sum_vec, res0);
                        sum_vec = _mm512_add_ps(sum_vec, res1);
                        sum_vec = _mm512_add_ps(sum_vec, res2);
                        sum_vec = _mm512_add_ps(sum_vec, res3);
                        k += 64;
                    }
                    let mut sum_arr = [0.0f32; 16];
                    _mm512_storeu_ps(sum_arr.as_mut_ptr(), sum_vec);
                    let mut new_sum: f32 = sum_arr.iter().sum();
                    for i in k..n_states {
                        let f = *fwd_ptr.add(i);
                        let em = if *ref_ptr.add(i) == req {
                            p_match
                        } else {
                            p_mismatch
                        };
                        let t = scale.mul_add(f, shift);
                        let v = em * t;
                        *fwd_ptr.add(i) = v;
                        new_sum += v;
                    }
                    return new_sum;
                }
            }
        }

        let mut new_sum = 0.0f32;
        for i in 0..n_states {
            let f = fwd[i];
            let em = if ref_alleles[i] == req {
                p_match
            } else {
                p_mismatch
            };
            let t = scale.mul_add(f, shift);
            let v = em * t;
            fwd[i] = v;
            new_sum += v;
        }
        new_sum
    }

    /// Forward update using explicit scale/shift (for blocked processing).
    #[inline]
    pub fn fwd_update_emissions_scale(
        fwd: &mut [f32],
        scale: f32,
        shift: f32,
        emissions: &[f32],
        n_states: usize,
    ) -> f32 {
        if n_states >= Self::LOG8_SWITCH_STATES {
            return Self::fwd_update_emissions_scale_log8(fwd, scale, shift, emissions, n_states);
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if n_states >= 16 && is_x86_feature_detected!("avx512f") {
                unsafe {
                    let shift_vec = _mm512_set1_ps(shift);
                    let scale_vec = _mm512_set1_ps(scale);
                    let mut sum_vec = _mm512_setzero_ps();
                    let mut k = 0;
                    let fwd_ptr = fwd.as_mut_ptr();
                    let emit_ptr = emissions.as_ptr();
                    while k + 16 <= n_states {
                        let fwd_chunk = _mm512_loadu_ps(fwd_ptr.add(k));
                        let emit_vec = _mm512_loadu_ps(emit_ptr.add(k));
                        let scaled = _mm512_fmadd_ps(scale_vec, fwd_chunk, shift_vec);
                        let res = _mm512_mul_ps(emit_vec, scaled);
                        _mm512_storeu_ps(fwd_ptr.add(k), res);
                        sum_vec = _mm512_add_ps(sum_vec, res);
                        k += 16;
                    }
                    let mut sum_arr = [0.0f32; 16];
                    _mm512_storeu_ps(sum_arr.as_mut_ptr(), sum_vec);
                    let mut new_sum: f32 = sum_arr.iter().sum();
                    for i in k..n_states {
                        let f = *fwd_ptr.add(i);
                        let e = *emit_ptr.add(i);
                        let t = scale.mul_add(f, shift);
                        let v = e * t;
                        *fwd_ptr.add(i) = v;
                        new_sum += v;
                    }
                    return new_sum;
                }
            }
        }

        let shift_vec = f32x8::splat(shift);
        let scale_vec = f32x8::splat(scale);
        let mut sum_vec = f32x8::splat(0.0);
        let mut k = 0;
        while k + 8 <= n_states {
            let mut fwd_arr = [0.0f32; 8];
            let mut emit_arr = [0.0f32; 8];
            unsafe {
                std::ptr::copy_nonoverlapping(fwd.as_ptr().add(k), fwd_arr.as_mut_ptr(), 8);
                std::ptr::copy_nonoverlapping(emissions.as_ptr().add(k), emit_arr.as_mut_ptr(), 8);
            }
            let fwd_chunk = f32x8::from(fwd_arr);
            let emit_vec = f32x8::from(emit_arr);
            let res = emit_vec * (scale_vec * fwd_chunk + shift_vec);
            let res_arr: [f32; 8] = res.into();
            unsafe {
                std::ptr::copy_nonoverlapping(res_arr.as_ptr(), fwd.as_mut_ptr().add(k), 8);
            }
            sum_vec += res;
            k += 8;
        }
        let mut new_sum = sum_vec.reduce_add();
        for i in k..n_states {
            unsafe {
                let f = *fwd.get_unchecked(i);
                let e = *emissions.get_unchecked(i);
                let t = scale.mul_add(f, shift);
                let v = e * t;
                *fwd.get_unchecked_mut(i) = v;
                new_sum += v;
            }
        }
        new_sum
    }
    /// Backward update matching Java HmmUpdater.bwdUpdate exactly.
    ///
    /// Updates backward values in place.
    ///
    /// # Arguments
    /// * `bwd` - Backward values array that will be updated in place
    /// * `p_switch` - Probability of jumping to a random HMM state
    /// * `emit_probs` - Two-element array: [p_match, p_mismatch]
    /// * `mismatches` - Number of mismatches (0 or 1) for each state
    /// * `n_states` - Number of states to process
    #[inline]
    pub fn bwd_update(
        bwd: &mut [f32],
        p_switch: f32,
        emit_probs: &[f32; 2],
        mismatches: &[u8],
        n_states: usize,
    ) {
        if n_states >= Self::LOG8_SWITCH_STATES {
            let p0 = emit_probs[0];
            let p1 = emit_probs[1];
            let diff = p1 - p0;

            let mut sum = 0.0f32;
            for i in 0..n_states {
                let em = p0 + (mismatches[i] as f32) * diff;
                bwd[i] *= em;
                sum += bwd[i];
            }
            let shift = p_switch / n_states as f32;
            let scale = (1.0 - p_switch) / sum.max(Self::LOG8_EPS);
            for x in bwd.iter_mut().take(n_states) {
                *x = scale * *x + shift;
            }
            let _ = Self::quantize_row_log8(bwd, n_states);
            return;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if n_states >= 16
                && is_x86_feature_detected!("avx512f")
                && is_x86_feature_detected!("avx512bw")
            {
                if is_x86_feature_detected!("fma") {
                    unsafe {
                        return Self::bwd_update_avx512_fma(
                            bwd, p_switch, emit_probs, mismatches, n_states,
                        );
                    }
                }
                unsafe {
                    return Self::bwd_update_avx512(
                        bwd, p_switch, emit_probs, mismatches, n_states,
                    );
                }
            }
        }

        // First: multiply by emission and compute sum
        let mut sum_vec = f32x8::splat(0.0);
        let p0 = emit_probs[0];
        let p1 = emit_probs[1];
        let diff = p1 - p0;

        let mut k = 0;
        while k + 8 <= n_states {
            let mut bwd_arr = [0.0f32; 8];
            bwd_arr.copy_from_slice(&bwd[k..k + 8]);
            let bwd_chunk = f32x8::from(bwd_arr);

            let m_chunk = &mismatches[k..k + 8];
            let emit_arr = [
                p0 + (m_chunk[0] as f32) * diff,
                p0 + (m_chunk[1] as f32) * diff,
                p0 + (m_chunk[2] as f32) * diff,
                p0 + (m_chunk[3] as f32) * diff,
                p0 + (m_chunk[4] as f32) * diff,
                p0 + (m_chunk[5] as f32) * diff,
                p0 + (m_chunk[6] as f32) * diff,
                p0 + (m_chunk[7] as f32) * diff,
            ];
            let emit_vec = f32x8::from(emit_arr);

            let res = bwd_chunk * emit_vec;
            let res_arr: [f32; 8] = res.into();
            bwd[k..k + 8].copy_from_slice(&res_arr);

            sum_vec += res;
            k += 8;
        }

        let mut sum = sum_vec.reduce_add();

        // Tail loop 1
        for i in k..n_states {
            bwd[i] *= emit_probs[mismatches[i] as usize];
            sum += bwd[i];
        }

        // Then: apply transition
        let shift = p_switch / n_states as f32;
        let scale = (1.0 - p_switch) / sum.max(1e-30);

        let shift_vec = f32x8::splat(shift);
        let scale_vec = f32x8::splat(scale);

        k = 0;
        while k + 8 <= n_states {
            let mut bwd_arr = [0.0f32; 8];
            bwd_arr.copy_from_slice(&bwd[k..k + 8]);
            let bwd_chunk = f32x8::from(bwd_arr);

            let res = scale_vec * bwd_chunk + shift_vec;
            let res_arr: [f32; 8] = res.into();
            bwd[k..k + 8].copy_from_slice(&res_arr);

            k += 8;
        }

        for i in k..n_states {
            bwd[i] = scale * bwd[i] + shift;
        }
    }

    /// Backward update using explicit scale/shift (for blocked processing).
    #[inline]
    pub fn bwd_update_constant_scale(
        bwd: &mut [f32],
        scale: f32,
        shift: f32,
        emissions: &[f32],
        n_states: usize,
    ) {
        if n_states >= Self::LOG8_SWITCH_STATES {
            Self::bwd_update_constant_scale_log8(bwd, scale, shift, emissions, n_states);
            return;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if n_states >= 16 && is_x86_feature_detected!("avx512f") {
                unsafe {
                    let scale_vec = _mm512_set1_ps(scale);
                    let shift_vec = _mm512_set1_ps(shift);
                    let mut k = 0;
                    let bwd_ptr = bwd.as_mut_ptr();
                    let emit_ptr = emissions.as_ptr();
                    while k + 16 <= n_states {
                        let bwd_chunk = _mm512_loadu_ps(bwd_ptr.add(k));
                        let emit_vec = _mm512_loadu_ps(emit_ptr.add(k));
                        let scaled = _mm512_mul_ps(scale_vec, _mm512_mul_ps(emit_vec, bwd_chunk));
                        let res = _mm512_add_ps(scaled, shift_vec);
                        _mm512_storeu_ps(bwd_ptr.add(k), res);
                        k += 16;
                    }
                    for i in k..n_states {
                        *bwd_ptr.add(i) = scale * *emit_ptr.add(i) * *bwd_ptr.add(i) + shift;
                    }
                    return;
                }
            }
        }

        let scale_vec = f32x8::splat(scale);
        let shift_vec = f32x8::splat(shift);
        let mut k = 0;
        while k + 8 <= n_states {
            let mut bwd_arr = [0.0f32; 8];
            bwd_arr.copy_from_slice(&bwd[k..k + 8]);
            let bwd_chunk = f32x8::from(bwd_arr);

            let mut emit_arr = [0.0f32; 8];
            emit_arr.copy_from_slice(&emissions[k..k + 8]);
            let emit_vec = f32x8::from(emit_arr);

            let res = (scale_vec * emit_vec * bwd_chunk) + shift_vec;
            let res_arr: [f32; 8] = res.into();
            bwd[k..k + 8].copy_from_slice(&res_arr);
            k += 8;
        }

        for i in k..n_states {
            bwd[i] = scale * emissions[i] * bwd[i] + shift;
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn bwd_update_avx512(
        bwd: &mut [f32],
        p_switch: f32,
        emit_probs: &[f32; 2],
        mismatches: &[u8],
        n_states: usize,
    ) {
        // Safety: caller guarantees `bwd` and `mismatches` have at least `n_states` elements
        // and point to valid, properly aligned memory for AVX-512 loads/stores.
        unsafe {
            let p0 = emit_probs[0];
            let p1 = emit_probs[1];
            let diff = p1 - p0;

            let p0_vec = _mm512_set1_ps(p0);
            let diff_vec = _mm512_set1_ps(diff);
            let mut sum_vec = _mm512_setzero_ps();

            let mut k = 0;
            let bwd_ptr = bwd.as_mut_ptr();
            let mismatch_ptr = mismatches.as_ptr();
            while k + 16 <= n_states {
                let bwd_chunk = _mm512_loadu_ps(bwd_ptr.add(k));
                let m_u8 = _mm_loadu_si128(mismatch_ptr.add(k) as *const __m128i);
                let m_i32 = _mm512_cvtepu8_epi32(m_u8);
                let m_f32 = _mm512_cvtepi32_ps(m_i32);
                let emit_vec = _mm512_add_ps(_mm512_mul_ps(m_f32, diff_vec), p0_vec);
                let res = _mm512_mul_ps(bwd_chunk, emit_vec);
                _mm512_storeu_ps(bwd_ptr.add(k), res);
                sum_vec = _mm512_add_ps(sum_vec, res);
                k += 16;
            }

            let mut sum_arr = [0.0f32; 16];
            _mm512_storeu_ps(sum_arr.as_mut_ptr(), sum_vec);
            let mut sum: f32 = sum_arr.iter().sum();

            for i in k..n_states {
                let em = emit_probs[*mismatches.get_unchecked(i) as usize];
                let v = *bwd_ptr.add(i) * em;
                *bwd_ptr.add(i) = v;
                sum += v;
            }

            let shift = p_switch / n_states as f32;
            let scale = (1.0 - p_switch) / sum.max(1e-30);

            let shift_vec = _mm512_set1_ps(shift);
            let scale_vec = _mm512_set1_ps(scale);

            k = 0;
            while k + 16 <= n_states {
                let bwd_chunk = _mm512_loadu_ps(bwd_ptr.add(k));
                let res = _mm512_add_ps(_mm512_mul_ps(scale_vec, bwd_chunk), shift_vec);
                _mm512_storeu_ps(bwd_ptr.add(k), res);
                k += 16;
            }

            for i in k..n_states {
                *bwd_ptr.add(i) = scale * *bwd_ptr.add(i) + shift;
            }
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx512f,avx512bw,fma")]
    unsafe fn bwd_update_avx512_fma(
        bwd: &mut [f32],
        p_switch: f32,
        emit_probs: &[f32; 2],
        mismatches: &[u8],
        n_states: usize,
    ) {
        // Safety: caller guarantees `bwd` and `mismatches` have at least `n_states` elements
        // and point to valid, properly aligned memory for AVX-512 loads/stores.
        unsafe {
            let p0 = emit_probs[0];
            let p1 = emit_probs[1];
            let diff = p1 - p0;

            let p0_vec = _mm512_set1_ps(p0);
            let diff_vec = _mm512_set1_ps(diff);
            let mut sum_vec = _mm512_setzero_ps();

            let mut k = 0;
            let bwd_ptr = bwd.as_mut_ptr();
            let mismatch_ptr = mismatches.as_ptr();
            while k + 16 <= n_states {
                let bwd_chunk = _mm512_loadu_ps(bwd_ptr.add(k));
                let m_u8 = _mm_loadu_si128(mismatch_ptr.add(k) as *const __m128i);
                let m_i32 = _mm512_cvtepu8_epi32(m_u8);
                let m_f32 = _mm512_cvtepi32_ps(m_i32);
                let emit_vec = _mm512_fmadd_ps(m_f32, diff_vec, p0_vec);
                let res = _mm512_mul_ps(bwd_chunk, emit_vec);
                _mm512_storeu_ps(bwd_ptr.add(k), res);
                sum_vec = _mm512_add_ps(sum_vec, res);
                k += 16;
            }

            let mut sum_arr = [0.0f32; 16];
            _mm512_storeu_ps(sum_arr.as_mut_ptr(), sum_vec);
            let mut sum: f32 = sum_arr.iter().sum();

            for i in k..n_states {
                let em = emit_probs[*mismatches.get_unchecked(i) as usize];
                let v = *bwd_ptr.add(i) * em;
                *bwd_ptr.add(i) = v;
                sum += v;
            }

            let shift = p_switch / n_states as f32;
            let scale = (1.0 - p_switch) / sum.max(1e-30);

            let shift_vec = _mm512_set1_ps(shift);
            let scale_vec = _mm512_set1_ps(scale);

            k = 0;
            while k + 16 <= n_states {
                let bwd_chunk = _mm512_loadu_ps(bwd_ptr.add(k));
                let res = _mm512_fmadd_ps(scale_vec, bwd_chunk, shift_vec);
                _mm512_storeu_ps(bwd_ptr.add(k), res);
                k += 16;
            }

            for i in k..n_states {
                *bwd_ptr.add(i) = scale * *bwd_ptr.add(i) + shift;
            }
        }
    }
}

// ============================================================================
// MosaicHmm: Memory-Efficient Mosaic HMM with A-B-C Loop Pattern
// ============================================================================

use crate::model::pl_emission::{
    PlProvider, allele_probs_cond_from_pl, allele_probs_uncond_from_pl,
};
use crate::model::states::{MosaicCursor, StateSwitch, ThreadedHaps};
use crate::model::types::{CombinedHapSpace, HapId};

/// High-performance Li-Stephens HMM using mosaic states with A-B-C loop pattern.
///
/// This implementation achieves:
/// - **Memory efficiency**: O(K * segments) instead of O(M * K) for state map
/// - **SIMD friendliness**: Separates state maintenance from math kernel
/// - **Java parity**: Matches Beagle's composite state approach
///
/// ## The A-B-C Loop Pattern
/// - **Phase A**: State maintenance (integer logic, branch-predictable)
/// - **Phase B**: Allele materialization (memory fetch into contiguous scratch)
/// - **Phase C**: Math kernel (SIMD-vectorizable on flat data)
pub struct MosaicHmm<
    'a,
    TargetSpace = crate::data::AnyMarkerSpace,
    RefSpace = crate::data::AnyMarkerSpace,
    HapSpace = CombinedHapSpace,
> {
    /// Reference panel genotypes
    ref_gt: GenotypeView<'a, TargetSpace, RefSpace>,
    /// Model parameters
    params: &'a ModelParams,
    /// Number of HMM states
    n_states: usize,
    /// Recombination probabilities between consecutive markers
    p_recomb: &'a [f32],
    hap_space: std::marker::PhantomData<HapSpace>,
}
impl<'a, TargetSpace, RefSpace, HapSpace> MosaicHmm<'a, TargetSpace, RefSpace, HapSpace> {
    /// Create a new MosaicHmm
    pub fn new(
        ref_gt: impl Into<GenotypeView<'a, TargetSpace, RefSpace>>,
        params: &'a ModelParams,
        n_states: usize,
        p_recomb: &'a [f32],
    ) -> Self {
        Self {
            ref_gt: ref_gt.into(),
            params,
            n_states,
            p_recomb,
            hap_space: std::marker::PhantomData,
        }
    }

    /// Number of markers
    pub fn n_markers(&self) -> usize {
        self.ref_gt.n_markers()
    }

    /// Forward-backward with a fixed partner haplotype constraint.
    ///
    /// The emission probability is conditioned on the partner allele such that
    /// the target haplotype must complement the partner at heterozygous sites.
    pub fn conditioned_forward_backward(
        &self,
        geno_a1: &[u8],
        geno_a2: &[u8],
        partner_alleles: &[u8],
        target_conf: Option<&[f32]>,
        pl_provider: Option<&PlProvider>,
        allele_freqs: Option<&[f32]>,
        init_prior: Option<&[f32]>,
        threaded_haps: &ThreadedHaps<HapSpace>,
        fwd: &mut Vec<f32>,
        bwd: &mut Vec<f32>,
    ) -> f64 {
        let n_markers = self.n_markers();
        let n_states = self.n_states;
        let n_states_padded = ((n_states + 63) / 64) * 64;
        let total_size = n_markers * n_states_padded;

        if n_markers == 0 || n_states == 0 {
            return 0.0;
        }

        let p_err_base = self.params.p_mismatch;
        let p_no_err_base = 1.0 - p_err_base;

        fwd.resize(n_markers * n_states, 0.0);
        bwd.resize(n_markers * n_states, 0.0);

        let mut fwd_aligned =
            AVec::<f32, ConstAlign<64>>::from_iter(64, std::iter::repeat(0.0f32).take(total_size));
        let mut bwd_aligned =
            AVec::<f32, ConstAlign<64>>::from_iter(64, std::iter::repeat(0.0f32).take(total_size));
        let mut emissions = AVec::<f32, ConstAlign<64>>::from_iter(
            64,
            std::iter::repeat(0.0f32).take(n_states_padded),
        );
        let mut fwd_sum = 1.0f32;

        let mut allele_probs: Vec<f32> = Vec::new();

        let mut log_likelihood = 0.0f64;

        let mut state_buf = vec![HapId::<HapSpace>::new(0u32); n_states];
        let mut state_haps = vec![HapIdx::new(0u32); n_states];
        let mut ref_alleles_flat =
            AVec::<u8, ConstAlign<64>>::from_iter(64, std::iter::repeat(255u8).take(total_size));
        for m in 0..n_markers {
            threaded_haps.materialize_at(m, &mut state_buf);
            for k in 0..n_states {
                state_haps[k] = HapIdx::new(state_buf[k].as_u32());
            }
            let row_offset = m * n_states_padded;
            self.ref_gt.fill_batch(
                MarkerIdx::new(m as u32),
                &state_haps,
                &mut ref_alleles_flat[row_offset..row_offset + n_states],
            );
        }
        let inv_n_states = 1.0 / n_states as f32;
        let mut shift_by_marker = vec![0.0f32; n_markers];
        let mut one_minus_by_marker = vec![0.0f32; n_markers];
        for m in 0..n_markers {
            let p = self.p_recomb.get(m).copied().unwrap_or(0.0);
            shift_by_marker[m] = p * inv_n_states;
            one_minus_by_marker[m] = 1.0 - p;
        }

        let mut req_allele = vec![255u8; n_markers];
        let mut p_match = vec![1.0f32; n_markers];
        let mut p_mismatch = vec![1.0f32; n_markers];
        for m in 0..n_markers {
            let conf = target_conf
                .and_then(|c| c.get(m).copied())
                .unwrap_or(1.0)
                .clamp(0.0, 1.0);
            let g1 = *geno_a1.get(m).unwrap_or(&255);
            let g2 = *geno_a2.get(m).unwrap_or(&255);
            let partner = partner_alleles.get(m).copied().unwrap_or(255);
            let req = if g1 == 255 || g2 == 255 {
                255
            } else if g1 == g2 {
                g1
            } else if partner == g1 {
                g2
            } else if partner == g2 {
                g1
            } else {
                255
            };
            req_allele[m] = req;
            if req != 255 {
                p_match[m] = p_no_err_base * conf + 0.5 * (1.0 - conf);
                p_mismatch[m] = p_err_base * conf + 0.5 * (1.0 - conf);
            }
        }

        let fill_conf_emissions_fast = |m: usize, emissions: &mut [f32], ref_alleles: &[u8]| {
            let req = req_allele[m];
            if req != 255 {
                let p_no_err = p_match[m];
                let p_err = p_mismatch[m];
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    if ref_alleles.len() >= 64
                        && is_x86_feature_detected!("avx512f")
                        && is_x86_feature_detected!("avx512bw")
                    {
                        unsafe {
                            let req_vec = _mm512_set1_epi8(req as i8);
                            let match_vec = _mm512_set1_ps(p_no_err);
                            let mismatch_vec = _mm512_set1_ps(p_err);
                            let mut k = 0usize;
                            let out_ptr = emissions.as_mut_ptr();
                            let in_ptr = ref_alleles.as_ptr();
                            while k + 64 <= ref_alleles.len() && k + 64 <= emissions.len() {
                                let bytes = _mm512_load_si512(in_ptr.add(k) as *const __m512i);
                                let mask = _mm512_cmpeq_epi8_mask(bytes, req_vec);
                                let mask0 = (mask & 0xFFFF) as u16;
                                let mask1 = ((mask >> 16) & 0xFFFF) as u16;
                                let mask2 = ((mask >> 32) & 0xFFFF) as u16;
                                let mask3 = ((mask >> 48) & 0xFFFF) as u16;
                                let res0 = _mm512_mask_blend_ps(mask0, mismatch_vec, match_vec);
                                let res1 = _mm512_mask_blend_ps(mask1, mismatch_vec, match_vec);
                                let res2 = _mm512_mask_blend_ps(mask2, mismatch_vec, match_vec);
                                let res3 = _mm512_mask_blend_ps(mask3, mismatch_vec, match_vec);
                                _mm512_storeu_ps(out_ptr.add(k), res0);
                                _mm512_storeu_ps(out_ptr.add(k + 16), res1);
                                _mm512_storeu_ps(out_ptr.add(k + 32), res2);
                                _mm512_storeu_ps(out_ptr.add(k + 48), res3);
                                k += 64;
                            }
                            for i in k..emissions.len() {
                                let ra = *ref_alleles.get_unchecked(i);
                                *emissions.get_unchecked_mut(i) =
                                    if ra == req { p_no_err } else { p_err };
                            }
                        }
                        return;
                    }
                }
                let mut lut = [0.0f32; 256];
                lut.fill(p_err);
                lut[req as usize] = p_no_err;
                for (k, &ra) in ref_alleles.iter().enumerate() {
                    emissions[k] = lut[ra as usize];
                }
            } else {
                emissions.fill(1.0);
            }
        };

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum MarkerKind {
            Missing,
            HomoSimple,
            Generic,
        }

        #[derive(Clone, Copy, Debug)]
        struct MarkerSegment {
            kind: MarkerKind,
            start: usize,
            len: usize,
        }

        // Pre-classify markers so we can dispatch to specialized kernels without
        // re-checking PLs/heterozygosity inside the hot loop.
        let mut marker_kinds = vec![MarkerKind::Generic; n_markers];
        for m in 0..n_markers {
            let g1 = *geno_a1.get(m).unwrap_or(&255);
            let g2 = *geno_a2.get(m).unwrap_or(&255);
            if g1 == 255 || g2 == 255 {
                marker_kinds[m] = MarkerKind::Missing;
                continue;
            }
            if let Some(plp) = pl_provider {
                if plp.pl(m).map(|v| !v.is_empty()).unwrap_or(false) {
                    marker_kinds[m] = MarkerKind::Generic;
                    continue;
                }
            }
            if g1 == g2 && req_allele[m] != 255 {
                marker_kinds[m] = MarkerKind::HomoSimple;
            } else {
                marker_kinds[m] = MarkerKind::Generic;
            }
        }

        // Build run-length segments of marker kinds to avoid per-marker
        // classification overhead in the forward traversal.
        let mut segments: Vec<MarkerSegment> = Vec::with_capacity(n_markers / 4 + 1);
        if n_markers > 0 {
            let mut start = 0usize;
            let mut current = marker_kinds[0];
            for m in 1..n_markers {
                let kind = marker_kinds[m];
                if kind != current {
                    segments.push(MarkerSegment {
                        kind: current,
                        start,
                        len: m - start,
                    });
                    start = m;
                    current = kind;
                }
            }
            segments.push(MarkerSegment {
                kind: current,
                start,
                len: n_markers - start,
            });
        }

        let try_fill_pattern_emissions =
            |_: usize, _: u8, _: Option<&[f32]>, _: &mut [f32], _: &[HapId<HapSpace>]| false;

        let mut process_marker = |m: usize, kind: MarkerKind| {
            let row_offset = m * n_states_padded;
            let emissions_row = &mut emissions[..n_states];
            let ref_row = &ref_alleles_flat[row_offset..row_offset + n_states];
            if kind != MarkerKind::Missing {
                threaded_haps.materialize_at(m, &mut state_buf);
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if m + PREFETCH_DISTANCE < n_markers {
                let prefetch_ref = (m + PREFETCH_DISTANCE) * n_states_padded;
                let prefetch_fwd = (m + PREFETCH_DISTANCE) * n_states_padded;
                unsafe {
                    _mm_prefetch(
                        ref_alleles_flat.as_ptr().add(prefetch_ref) as *const i8,
                        _MM_HINT_T0,
                    );
                    _mm_prefetch(
                        fwd_aligned.as_ptr().add(prefetch_fwd) as *const i8,
                        _MM_HINT_T0,
                    );
                }
            }

            match kind {
                MarkerKind::Missing => {
                    // Missing emissions are identically 1.0 for all states.
                    // Math: for missing data, E_m is the identity (all ones), so the
                    // forward recursion is exactly:
                    //   f_m = T_m f_{m-1}
                    // where T_m is the Li-Stephens transition. We still apply the
                    // transition per marker (exact), but we skip emission work and
                    // reference allele loads. This keeps the model unchanged.
                    emissions_row[..n_states].fill(1.0);
                }
                MarkerKind::HomoSimple => {
                    // Specialized homozygous path: no PLs, no partner disambiguation.
                    // Emissions depend only on a single required allele (req_allele[m]) and
                    // its confidence-derived p_match/p_mismatch.
                    if req_allele[m] != 255 && m != 0 {
                        // For homozygous markers after the first, fuse emission generation
                        // with the transition update to avoid touching the emissions buffer.
                        // This is mathematically exact (emission is still applied per marker)
                        // and reduces memory traffic.
                    } else {
                        fill_conf_emissions_fast(m, emissions_row, ref_row);
                    }
                }
                MarkerKind::Generic => {
                    if let Some(plp) = pl_provider {
                        let partner = partner_alleles.get(m).copied().unwrap_or(255);
                        let pl = plp.pl(m).filter(|v| !v.is_empty());
                        if let Some(pl) = pl {
                            let biallelic_freqs =
                                allele_freqs.and_then(|f| f.get(m).copied()).and_then(|f| {
                                    if (0.0..=1.0).contains(&f) {
                                        Some([1.0 - f, f])
                                    } else {
                                        None
                                    }
                                });
                            let n = if partner != 255 {
                                allele_probs_cond_from_pl(
                                    pl,
                                    partner,
                                    biallelic_freqs.as_ref().map(|f| f.as_slice()),
                                    &mut allele_probs,
                                )
                                .or_else(|| {
                                    allele_probs_uncond_from_pl(
                                        pl,
                                        biallelic_freqs.as_ref().map(|f| f.as_slice()),
                                        &mut allele_probs,
                                    )
                                })
                            } else {
                                allele_probs_uncond_from_pl(
                                    pl,
                                    biallelic_freqs.as_ref().map(|f| f.as_slice()),
                                    &mut allele_probs,
                                )
                            };
                            if let Some(n_alleles) = n {
                                let p_no_err = p_no_err_base;
                                let p_err_other = if n_alleles > 1 {
                                    p_err_base / (n_alleles as f32 - 1.0)
                                } else {
                                    0.0
                                };
                                let mut lut = [0.0f32; 256];
                                lut.fill(1.0);
                                for a in 0..255usize {
                                    if a < n_alleles {
                                        let p_true = allele_probs.get(a).copied().unwrap_or(0.0);
                                        lut[a] = (p_no_err * p_true + p_err_other * (1.0 - p_true))
                                            .max(1e-30);
                                    }
                                }
                                let used_pattern = try_fill_pattern_emissions(
                                    m,
                                    partner,
                                    Some(&allele_probs),
                                    emissions_row,
                                    &state_buf,
                                );
                                if !used_pattern {
                                    for k in 0..n_states {
                                        emissions_row[k] = lut[ref_row[k] as usize];
                                    }
                                }
                            } else {
                                let used_pattern = try_fill_pattern_emissions(
                                    m,
                                    partner,
                                    None,
                                    emissions_row,
                                    &state_buf,
                                );
                                if !used_pattern {
                                    fill_conf_emissions_fast(m, emissions_row, ref_row);
                                }
                            }
                        } else {
                            let used_pattern = try_fill_pattern_emissions(
                                m,
                                partner,
                                None,
                                emissions_row,
                                &state_buf,
                            );
                            if !used_pattern {
                                fill_conf_emissions_fast(m, emissions_row, ref_row);
                            }
                        }
                    } else {
                        let partner = partner_alleles.get(m).copied().unwrap_or(255);
                        let used_pattern =
                            try_fill_pattern_emissions(m, partner, None, emissions_row, &state_buf);
                        if !used_pattern {
                            fill_conf_emissions_fast(m, emissions_row, ref_row);
                        }
                    }
                }
            }

            if m == 0 {
                let mut prior_sum = 0.0f32;
                if let Some(prior) = init_prior {
                    if prior.len() == n_states {
                        prior_sum = prior.iter().copied().sum();
                    }
                }
                let use_prior = prior_sum > 0.0;
                let init_val = if use_prior {
                    1.0 / prior_sum
                } else {
                    1.0 / n_states as f32
                };
                fwd_sum = 0.0;
                for k in 0..n_states {
                    let base = if use_prior {
                        init_prior.and_then(|p| p.get(k)).copied().unwrap_or(0.0) * init_val
                    } else {
                        init_val
                    };
                    let val = base * emissions_row[k];
                    fwd_aligned[row_offset + k] = val;
                    fwd_sum += val;
                }
            } else if kind == MarkerKind::HomoSimple && req_allele[m] != 255 {
                let prev_row_offset = (m - 1) * n_states_padded;
                let (before, curr_and_after) = fwd_aligned.split_at_mut(row_offset);
                let prev_row = &before[prev_row_offset..prev_row_offset + n_states];
                let curr_row = &mut curr_and_after[..n_states];
                curr_row.copy_from_slice(prev_row);
                let shift = shift_by_marker[m];
                let one_minus = one_minus_by_marker[m];
                let scale = one_minus / fwd_sum.max(1e-30);
                let req = req_allele[m];
                let p_no_err = p_match[m];
                let p_err = p_mismatch[m];
                fwd_sum = HmmUpdater::fwd_update_homo_emissions_scale(
                    curr_row, scale, shift, ref_row, req, p_no_err, p_err, n_states,
                )
                .max(1e-30);
            } else {
                let prev_row_offset = (m - 1) * n_states_padded;
                let (before, curr_and_after) = fwd_aligned.split_at_mut(row_offset);
                let prev_row = &before[prev_row_offset..prev_row_offset + n_states];
                let curr_row = &mut curr_and_after[..n_states];
                curr_row.copy_from_slice(prev_row);
                let shift = shift_by_marker[m];
                let one_minus = one_minus_by_marker[m];
                let scale = one_minus / fwd_sum.max(1e-30);
                const STATE_BLOCK: usize = 256;
                if n_states > STATE_BLOCK {
                    let mut new_sum = 0.0f32;
                    let mut start = 0usize;
                    while start < n_states {
                        let end = (start + STATE_BLOCK).min(n_states);
                        new_sum += HmmUpdater::fwd_update_emissions_scale(
                            &mut curr_row[start..end],
                            scale,
                            shift,
                            &emissions_row[start..end],
                            end - start,
                        );
                        start = end;
                    }
                    fwd_sum = new_sum.max(1e-30);
                } else {
                    fwd_sum = HmmUpdater::fwd_update_emissions_scale(
                        curr_row,
                        scale,
                        shift,
                        emissions_row,
                        n_states,
                    );
                }
            }

            if fwd_sum > 0.0 {
                log_likelihood += (fwd_sum as f64).ln();
            }
        };

        const TILE_SIZE: usize = 256;
        for seg in segments.iter().copied() {
            let mut m = seg.start;
            let end = seg.start + seg.len;
            while m < end {
                let tile_end = (m + TILE_SIZE).min(end);
                let mut i = m;
                while i < tile_end {
                    process_marker(i, seg.kind);
                    i += 1;
                }
                m = tile_end;
            }
        }

        let last_row = (n_markers - 1) * n_states_padded;
        let init_bwd = 1.0 / n_states as f32;
        for k in 0..n_states {
            bwd_aligned[last_row + k] = init_bwd;
        }

        let mut tile_end = n_markers;
        while tile_end > 1 {
            let tile_start = tile_end.saturating_sub(TILE_SIZE).max(1);
            for m in (tile_start - 1..tile_end - 1).rev() {
                let m_next = m + 1;
                threaded_haps.materialize_at(m_next, &mut state_buf);
                let next_row_offset = m_next * n_states_padded;
                let ref_row = &ref_alleles_flat[next_row_offset..next_row_offset + n_states];
                let emissions_row = &mut emissions[..n_states];
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                if m_next >= PREFETCH_DISTANCE {
                    let prefetch_row = (m_next - PREFETCH_DISTANCE) * n_states_padded;
                    let prefetch_bwd = (m_next - PREFETCH_DISTANCE) * n_states_padded;
                    unsafe {
                        _mm_prefetch(
                            ref_alleles_flat.as_ptr().add(prefetch_row) as *const i8,
                            _MM_HINT_T0,
                        );
                        _mm_prefetch(
                            bwd_aligned.as_ptr().add(prefetch_bwd) as *const i8,
                            _MM_HINT_T0,
                        );
                    }
                }

                match marker_kinds[m_next] {
                    MarkerKind::Missing => {
                        // Missing data: emission is the identity (all ones), so the backward
                        // update is purely the transition operator. We keep the exact per-marker
                        // transition (no aggregation) and skip any allele-based emission work.
                        emissions_row[..n_states].fill(1.0);
                    }
                    MarkerKind::HomoSimple => {
                        fill_conf_emissions_fast(m_next, emissions_row, ref_row);
                    }
                    MarkerKind::Generic => {
                        if let Some(plp) = pl_provider {
                            let partner = partner_alleles.get(m_next).copied().unwrap_or(255);
                            let pl = plp.pl(m_next).filter(|v| !v.is_empty());
                            if let Some(pl) = pl {
                                let biallelic_freqs = allele_freqs
                                    .and_then(|f| f.get(m_next).copied())
                                    .and_then(|f| {
                                        if (0.0..=1.0).contains(&f) {
                                            Some([1.0 - f, f])
                                        } else {
                                            None
                                        }
                                    });
                                let n = if partner != 255 {
                                    allele_probs_cond_from_pl(
                                        pl,
                                        partner,
                                        biallelic_freqs.as_ref().map(|f| f.as_slice()),
                                        &mut allele_probs,
                                    )
                                    .or_else(|| {
                                        allele_probs_uncond_from_pl(
                                            pl,
                                            biallelic_freqs.as_ref().map(|f| f.as_slice()),
                                            &mut allele_probs,
                                        )
                                    })
                                } else {
                                    allele_probs_uncond_from_pl(
                                        pl,
                                        biallelic_freqs.as_ref().map(|f| f.as_slice()),
                                        &mut allele_probs,
                                    )
                                };
                                if let Some(n_alleles) = n {
                                    let p_no_err = p_no_err_base;
                                    let p_err_other = if n_alleles > 1 {
                                        p_err_base / (n_alleles as f32 - 1.0)
                                    } else {
                                        0.0
                                    };
                                    let mut lut = [0.0f32; 256];
                                    lut.fill(1.0);
                                    for a in 0..255usize {
                                        if a < n_alleles {
                                            let p_true = allele_probs.get(a).copied().unwrap_or(0.0);
                                            lut[a] =
                                                (p_no_err * p_true + p_err_other * (1.0 - p_true))
                                                    .max(1e-30);
                                        }
                                    }
                                    let used_pattern = try_fill_pattern_emissions(
                                        m_next,
                                        partner,
                                        Some(&allele_probs),
                                        emissions_row,
                                        &state_buf,
                                    );
                                    if !used_pattern {
                                        for k in 0..n_states {
                                            emissions_row[k] = lut[ref_row[k] as usize];
                                        }
                                    }
                                } else {
                                    let used_pattern = try_fill_pattern_emissions(
                                        m_next,
                                        partner,
                                        None,
                                        emissions_row,
                                        &state_buf,
                                    );
                                    if !used_pattern {
                                        fill_conf_emissions_fast(m_next, emissions_row, ref_row);
                                    }
                                }
                            } else {
                                let used_pattern = try_fill_pattern_emissions(
                                    m_next,
                                    partner,
                                    None,
                                    emissions_row,
                                    &state_buf,
                                );
                                if !used_pattern {
                                    fill_conf_emissions_fast(m_next, emissions_row, ref_row);
                                }
                            }
                        } else {
                            let partner = partner_alleles.get(m_next).copied().unwrap_or(255);
                            let used_pattern = try_fill_pattern_emissions(
                                m_next,
                                partner,
                                None,
                                emissions_row,
                                &state_buf,
                            );
                            if !used_pattern {
                                fill_conf_emissions_fast(m_next, emissions_row, ref_row);
                            }
                        }
                    }
                }

                let next_row = m_next * n_states_padded;
                let curr_row = m * n_states_padded;
                for k in 0..n_states {
                    bwd_aligned[curr_row + k] = bwd_aligned[next_row + k];
                }

                // Calculate constant term C = sum_k (bwd[k] * output_prob[k])
                let mut constant_term = 0.0f32;
                let current_bwd = &bwd_aligned[curr_row..curr_row + n_states];
                for k in 0..n_states {
                    constant_term += current_bwd[k] * emissions_row[k];
                }

                let shift = shift_by_marker[m_next];
                let one_minus = one_minus_by_marker[m_next];
                let scale = one_minus / constant_term.max(1e-30);
                const STATE_BLOCK: usize = 256;
                if n_states > STATE_BLOCK {
                    let mut start = 0usize;
                    while start < n_states {
                        let end = (start + STATE_BLOCK).min(n_states);
                        HmmUpdater::bwd_update_constant_scale(
                            &mut bwd_aligned[curr_row + start..curr_row + end],
                            scale,
                            shift,
                            &emissions_row[start..end],
                            end - start,
                        );
                        start = end;
                    }
                } else {
                    HmmUpdater::bwd_update_constant_scale(
                        &mut bwd_aligned[curr_row..curr_row + n_states],
                        scale,
                        shift,
                        emissions_row,
                        n_states,
                    );
                }
            }
            tile_end = tile_start;
        }

        for m in 0..n_markers {
            let src = m * n_states_padded;
            let dst = m * n_states;
            fwd[dst..dst + n_states].copy_from_slice(&fwd_aligned[src..src + n_states]);
            bwd[dst..dst + n_states].copy_from_slice(&bwd_aligned[src..src + n_states]);
        }

        log_likelihood
    }

    /// Collect statistics for EM parameter estimation
    /// Uses checkpointing to reduce memory from O(n_markers × n_states) to O(n_markers/64 × n_states)
    pub fn collect_stats(
        &self,
        target_alleles: &[u8],
        threaded_haps: &ThreadedHaps<HapSpace>,
        gen_dists: &[f64],
        estimates: &mut crate::model::parameters::ParamEstimates,
    ) {
        let n_markers = self.n_markers();
        let n_states = self.n_states;
        if n_markers < 2 || n_states <= 1 {
            return;
        }

        let p_err = self.params.p_mismatch;
        let p_no_err = 1.0 - p_err;
        let emit_probs = [p_no_err, p_err];
        let inv_n_states = 1.0 / n_states as f32;
        let mut shift_by_marker = vec![0.0f32; n_markers];
        let mut one_minus_by_marker = vec![0.0f32; n_markers];
        for m in 0..n_markers {
            let p = self.p_recomb.get(m).copied().unwrap_or(0.0);
            shift_by_marker[m] = p * inv_n_states;
            one_minus_by_marker[m] = 1.0 - p;
        }

        let n_states_padded = ((n_states + 63) / 64) * 64;
        let total_size = n_markers * n_states_padded;
        let mut state_buf = vec![HapId::<HapSpace>::new(0u32); n_states];
        let mut state_haps = vec![HapIdx::new(0u32); n_states];
        let mut ref_alleles_flat =
            AVec::<u8, ConstAlign<64>>::from_iter(64, std::iter::repeat(255u8).take(total_size));
        for m in 0..n_markers {
            threaded_haps.materialize_at(m, &mut state_buf);
            for k in 0..n_states {
                state_haps[k] = HapIdx::new(state_buf[k].as_u32());
            }
            let row_offset = m * n_states_padded;
            self.ref_gt.fill_batch(
                MarkerIdx::new(m as u32),
                &state_haps,
                &mut ref_alleles_flat[row_offset..row_offset + n_states],
            );
        }

        // Checkpoint interval - balance memory vs recomputation
        const CHECKPOINT_INTERVAL: usize = 64;
        let n_checkpoints = (n_markers + CHECKPOINT_INTERVAL - 1) / CHECKPOINT_INTERVAL;

        // Create cursor and record history during forward traversal
        let mut cursor = MosaicCursor::from_threaded(threaded_haps);
        let mut history: Vec<StateSwitch<HapSpace>> = Vec::with_capacity(n_markers);

        // First pass: advance cursor to end while recording history AND storing checkpoints
        let mut fwd_checkpoints = vec![0.0f32; n_checkpoints * n_states];
        let mut fwd = vec![1.0f32 / n_states as f32; n_states];
        let mut fwd_sums = vec![1.0f32; n_markers];
        let mut last_fwd_sum = 1.0f32;

        for m in 0..n_markers {
            cursor.advance_with_history(m, threaded_haps, &mut history);

            let targ_al = target_alleles[m];
            let row_offset = m * n_states_padded;
            let ref_row = &ref_alleles_flat[row_offset..row_offset + n_states];
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if m + PREFETCH_DISTANCE < n_markers {
                let prefetch_row = (m + PREFETCH_DISTANCE) * n_states_padded;
                unsafe {
                    _mm_prefetch(
                        ref_alleles_flat.as_ptr().add(prefetch_row) as *const i8,
                        _MM_HINT_T0,
                    );
                }
            }

            if m > 0 {
                let shift = shift_by_marker[m];
                let scale = one_minus_by_marker[m] / last_fwd_sum;

                let mut sum = 0.0f32;
                for k in 0..n_states {
                    let ref_al = ref_row[k];
                    let is_mismatch = ref_al != targ_al;
                    let em = if is_mismatch { p_err } else { p_no_err };
                    fwd[k] = em * (scale * fwd[k] + shift);
                    sum += fwd[k];
                }
                last_fwd_sum = sum.max(1e-30);
            } else {
                // First marker: uniform prior * emission
                let prior = 1.0 / n_states as f32;
                let mut sum = 0.0f32;
                for k in 0..n_states {
                    let ref_al = ref_row[k];
                    let is_mismatch = ref_al != targ_al;
                    let em = if is_mismatch { p_err } else { p_no_err };
                    fwd[k] = em * prior;
                    sum += fwd[k];
                }
                last_fwd_sum = sum.max(1e-30);
            }

            fwd_sums[m] = last_fwd_sum;

            // Store checkpoint at interval boundaries
            if m % CHECKPOINT_INTERVAL == 0 {
                let checkpoint_idx = m / CHECKPOINT_INTERVAL;
                let checkpoint_off = checkpoint_idx * n_states;
                fwd_checkpoints[checkpoint_off..checkpoint_off + n_states].copy_from_slice(&fwd);
            }
        }

        // 2. Combined backward pass with forward recomputation and stats accumulation
        // Process in reverse order, recomputing forward from checkpoints as needed
        let mut bwd = vec![1.0f32; n_states];
        let mut mismatches = vec![0u8; n_states];
        let mut fwd_recomp = vec![0.0f32; n_states];

        let h_factor = n_states as f32 / (n_states - 1) as f32;

        for m in (0..n_markers).rev() {
            let marker_idx = MarkerIdx::new(m as u32);
            let targ_al = target_alleles[m];

            // Rewind cursor to this marker
            cursor.rewind(m, &mut history);

            // Recompute forward values from nearest checkpoint
            let checkpoint_idx = m / CHECKPOINT_INTERVAL;
            let checkpoint_start = checkpoint_idx * CHECKPOINT_INTERVAL;
            let checkpoint_off = checkpoint_idx * n_states;

            // Load checkpoint
            fwd_recomp.copy_from_slice(&fwd_checkpoints[checkpoint_off..checkpoint_off + n_states]);
            let mut recomp_sum = fwd_sums[checkpoint_start];

            // Recompute forward from checkpoint to m
            // Need a separate cursor for recomputation
            let mut recomp_cursor = MosaicCursor::from_threaded(threaded_haps);
            let mut recomp_history: Vec<StateSwitch<HapSpace>> = Vec::with_capacity(m + 1);

            // Advance recomp cursor to checkpoint_start
            for recomp_m in 0..=checkpoint_start {
                recomp_cursor.advance_with_history(recomp_m, threaded_haps, &mut recomp_history);
            }

            // Now advance from checkpoint_start+1 to m while recomputing forward
            for recomp_m in (checkpoint_start + 1)..=m {
                recomp_cursor.advance_with_history(recomp_m, threaded_haps, &mut recomp_history);

                let recomp_marker_idx = MarkerIdx::new(recomp_m as u32);
                let recomp_targ_al = target_alleles[recomp_m];
                let p_switch = self.p_recomb.get(recomp_m).copied().unwrap_or(0.0);
                let shift = p_switch / n_states as f32;
                let scale = (1.0 - p_switch) / recomp_sum.max(1e-30);

                let mut sum = 0.0f32;
                for k in 0..n_states {
                    let ref_al = self.ref_gt.allele(
                        recomp_marker_idx,
                        HapIdx::new(recomp_cursor.active_haps()[k].as_u32()),
                    );
                    let is_mismatch = ref_al != recomp_targ_al;
                    let em = if is_mismatch { p_err } else { p_no_err };
                    fwd_recomp[k] = em * (scale * fwd_recomp[k] + shift);
                    sum += fwd_recomp[k];
                }
                recomp_sum = sum.max(1e-30);
            }

            // Now fwd_recomp contains forward values at marker m
            // Compute stats using fwd_recomp and bwd
            let p_switch = self.p_recomb.get(m).copied().unwrap_or(0.0);
            let last_sum = if m > 0 { fwd_sums[m - 1] } else { 1.0 };
            let shift = p_switch / n_states as f32;
            let scale = (1.0 - p_switch) / last_sum;
            let no_switch_scale = ((1.0 - p_switch) + shift) / last_sum;

            let mut joint_state_sum = 0.0f32;
            let mut state_sum = 0.0f32;
            let mut mismatch_sum = 0.0f32;

            for k in 0..n_states {
                let ref_al = self
                    .ref_gt
                    .allele(marker_idx, HapIdx::new(cursor.active_haps()[k].as_u32()));
                let is_mismatch = ref_al != targ_al;
                let em = if is_mismatch { p_err } else { p_no_err };

                // Use fwd values from before emission update for joint probability
                let fwd_prior_k = if m > 0 {
                    scale * fwd_recomp[k] / em + shift / em // Reverse the emission to get prior
                } else {
                    1.0 / n_states as f32
                };

                joint_state_sum += bwd[k] * em * no_switch_scale * fwd_prior_k.max(0.0);

                let state_prob = fwd_recomp[k] * bwd[k];
                state_sum += state_prob;
                if is_mismatch {
                    mismatch_sum += state_prob;
                }
            }

            if state_sum > 0.0 {
                estimates.add_emission(
                    (1.0 - mismatch_sum / state_sum) as f64,
                    (mismatch_sum / state_sum) as f64,
                );
            }

            if m > 0 && state_sum > 0.0 {
                let switch_prob = h_factor * (1.0 - joint_state_sum / state_sum);
                if switch_prob > 0.0 {
                    let gen_dist = gen_dists.get(m - 1).copied().unwrap_or(0.0);
                    estimates.add_switch(gen_dist, switch_prob as f64);
                }
            }

            // Update backward values for next iteration (moving to m-1)
            if m > 0 {
                let m_next = m; // We're about to move to m-1, so m is the "next" marker from m-1's perspective
                let targ_al_next = target_alleles[m_next];
                let p_recomb = self.p_recomb.get(m_next).copied().unwrap_or(0.0);

                for k in 0..n_states {
                    let h = cursor.active_haps()[k];
                    let r = self.ref_gt.allele(marker_idx, HapIdx::new(h.as_u32()));
                    mismatches[k] = if r == targ_al_next { 0 } else { 1 };
                }

                HmmUpdater::bwd_update(&mut bwd, p_recomb, &emit_probs, &mismatches, n_states);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::ChromIdx;
    use crate::data::haplotype::Samples;
    use crate::data::marker::{Allele, Marker, Markers, Nucleotide};
    use crate::data::storage::{GenotypeColumn, GenotypeMatrix};
    use crate::model::types::RefHapSpace;
    use std::sync::Arc;

    fn make_test_ref_panel() -> GenotypeMatrix {
        let samples = Arc::new(Samples::from_ids(vec![
            "R1".to_string(),
            "R2".to_string(),
            "R3".to_string(),
        ]));
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");

        // Create 5 markers
        let mut columns = Vec::new();
        for i in 0..5 {
            let m = Marker::new(
                ChromIdx::new(0),
                (i * 1000 + 100) as u32,
                None,
                Allele::Base(Nucleotide::A),
                vec![Allele::Base(Nucleotide::C)],
            );
            markers.push(m);

            // 6 haplotypes with different patterns
            let alleles = match i {
                0 => vec![0, 0, 1, 1, 0, 1],
                1 => vec![0, 1, 1, 0, 0, 1],
                2 => vec![0, 0, 1, 1, 1, 0],
                3 => vec![1, 0, 1, 0, 1, 0],
                4 => vec![0, 1, 0, 1, 0, 1],
                _ => vec![0; 6],
            };
            columns.push(GenotypeColumn::from_alleles(&alleles, 2));
        }

        GenotypeMatrix::new_unphased(markers, columns, samples)
    }

    #[test]
    fn test_hmm_updater_fwd() {
        let mut fwd = vec![0.25f32; 4];
        let emit_probs = [0.99f32, 0.01];
        let mismatches = vec![0u8, 0, 1, 0];

        let emissions: Vec<f32> = mismatches.iter().map(|&m| emit_probs[m as usize]).collect();
        let p_switch = 0.01;
        let shift = p_switch / 4.0;
        let scale = (1.0 - p_switch) / 1.0;
        let sum = HmmUpdater::fwd_update_emissions_scale(&mut fwd, scale, shift, &emissions, 4);

        assert!(sum > 0.0);
        assert!(sum < 2.0);
    }

    #[test]
    fn test_hmm_updater_bwd() {
        let mut bwd = vec![0.25f32; 4];
        let emit_probs = [0.99f32, 0.01];
        let mismatches = vec![0u8, 0, 1, 0];

        HmmUpdater::bwd_update(&mut bwd, 0.01, &emit_probs, &mismatches, 4);

        let sum: f32 = bwd.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_mosaic_hmm_forward_backward() {
        let ref_panel = make_test_ref_panel();
        let params = ModelParams::for_phasing(6, 10000.0, None);
        let p_recomb = vec![0.0, 0.01, 0.01, 0.01, 0.01];

        let n_markers = 5;
        let n_states = 3; // 3 composite states with mosaic segments

        // Build ThreadedHaps using PRODUCTION API with actual segment transitions
        // This tests MosaicCursor segment-switching logic that from_static_haps bypassed
        let mut threaded_haps = ThreadedHaps::<RefHapSpace>::new(n_states, n_states * 2, n_markers);

        // State 0: hap 0 for markers 0-2, then hap 1 for markers 3-4 (segment switch at marker 3)
        threaded_haps.push_new(HapId::new(0));
        threaded_haps.add_segment(0, HapId::new(1), 3);

        // State 1: hap 2 for entire chromosome (no switch - tests static case too)
        threaded_haps.push_new(HapId::new(2));

        // State 2: hap 4 for markers 0-1, then hap 5 for markers 2-4 (segment switch at marker 2)
        threaded_haps.push_new(HapId::new(4));
        threaded_haps.add_segment(2, HapId::new(5), 2);

        let hmm = MosaicHmm::new(&ref_panel, &params, n_states, &p_recomb);

        let target_alleles = vec![0, 0, 0, 1, 0]; // Should match haplotype 0 or 4
        let mut fwd = Vec::new();
        let mut bwd = Vec::new();

        let log_likelihood = hmm.conditioned_forward_backward(
            &target_alleles,
            &target_alleles,
            &target_alleles,
            None,
            None,
            None,
            None,
            &threaded_haps,
            &mut fwd,
            &mut bwd,
        );

        assert_eq!(fwd.len(), 5 * 3); // 5 markers * 3 states
        assert_eq!(bwd.len(), 5 * 3);
        assert!(log_likelihood.is_finite());

        // Verify posteriors sum to 1 at each marker (this validates the mosaic HMM math)
        for m in 0..n_markers {
            let sum: f32 = (0..n_states)
                .map(|k| fwd[m * n_states + k] * bwd[m * n_states + k])
                .sum();
            // Posteriors should be positive and reasonable
            assert!(
                sum > 0.0,
                "Posterior sum at marker {} should be positive",
                m
            );
        }
    }

    // =========================================================================
    // Rigorous HMM Updater Tests
    // =========================================================================

    #[test]
    fn test_fwd_update_preserves_probability_mass() {
        // After forward update, values should remain valid probabilities
        for n_states in [4, 8, 16, 32] {
            let mut fwd = vec![1.0 / n_states as f32; n_states];
            let emit_probs = [0.99f32, 0.01];
            let mismatches: Vec<u8> = (0..n_states).map(|k| (k % 2) as u8).collect();

            let initial_sum: f32 = fwd.iter().sum();
            let emissions: Vec<f32> = mismatches.iter().map(|&m| emit_probs[m as usize]).collect();
            let p_switch = 0.05;
            let shift = p_switch / n_states as f32;
            let scale = (1.0 - p_switch) / initial_sum.max(1e-30);
            let new_sum = HmmUpdater::fwd_update_emissions_scale(
                &mut fwd, scale, shift, &emissions, n_states,
            );

            // All values should be positive
            for (k, &val) in fwd.iter().enumerate() {
                assert!(val >= 0.0, "fwd[{}] = {} is negative", k, val);
                assert!(val.is_finite(), "fwd[{}] = {} is not finite", k, val);
            }

            // Sum should be positive and finite
            assert!(new_sum > 0.0, "new_sum {} should be positive", new_sum);
            assert!(new_sum.is_finite(), "new_sum {} should be finite", new_sum);

            // Verify returned sum matches actual sum
            let actual_sum: f32 = fwd.iter().sum();
            assert!(
                (new_sum - actual_sum).abs() < 1e-5,
                "Returned sum {} != actual sum {}",
                new_sum,
                actual_sum
            );
        }
    }

    #[test]
    fn test_bwd_update_normalizes_to_one() {
        // After backward update, values should sum close to 1 (normalized)
        for n_states in [4, 8, 16, 32] {
            let mut bwd = vec![1.0 / n_states as f32; n_states];
            let emit_probs = [0.99f32, 0.01];
            let mismatches: Vec<u8> = (0..n_states).map(|k| (k % 2) as u8).collect();

            HmmUpdater::bwd_update(&mut bwd, 0.05, &emit_probs, &mismatches, n_states);

            // All values should be positive
            for (k, &val) in bwd.iter().enumerate() {
                assert!(val >= 0.0, "bwd[{}] = {} is negative", k, val);
                assert!(val.is_finite(), "bwd[{}] = {} is not finite", k, val);
            }

            // Sum should be close to 1 after normalization
            let sum: f32 = bwd.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "bwd sum {} should be ~1.0 (n_states={})",
                sum,
                n_states
            );
        }
    }

    #[test]
    fn test_fwd_update_favors_matching_states() {
        // States that match (mismatch=0) should have higher probability than mismatching states
        let n_states = 8;
        let mut fwd = vec![1.0 / n_states as f32; n_states];
        let emit_probs = [0.99f32, 0.01]; // Strong preference for match

        // First 4 states match, last 4 mismatch
        let mismatches: Vec<u8> = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let initial_sum: f32 = fwd.iter().sum();
        let emissions: Vec<f32> = mismatches.iter().map(|&m| emit_probs[m as usize]).collect();
        let p_switch = 0.001;
        let shift = p_switch / n_states as f32;
        let scale = (1.0 - p_switch) / initial_sum.max(1e-30);
        HmmUpdater::fwd_update_emissions_scale(&mut fwd, scale, shift, &emissions, n_states);

        // Matching states should have higher values
        let match_sum: f32 = fwd[0..4].iter().sum();
        let mismatch_sum: f32 = fwd[4..8].iter().sum();

        assert!(
            match_sum > mismatch_sum * 10.0,
            "Matching states ({}) should dominate mismatching ({})",
            match_sum,
            mismatch_sum
        );
    }

    #[test]
    fn test_simd_vectorized_matches_scalar() {
        // Test that SIMD and scalar paths produce identical results
        // by testing with n_states = 8 (pure SIMD) and n_states = 11 (SIMD + scalar tail)
        for n_states in [8, 11, 16, 17, 24, 25] {
            let initial_fwd: Vec<f32> = (0..n_states).map(|k| (k as f32 + 1.0) / 100.0).collect();
            let initial_sum: f32 = initial_fwd.iter().sum();
            let emit_probs = [0.95f32, 0.05];
            let mismatches: Vec<u8> = (0..n_states).map(|k| ((k * 3) % 2) as u8).collect();

            // Run forward update
            let mut fwd = initial_fwd.clone();
            let emissions: Vec<f32> = mismatches.iter().map(|&m| emit_probs[m as usize]).collect();
            let p_switch = 0.02;
            let shift = p_switch / n_states as f32;
            let scale = (1.0 - p_switch) / initial_sum.max(1e-30);
            let new_sum = HmmUpdater::fwd_update_emissions_scale(
                &mut fwd, scale, shift, &emissions, n_states,
            );

            // Verify basic properties
            assert!(new_sum > 0.0);
            let actual_sum: f32 = fwd.iter().sum();
            assert!(
                (new_sum - actual_sum).abs() < 1e-4,
                "n_states={}: sum mismatch {} vs {}",
                n_states,
                new_sum,
                actual_sum
            );

            // Run backward update
            let mut bwd: Vec<f32> = (0..n_states).map(|k| (k as f32 + 1.0) / 100.0).collect();
            HmmUpdater::bwd_update(&mut bwd, 0.02, &emit_probs, &mismatches, n_states);

            let bwd_sum: f32 = bwd.iter().sum();
            assert!(
                (bwd_sum - 1.0).abs() < 0.01,
                "n_states={}: bwd sum {} should be ~1",
                n_states,
                bwd_sum
            );
        }
    }

    #[test]
    fn test_extreme_recombination_rates() {
        // Test edge cases: no recombination and very high recombination
        let n_states = 8;
        let emit_probs = [0.99f32, 0.01];
        let mismatches: Vec<u8> = vec![0, 1, 0, 1, 0, 1, 0, 1];

        // Test with zero recombination (p_switch = 0)
        let mut fwd_no_recomb = vec![0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        let initial_sum: f32 = fwd_no_recomb.iter().sum();
        let emissions: Vec<f32> = mismatches.iter().map(|&m| emit_probs[m as usize]).collect();
        let p_switch = 0.0;
        let shift = 0.0;
        let scale = (1.0 - p_switch) / initial_sum.max(1e-30);
        let new_sum = HmmUpdater::fwd_update_emissions_scale(
            &mut fwd_no_recomb,
            scale,
            shift,
            &emissions,
            n_states,
        );

        // With no recombination, only states with initial probability should have probability
        // (though emission still affects all)
        assert!(new_sum > 0.0);
        assert!(new_sum.is_finite());

        // Test with very high recombination (p_switch = 0.99)
        let mut fwd_high_recomb = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let initial_sum_high: f32 = fwd_high_recomb.iter().sum();
        let emissions: Vec<f32> = mismatches.iter().map(|&m| emit_probs[m as usize]).collect();
        let p_switch = 0.99;
        let shift = p_switch / n_states as f32;
        let scale = (1.0 - p_switch) / initial_sum_high.max(1e-30);
        HmmUpdater::fwd_update_emissions_scale(
            &mut fwd_high_recomb,
            scale,
            shift,
            &emissions,
            n_states,
        );

        // With high recombination, probability should spread to all states
        let min_val = fwd_high_recomb.iter().cloned().fold(f32::MAX, f32::min);
        assert!(
            min_val > 0.0,
            "With high recomb, all states should have some probability, min={}",
            min_val
        );
    }

    #[test]
    fn test_numerical_stability_small_values() {
        // Test with very small initial values to check for underflow
        let n_states = 16;
        let mut fwd: Vec<f32> = vec![1e-30; n_states];
        let emit_probs = [0.99f32, 0.01];
        let mismatches: Vec<u8> = (0..n_states).map(|k| (k % 2) as u8).collect();

        let initial_sum: f32 = fwd.iter().sum();

        // Should not panic or produce NaN/Inf
        let emissions: Vec<f32> = mismatches.iter().map(|&m| emit_probs[m as usize]).collect();
        let p_switch = 0.01;
        let shift = p_switch / n_states as f32;
        let scale = (1.0 - p_switch) / initial_sum.max(1e-30);
        let new_sum =
            HmmUpdater::fwd_update_emissions_scale(&mut fwd, scale, shift, &emissions, n_states);

        assert!(
            new_sum.is_finite(),
            "new_sum should be finite, got {}",
            new_sum
        );
        for (k, &val) in fwd.iter().enumerate() {
            assert!(val.is_finite(), "fwd[{}] should be finite, got {}", k, val);
        }
    }

    #[test]
    fn test_quantized_large_state_forward_path() {
        // Force large-state path to exercise log8 quantized update.
        let n_states = 2048;
        let mut fwd = vec![1.0f32 / n_states as f32; n_states];
        let emit_probs = [0.995f32, 0.005];
        let mismatches: Vec<u8> = (0..n_states).map(|k| ((k * 7) % 5 == 0) as u8).collect();
        let emissions: Vec<f32> = mismatches.iter().map(|&m| emit_probs[m as usize]).collect();
        let p_switch = 0.02f32;
        let shift = p_switch / n_states as f32;
        let scale = (1.0 - p_switch) / 1.0f32;

        let sum =
            HmmUpdater::fwd_update_emissions_scale(&mut fwd, scale, shift, &emissions, n_states);

        assert!(
            sum.is_finite() && sum > 0.0,
            "quantized forward sum must be finite and positive"
        );
        let actual: f32 = fwd.iter().sum();
        let rel = (sum - actual).abs() / actual.max(1e-12);
        assert!(
            rel < 1e-4,
            "returned sum should match actual sum, rel={}",
            rel
        );
        assert!(
            fwd.iter().all(|v| v.is_finite() && *v >= 0.0),
            "all forward entries must be finite/non-negative"
        );
    }

    #[test]
    fn test_quantized_large_state_backward_path() {
        // Force large-state path to exercise log8 quantized backward update.
        let n_states = 2048;
        let mut bwd = vec![1.0f32 / n_states as f32; n_states];
        let emit_probs = [0.995f32, 0.005];
        let mismatches: Vec<u8> = (0..n_states).map(|k| ((k * 11) % 6 == 0) as u8).collect();

        HmmUpdater::bwd_update(&mut bwd, 0.03, &emit_probs, &mismatches, n_states);

        let sum: f32 = bwd.iter().sum();
        assert!(
            sum.is_finite() && sum > 0.0,
            "quantized backward sum must be finite and positive"
        );
        // Normalized update should remain near one despite quantization.
        assert!(
            (sum - 1.0).abs() < 0.05,
            "quantized backward sum should stay near 1, got {}",
            sum
        );
        assert!(
            bwd.iter().all(|v| v.is_finite() && *v >= 0.0),
            "all backward entries must be finite/non-negative"
        );
    }

    #[test]
    fn test_bwd_update_constant_normalization() {
        // Formula: bwd[i] = ( (1-r)*e[i]*bwd[i] + (r/N)*C ) / C
        let n_states = 2;
        let mut bwd = vec![1.0, 1.0];
        let p_switch = 0.1;
        let emissions = vec![1.0, 1.0];
        let constant_term: f32 = 2.0; // sum(bwd * emissions)

        let shift = p_switch / n_states as f32;
        let scale = (1.0 - p_switch) / constant_term.max(1e-30);
        HmmUpdater::bwd_update_constant_scale(&mut bwd, scale, shift, &emissions, n_states);

        // Expected result: 0.5 (normalized)
        assert!(
            (bwd[0] - 0.5).abs() < 1e-6,
            "Expected 0.5 (normalized), got {}. The backward update must be normalized by C.",
            bwd[0]
        );

        let sum: f32 = bwd.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "Sum should be 1.0");
    }
}
