use crate::data::marker::MarkerIdx;
use crate::data::storage::GenotypeMatrix;

#[inline]
fn smooth_probs(probs: &mut [f32]) {
    let n = probs.len();
    if n == 0 {
        return;
    }
    let lambda = 1e-4f32;
    let eps = 1e-8f32;
    let n_inv = 1.0f32 / (n as f32);
    let mut sum = 0.0f32;
    for p in probs.iter_mut() {
        *p = (1.0 - lambda) * (*p) + lambda * n_inv;
        if *p < eps {
            *p = eps;
        }
        sum += *p;
    }
    if sum > 0.0 {
        for p in probs.iter_mut() {
            *p /= sum;
        }
    }
}

#[derive(Clone, Copy)]
pub struct PlProvider<'a> {
    pub gt: &'a GenotypeMatrix,
    pub sample: usize,
    pub subset_to_orig: Option<&'a [usize]>,
}

impl<'a> PlProvider<'a> {
    #[inline]
    pub fn pl(&self, local_marker: usize) -> Option<&'a [u16]> {
        let orig = self
            .subset_to_orig
            .and_then(|m| m.get(local_marker).copied())
            .unwrap_or(local_marker);
        self.gt.sample_pl(MarkerIdx::new(orig as u32), self.sample)
    }
}

#[inline]
pub fn infer_n_alleles_from_pl_len(len: usize) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let disc = 8usize.checked_mul(len).and_then(|x| x.checked_add(1))? as f64;
    let n = (((disc.sqrt() - 1.0) / 2.0).floor() as usize).max(1);
    if n * (n + 1) / 2 == len {
        Some(n)
    } else {
        None
    }
}

#[inline]
fn phred_weight(pl: u16) -> f32 {
    (10.0f32).powf(-(pl as f32) * 0.1)
}

pub fn allele_probs_uncond_from_pl(pl: &[u16], probs: &mut Vec<f32>) -> Option<usize> {
    let n_alleles = infer_n_alleles_from_pl_len(pl.len())?;
    probs.clear();
    probs.resize(n_alleles, 0.0);
    let mut sum_w = 0.0f32;
    let mut idx = 0usize;
    for j in 0..n_alleles {
        for i in 0..=j {
            let w = phred_weight(*pl.get(idx).unwrap_or(&0));
            sum_w += w;
            if i == j {
                probs[i] += w;
            } else {
                probs[i] += 0.5 * w;
                probs[j] += 0.5 * w;
            }
            idx += 1;
        }
    }
    if sum_w <= 0.0 {
        return None;
    }
    for p in probs.iter_mut() {
        *p /= sum_w;
    }
    smooth_probs(probs);
    Some(n_alleles)
}

pub fn allele_probs_cond_from_pl(pl: &[u16], partner: u8, probs: &mut Vec<f32>) -> Option<usize> {
    let n_alleles = infer_n_alleles_from_pl_len(pl.len())?;
    let partner = partner as usize;
    if partner >= n_alleles {
        return None;
    }
    probs.clear();
    probs.resize(n_alleles, 0.0);
    let mut sum_w = 0.0f32;
    let mut idx = 0usize;
    for j in 0..n_alleles {
        for i in 0..=j {
            let w = phred_weight(*pl.get(idx).unwrap_or(&0));
            let other = if i == partner && j == partner {
                Some(partner)
            } else if i == partner && j != partner {
                Some(j)
            } else if j == partner && i != partner {
                Some(i)
            } else {
                None
            };
            if let Some(o) = other {
                probs[o] += w;
                sum_w += w;
            }
            idx += 1;
        }
    }
    if sum_w <= 0.0 {
        return None;
    }
    for p in probs.iter_mut() {
        *p /= sum_w;
    }
    smooth_probs(probs);
    Some(n_alleles)
}

#[inline]
pub fn emit_from_allele_probs(ref_al: u8, probs: &[f32], p_no_err: f32, p_err_other: f32) -> f32 {
    if ref_al == 255 {
        return 1.0;
    }
    let a = ref_al as usize;
    let p_true = probs.get(a).copied().unwrap_or(0.0);
    (p_no_err * p_true + p_err_other * (1.0 - p_true)).max(1e-30)
}
