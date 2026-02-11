use crate::data::marker::MarkerIdx;
use crate::data::storage::GenotypeMatrix;

const NEG_INF: f32 = f32::NEG_INFINITY;
const LN_10_DIV_10: f32 = std::f32::consts::LN_10 * 0.1;
const HWE_HET_SCALE: f32 = 2.0;
/// Minimum allele frequency for prior calculation to prevent collapse
const MIN_AF: f32 = 1e-4;

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
    match len {
        0 => return None,
        1 => return Some(1),
        3 => return Some(2),
        _ => {}
    }
    let disc = 8usize.checked_mul(len).and_then(|x| x.checked_add(1))? as f64;
    let n = (((disc.sqrt() - 1.0) / 2.0).floor() as usize).max(1);
    if n * (n + 1) / 2 == len {
        Some(n)
    } else {
        None
    }
}

struct PriorModel<'a> {
    freqs: &'a [f32],
    inv_sum: f32,
}

fn build_prior_model(allele_freqs: Option<&[f32]>, n_alleles: usize) -> Option<PriorModel<'_>> {
    // Floor allele frequency to prevent zero-MAF priors from overwhelming
    // strong PL evidence. With 1e-6, ln(2*1e-6) ≈ -13 which is comparable
    // to PL=50 evidence (-11.5), causing AF collapse for observed hets.
    // 1e-4 keeps the prior weak enough that PL evidence dominates.
    let freqs = allele_freqs?;
    if freqs.len() != n_alleles || freqs.is_empty() {
        return None;
    }

    let mut sum = 0.0f32;
    for &f in freqs {
        if !f.is_finite() || f < 0.0 {
            return None;
        }
        sum += f.max(MIN_AF);
    }

    if sum <= 0.0 || !sum.is_finite() {
        return None;
    }
    Some(PriorModel {
        freqs,
        inv_sum: 1.0 / sum,
    })
}

#[inline]
fn pl_to_log_likelihood(pl: u16) -> f32 {
    -(pl as f32) * LN_10_DIV_10
}

#[inline]
fn genotype_log_prior(i: usize, j: usize, prior_model: &PriorModel<'_>) -> f32 {
    let fi = prior_model.freqs[i].max(MIN_AF) * prior_model.inv_sum;
    let fj = prior_model.freqs[j].max(MIN_AF) * prior_model.inv_sum;
    let prior = if i == j {
        fi * fi
    } else {
        HWE_HET_SCALE * fi * fj
    };
    prior.ln()
}

#[inline]
fn biallelic_log_weights(pl: &[u16], prior_model: Option<&PriorModel<'_>>) -> [f32; 3] {
    let mut w00 = pl_to_log_likelihood(pl[0]);
    let mut w01 = pl_to_log_likelihood(pl[1]);
    let mut w11 = pl_to_log_likelihood(pl[2]);
    if let Some(model) = prior_model {
        w00 += genotype_log_prior(0, 0, model);
        w01 += genotype_log_prior(0, 1, model);
        w11 += genotype_log_prior(1, 1, model);
    }
    [w00, w01, w11]
}

#[inline]
fn softmax3(log_w: [f32; 3]) -> Option<([f32; 3], f32)> {
    let max_w = log_w[0].max(log_w[1]).max(log_w[2]);
    if !max_w.is_finite() {
        return None;
    }
    let mut w = [0.0f32; 3];
    w[0] = if log_w[0].is_finite() {
        (log_w[0] - max_w).exp()
    } else {
        0.0
    };
    w[1] = if log_w[1].is_finite() {
        (log_w[1] - max_w).exp()
    } else {
        0.0
    };
    w[2] = if log_w[2].is_finite() {
        (log_w[2] - max_w).exp()
    } else {
        0.0
    };
    let sum = w[0] + w[1] + w[2];
    if sum > 0.0 { Some((w, sum)) } else { None }
}

pub fn allele_probs_uncond_from_pl(
    pl: &[u16],
    allele_freqs: Option<&[f32]>,
    probs: &mut Vec<f32>,
) -> Option<usize> {
    let n_alleles = infer_n_alleles_from_pl_len(pl.len())?;
    probs.clear();
    probs.resize(n_alleles, 0.0);
    let prior_model = build_prior_model(allele_freqs, n_alleles);
    if n_alleles == 2 {
        let (w, sum_w) = softmax3(biallelic_log_weights(pl, prior_model.as_ref()))?;
        probs[0] = (w[0] + 0.5 * w[1]) / sum_w;
        probs[1] = (w[2] + 0.5 * w[1]) / sum_w;
        return Some(n_alleles);
    }
    let mut max_log_w = NEG_INF;
    let mut idx = 0usize;
    for j in 0..n_alleles {
        for i in 0..=j {
            let mut log_w = pl_to_log_likelihood(pl[idx]);
            if let Some(model) = &prior_model {
                log_w += genotype_log_prior(i, j, model);
            }
            max_log_w = max_log_w.max(log_w);
            idx += 1;
        }
    }
    if !max_log_w.is_finite() {
        return None;
    }

    let mut sum_w = 0.0f32;
    idx = 0;
    for j in 0..n_alleles {
        for i in 0..=j {
            let mut log_w = pl_to_log_likelihood(pl[idx]);
            if let Some(model) = &prior_model {
                log_w += genotype_log_prior(i, j, model);
            }
            if !log_w.is_finite() {
                idx += 1;
                continue;
            }
            let w = (log_w - max_log_w).exp();
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
    Some(n_alleles)
}

pub fn genotype_probs_from_pl(
    pl: &[u16],
    allele_freqs: Option<&[f32]>,
    probs: &mut Vec<f32>,
) -> Option<usize> {
    let n_alleles = infer_n_alleles_from_pl_len(pl.len())?;
    let n_genotypes = n_alleles * (n_alleles + 1) / 2;
    probs.clear();
    probs.resize(n_genotypes, 0.0);

    let prior_model = build_prior_model(allele_freqs, n_alleles);
    if n_alleles == 2 {
        let (w, sum_w) = softmax3(biallelic_log_weights(pl, prior_model.as_ref()))?;
        probs[0] = w[0] / sum_w;
        probs[1] = w[1] / sum_w;
        probs[2] = w[2] / sum_w;
        return Some(n_alleles);
    }

    let mut max_log_w = NEG_INF;
    let mut idx = 0usize;
    for j in 0..n_alleles {
        for i in 0..=j {
            let mut w = pl_to_log_likelihood(pl[idx]);
            if let Some(model) = &prior_model {
                w += genotype_log_prior(i, j, model);
            }
            probs[idx] = w;
            max_log_w = max_log_w.max(w);
            idx += 1;
        }
    }
    if !max_log_w.is_finite() {
        return None;
    }

    let mut sum_w = 0.0f32;
    for p in probs.iter_mut() {
        if p.is_finite() {
            *p = (*p - max_log_w).exp();
            sum_w += *p;
        } else {
            *p = 0.0;
        }
    }
    if sum_w <= 0.0 {
        return None;
    }
    for p in probs.iter_mut() {
        *p /= sum_w;
    }
    Some(n_alleles)
}

pub fn allele_probs_cond_from_pl(
    pl: &[u16],
    partner: u8,
    allele_freqs: Option<&[f32]>,
    probs: &mut Vec<f32>,
) -> Option<usize> {
    let n_alleles = infer_n_alleles_from_pl_len(pl.len())?;
    let partner = partner as usize;
    if partner >= n_alleles {
        return None;
    }
    probs.clear();
    probs.resize(n_alleles, 0.0);
    let prior_model = build_prior_model(allele_freqs, n_alleles);
    if n_alleles == 2 {
        let (w, _) = softmax3(biallelic_log_weights(pl, prior_model.as_ref()))?;
        if partner == 0 {
            let denom = w[0] + w[1];
            if denom <= 0.0 {
                return None;
            }
            probs[0] = w[0] / denom;
            probs[1] = w[1] / denom;
        } else {
            let denom = w[1] + w[2];
            if denom <= 0.0 {
                return None;
            }
            probs[0] = w[1] / denom;
            probs[1] = w[2] / denom;
        }
        return Some(n_alleles);
    }

    let mut max_log_w = NEG_INF;
    let mut idx = 0usize;
    for j in 0..n_alleles {
        for i in 0..=j {
            let mut log_w = pl_to_log_likelihood(pl[idx]);
            if let Some(model) = &prior_model {
                log_w += genotype_log_prior(i, j, model);
            }
            let other = if i == partner && j == partner {
                Some(partner)
            } else if i == partner && j != partner {
                Some(j)
            } else if j == partner && i != partner {
                Some(i)
            } else {
                None
            };
            if other.is_some() {
                max_log_w = max_log_w.max(log_w);
            }
            idx += 1;
        }
    }
    if !max_log_w.is_finite() {
        return None;
    }

    let mut sum_w = 0.0f32;
    idx = 0;
    for j in 0..n_alleles {
        for i in 0..=j {
            let mut log_w = pl_to_log_likelihood(pl[idx]);
            if let Some(model) = &prior_model {
                log_w += genotype_log_prior(i, j, model);
            }
            if !log_w.is_finite() {
                idx += 1;
                continue;
            }
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
                let w = (log_w - max_log_w).exp();
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
