use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Instant;

#[derive(Debug, Clone)]
pub struct FastMetrics {
    pub sites_compared: usize,
    pub genotypes_compared: usize,
    pub r_squared: Option<f64>,
    pub iqs: Option<f64>,
    pub iqs_sites: usize,
    pub hellinger_score: Option<f64>,
    pub switch_error_rate: Option<f64>,
    pub switch_errors: usize,
    pub switch_opportunities: usize,
    pub phase_concordance: Option<f64>,
    pub phase_concordant: usize,
    pub phase_total: usize,
}

fn running_r2(n: f64, sum_t: f64, sum_i: f64, sum_tt: f64, sum_ii: f64, sum_ti: f64) -> Option<f64> {
    if n <= 1.0 {
        return None;
    }
    let num = n * sum_ti - sum_t * sum_i;
    let den_t = n * sum_tt - sum_t * sum_t;
    let den_i = n * sum_ii - sum_i * sum_i;
    if den_t > 0.0 && den_i > 0.0 {
        Some((num / (den_t * den_i).sqrt()).powi(2))
    } else {
        None
    }
}

fn dosage_from_gt(gt: &str) -> Option<f64> {
    let norm = gt.replace('|', "/");
    if norm == "./." || norm == ".|." {
        return None;
    }
    let parts: Vec<&str> = norm.split('/').collect();
    if parts.len() != 2 {
        return None;
    }
    let a0: i32 = parts[0].parse().ok()?;
    let a1: i32 = parts[1].parse().ok()?;
    Some((a0 + a1) as f64)
}

fn gt_class_unphased(gt: &str) -> Option<usize> {
    let norm = gt.replace('|', "/");
    match norm.as_str() {
        "0/0" => Some(0),
        "0/1" | "1/0" => Some(1),
        "1/1" => Some(2),
        _ => None,
    }
}

fn parse_het_phase(gt: &str) -> Option<bool> {
    let mut it = gt.split('|');
    let a = it.next()?.trim();
    let b = it.next()?.trim();
    if it.next().is_some() {
        return None;
    }
    if a == "0" && b == "1" {
        Some(true)
    } else if a == "1" && b == "0" {
        Some(false)
    } else {
        None
    }
}

fn next_vcf_data_line<R: BufRead>(reader: &mut R, line: &mut String) -> usize {
    loop {
        line.clear();
        let n = reader.read_line(line).expect("read VCF line");
        if n == 0 {
            return 0;
        }
        if !line.starts_with('#') {
            return n;
        }
    }
}

fn format_field_idx(format_col: &str, needle: &str) -> Option<usize> {
    format_col.split(':').position(|f| f == needle)
}

fn sample_field<'a>(sample_col: &'a str, idx: Option<usize>) -> &'a str {
    idx.and_then(|i| sample_col.split(':').nth(i)).unwrap_or(".")
}

fn parse_gp_probs(raw: &str) -> Option<[f64; 3]> {
    if raw == "." || raw.is_empty() {
        return None;
    }
    let mut it = raw.split(',');
    let a = it.next()?.parse::<f64>().ok()?;
    let b = it.next()?.parse::<f64>().ok()?;
    let c = it.next()?.parse::<f64>().ok()?;
    if it.next().is_some() {
        return None;
    }
    Some([a, b, c])
}

pub fn compute_fast_metrics(truth_vcf: &Path, imputed_vcf: &Path) -> FastMetrics {
    let mut truth_cmd = Command::new("gzip");
    truth_cmd
        .args(["-dc"])
        .arg(truth_vcf)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let mut truth_proc = truth_cmd.spawn().expect("spawn gzip (truth)");

    let mut imp_cmd = Command::new("gzip");
    imp_cmd
        .args(["-dc"])
        .arg(imputed_vcf)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let mut imp_proc = imp_cmd.spawn().expect("spawn gzip (imputed)");

    let truth_out = truth_proc.stdout.take().expect("truth stdout");
    let imp_out = imp_proc.stdout.take().expect("imputed stdout");
    let mut truth_reader = BufReader::new(truth_out);
    let mut imp_reader = BufReader::new(imp_out);

    let mut truth_line = String::new();
    let mut imp_line = String::new();

    let mut sites_compared = 0usize;
    let mut genotypes_compared = 0usize;

    let mut r2_n = 0.0f64;
    let mut r2_sum_t = 0.0f64;
    let mut r2_sum_i = 0.0f64;
    let mut r2_sum_tt = 0.0f64;
    let mut r2_sum_ii = 0.0f64;
    let mut r2_sum_ti = 0.0f64;

    let mut hellinger_sum = 0.0f64;
    let mut hellinger_n = 0usize;
    let mut iqs_sum = 0.0f64;
    let mut iqs_sites = 0usize;

    let mut prev_orientation: Vec<Option<bool>> = Vec::new();
    let mut phase_match_counts: Vec<usize> = Vec::new();
    let mut phase_mismatch_counts: Vec<usize> = Vec::new();
    let mut phase_total = 0usize;
    let mut switch_errors = 0usize;
    let mut switch_opportunities = 0usize;
    let started = Instant::now();
    let progress_every = std::env::var("REFERENCE_METRICS_PROGRESS_EVERY")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(50_000);

    loop {
        let tr_n = next_vcf_data_line(&mut truth_reader, &mut truth_line);
        let im_n = next_vcf_data_line(&mut imp_reader, &mut imp_line);
        if tr_n == 0 && im_n == 0 {
            break;
        }
        assert!(
            tr_n > 0 && im_n > 0,
            "truth/imputed VCF lengths differ while computing metrics"
        );

        let t_cols: Vec<&str> = truth_line.trim_end().split('\t').collect();
        let i_cols: Vec<&str> = imp_line.trim_end().split('\t').collect();
        assert!(t_cols.len() >= 10, "malformed truth VCF row");
        assert!(i_cols.len() >= 10, "malformed imputed VCF row");
        assert_eq!(t_cols[0], i_cols[0], "CHROM mismatch between truth/imputed");
        assert_eq!(t_cols[1], i_cols[1], "POS mismatch between truth/imputed");

        let t_gt_idx = format_field_idx(t_cols[8], "GT");
        let i_gt_idx = format_field_idx(i_cols[8], "GT");
        let i_ds_idx = format_field_idx(i_cols[8], "DS");
        let i_gp_idx = format_field_idx(i_cols[8], "GP");
        assert!(t_gt_idx.is_some(), "truth row missing GT in FORMAT");
        assert!(i_gt_idx.is_some(), "imputed row missing GT in FORMAT");

        let n_samples_truth = t_cols.len() - 9;
        let n_samples_imp = i_cols.len() - 9;
        assert_eq!(
            n_samples_truth, n_samples_imp,
            "truth/imputed sample counts differ"
        );

        if prev_orientation.is_empty() {
            prev_orientation.resize(n_samples_truth, None);
            phase_match_counts.resize(n_samples_truth, 0);
            phase_mismatch_counts.resize(n_samples_truth, 0);
        }

        sites_compared += 1;
        let mut site_truth_counts = [0.0f64; 3];
        let mut site_imputed_marginals = [0.0f64; 3];
        let mut site_correct_mass = 0.0f64;
        let mut site_n = 0.0f64;
        for sample_idx in 0..n_samples_truth {
            let t_sample = t_cols[9 + sample_idx];
            let i_sample = i_cols[9 + sample_idx];
            let t_gt = sample_field(t_sample, t_gt_idx);
            let i_gt = sample_field(i_sample, i_gt_idx);
            let i_ds = sample_field(i_sample, i_ds_idx);
            let i_gp = sample_field(i_sample, i_gp_idx);

            let t_ds = dosage_from_gt(t_gt);
            let i_ds_val = i_ds.parse::<f64>().ok();

            if let (Some(t), Some(i)) = (t_ds, i_ds_val) {
                genotypes_compared += 1;
                r2_n += 1.0;
                r2_sum_t += t;
                r2_sum_i += i;
                r2_sum_tt += t * t;
                r2_sum_ii += i * i;
                r2_sum_ti += t * i;
            }

            if let Some(t_class) = gt_class_unphased(t_gt) {
                if let Some(probs) = parse_gp_probs(i_gp) {
                    site_truth_counts[t_class] += 1.0;
                    site_imputed_marginals[0] += probs[0];
                    site_imputed_marginals[1] += probs[1];
                    site_imputed_marginals[2] += probs[2];
                    site_correct_mass += probs[t_class].clamp(0.0, 1.0);
                    site_n += 1.0;
                } else if let Some(i_class) = gt_class_unphased(i_gt) {
                    site_truth_counts[t_class] += 1.0;
                    site_imputed_marginals[i_class] += 1.0;
                    if i_class == t_class {
                        site_correct_mass += 1.0;
                    }
                    site_n += 1.0;
                }
            }

            if let Some(t_class) = gt_class_unphased(t_gt) {
                if let Some(probs) = parse_gp_probs(i_gp) {
                    let p_true = probs[t_class].clamp(0.0, 1.0);
                    let bc = p_true.sqrt();
                    let h = (1.0 - bc).max(0.0).sqrt();
                    hellinger_sum += h;
                    hellinger_n += 1;
                }
            }

            let t_ori = parse_het_phase(t_gt);
            let i_ori = parse_het_phase(i_gt);
            if let (Some(to), Some(io)) = (t_ori, i_ori) {
                let orientation = to == io;
                phase_total += 1;
                if orientation {
                    phase_match_counts[sample_idx] += 1;
                } else {
                    phase_mismatch_counts[sample_idx] += 1;
                }
                if let Some(prev) = prev_orientation[sample_idx] {
                    switch_opportunities += 1;
                    if prev != orientation {
                        switch_errors += 1;
                    }
                }
                prev_orientation[sample_idx] = Some(orientation);
            }
        }

        if site_n > 0.0 {
            let po = site_correct_mass / site_n;
            let mut pe = 0.0f64;
            for k in 0..3 {
                pe += (site_truth_counts[k] / site_n) * (site_imputed_marginals[k] / site_n);
            }
            let denom = 1.0 - pe;
            if denom > 1e-12 {
                iqs_sum += (po - pe) / denom;
                iqs_sites += 1;
            }
        }

        if progress_every > 0 && (sites_compared % progress_every == 0) {
            let r2_now = running_r2(r2_n, r2_sum_t, r2_sum_i, r2_sum_tt, r2_sum_ii, r2_sum_ti);
            eprintln!(
                "[fast-metrics] sites={} genotypes={} r2_now={:?} elapsed_sec={:.1}",
                sites_compared,
                genotypes_compared,
                r2_now,
                started.elapsed().as_secs_f64()
            );
        }
    }

    let truth_status = truth_proc.wait().expect("wait truth gzip");
    assert!(truth_status.success(), "truth VCF decompression failed");
    let imp_status = imp_proc.wait().expect("wait imputed gzip");
    assert!(imp_status.success(), "imputed VCF decompression failed");

    let r_squared = running_r2(r2_n, r2_sum_t, r2_sum_i, r2_sum_tt, r2_sum_ii, r2_sum_ti);

    let hellinger_score = if hellinger_n > 0 {
        Some(hellinger_sum / hellinger_n as f64)
    } else {
        None
    };

    let iqs = if iqs_sites > 0 {
        Some(iqs_sum / iqs_sites as f64)
    } else {
        None
    };

    let mut phase_concordant = 0usize;
    for idx in 0..phase_match_counts.len() {
        phase_concordant += phase_match_counts[idx].max(phase_mismatch_counts[idx]);
    }

    let phase_concordance = if phase_total > 0 {
        Some(phase_concordant as f64 / phase_total as f64)
    } else {
        None
    };

    let switch_error_rate = if switch_opportunities > 0 {
        Some(switch_errors as f64 / switch_opportunities as f64)
    } else {
        None
    };

    FastMetrics {
        sites_compared,
        genotypes_compared,
        r_squared,
        iqs,
        iqs_sites,
        hellinger_score,
        switch_error_rate,
        switch_errors,
        switch_opportunities,
        phase_concordance,
        phase_concordant,
        phase_total,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gt_class_unphased_is_hap_label_invariant() {
        assert_eq!(gt_class_unphased("0|1"), Some(1));
        assert_eq!(gt_class_unphased("1|0"), Some(1));
        assert_eq!(gt_class_unphased("0/1"), Some(1));
        assert_eq!(gt_class_unphased("1/0"), Some(1));
    }

    #[test]
    fn iqs_fallback_path_is_hap_label_invariant() {
        // When GP is missing, IQS falls back to GT class. The heterozygote
        // class must remain identical under hap-label swap.
        let t_class = gt_class_unphased("0|1").unwrap();
        let i_class_a = gt_class_unphased("0|1").unwrap();
        let i_class_b = gt_class_unphased("1|0").unwrap();
        assert_eq!(t_class, 1);
        assert_eq!(i_class_a, 1);
        assert_eq!(i_class_b, 1);
        assert_eq!(usize::from(i_class_a == t_class), usize::from(i_class_b == t_class));
    }
}
