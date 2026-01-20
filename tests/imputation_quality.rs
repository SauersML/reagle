use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use noodles::vcf;
use noodles::vcf::variant::record::samples::series::Value;
use noodles::vcf::variant::record::samples::series::value::Array; 
use noodles::vcf::variant::record::samples::Sample;

#[derive(Debug, Default, serde::Serialize)]
struct ImputationMetrics {
    n_sites: usize,
    overall_sen: f64,
    overall_concordance: f64,
    r2_aggregate: f64,
    sen_distribution: Vec<f64>,
    calibration: CalibrationMetrics,
    accuracy_by_maf: MafMetrics,
}

#[derive(Debug, Default, serde::Serialize)]
struct CalibrationMetrics {
    bins: Vec<f64>,
    observed_frequencies: Vec<f64>,
}

#[derive(Debug, Default, serde::Serialize)]
struct MafMetrics {
    bins: Vec<f64>,
    accuracy: Vec<f64>,
}

struct RecordData {
    gt_dose: f64,
    ds: f64,
    gp: f64,
}

fn parse_dosage(field: Option<Value>) -> Option<f64> {
    match field {
        Some(Value::Float(f)) => Some(f as f64),
        Some(Value::String(s)) => s.parse::<f64>().ok(),
        _ => None,
    }
}

fn parse_genotype_dose(field: Option<Value>) -> Option<f64> {
    match field {
        Some(Value::String(s)) => {
            let alleles: Vec<&str> = s.split(['|', '/']).collect();
            if alleles.len() != 2 { return None; }
            let a1 = alleles[0].parse::<u8>().ok()?;
            let a2 = alleles[1].parse::<u8>().ok()?;
            Some((a1 + a2) as f64)
        }
        _ => None,
    }
}

fn parse_gp_max(field: Option<Value>) -> Option<f64> {
    match field {
        Some(Value::Array(Array::Float(vals))) => {
             vals.iter()
                 .filter_map(|v| v.ok().flatten())
                 .map(|f| f as f64)
                 .max_by(|a, b| a.partial_cmp(b).unwrap())
        }
        _ => None,
    }
}

fn run_metrics(truth_path: &Path, imputed_path: &Path, output_json: &Path) {
    let mut truth_reader = vcf::io::Reader::new(BufReader::new(File::open(truth_path).expect("Open Truth")));
    let truth_header = truth_reader.read_header().expect("Read Truth Header");

    let mut imp_reader = vcf::io::Reader::new(BufReader::new(File::open(imputed_path).expect("Open Imputed")));
    let imp_header = imp_reader.read_header().expect("Read Imputed Header");

    let mut metrics_data: Vec<RecordData> = Vec::new();
    
    let mut truth_iter = truth_reader.records().peekable();
    let mut imp_iter = imp_reader.records().peekable();
    
    let mut n_processed = 0;
    
    while let (Some(Ok(t)), Some(Ok(i))) = (truth_iter.peek(), imp_iter.peek()) {
        let t_pos = t.variant_start().expect("pos").expect("pos");
        let i_pos = i.variant_start().expect("pos").expect("pos");
        
        if t_pos < i_pos {
            truth_iter.next();
            continue;
        }
        if i_pos < t_pos {
            imp_iter.next();
            continue;
        }
        
        let t = truth_iter.next().unwrap().unwrap();
        let i = imp_iter.next().unwrap().unwrap();
        
        let t_samples = t.samples();
        let i_samples = i.samples();
        
        let t_gt = t_samples.get_index(0);
        let i_gt = i_samples.get_index(0);
        
        if let (Some(tgt), Some(igt)) = (t_gt, i_gt) {
            let truth_dose = parse_genotype_dose(tgt.get(&truth_header, "GT").transpose().ok().flatten().flatten());
            
            let imp_dose = parse_dosage(igt.get(&imp_header, "DS").transpose().ok().flatten().flatten())
                .or_else(|| parse_genotype_dose(igt.get(&imp_header, "GT").transpose().ok().flatten().flatten()));

            if let (Some(td), Some(id)) = (truth_dose, imp_dose) {
                let gp = parse_gp_max(igt.get(&imp_header, "GP").transpose().ok().flatten().flatten()).unwrap_or(1.0); 

                metrics_data.push(RecordData { gt_dose: td, ds: id, gp });
                n_processed += 1;
            }
        }
    }
    
    let mut sen_sum = 0.0;
    let mut conc_sum = 0.0;
    let mut sen_dist = Vec::new();
    
    let n_bins = 10;
    let mut cal_bins: Vec<Vec<f64>> = vec![vec![]; n_bins];
    
    for d in &metrics_data {
        let sen = 1.0 - (d.gt_dose - d.ds).powi(2) / 4.0;
        sen_sum += sen;
        sen_dist.push(sen);
        
        let hard_call = d.ds.round();
        if (hard_call - d.gt_dose).abs() < 0.1 {
            conc_sum += 1.0;
        }
        
        let bin_idx = ((d.gp * n_bins as f64).floor() as usize).min(n_bins - 1);
        cal_bins[bin_idx].push(if (d.ds.round() - d.gt_dose).abs() < 0.1 { 1.0 } else { 0.0 });
    }
    
    let n = metrics_data.len().max(1) as f64;
    
    let mut cal_x = Vec::new();
    let mut cal_y = Vec::new();
    for (i, bin) in cal_bins.iter().enumerate() {
        if !bin.is_empty() {
            cal_x.push((i as f64 + 0.5) / n_bins as f64);
            let correct = bin.iter().sum::<f64>();
            cal_y.push(correct / bin.len() as f64);
        }
    }
    
    let metrics = ImputationMetrics {
        n_sites: n_processed,
        overall_sen: sen_sum / n,
        overall_concordance: conc_sum / n,
        r2_aggregate: 0.0,
        sen_distribution: sen_dist,
        calibration: CalibrationMetrics {
            bins: cal_x,
            observed_frequencies: cal_y,
        },
        accuracy_by_maf: MafMetrics::default(),
    };
    
    let f = File::create(output_json).expect("Create JSON");
    serde_json::to_writer_pretty(f, &metrics).expect("Write JSON");
}
