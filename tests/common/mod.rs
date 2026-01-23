use noodles::bgzf::io as bgzf_io;
use noodles::vcf as noodles_vcf;
use noodles_vcf::variant::record::samples::series::value::Array;
use noodles_vcf::variant::record::samples::series::Value;
use noodles_vcf::variant::record::samples::Sample; // Added import
use std::fs::File;
use std::path::Path;

pub fn read_single_sample_ds(path: &Path) -> Vec<f64> {
    let file = File::open(path).expect("open output VCF");
    let decoder = bgzf_io::Reader::new(file);
    let mut reader = noodles_vcf::io::Reader::new(decoder);

    let header = reader.read_header().expect("read VCF header");
    assert!(header.formats().contains_key("DS"), "DS missing from output");

    let mut ds = Vec::new();

    for result in reader.records() {
        let record = result.expect("read VCF record");
        let samples = record.samples();
        let sample0 = samples.get_index(0).expect("sample 0");

        let val = sample0
            .get(&header, "DS")
            .transpose()
            .ok()
            .flatten()
            .flatten();

        let parsed = match val {
            Some(Value::Float(f)) => f as f64,
            Some(Value::String(s)) => s.parse::<f64>().expect("parse DS"),
            Some(Value::Array(Array::Float(values))) => {
                // DS is Number=A (one per ALT) or 1?
                // The output writer says `##FORMAT=<ID=DS,Number=A,Type=Float,...`
                // But for biallelic sites there is 1 ALT, so 1 DS.
                // The iterator yields Result<Option<f32>>
                match values.iter().next() {
                    Some(Ok(Some(v))) => v as f64,
                    Some(Ok(None)) => panic!("DS value is None inside array"),
                    Some(Err(e)) => panic!("Error reading DS array: {:?}", e),
                    None => panic!("Empty DS array"),
                }
            }
            Some(other) => panic!("Unexpected DS value: {other:?}"),
            None => panic!("Missing DS"),
        };

        ds.push(parsed);
    }

    ds
}
