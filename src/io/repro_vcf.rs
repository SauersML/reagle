#[cfg(test)]
mod reproduction_test {
    use crate::data::marker::{Allele, Marker, Markers, Nucleotide};
    use crate::data::haplotype::Samples;
    use crate::io::vcf::{VcfWriter, ImputationQuality};
    use crate::data::storage::AlleleCode;
    use std::sync::Arc;

    #[test]
    fn test_vcf_writer_handles_u8_max_as_missing() {
        assert_eq!(AlleleCode::MISSING.raw(), 255, "MISSING should be 255");
        
        let tmp = tempfile::NamedTempFile::new().expect("tmp vcf");
        let samples = Arc::new(Samples::from_ids(vec!["s1".to_string()]));
        let mut writer = VcfWriter::create(tmp.path(), samples.clone()).expect("writer");
        
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");
        let marker = Marker::new(
            crate::data::ChromIdx::new(0),
            100,
            None,
            Allele::Base(Nucleotide::A),
            vec![Allele::Base(Nucleotide::C)],
        );
        markers.push(marker);
        
        writer.write_header(&markers).unwrap();
        
        let get_dosage = |_: usize, _: usize| 0.0;
        let get_best_gt = |_: usize, _: usize| (255, 255);
        let quality = ImputationQuality::new(&[2]);
        
        writer.write_imputed_streaming(
            &markers,
            get_dosage,
            get_best_gt,
            None::<fn(usize, usize) -> (crate::pipelines::imputation::AllelePosteriors, crate::pipelines::imputation::AllelePosteriors)>,
            None::<fn(usize, usize) -> Option<Vec<f32>>>,
            &quality,
            0,
            1,
            false,
            false,
            None
        ).unwrap();
        
        writer.flush().unwrap();
        
        let content = std::fs::read_to_string(tmp.path()).unwrap();
        println!("VCF Content:\n{}", content);
        assert!(content.contains("\t.|."), "Output should contain missing genotype .|., got something else");
    }
}
