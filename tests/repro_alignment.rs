use reagle::data::marker::{Marker, Allele, Nucleotide, ChromIdx, compute_allele_mapping};

#[test]
fn test_compute_allele_mapping_diff_chrom_idx() {
    let m1 = Marker::new(
        ChromIdx::new(0),
        100,
        None,
        Allele::Base(Nucleotide::A),
        vec![Allele::Base(Nucleotide::C)],
    );
    let m2 = Marker::new(
        ChromIdx::new(1), // Different index
        100,
        None,
        Allele::Base(Nucleotide::A),
        vec![Allele::Base(Nucleotide::C)],
    );
    
    // Should return Some if checks are removed/relaxed. Currently returns None.
    // The issue is that ChromIdx is context-specific, so different indices might map to the same chromosome name.
    // compute_allele_mapping shouldn't enforce ChromIdx equality.
    let mapping = compute_allele_mapping(&m1, &m2);
    
    // We assert mapping.is_some() to prove the fix works. 
    // Before the fix, this will fail.
    assert!(mapping.is_some(), "Mapping failed due to ChromIdx mismatch");
}

#[test]
fn test_compute_allele_mapping_diff_pos() {
    let m1 = Marker::new(
        ChromIdx::new(0),
        100,
        None,
        Allele::Base(Nucleotide::A),
        vec![Allele::Base(Nucleotide::C)],
    );
    let m2 = Marker::new(
        ChromIdx::new(0),
        200, // Different pos
        None,
        Allele::Base(Nucleotide::A),
        vec![Allele::Base(Nucleotide::C)],
    );
    
    // We might keep pos check or remove it. If removed, this passes. 
    // If we assume caller handles pos matching, then this function shouldn't care.
    // However, for safety, we might expect it to fail if we kept the check.
    // But since we plan to remove the entire check block, this will likely pass (return Some).
    let mapping = compute_allele_mapping(&m1, &m2);
    assert!(mapping.is_some(), "Mapping failed due to Pos mismatch");
}
