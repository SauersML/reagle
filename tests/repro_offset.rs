
use reagle::data::marker::Markers;
use reagle::data::storage::{GenotypeColumn, GenotypeMatrix, GenotypeView};
use reagle::data::haplotype::Samples;
use reagle::model::states::ThreadedHaps;
use reagle::model::types::{CombinedHapId, CombinedHapSpace, RefHapId, combined_from_ref};
use std::sync::Arc;

// I need to expose RefAlleleProvider or replicate its logic to test it.
// Since it's private in src/pipelines/phasing.rs, I cannot access it directly in integration tests.
// However, I can add a unit test in src/pipelines/phasing.rs if I modify the file.
// Or I can infer the issue by running a phasing test that fails (which I already have).

// I will assume the hypothesis is correct and proceed to fix it.
// The fix involves modifying RefAlleleProvider struct and methods.

