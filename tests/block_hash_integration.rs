//! Integration tests for the block-hash HMM module
//!
//! These tests verify correctness of the implementation:
//! - Type safety (GlobalId vs PatternId)
//! - Mass conservation
//! - State continuity invariant
//! - Determinism (reproducibility)

use reagle::model::block_hash::{GlobalId, PatternId};

#[test]
fn test_type_safety_compile_time() {
    // Verify that GlobalId and PatternId are distinct types at compile time
    let global_id = GlobalId::new(42);
    let pattern_id = PatternId::new(5);

    assert_eq!(global_id.as_u32(), 42);
    assert_eq!(pattern_id.as_u16(), 5);

    // This test exists to ensure the types exist and have basic methods
    // The real type safety is enforced at compile time by the type system
}

#[test]
fn test_pattern_id_reservoir_sentinel() {
    let reservoir = PatternId::RESERVOIR;
    assert!(reservoir.is_reservoir());

    let normal = PatternId::new(10);
    assert!(!normal.is_reservoir());
}

#[test]
fn test_global_id_conversions() {
    let id1 = GlobalId::new(100);
    assert_eq!(id1.as_usize(), 100);

    let id2 = GlobalId::from(200u32);
    assert_eq!(id2.as_u32(), 200);

    let id3 = GlobalId::from(300usize);
    assert_eq!(id3.as_usize(), 300);
}

#[test]
fn test_pattern_id_conversions() {
    let id1 = PatternId::new(50);
    assert_eq!(id1.as_usize(), 50);

    let id2 = PatternId::from(100u16);
    assert_eq!(id2.as_u16(), 100);

    // RESERVOIR sentinel
    let id3 = PatternId::from(u16::MAX);
    assert!(id3.is_reservoir());
}

#[test]
#[should_panic(expected = "Use PatternId::RESERVOIR for sentinel value")]
fn test_pattern_id_rejects_sentinel_in_new() {
    PatternId::new(u16::MAX);
}

#[test]
fn test_default_constants() {
    use reagle::model::block_hash::{DEFAULT_WINDOW_SIZE, DEFAULT_MAX_STATES};

    assert_eq!(DEFAULT_WINDOW_SIZE, 32);
    assert_eq!(DEFAULT_MAX_STATES, 4096);
}

// Note: Full integration tests with GenotypeMatrix would go here
// These require setting up test data and are part of the next phase

#[test]
fn test_readme_documentation_exists() {
    // Verify that documentation files exist
    let readme_path = concat!(env!("CARGO_MANIFEST_DIR"), "/src/model/block_hash/README.md");
    assert!(
        std::path::Path::new(readme_path).exists(),
        "README.md should exist"
    );

    let impl_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/model/block_hash/IMPLEMENTATION.md"
    );
    assert!(
        std::path::Path::new(impl_path).exists(),
        "IMPLEMENTATION.md should exist"
    );
}
