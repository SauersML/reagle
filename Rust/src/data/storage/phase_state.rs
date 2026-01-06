//! # Phase State Marker Types
//!
//! Zero-sized marker types for compile-time phase state tracking.
//! Uses the Type State pattern to enforce pipeline correctness at compile time.
//!
//! # Example
//!
//! ```ignore
//! // VCF reader produces unphased data
//! let unphased: GenotypeMatrix<Unphased> = vcf_reader.read()?;
//!
//! // Phasing pipeline transforms Unphased -> Phased
//! let phased: GenotypeMatrix<Phased> = phasing_pipeline.run(unphased)?;
//!
//! // Imputation requires phased input - this is enforced at compile time!
//! imputation_pipeline.run(&phased)?;
//!
//! // This would NOT compile:
//! // imputation_pipeline.run(&unphased);  // Error: expected Phased, found Unphased
//! ```

use std::fmt::Debug;

/// Marker trait for phase states.
///
/// This trait is sealed and cannot be implemented outside this module,
/// ensuring only `Phased` and `Unphased` can be used as state parameters.
pub trait PhaseState: Copy + Clone + Default + Debug + private::Sealed {
    /// Whether data in this state is phased
    const IS_PHASED: bool;
}

mod private {
    pub trait Sealed {}
    impl Sealed for super::Phased {}
    impl Sealed for super::Unphased {}
}

/// Type state marker: Data is phased (haplotype-resolved).
///
/// Phased data has known phase for each heterozygous genotype,
/// meaning we know which allele came from which parent.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct Phased;

impl PhaseState for Phased {
    const IS_PHASED: bool = true;
}

/// Type state marker: Data is unphased (genotype-level only).
///
/// Unphased data only knows the pair of alleles at each site,
/// not which allele came from which parent.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct Unphased;

impl PhaseState for Unphased {
    const IS_PHASED: bool = false;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phase_state_constants() {
        assert!(Phased::IS_PHASED);
        assert!(!Unphased::IS_PHASED);
    }

    #[test]
    fn test_zero_sized() {
        // Verify these are zero-sized types
        assert_eq!(std::mem::size_of::<Phased>(), 0);
        assert_eq!(std::mem::size_of::<Unphased>(), 0);
    }
}
