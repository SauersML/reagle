//! # Phase State Marker Types
//!
//! Zero-sized marker types for compile-time phase state tracking.
//! Uses the Type State pattern to enforce pipeline correctness at compile time.
//!
//! # Example
//!
//! ```
//! use reagle::data::storage::phase_state::{Phased, Unphased, PhaseState};
//!
//! // Phase state types are zero-sized markers
//! assert_eq!(std::mem::size_of::<Phased>(), 0);
//! assert_eq!(std::mem::size_of::<Unphased>(), 0);
//! ```

use std::fmt::Debug;

/// Marker trait for phase states.
///
/// This trait is sealed and cannot be implemented outside this module,
/// ensuring only `Phased` and `Unphased` can be used as state parameters.
pub trait PhaseState: Copy + Clone + Default + Debug + private::Sealed {}

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

impl PhaseState for Phased {}

/// Type state marker: Data is unphased (genotype-level only).
///
/// Unphased data only knows the pair of alleles at each site,
/// not which allele came from which parent.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct Unphased;

impl PhaseState for Unphased {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_sized() {
        // Verify these are zero-sized types
        assert_eq!(std::mem::size_of::<Phased>(), 0);
        assert_eq!(std::mem::size_of::<Unphased>(), 0);
    }
}
