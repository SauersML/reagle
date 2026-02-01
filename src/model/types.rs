//! # Type-Safe Index Types for the Imputation HMM
//!
//! This module defines strict newtype wrappers to prevent index confusion bugs.
//! Using distinct types for Global Haplotype IDs vs. Local Pattern IDs ensures
//! that the compiler prevents accidental misuse at compile time.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::marker::PhantomData;

/// Marker type: reference haplotype space (0..N_ref_haplotypes)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RefHapSpace;


/// Marker type: combined haplotype space (target first, then reference)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CombinedHapSpace;

/// Zero-cost haplotype ID tagged by space.
#[repr(transparent)]
#[derive(PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HapId<Space>(u32, PhantomData<Space>);

impl<Space> Copy for HapId<Space> {}

impl<Space> Clone for HapId<Space> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<Space> HapId<Space> {
    #[inline]
    pub fn new(id: u32) -> Self {
        Self(id, PhantomData)
    }

    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl<Space> fmt::Debug for HapId<Space> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HapId({})", self.0)
    }
}

impl<Space> fmt::Display for HapId<Space> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "H{}", self.0)
    }
}

impl<Space> From<u32> for HapId<Space> {
    fn from(id: u32) -> Self {
        Self::new(id)
    }
}

impl<Space> From<usize> for HapId<Space> {
    fn from(id: usize) -> Self {
        Self::new(id as u32)
    }
}

pub type RefHapId = HapId<RefHapSpace>;
pub type CombinedHapId = HapId<CombinedHapSpace>;

/// HMM state index (0..n_states)
///
/// Distinct from haplotype IDs to avoid mixing reference hap IDs with local state indices.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct StateId(pub u32);

impl StateId {
    #[inline]
    pub fn new(id: u32) -> Self {
        Self(id)
    }

    #[inline]
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub fn as_u32(self) -> u32 {
        self.0
    }
}

impl fmt::Debug for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "StateId({})", self.0)
    }
}

impl fmt::Display for StateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S{}", self.0)
    }
}

impl From<u32> for StateId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<usize> for StateId {
    fn from(id: usize) -> Self {
        Self(id as u32)
    }
}

impl<T> std::ops::Index<StateId> for Vec<T> {
    type Output = T;

    #[inline]
    fn index(&self, id: StateId) -> &Self::Output {
        &self[id.as_usize()]
    }
}

impl<T> std::ops::IndexMut<StateId> for Vec<T> {
    #[inline]
    fn index_mut(&mut self, id: StateId) -> &mut Self::Output {
        &mut self[id.as_usize()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ref_hap_id_basic() {
        let hid = RefHapId::new(42);
        assert_eq!(hid.as_u32(), 42);
        assert_eq!(hid.as_usize(), 42);
    }

}
