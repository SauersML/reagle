//! # Type-Safe Index Types for the Imputation HMM
//!
//! This module defines strict newtype wrappers to prevent index confusion bugs.
//! Using distinct types for Global Haplotype IDs vs. Local Pattern IDs ensures
//! that the compiler prevents accidental misuse at compile time.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Global haplotype index (0..N_ref_haplotypes)
///
/// This identifies a specific haplotype in the reference panel across all windows.
/// It remains stable throughout the entire chromosome and enables tracking
/// probability flow through the correct physical DNA molecules.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GlobalId(pub u32);

impl GlobalId {
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

impl fmt::Debug for GlobalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GlobalId({})", self.0)
    }
}

impl fmt::Display for GlobalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "G{}", self.0)
    }
}

impl From<u32> for GlobalId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<usize> for GlobalId {
    fn from(id: usize) -> Self {
        Self(id as u32)
    }
}

/// HMM state index (0..n_states)
///
/// Distinct from GlobalId to avoid mixing reference hap IDs with local state indices.
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
    fn test_global_id_basic() {
        let gid = GlobalId::new(42);
        assert_eq!(gid.as_u32(), 42);
        assert_eq!(gid.as_usize(), 42);
    }

}
