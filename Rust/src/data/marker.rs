//! # Marker Definitions
//!
//! Genomic marker (variant site) representation. Replaces `vcf/Marker.java`.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::data::ChromIdx;

/// Zero-cost newtype for marker indices
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default, Serialize, Deserialize)]
pub struct MarkerIdx(pub u32);

impl MarkerIdx {
    pub fn new(idx: u32) -> Self {
        Self(idx)
    }

    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

impl From<u32> for MarkerIdx {
    fn from(idx: u32) -> Self {
        Self(idx)
    }
}

impl From<usize> for MarkerIdx {
    fn from(idx: usize) -> Self {
        Self(idx as u32)
    }
}

impl From<MarkerIdx> for usize {
    fn from(idx: MarkerIdx) -> usize {
        idx.0 as usize
    }
}

/// Allele representation
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Allele {
    /// Single nucleotide (A=0, C=1, G=2, T=3)
    Base(u8),
    /// Insertion/deletion or complex variant
    Seq(Arc<str>),
    /// Missing data marker
    Missing,
}

impl Allele {
    /// Create allele from a single character
    pub fn from_char(c: char) -> Self {
        match c {
            'A' | 'a' => Self::Base(0),
            'C' | 'c' => Self::Base(1),
            'G' | 'g' => Self::Base(2),
            'T' | 't' => Self::Base(3),
            'N' | 'n' | '.' | '*' => Self::Missing,
            _ => Self::Seq(c.to_string().into()),
        }
    }

    /// Create allele from a string
    pub fn from_str(s: &str) -> Self {
        if s.len() == 1 {
            Self::from_char(s.chars().next().unwrap())
        } else if s == "." || s == "*" || s == "<*>" || s == "<NON_REF>" {
            Self::Missing
        } else {
            Self::Seq(s.into())
        }
    }

    /// Check if this is a single nucleotide
    pub fn is_snv(&self) -> bool {
        matches!(self, Self::Base(_))
    }


    /// Get complement (for strand flipping)
    pub fn complement(&self) -> Self {
        match self {
            Self::Base(0) => Self::Base(3), // A -> T
            Self::Base(1) => Self::Base(2), // C -> G
            Self::Base(2) => Self::Base(1), // G -> C
            Self::Base(3) => Self::Base(0), // T -> A
            other => other.clone(),
        }
    }
}

impl std::fmt::Display for Allele {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Base(0) => write!(f, "A"),
            Self::Base(1) => write!(f, "C"),
            Self::Base(2) => write!(f, "G"),
            Self::Base(3) => write!(f, "T"),
            Self::Base(_) => write!(f, "N"),
            Self::Seq(s) => write!(f, "{}", s),
            Self::Missing => write!(f, "."),
        }
    }
}

/// A genomic marker (variant site)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Marker {
    /// Chromosome index
    pub chrom: ChromIdx,
    /// 1-based genomic position (start)
    pub pos: u32,
    /// End position for gVCF blocks/SVs (None = pos, same as start)
    pub end: Option<u32>,
    /// Variant ID (rsID or similar), None if missing
    pub id: Option<Arc<str>>,
    /// Reference allele
    pub ref_allele: Allele,
    /// Alternate allele(s)
    pub alt_alleles: Vec<Allele>,
}

impl Marker {
    /// Create a new marker
    pub fn new(
        chrom: ChromIdx,
        pos: u32,
        id: Option<Arc<str>>,
        ref_allele: Allele,
        alt_alleles: Vec<Allele>,
    ) -> Self {
        Self {
            chrom,
            pos,
            end: None,
            id,
            ref_allele,
            alt_alleles,
        }
    }

    /// Total number of alleles (ref + alts)
    pub fn n_alleles(&self) -> usize {
        1 + self.alt_alleles.len()
    }

    /// Check if this is a biallelic variant
    pub fn is_biallelic(&self) -> bool {
        self.alt_alleles.len() == 1
    }

    /// Check if this is a SNV (all alleles are single nucleotides)
    pub fn is_snv(&self) -> bool {
        self.ref_allele.is_snv() && self.alt_alleles.iter().all(|a| a.is_snv())
    }

    /// Create a new marker with explicit end position (for SVs and gVCF blocks)
    ///
    /// # Arguments
    /// * `chrom` - Chromosome index
    /// * `pos` - Start position (1-based)
    /// * `end` - End position from INFO/END field (None if not present)
    /// * `id` - Variant ID
    /// * `ref_allele` - Reference allele
    /// * `alt_alleles` - Alternate alleles
    pub fn with_end(
        chrom: ChromIdx,
        pos: u32,
        end: Option<u32>,
        id: Option<Arc<str>>,
        ref_allele: Allele,
        alt_alleles: Vec<Allele>,
    ) -> Self {
        Self {
            chrom,
            pos,
            end,
            id,
            ref_allele,
            alt_alleles,
        }
    }
}

/// Allele mapping from target to reference panel
///
/// This handles Ref/Alt swaps and strand flips following Java Marker.targToRefAllele().
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlleleMapping {
    /// For each target allele index, the corresponding reference allele index (-1 if no match)
    pub targ_to_ref: Vec<i8>,
    /// For each reference allele index, the corresponding target allele index (-1 if no match)
    pub ref_to_targ: Vec<i8>,
    /// Whether strand was flipped
    pub strand_flipped: bool,
    /// Whether Ref/Alt were swapped
    pub alleles_swapped: bool,
}

impl AlleleMapping {
    /// Create an AlleleMapping with automatic reverse mapping computation
    pub fn new(targ_to_ref: Vec<i8>, n_ref_alleles: usize, strand_flipped: bool, alleles_swapped: bool) -> Self {
        // Build reverse mapping
        let mut ref_to_targ = vec![-1i8; n_ref_alleles];
        for (targ_idx, &ref_idx) in targ_to_ref.iter().enumerate() {
            if ref_idx >= 0 && (ref_idx as usize) < n_ref_alleles {
                ref_to_targ[ref_idx as usize] = targ_idx as i8;
            }
        }
        Self {
            targ_to_ref,
            ref_to_targ,
            strand_flipped,
            alleles_swapped,
        }
    }

    /// Map a target allele to reference allele
    /// Returns None if the allele cannot be mapped
    pub fn map_allele(&self, targ_allele: u8) -> Option<u8> {
        self.targ_to_ref
            .get(targ_allele as usize)
            .and_then(|&r| if r >= 0 { Some(r as u8) } else { None })
    }

    /// Map a reference allele to target allele (reverse mapping)
    /// Returns None if the allele cannot be mapped
    pub fn reverse_map_allele(&self, ref_allele: u8) -> Option<u8> {
        self.ref_to_targ
            .get(ref_allele as usize)
            .and_then(|&t| if t >= 0 { Some(t as u8) } else { None })
    }

    /// Check if all target alleles can be mapped
    pub fn is_valid(&self) -> bool {
        self.targ_to_ref.iter().all(|&r| r >= 0)
    }
}

/// Compute allele mapping from target marker to reference marker
///
/// This implements the Java Marker.targToRefAllele() logic:
/// 1. Check if alleles match directly
/// 2. If not, check if Ref/Alt are swapped
/// 3. If not, check if strand is flipped (A<->T, C<->G)
/// 4. If not, check if both swapped and flipped
///
/// # Arguments
/// * `targ` - Target marker
/// * `ref_marker` - Reference marker
///
/// # Returns
/// AlleleMapping if markers can be aligned, or None if incompatible
pub fn compute_allele_mapping(targ: &Marker, ref_marker: &Marker) -> Option<AlleleMapping> {
    // Must be at same position
    if targ.chrom != ref_marker.chrom || targ.pos != ref_marker.pos {
        return None;
    }

    // Only handle SNVs for now (strand flip only makes sense for SNVs)
    if !targ.is_snv() || !ref_marker.is_snv() {
        // For non-SNVs, try direct match only
        return try_direct_match(targ, ref_marker);
    }

    // Try direct match
    if let Some(mapping) = try_direct_match(targ, ref_marker) {
        return Some(mapping);
    }

    // Try strand flip
    if let Some(mapping) = try_strand_flip(targ, ref_marker) {
        return Some(mapping);
    }

    // No valid mapping found
    None
}

/// Try to match alleles directly without any transformation
fn try_direct_match(targ: &Marker, ref_marker: &Marker) -> Option<AlleleMapping> {
    let n_targ_alleles = targ.n_alleles();
    let mut targ_to_ref = vec![-1i8; n_targ_alleles];
    let mut all_matched = true;

    // Build reference allele lookup
    let ref_alleles: Vec<&Allele> = std::iter::once(&ref_marker.ref_allele)
        .chain(ref_marker.alt_alleles.iter())
        .collect();

    // Try to match each target allele
    for (t_idx, t_allele) in std::iter::once(&targ.ref_allele)
        .chain(targ.alt_alleles.iter())
        .enumerate()
    {
        let mut found = false;
        for (r_idx, &r_allele) in ref_alleles.iter().enumerate() {
            if t_allele == r_allele {
                targ_to_ref[t_idx] = r_idx as i8;
                found = true;
                break;
            }
        }
        if !found {
            all_matched = false;
        }
    }

    // Check if Ref alleles match (identity) or are swapped
    let alleles_swapped = targ_to_ref.get(0) == Some(&1);
    let n_ref_alleles = ref_marker.n_alleles();

    if all_matched {
        Some(AlleleMapping::new(targ_to_ref, n_ref_alleles, false, alleles_swapped))
    } else {
        None
    }
}

/// Try to match alleles with strand flip (A<->T, C<->G)
fn try_strand_flip(targ: &Marker, ref_marker: &Marker) -> Option<AlleleMapping> {
    let n_targ_alleles = targ.n_alleles();
    let mut targ_to_ref = vec![-1i8; n_targ_alleles];
    let mut all_matched = true;

    // Build reference allele lookup
    let ref_alleles: Vec<&Allele> = std::iter::once(&ref_marker.ref_allele)
        .chain(ref_marker.alt_alleles.iter())
        .collect();

    // Try to match each target allele (with complement)
    for (t_idx, t_allele) in std::iter::once(&targ.ref_allele)
        .chain(targ.alt_alleles.iter())
        .enumerate()
    {
        let flipped = t_allele.complement();
        let mut found = false;
        for (r_idx, &r_allele) in ref_alleles.iter().enumerate() {
            if &flipped == r_allele {
                targ_to_ref[t_idx] = r_idx as i8;
                found = true;
                break;
            }
        }
        if !found {
            all_matched = false;
        }
    }

    let alleles_swapped = targ_to_ref.get(0) == Some(&1);
    let n_ref_alleles = ref_marker.n_alleles();

    if all_matched {
        Some(AlleleMapping::new(targ_to_ref, n_ref_alleles, true, alleles_swapped))
    } else {
        None
    }
}

/// Check if a marker can potentially have ambiguous strand (A/T or C/G SNV)
pub fn is_strand_ambiguous(marker: &Marker) -> bool {
    if !marker.is_snv() || !marker.is_biallelic() {
        return false;
    }

    let ref_allele = &marker.ref_allele;
    let alt_allele = marker.alt_alleles.first();

    match (ref_allele, alt_allele) {
        (Allele::Base(0), Some(Allele::Base(3))) => true, // A/T
        (Allele::Base(3), Some(Allele::Base(0))) => true, // T/A
        (Allele::Base(1), Some(Allele::Base(2))) => true, // C/G
        (Allele::Base(2), Some(Allele::Base(1))) => true, // G/C
        _ => false,
    }
}

impl PartialEq for Marker {
    fn eq(&self, other: &Self) -> bool {
        self.chrom == other.chrom && self.pos == other.pos
    }
}

impl Eq for Marker {}

impl PartialOrd for Marker {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Marker {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match self.chrom.cmp(&other.chrom) {
            std::cmp::Ordering::Equal => self.pos.cmp(&other.pos),
            other => other,
        }
    }
}

/// A collection of markers
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Markers {
    /// The markers in order
    markers: Vec<Marker>,
    /// Chromosome names (indexed by ChromIdx)
    chrom_names: Vec<Arc<str>>,
}

impl Markers {
    /// Create an empty marker collection
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of markers
    pub fn len(&self) -> usize {
        self.markers.len()
    }

    /// Get marker by index
    pub fn get(&self, idx: MarkerIdx) -> Option<&Marker> {
        self.markers.get(idx.as_usize())
    }

    /// Get marker by index (unchecked)
    pub fn marker(&self, idx: MarkerIdx) -> &Marker {
        &self.markers[idx.as_usize()]
    }

    /// Get chromosome name by index
    pub fn chrom_name(&self, idx: ChromIdx) -> Option<&str> {
        self.chrom_names.get(idx.as_usize()).map(|s| s.as_ref())
    }

    /// Add a chromosome name and return its index
    pub fn add_chrom(&mut self, name: &str) -> ChromIdx {
        // Check if already exists
        for (i, existing) in self.chrom_names.iter().enumerate() {
            if existing.as_ref() == name {
                return ChromIdx::new(i as u16);
            }
        }
        let idx = ChromIdx::new(self.chrom_names.len() as u16);
        self.chrom_names.push(name.into());
        idx
    }

    /// Add a marker
    pub fn push(&mut self, marker: Marker) {
        self.markers.push(marker);
    }

    /// Get all chromosome names
    pub fn chrom_names(&self) -> &[Arc<str>] {
        &self.chrom_names
    }
}

impl std::ops::Index<MarkerIdx> for Markers {
    type Output = Marker;

    fn index(&self, idx: MarkerIdx) -> &Self::Output {
        &self.markers[idx.as_usize()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allele_from_char() {
        assert_eq!(Allele::from_char('A'), Allele::Base(0));
        assert_eq!(Allele::from_char('C'), Allele::Base(1));
        assert_eq!(Allele::from_char('G'), Allele::Base(2));
        assert_eq!(Allele::from_char('T'), Allele::Base(3));
    }

    #[test]
    fn test_marker_is_snv() {
        let marker = Marker::new(
            ChromIdx(0),
            100,
            None,
            Allele::Base(0),
            vec![Allele::Base(1)],
        );
        assert!(marker.is_snv());
        assert!(marker.is_biallelic());
    }

    // TODO: bits_per_allele method not implemented on Marker
    // #[test]
    // fn test_bits_per_allele() {
    //     let marker2 = Marker::new(
    //         ChromIdx(0),
    //         100,
    //         None,
    //         Allele::Base(0),
    //         vec![Allele::Base(1)],
    //     );
    //     assert_eq!(marker2.bits_per_allele(), 1);
    //
    //     let marker4 = Marker::new(
    //         ChromIdx(0),
    //         100,
    //         None,
    //         Allele::Base(0),
    //         vec![Allele::Base(1), Allele::Base(2), Allele::Base(3)],
    //     );
    //     assert_eq!(marker4.bits_per_allele(), 2);
    // }
}
