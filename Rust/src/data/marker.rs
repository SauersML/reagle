//! # Marker Definitions
//!
//! Genomic marker (variant site) representation. Replaces `vcf/Marker.java`.

use std::sync::Arc;

use crate::data::ChromIdx;

/// Zero-cost newtype for marker indices
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
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

    /// Check if this is missing
    pub fn is_missing(&self) -> bool {
        matches!(self, Self::Missing)
    }

    /// Get the length of this allele in bases
    pub fn len(&self) -> usize {
        match self {
            Self::Base(_) => 1,
            Self::Seq(s) => s.len(),
            Self::Missing => 0,
        }
    }

    /// Check if allele is empty (missing)
    pub fn is_empty(&self) -> bool {
        self.len() == 0
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
#[derive(Clone, Debug)]
pub struct Marker {
    /// Chromosome index
    pub chrom: ChromIdx,
    /// 1-based genomic position (start)
    pub pos: u32,
    /// 1-based end position (from INFO/END tag for SVs, or pos + len(ref) - 1 for SNVs/indels)
    /// This is used for gVCF blocks and structural variants.
    pub end: u32,
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
            end: pos,
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

    /// Get allele by index (0 = ref, 1+ = alt)
    pub fn allele(&self, idx: usize) -> Option<&Allele> {
        if idx == 0 {
            Some(&self.ref_allele)
        } else {
            self.alt_alleles.get(idx - 1)
        }
    }

    /// Number of bits needed to store an allele index
    pub fn bits_per_allele(&self) -> u32 {
        let n = self.n_alleles();
        if n <= 1 {
            0
        } else {
            usize::BITS - (n - 1).leading_zeros()
        }
    }

    /// Create a new marker with explicit end position (for SVs and gVCF blocks)
    pub fn with_end(
        chrom: ChromIdx,
        pos: u32,
        end: u32,
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

    /// Get the span of this marker (end - pos + 1)
    pub fn span(&self) -> u32 {
        self.end.saturating_sub(self.pos) + 1
    }

    /// Check if this marker overlaps with a position range
    pub fn overlaps(&self, start: u32, end: u32) -> bool {
        self.pos <= end && self.end >= start
    }

    /// Check if this marker has a non-trivial END value (spans multiple positions)
    pub fn has_end_value(&self) -> bool {
        self.end > self.pos
    }
}

/// Allele mapping from target to reference panel
///
/// This handles Ref/Alt swaps and strand flips following Java Marker.targToRefAllele().
#[derive(Clone, Debug)]
pub struct AlleleMapping {
    /// For each target allele index, the corresponding reference allele index (-1 if no match)
    pub targ_to_ref: Vec<i8>,
    /// Whether strand was flipped
    pub strand_flipped: bool,
    /// Whether Ref/Alt were swapped
    pub alleles_swapped: bool,
}

impl AlleleMapping {
    /// Create an identity mapping (no transformation needed)
    pub fn identity(n_alleles: usize) -> Self {
        Self {
            targ_to_ref: (0..n_alleles as i8).collect(),
            strand_flipped: false,
            alleles_swapped: false,
        }
    }

    /// Create a mapping indicating no match is possible
    pub fn no_match(n_alleles: usize) -> Self {
        Self {
            targ_to_ref: vec![-1; n_alleles],
            strand_flipped: false,
            alleles_swapped: false,
        }
    }

    /// Map a target allele to reference allele
    /// Returns None if the allele cannot be mapped
    pub fn map_allele(&self, targ_allele: u8) -> Option<u8> {
        self.targ_to_ref
            .get(targ_allele as usize)
            .and_then(|&r| if r >= 0 { Some(r as u8) } else { None })
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

    if all_matched {
        Some(AlleleMapping {
            targ_to_ref,
            strand_flipped: false,
            alleles_swapped,
        })
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

    if all_matched {
        Some(AlleleMapping {
            targ_to_ref,
            strand_flipped: true,
            alleles_swapped,
        })
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
#[derive(Clone, Debug, Default)]
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

    /// Create from a vector of markers and chromosome names
    pub fn from_vec(markers: Vec<Marker>, chrom_names: Vec<Arc<str>>) -> Self {
        Self {
            markers,
            chrom_names,
        }
    }

    /// Number of markers
    pub fn len(&self) -> usize {
        self.markers.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.markers.is_empty()
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

    /// Iterate over markers
    pub fn iter(&self) -> impl Iterator<Item = &Marker> {
        self.markers.iter()
    }

    /// Get a slice of markers
    pub fn restrict(&self, start: usize, end: usize) -> Self {
        Self {
            markers: self.markers[start..end].to_vec(),
            chrom_names: self.chrom_names.clone(),
        }
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

    #[test]
    fn test_bits_per_allele() {
        let marker2 = Marker::new(
            ChromIdx(0),
            100,
            None,
            Allele::Base(0),
            vec![Allele::Base(1)],
        );
        assert_eq!(marker2.bits_per_allele(), 1);

        let marker4 = Marker::new(
            ChromIdx(0),
            100,
            None,
            Allele::Base(0),
            vec![Allele::Base(1), Allele::Base(2), Allele::Base(3)],
        );
        assert_eq!(marker4.bits_per_allele(), 2);
    }
}