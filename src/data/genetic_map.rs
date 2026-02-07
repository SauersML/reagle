//! # Genetic Map
//!
//! Physical-to-genetic distance interpolation.
//! This module provides:
//! - `GeneticMap`: Raw genetic map data from PLINK files
//! - `MarkerMap`: Pre-computed genetic positions and distances for markers
//!
//! Replaces Java `vcf/GeneticMap.java`, `vcf/PlinkGenMap.java`, and `vcf/MarkerMap.java`.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::data::ChromIdx;
use crate::data::marker::{MarkerIdx, Markers};
use crate::error::{ReagleError, Result};
use tracing::info_span;

/// Default scale factor: 1 cM per Mb (1e-6 cM per bp)
pub const DEFAULT_SCALE_FACTOR: f64 = 1e-6;

/// A genetic map for interpolating physical positions to genetic distances (cM)
///
/// This matches Java `vcf/GeneticMap.java`.
#[derive(Clone, Debug)]
pub struct GeneticMap {
    /// Chromosome index
    chrom: ChromIdx,

    /// Physical positions (bp), sorted
    positions: StrictlyIncreasingU32,

    /// Genetic positions (cM) corresponding to physical positions
    gen_positions: NonDecreasingF64,
}

impl GeneticMap {
    /// Load from PLINK format map file
    ///
    /// Format: chrom position_bp rate_cM_per_Mb position_cM
    /// (Note: rate column is ignored, we use the cumulative position)
    pub fn from_plink_file(path: &Path, target_chrom: &str) -> Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut positions = Vec::new();
        let mut gen_positions = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = line?;
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() < 4 {
                return Err(ReagleError::parse(
                    line_num + 1,
                    format!("Expected 4 columns, got {}", parts.len()),
                ));
            }

            let chrom = parts[0];
            if !chrom_name_eq(chrom, target_chrom) {
                continue;
            }

            let pos: u32 = parts[1]
                .parse()
                .map_err(|_| ReagleError::parse(line_num + 1, "Invalid position"))?;

            // Column 3 is rate (ignored), column 4 is genetic position
            let gen_pos: f64 = parts[3]
                .parse()
                .map_err(|_| ReagleError::parse(line_num + 1, "Invalid genetic position"))?;

            if !gen_pos.is_finite() {
                return Err(ReagleError::parse(
                    line_num + 1,
                    "Genetic position is not finite",
                ));
            }

            positions.push(pos);
            gen_positions.push(gen_pos);
        }

        let positions = StrictlyIncreasingU32::new(positions)?;
        let gen_positions = NonDecreasingF64::new(gen_positions, positions.as_slice())?;

        Ok(Self {
            chrom: ChromIdx::new(0), // Will be set by caller
            positions,
            gen_positions,
        })
    }

    /// Interpolate genetic position (cM) from physical position (bp)
    ///
    /// From Java `GeneticMap.genPos(int chrom, int basePosition)`
    pub fn gen_pos(&self, phys_pos: u32) -> f64 {
        if self.positions.is_empty() {
            // Default: 1 cM per Mb
            return phys_pos as f64 * DEFAULT_SCALE_FACTOR;
        }

        // Binary search for position
        match self.positions.as_slice().binary_search(&phys_pos) {
            Ok(idx) => self.gen_positions.as_slice()[idx],
            Err(idx) => {
                if idx == 0 {
                    // Before first position: extrapolate
                    let rate = if self.positions.len() > 1 {
                        (self.gen_positions.as_slice()[1] - self.gen_positions.as_slice()[0])
                            / (self.positions.as_slice()[1] - self.positions.as_slice()[0]) as f64
                    } else {
                        DEFAULT_SCALE_FACTOR
                    };
                    self.gen_positions.as_slice()[0]
                        - rate * (self.positions.as_slice()[0] - phys_pos) as f64
                } else if idx == self.positions.len() {
                    // After last position: extrapolate
                    let last = self.positions.len() - 1;
                    let rate = if last > 0 {
                        (self.gen_positions.as_slice()[last]
                            - self.gen_positions.as_slice()[last - 1])
                            / (self.positions.as_slice()[last]
                                - self.positions.as_slice()[last - 1])
                                as f64
                    } else {
                        DEFAULT_SCALE_FACTOR
                    };
                    self.gen_positions.as_slice()[last]
                        + rate * (phys_pos - self.positions.as_slice()[last]) as f64
                } else {
                    // Interpolate between idx-1 and idx
                    let p0 = self.positions.as_slice()[idx - 1] as f64;
                    let p1 = self.positions.as_slice()[idx] as f64;
                    let g0 = self.gen_positions.as_slice()[idx - 1];
                    let g1 = self.gen_positions.as_slice()[idx];
                    let t = (phys_pos as f64 - p0) / (p1 - p0);
                    g0 + t * (g1 - g0)
                }
            }
        }
    }

    /// Get genetic distance between two physical positions (cM)
    pub fn gen_dist(&self, pos1: u32, pos2: u32) -> f64 {
        (self.gen_pos(pos2) - self.gen_pos(pos1)).abs()
    }

    /// Set chromosome index
    pub fn set_chrom(&mut self, chrom: ChromIdx) {
        self.chrom = chrom;
    }
}

/// Position-based genetic map (no recombination rate map)
///
/// This matches Java `vcf/PositionMap.java`
#[derive(Clone, Debug)]
pub struct PositionMap {
    scale_factor: f64,
}

impl PositionMap {
    /// Create a new position map with default scale factor (1 cM per Mb)
    pub fn new() -> Self {
        Self {
            scale_factor: DEFAULT_SCALE_FACTOR,
        }
    }

    /// Get genetic position from physical position
    pub fn gen_pos(&self, phys_pos: u32) -> f64 {
        phys_pos as f64 * self.scale_factor
    }
}

/// Pre-computed genetic positions for a set of markers
///
/// This matches Java `vcf/MarkerMap.java`
#[derive(Clone, Debug)]
pub struct MarkerMap {
    /// Genetic positions (cM) for each marker
    gen_pos: Vec<f64>,
}

impl MarkerMap {
    /// Create a MarkerMap from markers and genetic map
    ///
    /// This matches Java `MarkerMap.create(GeneticMap genMap, Markers markers)`
    pub fn create<Space>(markers: &Markers<Space>, gen_map: &GeneticMap) -> Self {
        let n = markers.len();
        if n == 0 {
            return Self {
                gen_pos: Vec::new(),
            };
        }

        // Calculate mean single-base genetic distance
        // Keep map mass faithful; rely on numerically stable transition math for tiny distances.
        Self::from_gen_map(markers, gen_map)
    }

    /// Create from genetic map with non-negative incremental distances.
    ///
    /// From Java `GeneticMap.genPos(GeneticMap genMap, double minGenDist, Markers markers)`
    pub fn from_gen_map<Space>(markers: &Markers<Space>, gen_map: &GeneticMap) -> Self {
        let n = markers.len();
        if n == 0 {
            return Self {
                gen_pos: Vec::new(),
            };
        }

        let mut gen_pos = Vec::with_capacity(n);

        // First marker
        let first_pos = markers
            .get(MarkerIdx::from(0usize))
            .map(|m| m.pos)
            .unwrap_or(0);
        gen_pos.push(gen_map.gen_pos(first_pos));

        let mut last_map_pos = gen_pos[0];

        // Subsequent markers
        for i in 1..n {
            let pos = markers.get(MarkerIdx::from(i)).map(|m| m.pos).unwrap_or(0);
            let map_pos = gen_map.gen_pos(pos);
            let dist = (map_pos - last_map_pos).max(0.0);
            gen_pos.push(gen_pos[i - 1] + dist);
            last_map_pos = map_pos;
        }

        Self { gen_pos }
    }

    /// Create using default position-based map (1 cM per Mb)
    pub fn from_positions<Space>(markers: &Markers<Space>) -> Self {
        let n = markers.len();
        if n == 0 {
            return Self {
                gen_pos: Vec::new(),
            };
        }

        let pos_map = PositionMap::new();
        let mut gen_pos = Vec::with_capacity(n);

        for i in 0..n {
            let pos = markers.get(MarkerIdx::from(i)).map(|m| m.pos).unwrap_or(0);
            gen_pos.push(pos_map.gen_pos(pos));
        }

        Self { gen_pos }
    }

    /// Get all genetic positions
    pub fn gen_positions(&self) -> &[f64] {
        &self.gen_pos
    }
}

/// A collection of genetic maps for multiple chromosomes
#[derive(Clone, Debug, Default)]
pub struct GeneticMaps {
    maps: Vec<Option<GeneticMap>>,
    chrom_name_to_idx: HashMap<String, ChromIdx>,
}

impl GeneticMaps {
    /// Create empty collection
    pub fn new() -> Self {
        Self::default()
    }

    /// Load all chromosomes from a PLINK map file
    pub fn from_plink_file(path: &Path, chrom_names: &[&str]) -> Result<Self> {
        info_span!("genetic_maps_from_plink_file", path = ?path).in_scope(|| {
            let mut maps = Vec::with_capacity(chrom_names.len());
            let mut chrom_name_to_idx: HashMap<String, ChromIdx> = HashMap::new();
            let mut missing_names: Vec<String> = Vec::new();
            for (i, &name) in chrom_names.iter().enumerate() {
                let mut map = GeneticMap::from_plink_file(path, name)?;
                let chrom_idx = ChromIdx::new(i as u16);
                if map.positions.is_empty() {
                    missing_names.push(name.to_string());
                }
                map.set_chrom(chrom_idx);
                maps.push(Some(map));
                chrom_name_to_idx
                    .entry(chrom_name_key(name))
                    .or_insert(chrom_idx);
            }
            if !missing_names.is_empty() {
                eprintln!(
                    "Warning: genetic map has no rows for {} requested chromosome name(s) in {:?}; default {:.1} cM/Mb will be used where needed",
                    missing_names.len(),
                    path,
                    DEFAULT_SCALE_FACTOR * 1_000_000.0
                );
                let preview = missing_names.iter().take(8).cloned().collect::<Vec<_>>();
                eprintln!("  Missing examples: {}", preview.join(", "));
            }
            Ok(Self {
                maps,
                chrom_name_to_idx,
            })
        })
    }

    /// Get genetic map for a chromosome
    pub fn get(&self, chrom: ChromIdx) -> Option<&GeneticMap> {
        self.maps.get(chrom.as_usize()).and_then(|m| m.as_ref())
    }

    /// Get chromosome index by chromosome name.
    pub fn chrom_idx_by_name(&self, chrom_name: &str) -> Option<ChromIdx> {
        self.chrom_name_to_idx
            .get(&chrom_name_key(chrom_name))
            .copied()
    }

    /// Get genetic map for a chromosome name.
    pub fn get_by_name(&self, chrom_name: &str) -> Option<&GeneticMap> {
        self.chrom_idx_by_name(chrom_name)
            .and_then(|idx| self.get(idx))
    }

    /// Get genetic position, falling back to default rate if no map
    pub fn gen_pos(&self, chrom: ChromIdx, phys_pos: u32) -> f64 {
        match self.get(chrom) {
            Some(map) => map.gen_pos(phys_pos),
            None => phys_pos as f64 * DEFAULT_SCALE_FACTOR,
        }
    }

    /// Get genetic position by chromosome name, falling back to default rate if no map.
    pub fn gen_pos_by_name(&self, chrom_name: &str, phys_pos: u32) -> f64 {
        match self.get_by_name(chrom_name) {
            Some(map) => map.gen_pos(phys_pos),
            None => phys_pos as f64 * DEFAULT_SCALE_FACTOR,
        }
    }

    /// Get genetic distance between two positions
    pub fn gen_dist(&self, chrom: ChromIdx, pos1: u32, pos2: u32) -> f64 {
        match self.get(chrom) {
            Some(map) => map.gen_dist(pos1, pos2),
            None => (pos2 as f64 - pos1 as f64).abs() * DEFAULT_SCALE_FACTOR,
        }
    }
}

fn chrom_name_key(name: &str) -> String {
    normalize_chrom(name).to_ascii_lowercase()
}

#[inline]
fn normalize_chrom(name: &str) -> &str {
    if name.len() >= 3 && name[..3].eq_ignore_ascii_case("chr") {
        &name[3..]
    } else {
        name
    }
}

#[inline]
fn chrom_name_eq(left: &str, right: &str) -> bool {
    normalize_chrom(left).eq_ignore_ascii_case(normalize_chrom(right))
}

#[derive(Clone, Debug)]
struct StrictlyIncreasingU32(Vec<u32>);

impl StrictlyIncreasingU32 {
    fn new(values: Vec<u32>) -> Result<Self> {
        for i in 1..values.len() {
            if values[i] <= values[i - 1] {
                return Err(ReagleError::Config {
                    message: format!(
                        "Genetic map positions not in ascending order at position {}",
                        values[i]
                    ),
                });
            }
        }
        Ok(Self(values))
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn as_slice(&self) -> &[u32] {
        &self.0
    }
}

#[derive(Clone, Debug)]
struct NonDecreasingF64(Vec<f64>);

impl NonDecreasingF64 {
    fn new(values: Vec<f64>, positions: &[u32]) -> Result<Self> {
        for i in 0..values.len() {
            if !values[i].is_finite() {
                return Err(ReagleError::Config {
                    message: format!("Genetic map cM position is not finite at index {}", i),
                });
            }
        }
        for i in 1..values.len() {
            if values[i] < values[i - 1] {
                let pos = positions.get(i).copied().unwrap_or(0);
                return Err(ReagleError::Config {
                    message: format!(
                        "Genetic map cM positions decrease at bp {} ({} -> {})",
                        pos,
                        values[i - 1],
                        values[i]
                    ),
                });
            }
        }
        Ok(Self(values))
    }

    fn as_slice(&self) -> &[f64] {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::marker::{Allele, Marker, Nucleotide};

    /// Create a temporary PLINK genetic map file for testing
    fn create_test_map_file() -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut file = tempfile::NamedTempFile::new().expect("Failed to create temp file");
        // PLINK format: chrom position_bp rate_cM_per_Mb position_cM
        writeln!(file, "chr1 1000000 1.0 0.0").unwrap();
        writeln!(file, "chr1 2000000 1.0 1.0").unwrap();
        writeln!(file, "chr1 3000000 1.5 2.5").unwrap();
        file.flush().unwrap();
        file
    }

    fn make_test_markers() -> Markers {
        let mut markers = Markers::<crate::data::AnyMarkerSpace>::new();
        markers.add_chrom("chr1");

        // 5 markers at 1Mb intervals
        for i in 0..5 {
            let m = Marker::new(
                ChromIdx::new(0),
                (i + 1) * 1_000_000,
                None,
                Allele::Base(Nucleotide::A),
                vec![Allele::Base(Nucleotide::C)],
            );
            markers.push(m);
        }
        markers
    }

    #[test]
    fn test_interpolation() {
        let map_file = create_test_map_file();
        let map =
            GeneticMap::from_plink_file(map_file.path(), "chr1").expect("Failed to load PLINK map");

        // Exact positions (1Mb=0.0, 2Mb=1.0, 3Mb=2.5)
        assert!((map.gen_pos(1_000_000) - 0.0).abs() < 0.001);
        assert!((map.gen_pos(2_000_000) - 1.0).abs() < 0.001);
        assert!((map.gen_pos(3_000_000) - 2.5).abs() < 0.001);

        // Interpolated position
        assert!((map.gen_pos(1_500_000) - 0.5).abs() < 0.001);
        assert!((map.gen_pos(2_500_000) - 1.75).abs() < 0.001);
    }

    #[test]
    fn test_extrapolation() {
        let map_file = create_test_map_file();
        let map =
            GeneticMap::from_plink_file(map_file.path(), "chr1").expect("Failed to load PLINK map");

        // Before first position (should extrapolate)
        let before = map.gen_pos(500_000);
        assert!(
            before < 0.0,
            "Position before first marker should extrapolate to < 0"
        );

        // After last position (should extrapolate)
        let after = map.gen_pos(3_500_000);
        assert!(
            after > 2.5,
            "Position after last marker should extrapolate to > 2.5"
        );
    }

    #[test]
    fn test_empty_map_for_missing_chrom() {
        let map_file = create_test_map_file();
        let map = GeneticMap::from_plink_file(map_file.path(), "chr99")
            .expect("Loading map for missing chrom should succeed with empty positions");

        // Empty map should use default rate of 1 cM per Mb
        assert!((map.gen_pos(1_000_000) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_gen_dist() {
        let map_file = create_test_map_file();
        let map =
            GeneticMap::from_plink_file(map_file.path(), "chr1").expect("Failed to load PLINK map");

        // Distance from 1Mb (0.0 cM) to 2Mb (1.0 cM) = 1.0 cM
        assert!((map.gen_dist(1_000_000, 2_000_000) - 1.0).abs() < 0.001);

        // Distance from 1Mb to 1.5Mb (interpolated to 0.5 cM) = 0.5 cM
        assert!((map.gen_dist(1_000_000, 1_500_000) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_marker_map_from_positions() {
        let markers = make_test_markers();
        let mm = MarkerMap::from_positions(&markers);

        // Verify genetic positions are computed correctly
        let positions = mm.gen_positions();
        assert_eq!(positions.len(), 5);

        // First marker at 1 Mb should have gen_pos = 1 cM (default rate: 1 cM/Mb)
        assert!((positions[0] - 1.0).abs() < 0.001);

        // Other markers: ~1 cM apart (1 Mb * 1e-6 = 1 cM)
        for i in 1..5 {
            assert!((positions[i] - positions[i - 1] - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_position_map() {
        let pm = PositionMap::new();

        assert!((pm.gen_pos(1_000_000) - 1.0).abs() < 0.001);
        assert!((pm.gen_pos(2_000_000) - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_genetic_maps_collection() {
        let map_file = create_test_map_file();
        let chrom_names = ["chr1"];
        let maps = GeneticMaps::from_plink_file(map_file.path(), &chrom_names)
            .expect("Failed to load maps");

        // Should have loaded chr1
        assert!(maps.get(ChromIdx::new(0)).is_some());

        // Test gen_pos through collection
        let pos = maps.gen_pos(ChromIdx::new(0), 1_500_000);
        assert!((pos - 0.5).abs() < 0.001);
    }
}
