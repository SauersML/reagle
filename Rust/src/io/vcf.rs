//! # VCF Reading and Writing
//!
//! ## Role
//! Parse VCF/BCF files into `GenotypeMatrix`. Write phased results back to VCF.
//! Replaces `blbutil/VcfFileIt.java` and related classes.
//!
//! ## Dependencies
//! - `noodles-vcf`: VCF parsing
//! - `noodles-bgzf`: Compressed file handling
//! - `noodles-csi`/`noodles-tabix`: Index support (optional)
//!
//! ## Spec
//!
//! ### VcfReader
//! ```rust,ignore
//! pub struct VcfReader {
//!     reader: noodles::vcf::Reader<...>,
//!     header: vcf::Header,
//!     samples: Samples,
//! }
//!
//! impl VcfReader {
//!     /// Open a VCF/BCF file (auto-detects compression).
//!     pub fn open(path: &Path) -> Result<Self>;
//!
//!     /// Read all records into a GenotypeMatrix.
//!     /// Automatically chooses Dense vs Sparse based on MAF.
//!     pub fn read_all(&mut self) -> Result<GenotypeMatrix>;
//!
//!     /// Iterate records one at a time (for streaming).
//!     pub fn records(&mut self) -> impl Iterator<Item = Result<VcfRecord>>;
//! }
//! ```
//!
//! ### VcfWriter
//! ```rust,ignore
//! pub struct VcfWriter {
//!     writer: noodles::vcf::Writer<...>,
//!     header: vcf::Header,
//! }
//!
//! impl VcfWriter {
//!     /// Create output VCF with appropriate headers.
//!     pub fn create(path: &Path, samples: &Samples) -> Result<Self>;
//!
//!     /// Write a phased genotype matrix.
//!     pub fn write_phased(&mut self, matrix: &GenotypeMatrix) -> Result<()>;
//!
//!     /// Write imputed dosages (DS) and probabilities (GP).
//!     pub fn write_imputed(
//!         &mut self,
//!         matrix: &GenotypeMatrix,
//!         dosages: &[f32],
//!         probs: &[f32],
//!     ) -> Result<()>;
//! }
//! ```
//!
//! ### Storage Selection Heuristic
//! ```rust,ignore
//! fn choose_storage(alleles: &[u8], n_haplotypes: usize) -> GenotypeColumn {
//!     let alt_count = alleles.iter().filter(|&&a| a > 0).count();
//!     let maf = alt_count as f64 / n_haplotypes as f64;
//!
//!     if maf < 0.01 {
//!         GenotypeColumn::Sparse(SparseColumn::from_alleles(alleles))
//!     } else {
//!         GenotypeColumn::Dense(DenseColumn::from_alleles(alleles))
//!     }
//! }
//! ```
