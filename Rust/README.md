# Reagle: High-Performance Genotype Phasing and Imputation

[![Rust CI](https://github.com/SauersML/reagle/actions/workflows/rust.yml/badge.svg)](https://github.com/SauersML/reagle/actions/workflows/rust.yml)

Reagle is a high-performance, memory-efficient reimplementation of the [BEAGLE](https://faculty.washington.edu/browning/beagle/beagle.html) software for genotype phasing and imputation, written in Rust. It is designed to be a drop-in replacement for the original, offering significant performance improvements by leveraging modern hardware and parallel processing.

## About the Project

This project is a port of the BEAGLE software from Java to Rust. The original Java implementation can be found in the `Java/` directory for reference. The goal of this project is to provide a faster, more memory-efficient tool for genotype phasing and imputation, while maintaining compatibility with the original BEAGLE software.

## Usage

### Phasing

To phase genotypes, use the `--gt` argument to specify the input VCF file. The `--out` argument is used to specify the prefix for the output files.

```bash
reagle --gt input.vcf.gz --out phased
```

### Imputation

For imputation, you need to provide a reference panel using the `--ref` argument, in addition to the target genotypes.

```bash
reagle --gt input.vcf.gz --ref reference.vcf.gz --out imputed
```

## Installation and Building

To build the project, you need to have Rust installed. You can then build the project using `cargo`:

```bash
git clone https://github.com/SauersML/reagle.git
cd reagle/Rust
cargo build --release
```

The executable will be located at `target/release/reagle`.

## License

The license for this project is not specified.
