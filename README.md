# Reagle: High-Performance Genotype Phasing and Imputation

Reagle is a reimplementation of the [BEAGLE](https://faculty.washington.edu/browning/beagle/beagle.html)
software for genotype phasing and imputation, written in Rust.

## Limitations

- **No pedigree support**: Unlike Java Beagle, this implementation does not support the `--ped` parameter for pedigree-constrained phasing. Trio/duo phasing is not available.

## Usage

### Phasing

To phase genotypes, use the `--gt` argument to specify the input VCF file. The
`--out` argument is used to specify the prefix for the output files.

```bash
reagle --gt input.vcf.gz --out phased
```

### Imputation

For imputation, provide a reference panel using the `--ref` argument, in
addition to the target genotypes.

```bash
reagle --gt input.vcf.gz --ref reference.vcf.gz --out imputed
```

## Installation and Building

To build the project, you need to have Rust installed. You can then build the
project using `cargo`:

```bash
git clone https://github.com/SauersML/reagle.git
cd reagle
cargo build --release
```

The executable will be located at `target/release/reagle`.

## License

The license for this project is not specified.
