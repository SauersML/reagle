<img width="204" height="204" alt="Reagle Logo" src="https://github.com/user-attachments/assets/7f1988e6-e1c9-4efe-b6a3-c3e39d6d4531" />

# Reagle: High-Performance Genotype Phasing and Imputation

Reagle is a phasing and imputation program written in Rust which uses modified algorithms from the [BEAGLE](https://faculty.washington.edu/browning/beagle/beagle.html)
software for genotype imputation, and algorithms mostly from [SHAPEIT5](https://github.com/odelaneau/shapeit5) for phasing.

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
