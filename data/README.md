# Data directory

This folder contains sample genomic data organized by person. Each person gets
an underscore-based folder name.

## Kat Suricata
- Folder: `data/kat_suricata/`
- The large VCF was split into <=95 MB chunks to stay under GitHub's 100 MB
  per-file limit.
- Parts are named:
  `KatSuricata-NG1N86S6FC-30x-WGS-Sequencing_com-03-18-24.snp-indel.genome.vcf.gz.part-00` ... `part-03`

To reconstruct the original VCF exactly:

```bash
cat data/kat_suricata/KatSuricata-NG1N86S6FC-30x-WGS-Sequencing_com-03-18-24.snp-indel.genome.vcf.gz.part-* \
  > data/kat_suricata/KatSuricata-NG1N86S6FC-30x-WGS-Sequencing_com-03-18-24.snp-indel.genome.vcf.gz
```

## Christopher Smith
- Folder: `data/christopher_smith/`
- The large VCF was split into <=95 MB chunks to stay under GitHub's 100 MB
  per-file limit.
- Parts are named:
  `NG1LRQNESI.hard-filtered.vcf.gz.part-00` ... `part-04`

To reconstruct the original VCF exactly:

```bash
cat data/christopher_smith/NG1LRQNESI.hard-filtered.vcf.gz.part-* \
  > data/christopher_smith/NG1LRQNESI.hard-filtered.vcf.gz
```

You can then decompress as usual (e.g., `gunzip`).
