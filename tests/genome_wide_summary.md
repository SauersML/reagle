# 🧬 Imputation Benchmark Results
**Chromosome 19**
*Metrics aggregated exactly across all sites (Dosage R²).*

**Test Setup:** All tools receive pre-phased genotype array data (GSA v2 sites) as input for fair comparison. Reference panel: HGDP+1kG phased haplotypes.

### 🏆 Highlights
- **Most Accurate (R²):** IMPUTE5 (0.9580)
- **Best Rare Variants (R² <1%):** IMPUTE5 (0.8703)
- **Best F1 Score:** IMPUTE5 (0.9631)
- **Fastest:** GLIMPSE2 (1232.0s)
- **Reagle Speedup:** 1.0x faster than Beagle
- **Best Phasing (SER):** IMPUTE5 (0.0021)
- **Longest Phase Blocks (N50):** IMPUTE5 (4186451 bp)

### 📊 Accuracy Metrics
*Primary imputation quality metrics*

| Tool | Dosage R² | Rare R² (<1%) | IQS | INFO Score | Concordance | Non-Ref Conc. | Precision | Recall | F1 Score | SEN (mean) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IMPUTE5** | 0.9580 (🔺0.6648) | 0.8703 | 0.7083 | 0.9551 | 0.9952 | 0.9287 | 0.9746 | 0.9520 | 0.9631 | 0.9990 |
| **Beagle 5.5** | 0.9399 (🔺0.6467) | 0.8305 | 0.6693 | 0.9917 | 0.9944 | 0.9248 | 0.9635 | 0.9510 | 0.9572 | 0.9985 |
| **Minimac4** | 0.8215 (🔺0.5283) | 0.6415 | 0.3487 | 0.9465 | 0.9831 | 0.7429 | 0.9132 | 0.8271 | 0.8680 | 0.9955 |
| **GLIMPSE2** | 0.5211 (🔺0.2279) | 0.3002 | 0.2305 | 0.5464 | 0.8053 | 0.3883 | 0.7756 | 0.5964 | 0.6743 | 0.9523 |
| **Reagle (Rust)** | 0.2932 | 0.3199 | 0.0920 | 0.7352 | 0.9522 | 0.2279 | 0.6728 | 0.3701 | 0.4775 | 0.9801 |

### 🔀 Phasing Quality
*Haplotype phasing accuracy metrics*

| Tool | Switch Error Rate | Switch Errors | Switch Opportunities | N50 Phase Block (bp) |
| :--- | :---: | :---: | :---: | :---: |
| **IMPUTE5** | 0.0021 | 92,126 | 42,942,886 | 4186451 |
| **Beagle 5.5** | 0.0033 | 142,227 | 42,735,055 | 2726010 |
| **Minimac4** | 0.0166 | 534,220 | 32,122,878 | 270732 |
| **GLIMPSE2** | 0.4091 | 234,679 | 573,600 | 406267 |
| **Reagle (Rust)** | 0.1021 | 574,988 | 5,632,391 | 206787 |

### 📈 Per-Class Accuracy
*Genotype calling accuracy by zygosity class*

| Tool | HomRef Acc. | Het Acc. | HomAlt Acc. | HomRef N | Het N | HomAlt N |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **IMPUTE5** | 0.9987 | 0.9128 | 0.9590 | 1,358,770,284 | 47,047,817 | 24,815,637 |
| **Beagle 5.5** | 0.9981 | 0.9082 | 0.9562 | 1,358,875,603 | 47,053,637 | 24,816,564 |
| **Minimac4** | 0.9958 | 0.6827 | 0.8569 | 1,358,875,603 | 47,053,637 | 24,816,564 |
| **GLIMPSE2** | 0.9430 | 0.3332 | 0.4899 | 8,058,904 | 1,724,192 | 936,794 |
| **Reagle (Rust)** | 0.9905 | 0.1197 | 0.4330 | 1,358,875,603 | 47,053,637 | 24,816,564 |

### 👥 Per-Sample Statistics
*Distribution of metrics across samples*

| Tool | Conc. Mean | Conc. Min | Conc. Max | R² Mean | R² Min | SEN Mean | SEN Min | SEN Max | SER Mean | SER Min | SER Max |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IMPUTE5** | 0.9952 | 0.9696 | 0.9999 | 0.9590 | 0.7630 | 0.9990 | 0.0000 | 1.0000 | 0.0022 | 0.0000 | 0.0330 |
| **Beagle 5.5** | 0.9944 | 0.9655 | 0.9999 | 0.9415 | 0.6888 | 0.9985 | 0.0000 | 1.0000 | 0.0035 | 0.0000 | 0.0355 |
| **Minimac4** | 0.9831 | 0.9605 | 0.9933 | 0.8251 | 0.6496 | 0.9955 | 0.0000 | 1.0000 | 0.0167 | 0.0085 | 0.0498 |
| **GLIMPSE2** | 0.8053 | 0.7898 | 0.8201 | 0.5216 | 0.4158 | 0.9523 | 0.0000 | 1.0000 | 0.4094 | 0.3478 | 0.5108 |
| **Reagle (Rust)** | 0.9522 | 0.9371 | 0.9610 | 0.2947 | 0.2287 | 0.9801 | 0.0000 | 1.0000 | 0.1026 | 0.0752 | 0.1579 |

### 📊 Overall Statistics
*Dataset size and runtime*

| Tool | Sites Compared | Genotypes | Time (s) | Speedup vs Beagle |
| :--- | :---: | :---: | :---: | :---: |
| **IMPUTE5** | 1,748,941 | 1,430,633,738 | 9118.9 | 0.7x |
| **Beagle 5.5** | 1,749,078 | 1,430,745,804 | 6643.3 | 1.0x |
| **Minimac4** | 1,749,078 | 1,430,745,804 | 6636.4 | 1.0x |
| **GLIMPSE2** | 13,105 | 10,719,890 | 1232.0 | 5.4x |
| **Reagle (Rust)** | 1,749,078 | 1,430,745,804 | 6598.9 | 1.0x |

### 📋 Confusion Matrices
*Truth (rows) vs Imputed (columns): HomRef, Het, HomAlt*


**IMPUTE5:**
```
              HomRef        Het     HomAlt
  HomRef  1,356,984,646  1,745,486     40,152
  Het        3,373,639 42,943,704    730,474
  HomAlt        78,758    939,080 23,797,799
```

**Beagle 5.5:**
```
              HomRef        Het     HomAlt
  HomRef  1,356,286,595  2,523,866     65,142
  Het        3,419,129 42,735,873    898,635
  HomAlt       102,415    985,055 23,729,094
```

**Minimac4:**
```
              HomRef        Het     HomAlt
  HomRef  1,353,225,398  5,481,757    168,448
  Het       12,100,426 32,123,696  2,829,515
  HomAlt       322,735  3,227,713 21,266,116
```

**GLIMPSE2:**
```
              HomRef        Het     HomAlt
  HomRef     7,599,853    402,226     56,825
  Het          927,261    574,418    222,513
  HomAlt       146,697    331,156    458,941
```

**Reagle (Rust):**
```
              HomRef        Het     HomAlt
  HomRef  1,345,943,207  8,172,562  4,759,834
  Het       34,965,478  5,633,209  6,454,950
  HomAlt    10,306,745  3,764,158 10,745,661
```

### 🧪 Masked-SNP Metrics
*Quality assessment on held-out proxy variants*

| Tool | Masked Total | Masked Concordance | Masked Non-Ref Conc. | Masked R² |
| :--- | :---: | :---: | :---: | :---: |
| **IMPUTE5** | 230 | 1.0000 | 1.0000 | 1.0000 |
| **Beagle 5.5** | 234 | 1.0000 | 1.0000 | 1.0000 |
| **Minimac4** | 235 | 0.9872 | 0.9667 | 0.9679 |
| **GLIMPSE2** | 224 | 0.8348 | 0.4043 | 0.4631 |
| **Reagle (Rust)** | 237 | 0.6624 | 0.3065 | 0.0165 |

### 🔢 Raw Counts
*Underlying counts used to calculate metrics above*

| Tool | True Positives | False Positives | False Negatives | Non-Ref Total | Switch Errors | Switch Opportunities |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **IMPUTE5** | 68,411,057 | 1,785,638 | 3,452,397 | 71,863,454 | 92,126 | 42,942,886 |
| **Beagle 5.5** | 68,348,657 | 2,589,008 | 3,521,544 | 71,870,201 | 142,227 | 42,735,055 |
| **Minimac4** | 59,447,040 | 5,650,205 | 12,423,161 | 71,870,201 | 534,220 | 32,122,878 |
| **GLIMPSE2** | 1,587,028 | 459,051 | 1,073,958 | 2,660,986 | 234,679 | 573,600 |
| **Reagle (Rust)** | 26,597,978 | 12,932,396 | 45,272,223 | 71,870,201 | 574,988 | 5,632,391 |

### 📈 MAF-Stratified Performance (R²)
*Dosage R² by Minor Allele Frequency bin - key metric for rare variant imputation quality*

| MAF Bin | IMPUTE5 | Beagle 5.5 | Minimac4 | GLIMPSE2 | Reagle (Rust) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ultra-rare (<0.1%) | 0.7491 | 0.6992 | 0.5019 | 0.0000 | 0.3502 |
| Very-rare (0.1-0.5%) | 0.8879 | 0.8496 | 0.6667 | 0.3870 | 0.3166 |
| Rare (0.5-1%) | 0.8984 | 0.8613 | 0.6635 | 0.3038 | 0.3057 |
| Low-freq (1-5%) | 0.9310 | 0.9041 | 0.7334 | 0.2998 | 0.3773 |
| Medium (5-20%) | 0.9506 | 0.9285 | 0.7813 | 0.4821 | 0.2159 |
| Common (>20%) | 0.9504 | 0.9263 | 0.7629 | 0.2263 | 0.0870 |
| **N genotypes** | 1,430,633,738 | 1,430,745,804 | 1,430,745,804 | 10,719,890 | 1,430,745,804 |