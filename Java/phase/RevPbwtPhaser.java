/*
 * Copyright (C) 2014-2021 Brian L. Browning
 *
 * This file is part of Beagle
 *
 * Beagle is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Beagle is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package phase;

import blbutil.BitArray;
import java.util.Random;
import java.util.stream.IntStream;
import vcf.GT;

/**
 * <p>Class {@code RevPbwtPhaser} phases input genotype data and imputes
 * missing alleles using the Positional Burrows-Wheeler Transform (PBWT).
 * The PBWT processes markers in order of decreasing marker index.  Any
 * heterozygote genotypes that cannot by phased using the PBWT are randomly
 * phased, and any alleles that cannot be imputed using the PBWT are randomly
 * imputed using the allele frequencies in the combined reference and target
 * input genotype data.</p>
 *
 * <p>Instances of class {@code RevPbwtPhaser} are immutable.</p>
 *
 * <p>Reference: Richard Durbin. (2014) Efficient haplotype matching and storage
 * using the Positional Burrows-Wheeler Transform (PBWT). Bioinformatics
 * 30(9):1266-72.</p>
 *
 * <p>Reference: Olivier Delaneau, Jean-Francois Zagury, Matthew R Robinson,
 * Jonathan Marchini, Emmanouil Dermitzakis. (2019) Accurate, scalable and
 * integrative haplotype estimation. Nature Communications 10(1):5436.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class RevPbwtPhaser {

    private final GT targGT;
    private final int start;
    private final int end;
    private final int[] bitsPerAllele;
    private final BitArray[] markerToBits;

    /**
     * Creates a new {@code RevPbwtPhaser} for the specified data.
     * @param fpd the input genotype data for phasing
     * @param start the index of the first marker (inclusive) to be phased
     * @param end the index of the last marker (exclusive) to be phased
     * @param seed seed for random number generation
     * @throws IllegalArgumentException if
     * {@code start < 0 || end > fpd.stage1TargGt().nMarkers() || start >= end}
     * @throws NullPointerException if {@code fpd == null}
     */
    public RevPbwtPhaser(FixedPhaseData fpd, int start, int end, long seed) {
        if (start<0 || end>fpd.stage1TargGT().nMarkers() || start>=end) {
            throw new IllegalArgumentException(String.valueOf(start));
        }
        this.targGT = fpd.stage1TargGT();
        this.start = start;
        this.end = end;
        this.bitsPerAllele = IntStream.range(start, end)
                .map(m -> targGT.markers().marker(m).bitsPerAllele())
                .toArray();
        this.markerToBits = phase(fpd, start, end, seed);
    }

    private static BitArray[] phase(FixedPhaseData fpd, int start, int end,
            long seed) {
        Random rand = new Random(seed);
        int overlap = fpd.stage1Overlap();
        GT targGT = fpd.stage1TargGT();
        PbwtRecPhaser recPhaser = new PbwtRecPhaser(fpd);
        boolean[] missingGT = new boolean[targGT.nSamples()];
        boolean[] unphHet = new boolean[targGT.nSamples()];
        int[] alleles = new int[fpd.nHaps()];
        
        BitArray[] markerToBits = new BitArray[end-start];
        int lastM = -1;
        for (int m=(end-1); m>=start; --m) {
            int[] alleleCDF = recPhaser.phase(lastM, alleles, m, missingGT,
                    unphHet);
            if (m>=overlap) {
                finishPhasing(alleles, unphHet, alleleCDF, rand);
            }
            markerToBits[m-start] = storePhasing(targGT, m, alleles);
            lastM = m;
        }
        return markerToBits;
    }

    private static void finishPhasing(int[] alleles, boolean[] unphHet,
            int[] alleleCDF, Random rand) {
        for (int s=0; s<unphHet.length; ++s) {
            int h1 = s<<1;
            int h2 = h1 | 0b1;
            if (unphHet[s]) {
                int a1 = alleles[h1];
                int a2 = alleles[h2];
                if (rand.nextBoolean()) {
                    alleles[h1] = a2;
                    alleles[h2] = a1;
                }
                unphHet[s] = false;
            }
            else {
                if (alleles[h1] == -1) {
                    alleles[h1] = imputeAllele(alleleCDF, rand);
                }
                if (alleles[h2] == -1) {
                    alleles[h2] = imputeAllele(alleleCDF, rand);
                }
            }
        }
    }

    private static int imputeAllele(int[] alleleCDF, Random rand) {
        int bound = alleleCDF[alleleCDF.length-1];
        if (bound==0) {
            return 0;
        }
        else {
            int r = rand.nextInt(alleleCDF[alleleCDF.length-1]);
            int allele = 0;
            while (r>=alleleCDF[allele]) {
                ++allele;
            }
            return allele;
        }
    }

    private static BitArray storePhasing(GT targGT, int m, int[] alleles) {
        int nTargHaps = targGT.nHaps();
        int bitsPerAllele = targGT.markers().marker(m).bitsPerAllele();
        BitArray bits = new BitArray(nTargHaps*bitsPerAllele);
        int bit=0;
        for (int h=0; h<nTargHaps; ++h) {
            int mask = 1;
            for (int j=0; j<bitsPerAllele; ++j, ++bit) {
                if ((alleles[h] & mask)==mask) {
                    bits.set(bit);
                }
                mask <<= 1;
            }
        }
        return bits;
    }

    /**
     * Returns the input target genotypes.
     * @return the input target genotypes
     */
    public GT targGT() {
        return targGT;
    }

    /**
     * Return the inclusive start marker index.
     * @return the inclusive start marker index
     */
    public int start() {
        return start;
    }

    /**
     * Return the exclusive end marker index.
     * @return the exclusive end marker index
     */
    public int end() {
        return end;
    }

    /**
     * Returns the minimum number of bits required to store a non-missing
     * allele for the specified marker
     * @param marker a marker index
     * @return the minimum number of bits required to store a non-missing
     * allele for the specified marker
     * @throws IndexOutOfBoundsException if
     * {@code marker < this.start() || marker >= this.end()}
     */
    public int bitsPerAllele(int marker) {
        return bitsPerAllele[marker-start];
    }

    /**
     * Returns the specified allele
     * @param marker a marker index
     * @param hap a haplotype index
     * @return the specified allele
     * @throws IndexOutOfBoundsException if
     * {@code marker < this.start() || marker >= this.end()}
     * @throws IndexOutOfBoundsException if
     * {@code hap < 0 || hap >= this.targGT().nHaps()}
     */
    public int allele(int marker, int hap) {
        int nBitsPerAllele = this.bitsPerAllele[marker-start];
        int bStart = hap*nBitsPerAllele;
        BitArray bits = markerToBits[marker-start];

        if (nBitsPerAllele==1) {
            return bits.get(bStart) ? 1 : 0;
        }
        int allele = 0;
        int mask = 1;
        int bEnd = bStart + nBitsPerAllele;
        for (int j=bStart; j<bEnd; ++j) {
            if (bits.get(j)) {
                allele |= mask;
            }
            mask <<= 1;
        }
        return allele;
    }
}
