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
import java.util.stream.IntStream;
import vcf.GT;

/**
 * <p>Class {@code FwdPbwtPhaser} phases input genotype data and imputes
 * missing alleles using the Positional Burrows-Wheeler Transform (PBWT).
 * The PBWT processes markers in order of increasing marker index.  If
 * a heterozygote genotypes cannot by phased using the increasing PBWT, an
 * attempt is made to phase the heterozygote using the decreasing PBWT.
 * Similarly, if a missing allele cannot be imputed using the increasing PBWT,
 * an attempt is made to impute the allele using the decreasing PBWT.
 * Any heterozygote genotypes that cannot be phased using the increasing
 * or decreasing PBWT will be randomly phased.  Similarly, any alleles that
 * cannot be imputed using the increasing or decreasing PBWT will be randomly
 * imputed using the allele frequencies in the combined reference and target
 * input genotype data.</p>
 *
 * <p>Instances of class {@code FwdPbwtPhaser} are immutable.</p>
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
public class FwdPbwtPhaser {

    private final GT targGT;
    private final int start;
    private final int end;
    private final int[] bitsPerAllele;
    private final BitArray[] markerToBits;

    /**
     * Creates a new {@code FwdPbwtPhaser} for the specified data.
     * @param fpd the input data for phasing
     * @param start the first marker (inclusive) to be phased
     * @param end the last marker (exclusive) to be phased
     * @param seed seed for random number generation
     * @throws IllegalArgumentException if
     * {@code start < 0 || end > targGT.nMarkers() || start >= end}
     * @throws NullPointerException if {@code fpd == null}
     */
    public FwdPbwtPhaser(FixedPhaseData fpd, int start, int end, long seed) {
        if (start<0 || end>fpd.targGT().nMarkers() || start>=end) {
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

    private static BitArray[] phase(FixedPhaseData fpd, int start, int end, long seed) {
        int overlap = fpd.stage1Overlap();
        GT targGT = fpd.stage1TargGT();
        PbwtRecPhaser recPhaser = new PbwtRecPhaser(fpd);
        RevPbwtPhaser revPbwt = new RevPbwtPhaser(fpd, start, end, seed);
        boolean[] missingGT = new boolean[targGT.nSamples()];
        boolean[] unphHet = new boolean[targGT.nSamples()];
        int[] lastHet = IntStream.range(0, targGT.nSamples())
                .map(j -> -1)
                .toArray();
        int[] alleles = new int[fpd.nHaps()];

        BitArray[] mkrToBits = new BitArray[end-start];
        int lastM = -1;
        for (int m=start; m<end; ++m) {
            recPhaser.phase(lastM, alleles, m, missingGT, unphHet);
            if (m>=overlap) {
                finishPhasing(mkrToBits, revPbwt, start, m, alleles, lastHet, unphHet);
            }
            mkrToBits[m-start] = storePhasing(targGT, m, alleles);
            updateLastHet(alleles, missingGT, lastHet, m);
            lastM = m;
        }
        return mkrToBits;
    }

    private static void finishPhasing(BitArray[] markerToBits, RevPbwtPhaser revPbwt,
            int start, int m,
            int[] alleles, int[] lastHet, boolean[] unphHet) {
        for (int s=0; s<unphHet.length; ++s) {
            int h1 = s<<1;
            int h2 = h1 | 0b1;
            if (unphHet[s]) {
                int prevHet = lastHet[s];
                if (prevHet>=0) {
                    int a1 = revPbwt.allele(prevHet, h1);
                    int a2 = revPbwt.allele(prevHet, h2);
                    int b1 = revPbwt.allele(m, h1);
                    int b2 = revPbwt.allele(m, h2);
                    boolean revSamePhase = ((a1<a2)==(b1<b2));
                    int bitsPerAllele = revPbwt.bitsPerAllele(prevHet);

                    int c1 = allele(markerToBits[prevHet-start], h1, bitsPerAllele);
                    int c2 = allele(markerToBits[prevHet-start], h2, bitsPerAllele);
                    boolean fwdSamePhase = (c1<c2)==(alleles[h1]<alleles[h2]);
                    if (revSamePhase!=fwdSamePhase) {
                        int tmp = alleles[h1];
                        alleles[h1] = alleles[h2];
                        alleles[h2] = tmp;
                    }
                }
                unphHet[s] = false;
            }
            else {
                if (alleles[h1] == -1) {
                    alleles[h1] = imputeAllele(markerToBits, revPbwt, start,
                            lastHet[s], m, h1);
                }
                if (alleles[h2] == -1) {
                    alleles[h2] = imputeAllele(markerToBits, revPbwt, start,
                            lastHet[s], m, h2);
                }
            }
        }
    }

    private static int imputeAllele(BitArray[] markerToBits, RevPbwtPhaser revPbwt,
            int start, int lastHet, int m, int hap) {
        if (lastHet<0) {
            return revPbwt.allele(m, hap);
        }
        int compHap = hap ^ 0b1;
        int a1 = revPbwt.allele(lastHet, hap);
        int a2 = revPbwt.allele(lastHet, compHap);

        int bitsPerAllele = revPbwt.bitsPerAllele(lastHet);
        int b1 = allele(markerToBits[lastHet-start], hap, bitsPerAllele);
        int b2 = allele(markerToBits[lastHet-start], compHap, bitsPerAllele);
        if (a1<a2==b1<b2) {
            return revPbwt.allele(m, hap);
        }
        else {
            return revPbwt.allele(m, compHap);
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

    private static void updateLastHet(int[] alleles, boolean[] missingGT, int[] lastHet, int m) {
        for (int s=0; s<missingGT.length; ++s) {
            int h1 = s<<1;
            int h2 = h1 | 0b1;
            if (missingGT[s]==false && alleles[h1]!=alleles[h2]) {
                lastHet[s] = m;
            }
        }
    }

    private static int allele(BitArray bits, int hap, int bitsPerAllele) {
        int bStart = hap*bitsPerAllele;
        if (bitsPerAllele==1) {
            return bits.get(bStart) ? 1 : 0;
        }
        else {
            int bEnd = bStart + bitsPerAllele;
            int allele = 0;
            int mask = 1;
            for (int j=bStart; j<bEnd; ++j) {
                if (bits.get(j)) {
                    allele |= mask;
                }
                mask <<= 1;
            }
            return allele;
        }
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
        int index = marker-start;
        return allele(markerToBits[index], hap, bitsPerAllele[index]);
    }
}
