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

import beagleutil.PbwtUpdater;
import java.util.BitSet;
import java.util.Optional;
import java.util.stream.IntStream;
import vcf.GT;
import vcf.RefGT;

/**
 * <p>Class {@code PbwtRecPhaser} partially phases and imputes genotypes
 * using the Positional Burrows-Wheeler transform.</p>
 *
 * <p>Instances of class {@code PbwtRecPhaser} are not thread-safe.</p>
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
public class PbwtRecPhaser {

    private final GT targGT;
    private final Optional<RefGT> optRef;
    private final int phasedOverlap;
    private final int nTargHaps;
    private final int nTargSamples;
    private final int nHaps;

    private final int[] a;
    private final int[] invA;
    private final PbwtUpdater pbwt;

    /**
     * Creates a new {@code PbwtPhaser} for the specified data.
     * @param fpd the input data for phasing
     * @throws NullPointerException if {@code fpd == null}
     */
    public PbwtRecPhaser(FixedPhaseData fpd) {
        this.targGT = fpd.stage1TargGT();
        this.optRef = fpd.stage1RefGT();
        this.phasedOverlap = fpd.stage1Overlap();
        this.nTargHaps = targGT.nHaps();
        this.nTargSamples = targGT.nSamples();
        this.nHaps = targGT.nHaps()
                + (optRef.isPresent() ? optRef.get().nHaps() : 0);

        this.a = IntStream.range(0, nHaps).toArray();
        this.invA = IntStream.range(0, nHaps).toArray();
        this.pbwt = new PbwtUpdater(nHaps);
    }

    /**
     * Returns the total number of target and reference haplotypes.
     * @return the total number of target and reference haplotypes
     */
    public int nHaps() {
        return nHaps;
    }

    /**
     * Returns the input target genotypes.
     * @return the input target genotypes
     */
    public GT targGT() {
        return targGT;
    }

    /**
     * Copies input target and reference alleles for the marker {@code nextMkr}
     * to the {@code alleles} array and partially phases and imputes the alleles.
     * When the method returns, the {@code alleles} array will contain the
     * partially phased and imputed genotypes at {@code nextMkr}, the
     * {@code true}  elements of the {@code missing} array will identify
     * target samples whose genotype  at marker {@code nextMkr} is missing
     * before partial phasing and imputation, and the {@code true} elements
     * of the {@code unphHet} array will identify samples with a non-missing,
     * unphased heteroygote genotype at marker {@code nextMkr} that remain
     * unphased after partial phasing and imputation.
     *
     * @param currentMkr the marker index corresponding to the specified
     * phased alleles or {@code -1} if the the alleles array does not contain
     * phased alleles
     * @param alleles phased alleles at marker {@code currentMkr} that
     * will be overwritten with partially phased genotypes at marker
     * {@code nextMkr}
     * @param nextMkr the marker whose partially phased genotypes will be
     * stored in the specified {@code alleles} array
     * @param missing an array whose {@code true} elements identify samples
     * with missing genotype at marker {@code nextMkr} before partial
     * phasing and imputation
     * @param unphHet an array whose {@code true} elements identify samples
     * with non-missing, unphased heterozygote genotype at marker
     * {@code nextMkr} that remain unphased after partial phasing
     * @return the CDF for the allele counts at marker {@code nextMkr}
     *
     * @throws IndexOutOfBoundsException if
     * {@code currentMkr < -1 || currentMkr >= this.targGT().nMarkers()}
     * @throws IndexOutOfBoundsException if
     * {@code nextMkr < 0 || nextMkr >= this.targGT().nMarkers()}
     * @throws IndexOutOfBoundsException if
     * {@code (0 <= currentMkr && currentMkr < this.targGT().nMarkers())} and
     * there exists {@code h} such that {@code (0 <= h && h < alleles.length)}
     * and {@code (alleles[h] < 0
     *          || alleles[h] >= this.targGT.marker(currentMkr).nAlleles())}
     * @throws IndexOutOfBoundsException if
     * {@code alleles.length != this.nHaps()}
     * @throws IndexOutOfBoundsException if
     * {@code missing.length != this.nTargGT().nSamples()}
     * @throws IndexOutOfBoundsException if
     * {@code unphHet.length != this.targGT().nSamples()}
     * @throws NullPointerException if any array is {@code null}
     */
    public int[] phase(int currentMkr, int[] alleles, int nextMkr,
            boolean[] missing, boolean[] unphHet) {
        checkArrays(alleles, missing, unphHet);
        if (currentMkr != -1) {
            pbwt.update(alleles, targGT.marker(currentMkr).nAlleles(), a);
        }
        int[] alCnts = setAlleles(nextMkr, alleles, unphHet, missing);
        if (nextMkr>=phasedOverlap) {
            phase(alleles, unphHet);
        }
        return alCnts;
    }

    private void checkArrays(int[] alleles, boolean[] missing, boolean[] unphHet) {
        if (alleles.length != nHaps) {
            throw new IllegalArgumentException(String.valueOf(alleles.length));
        }
        if (missing.length != nTargSamples) {
            throw new IllegalArgumentException(String.valueOf(missing.length));
        }
        if (unphHet.length != nTargSamples) {
            throw new IllegalArgumentException(String.valueOf(unphHet.length));
        }
    }

    private void phase(int[] alleles, boolean[] unphHet) {
        setInvA(a);
        int threshold = 2;
        boolean changeMade = true;
        while (threshold>0 || changeMade==true) {
            changeMade = false;
            for (int s=0; s<nTargSamples; ++s) {
                if (unphHet[s]) {
                    changeMade |= phase(s, threshold, alleles, unphHet, a);
                }
                else {
                    int h1 = s<<1;
                    int h2 = h1 | 0b1;
                    if (alleles[h1] == -1) {
                        alleles[h1] = impute(alleles, unphHet, a, invA[h1]);
                        changeMade |= (alleles[h1]>=0);
                    }
                    if (alleles[h2] == -1) {
                        alleles[h2] = impute(alleles, unphHet, a, invA[h2]);
                        changeMade |= (alleles[h2]>=0);
                    }
                }
            }
            if (changeMade==false) {
                --threshold;
            }
        }
    }

    private boolean phase(int s, int threshold, int[] alleles,
            boolean[] unphHet, int[] a) {
        int h1 = s<<1;
        int h2 = h1 | 0b1;
        int a1 = alleles[h1];
        int a2 = alleles[h2];
        assert a1>=0 && a2>=0 && a1!=a2;
        int cnt1 = phaseCnt(a, alleles, unphHet, invA[h1], a1, a2);
        int cnt2 = phaseCnt(a, alleles, unphHet, invA[h2], a2, a1);
        int cnt = cnt1 + cnt2;
        if (cnt>=threshold) {
            unphHet[s] = false;
            return true;
        }
        if (cnt <= -threshold) {
            alleles[h1] = a2;
            alleles[h2] = a1;
            unphHet[s] = false;
            return true;
        }
        return false;
    }

    private int phaseCnt(int[] a, int[] alleles, boolean[] unphasedHet,
            int ai, int a1, int a2) {
        int phaseCnt = 0;
        if (ai>0) {
            int h = a[ai-1];
            int s = h>>1;
            if (s>=unphasedHet.length || unphasedHet[s]==false) {
                phaseCnt += phaseCnt(alleles[h], a1, a2);
            }
        }
        if ((ai+1)<alleles.length) {
            int h = a[ai+1];
            int s = h>>1;
            if (s>=unphasedHet.length || unphasedHet[s]==false) {
                phaseCnt += phaseCnt(alleles[h], a1, a2);
            }
        }
        return phaseCnt;
    }

    private static int phaseCnt(int adjacentAllele, int a1, int a2) {
        if (adjacentAllele==a1) {
            return 1;
        }
        if (adjacentAllele==a2) {
            return -1;
        }
        return 0;
    }

    private int impute(int[] inputAlleles, boolean[] unphasedHet, int[] a,
            int ai) {
        int prev = -1;
        int next = -1;
        if (ai>0) {
            int h = a[ai-1];
            int s = h>>1;
            if (s>=unphasedHet.length || unphasedHet[s]==false) {
                prev = inputAlleles[h];
            }
        }
        if ((ai+1)<a.length) {
            int h = a[ai+1];
            int s = h>>1;
            if (s>=unphasedHet.length || unphasedHet[s]==false) {
                next = inputAlleles[h];
            }
        }
        if (prev>=0 && (prev==next || next<0)) {
            return prev;
        }
        else if (prev<0 && next>=0) {
            return next;
        }
        return -1;
    }

    private void setInvA(int[] a) {
        for (int j=0; j<a.length; ++j) {
            invA[a[j]] = j;
        }
    }

    private int[] setAlleles(int m, int[] inputAlleles, boolean[] unphHet,
            boolean[] missing) {
        int[] alCnts = new int[targGT.marker(m).nAlleles()];
        setTargAlleles(m, inputAlleles, unphHet, missing, alCnts);
        if (optRef.isPresent()) {
            setRefAlleles(m, inputAlleles, alCnts);
        }
        // convert allele counts to CDF
        for (int j=1; j<alCnts.length; ++j) {
            alCnts[j] += alCnts[j-1];
        }
        return alCnts;
    }

    private void setTargAlleles(int m, int[] inputAlleles, boolean[] unphHet,
            boolean[] missing, int[] alCnts) {
        for (int s=0; s<nTargSamples; ++s) {
            int h1 = s<<1;
            int h2 = h1 | 0b1;
            int a1 = targGT.allele(m, h1);
            int a2 = targGT.allele(m, h2);
            inputAlleles[h1] = a1;
            inputAlleles[h2] = a2;
            unphHet[s] = (m>=phasedOverlap && (a1>=0 && a2>=0 && a1!=a2));
            missing[s] = a1<0 || a2<0;
            if (a1>=0) {
                ++alCnts[a1];
            }
            if (a2>=0) {
                ++alCnts[a2];
            }
        }
    }

    private void setRefAlleles(int m, int[] inputAlleles, int[] alCnts) {
        assert optRef.isPresent();
        RefGT refGT = optRef.get();
        int refHap = 0;
        for (int h1=nTargHaps; h1<nHaps; h1+=2) {
            int h2 = h1 | 0b1;
            int a1 = refGT.allele(m, refHap++);
            int a2 = refGT.allele(m, refHap++);
            inputAlleles[h1] = a1;
            inputAlleles[h2] = a2;
            ++alCnts[a1];
            ++alCnts[a2];
        }
    }

    /**
     * Returns an array of bit sets.
     * @param nBitSets the size of the returned array
     * @param initBitSetCapacity the initial capacity of each bit set in
     * the returned array
     * @return an array of bit sets
     * @throws NegativeArraySizeException if {
     * {@code nBitSets < 0 || initBitSetCapacity < 0}
     */
    public static BitSet[] bitSets(int nBitSets, int initBitSetCapacity) {
        if (nBitSets<0) {
            throw new IllegalArgumentException(String.valueOf(nBitSets));
        }
        return IntStream.range(0, nBitSets)
                .mapToObj(j -> new BitSet(initBitSetCapacity))
                .toArray(BitSet[]::new);
    }
}
