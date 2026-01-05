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

import ints.IntArray;
import ints.SynchedIntList;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.stream.IntStream;
import vcf.BasicGT;
import vcf.GTRec;
import vcf.Markers;
import vcf.RefGTRec;
import vcf.Samples;
import vcf.Window;

/**
 * <p>Class {@code Stage2Haps} stores phased genotypes.</p>
 *
 * <p>Instances of {@code Stage2Haps} are thread-safe.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class Stage2Haps {

    private final FixedPhaseData fpd;
    private final GTRec[] stage1Recs;
    private final Markers markers;
    private final Samples targSamples;
    private final AtomicReferenceArray<SynchedIntList> rareCarriers;

    /**
     * Constructs a new {@code Stage2Haps} instance from the specified data.
     * @param phaseData the current genotype phase estimates and parameter
     * values
     * @throws NullPointerException if {@code phaseData == null}
     */
    public Stage2Haps(PhaseData phaseData) {
        this.fpd = phaseData.fpd();
        this.stage1Recs = phaseData.estPhase().toGTRecs();
        this.markers = fpd.targGT().markers();
        this.targSamples = fpd.targGT().samples();
        this.rareCarriers = rareCarriers(fpd);
    }

    private static AtomicReferenceArray<SynchedIntList> rareCarriers(
            FixedPhaseData fpd) {
        int size = fpd.targGT().markers().sumAlleles();
        AtomicReferenceArray<SynchedIntList> rareAlToHaps
                = new AtomicReferenceArray<>(size);
        IntStream.rangeClosed(0, fpd.stage1To2().size())
                .parallel()
                .forEach(j -> initList(fpd, j, rareAlToHaps));
        return rareAlToHaps;
    }

    private static void initList(FixedPhaseData fpd, int index,
            AtomicReferenceArray<SynchedIntList> rareCarriers) {
        Markers markers = fpd.targGT().markers();
        IntArray stage1To2 = fpd.stage1To2();
        int start = index==0 ? 0 : stage1To2.get(index-1)+1;
        int end = index==stage1To2.size() ? markers.size() : stage1To2.get(index);
        for (int m=start; m<end; ++m) {
            int nAlleles = markers.marker(m).nAlleles();
            int offset = markers.sumAlleles(m);
            for (int al=0; al<nAlleles; ++al) {
                IntArray ia = fpd.carriers(m, al);
                if (ia!=Window.HIGH_FREQ_ARRAY) {
                    rareCarriers.set(offset+al, new SynchedIntList(ia.size()));
                }
            }
        }
    }

    /**
     * Returns the input data for phasing that is the same in each iteration.
     * @return the input data for phasing that is the same in each iteration
     */
    public FixedPhaseData fpd() {
        return fpd;
    }

    /**
     * Sets the phased genotype if the specified marker was not phased
     * in the first stage. The contract for this method is unspecified if the
     * specified alleles are inconsistent with {@code this.fpd().targGT()}.
     * @param sample a sample index
     * @param marker a marker index
     * @param a1 the first allele index
     * @param a2 the second allele index
     * @throws IndexOutOfBoundsException if
     * {@code marker < 0 || marker >= this.fpd().targGT().markers().size()}
     * @throws IndexOutOfBoundsException if
     * {@code sample < 0 || sample >= this.fpd().targGT().samples().size()}
     * @throws IndexOutOfBoundsException if
     * {@code a1 < 0 || a1 >= this.fpd().targGT().markers().marker(marker).nAlleles()}
     * @throws IndexOutOfBoundsException if
     * {@code a2 < 0 || a2 >= this.fpd().targGT().markers().marker(marker).nAlleles()}
     */
    public void setPhasedGT(int marker, int sample, int a1, int a2) {
        int offset = markers.sumAlleles(marker);
        SynchedIntList list1 = rareCarriers.get(offset + a1);
        if (list1!=null) {
            list1.add(sample<<1);
        }
        SynchedIntList list2 = rareCarriers.get(offset + a2);
        if (list2!=null) {
            list2.add((sample<<1) | 0b1);
        }
    }

    /**
     * Returns a {@code BasicGT} instance containing phased genotypes for
     * the specified markers.
     * @param start the starting marker index (inclusive)
     * @param end the ending marker index (exclusive)
     * @return a {@code BasicGT} instance containing phased genotypes for
     * the specified range of marker indices
     *
     * @throws IndexOutOfBoundsException if
     * {@code start < 0 || end > this.nMarkers()}
     * @throws IllegalArgumentException if {@code start >= end}
     */
    public BasicGT toBasicGT(int start, int end) {
        GTRec[] recs = toGTRecs(start, end);
        if (start==0 && end==markers.size()) {
            return new BasicGT(markers, targSamples, recs);
        }
        else {
            return new BasicGT(targSamples, recs);
        }
    }

    /**
     * Returns a {@code GTRec[]} instance containing phased genotypes for
     * the specified markers.
     * @param start the starting marker index (inclusive)
     * @param end the ending marker index (exclusive)
     * @return a {@code GTRec[]} instance containing phased genotypes for
     * the specified range of marker indices
     *
     * @throws IndexOutOfBoundsException if
     * {@code start < 0 || end > this.nMarkers()}
     * @throws IllegalArgumentException if {@code start >= end}
     */
    public GTRec[] toGTRecs(int start, int end) {
        return IntStream.range(start, end)
                .parallel()
                .mapToObj(m -> gtRec(m))
                .toArray(GTRec[]::new);
    }

    private GTRec gtRec(int m) {
        int m1 = fpd.prevStage1Marker(m);
        int m2 = fpd.stage1To2().get(m1);
        return m==m2 ? stage1Recs[m1] : stage2Rec(m);
    }

    private RefGTRec stage2Rec(int m) {
        int alStart = markers.sumAlleles(m);
        int alEnd = markers.sumAlleles(m+1);
        int[][] hapIndices = new int[alEnd - alStart][];
        int majorAllele = -1;
        for (int j=0; j<hapIndices.length; ++j) {
            SynchedIntList list = rareCarriers.get(alStart + j);
            if (list!=null) {
                hapIndices[j] = list.toArray();
                Arrays.sort(hapIndices[j]);
            }
            else {
                majorAllele = j;
            }
        }
        if (majorAllele == -1) { // can occur if all alleles are rare due to high missing rate
            setMajorAlleleToNull(hapIndices);
        }
        return RefGTRec.alleleRefGTRec(markers.marker(m), targSamples, hapIndices);
    }

    private int setMajorAlleleToNull(int[][] hapIndices) {
        int majAllele = 0;
        for (int j=1; j<hapIndices.length; ++j) {
            if (hapIndices[j].length > hapIndices[majAllele].length) {
                majAllele = j;
            }
        }
        hapIndices[majAllele] = null;
        return majAllele;
    }
}
