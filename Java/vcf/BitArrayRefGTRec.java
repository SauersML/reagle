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
package vcf;

import blbutil.BitArray;
import java.util.stream.IntStream;
import phase.EstPhase;
import phase.SamplePhase;

/**
 * <p>Class {@code BitArrayRefGTRec} represents phased, nonmissing, genotypes
 * for a list of samples at a single marker. Instances of class
 * {@code BitArrayRefGTRec} store haplotype alleles in bit sets.</p>
 *
 * <p>Instances of class {@code BitArrayRefGTRec} are immutable.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class BitArrayRefGTRec implements GTRec {

    private final int bitsPerAllele;
    private final Marker marker;
    private final Samples samples;
    private final BitArray alleles;

    /**
     * Returns the current estimated phased, non-missing genotypes. This
     * method converts column-major data into row-major data.
     * @param estPhase the current estimated phased genotypes for each target
     * sample
     * @return the current estimated phased, non-missing genotypes
     * @throws NullPointerException if {@code estPhase == null}
     */
    public static BitArrayRefGTRec[] toBitArrayRefGTRecs(EstPhase estPhase) {
        GT gt = estPhase.fpd().stage1TargGT();
        Markers markers = gt.markers();
        Samples samples = gt.samples();
        BitArray[] bitLists = SamplePhase.toBitLists(estPhase);
        return IntStream.range(0, bitLists.length)
                .parallel()
                .mapToObj(m -> new BitArrayRefGTRec(markers.marker(m), samples, bitLists[m]))
                .toArray(BitArrayRefGTRec[]::new);
    }

//    NB: The toBitArrayRefGTRecs() method is commented-out because it is
//        not currently used
//    ToDo: decide whether to delete toBitArrayRefGTRecs() method after
//          XRefGT amd BrefGT code stabilizes.
//
//    /**
//     * Returns the phased, non-missing genotypes as a {@code BitArrayRefGTRec[]}
//     * array.  This method converts column-major data into row-major data.
//     * @param gt the genotype data
//     * @param nThreads the maximum number of computational threads for object
//     * construction
//     * @return the phased, non-missing genotypes as a {@code BitArrayRefGTRec[]}
//     * array
//     * @throws IllegalArgumentException if {@code nThreads < 1}
//     * @throws NullPointerException if {@code gt == null}
//     */
//    public static BitArrayRefGTRec[] toBitArrayRefGTRecs(XRefGT gt, int nThreads) {
//        Markers markers = gt.markers();
//        Samples samples = gt.samples();
//        BitArray[] bitLists = gt.toBitLists(nThreads);
//        return IntStream.range(0, bitLists.length)
//                .parallel()
//                .mapToObj(m -> new BitArrayRefGTRec(markers.marker(m), samples, bitLists[m]))
//                .toArray(BitArrayRefGTRec[]::new);
//    }

    private BitArrayRefGTRec(Marker marker, Samples samples, BitArray alleles) {
        this.bitsPerAllele = marker.bitsPerAllele();
        this.marker = marker;
        this.samples = samples;
        this.alleles = alleles;
    }

    @Override
    public Samples samples() {
        return samples;
    }

    @Override
    public int size() {
        return samples.size()<<1;
    }

    @Override
    public Marker marker() {
        return marker;
    }

    @Override
    public boolean isPhased() {
        return true;
    }

    @Override
    public boolean isPhased(int sample) {
        return true;
    }

    @Override
    public int get(int hap) {
        return allele(hap);
    }

    private int allele(int hap) {
        int start = bitsPerAllele*hap;
        int end = start + bitsPerAllele;
        int allele = 0;
        int mask = 1;
        for (int j=start; j<end; ++j) {
            if (alleles.get(j)) {
                allele += mask;
            }
            mask <<= 1;
        }
        return allele;
    }

    /**
     * Returns the data represented by {@code this} as a VCF
     * record with a GT format field. The returned VCF record
     * will have missing QUAL and INFO fields, will have "PASS"
     * in the filter field, and will have a GT format field.
     * @return the data represented by {@code this} as a VCF
     * record with a GT format field
     */
    @Override
    public String toString() {
        return GTRec.toVcfRec(this);
    }
}
