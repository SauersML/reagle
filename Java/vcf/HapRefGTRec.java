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

import ints.IndexArray;
import ints.IntArray;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * <p>Class {@code HapRefGTRec}  represents phased, non-missing
 * genotypes for a list of reference samples at a single marker.
 * Genotype emission probabilities are determined by the sample
 * genotypes.
 * </p>
 * <p>Instances of class {@code HapRefGTRec} are immutable.
 * </p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class HapRefGTRec implements RefGTRec {

    private final Marker marker;
    private final Samples samples;
    private final IntArray hapToSeq;
    private final IntArray seqToAllele;

    /**
     * Creates a new {@code HapRefGTRec} instance with phased,
     * non-missing genotypes from the specified marker, samples,
     * and haplotype alleles.  The contract for the constructed object
     * is undefined if any element of {@code hapToSeq} is negative or
     * greater than or equal to {@code hapToAllele.size()} or if any element
     * of {@code hapToAllele} is negative or greater than or equal to
     * {@code marker.nAlleles()}.
     *
     * @param marker the marker
     * @param samples the samples
     * @param hapToSeq an array whose {@code j}-th element is the index
     * of the distinct allele sequence carried by the {@code j}-th haplotype
     * @param seqToAllele an array whose {@code j}-th element is the marker
     * allele carried by the {@code j}-th distinct allele sequence
     *
     * @throws IllegalArgumentException if
     * {@code hapToSeq.size() != 2*samples.size()}
     * @throws NullPointerException if any parameter is {@code null}
     */
    public HapRefGTRec(Marker marker, Samples samples, IntArray hapToSeq,
        IntArray seqToAllele) {
        if (hapToSeq.size() != 2*samples.size()) {
            throw new IllegalArgumentException("inconsistent data");
        }
        this.marker = marker;
        this.samples = samples;
        this.hapToSeq = hapToSeq;
        this.seqToAllele = seqToAllele;
    }

    @Override
    public boolean isPhased(int sample) {
        if (sample < 0 || sample >= this.samples().size()) {
            throw new IndexOutOfBoundsException(String.valueOf(sample));
        }
        return true;
    }

    /**
     * Returns {@code true}.
     * @return {@code true}
     */
    @Override
    public boolean isPhased() {
        return true;
    }


    @Override
    public Samples samples() {
        return samples;
    }

    @Override
    public int size() {
        return hapToSeq.size();
    }

    @Override
    public Marker marker() {
        return marker;
    }

    @Override
    public int[][] alleleToHaps() {
        int[] alCnts = alleleCounts();
        int majAllele = 0;
        for (int al=1; al<alCnts.length; ++al) {
            if (alCnts[al] > alCnts[majAllele]) {
                majAllele = al;
            }
        }
        int[][] hapIndices = new int[alCnts.length][];
        for (int al=0; al<alCnts.length; ++al) {
            if (al!=majAllele) {
                hapIndices[al] = new int[alCnts[al]];
            }
        }
        Arrays.fill(alCnts, 0);
        for (int h=0, n=size(); h<n; ++h) {
            int al = get(h);
            if (al!=majAllele) {
                hapIndices[al][alCnts[al]++] = h;
            }
        }
        return hapIndices;
    }

    @Override
    public IndexArray hapToAllele() {
        int[] alleles = IntStream.range(0, size())
                .map(h -> seqToAllele.get(hapToSeq.get(h)))
                .toArray();
        return new IndexArray(alleles, marker.nAlleles());
    }

    @Override
    public int nAlleleCodedHaps() {
        return IntArrayRefGTRec.nonNullCnt(alleleToHaps());
    }

    @Override
    public boolean isAlleleCoded() {
        return false;
    }

    @Override
    public int majorAllele() {
        return majorAllele(alleleCounts());
    }

    private int majorAllele(int[] alCnts) {
        int majAl = 0;
        for (int al=1; al<alCnts.length; ++al) {
            if (alCnts[al]>alCnts[majAl]) {
                majAl = al;
            }
        }
        return majAl;
    }

    @Override
    public int[] alleleCounts() {
        int[] alCnts = new int[marker.nAlleles()];
        for (int h=0, n=size(); h<n; ++h) {
            ++alCnts[get(h)];
        }
        return alCnts;
    }

    @Override
    public int alleleCount(int allele) {
        int[] alCnts = alleleCounts();
        if (allele==majorAllele(alCnts)) {
            throw new IllegalArgumentException("major allele");
        }
        else {
            return alCnts[allele];
        }
    }

    @Override
    public int get(int hap) {
        return seqToAllele.get(hapToSeq.get(hap));
    }

    @Override
    public int hapIndex(int allele, int copy) {
        int[][] hapIndices = alleleToHaps();
        if (hapIndices[allele]==null) {
            throw new IllegalArgumentException("major allele");
        }
        else {
            return hapIndices[allele][copy];
        }
    }

    @Override
    public boolean isCarrier(int allele, int hap) {
        return get(hap)==allele;
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

    @Override
    public int nMaps() {
        return 2;
    }

    @Override
    public IntArray[] maps() {
        return new IntArray[] {hapToSeq, seqToAllele};
    }

    @Override
    public IntArray map(int index) {
        switch (index) {
            case 0:
                return hapToSeq;
            case 1:
                return seqToAllele;
            default:
                throw new IndexOutOfBoundsException(String.valueOf(index));
        }
    }
}
