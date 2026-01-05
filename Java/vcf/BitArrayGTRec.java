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

/**
 * <p>Class {@code BitArrayGT} represents genotypes for a list of samples
 * at a single marker. Instances of class {@code BitArrayGTRec} store
 * haplotype alleles and flags to indicate missing genotypes in bit sets.  All
 * genotypes are considered to be unphased if any sample has an
 * unphased or missing genotype.t</p>
 *
 * <p>Instances of class {@code BitArrayGTRec} are immutable.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class BitArrayGTRec implements GTRec {

    private final int bitsPerAllele;
    private final Marker marker;
    private final Samples samples;
    private final boolean isPhased;

    private final BitArray isMissing;
    private final BitArray alleles;

    /**
     * Constructs a new {@code BitArrayGT} instance representing
     * the specified VCF record's GT format field data.
     *
     * @param recParser the VCF record genotype data
     * @throws IllegalArgumentException if a format error is detected
     * in the VCF record
     * @throws NullPointerException if {@code recParser == null}
     */
    public BitArrayGTRec(VcfRecGTParser recParser) {
        int nSamples = recParser.samples().size();
        int nHaps = nSamples<<1;
        this.bitsPerAllele = recParser.marker().bitsPerAllele();
        this.marker = recParser.marker();
        this.samples = recParser.samples();
        BitArray alleleList = new BitArray(nHaps*bitsPerAllele);
        BitArray isMissingList = new BitArray(nSamples);
        this.isPhased = recParser.storeAlleles(alleleList, isMissingList);
        this.isMissing = isMissingList;
        this.alleles = alleleList;
    }

    /**
     * Constructs a new {@code BitArrayGT} instance representing
     * the specified VCF record's GT format field data.
     *
     * @param hlr the VCF record genotype data
     * @throws IllegalArgumentException if a format error is detected
     * in the VCF record
     * @throws NullPointerException if {@code hlr == null}
     */
    public BitArrayGTRec(VcfRecGTParser.HapListRep hlr) {
        int nSamples = hlr.samples().size();
        int nHaps = nSamples<<1;
        this.bitsPerAllele = hlr.marker().bitsPerAllele();
        this.marker = hlr.marker();
        this.samples = hlr.samples();
        this.isPhased = hlr.isPhased();
        this.isMissing = new BitArray(nSamples);
        this.alleles = new BitArray(nHaps*bitsPerAllele);
        boolean setMajorToNull = false;
        int[][] hapLists = hlr.hapLists(setMajorToNull);
        int[] missingSamples = hlr.missingSamples();
        for (int al=0; al<hapLists.length; ++al) {
            int[] list = hapLists[al];
            for (int h : list) {
                storeAllele(alleles, h, bitsPerAllele, al);
            }
        }
        for (int s : missingSamples) {
            isMissing.set(s);
        }
    }

    private static void storeAllele(BitArray alleles, int hap, int bitsPerAllele,
            int allele) {
        int index = hap*bitsPerAllele;
        int mask = 1;
        for (int k=0; k<bitsPerAllele; ++k) {
            if ((allele & mask)==mask) {
                alleles.set(index);
            }
            ++index;
            mask <<= 1;
        }
    }

    @Override
    public Samples samples() {
        return samples;
    }

    @Override
    public int size() {
        return 2*samples.size();
    }

    @Override
    public Marker marker() {
        return marker;
    }

    @Override
    public boolean isPhased() {
        return isPhased;
    }


    @Override
    public boolean isPhased(int sample) {
        return isPhased;
    }

    @Override
    public int get(int hap) {
        return isMissing.get(hap>>1) ? -1 : allele(hap);
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
