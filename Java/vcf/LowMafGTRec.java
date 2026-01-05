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

import java.util.Arrays;

/**
 * <p>Class {@code LowMafGTRc} stores genotypes for a list of samples
 * at a marker. Instances of class {@code LowMafGTRec} store lists of
 * haplotypes carrying each non-major or missing allele. All genotypes are
 * considered to be unphased if any sample has an unphased or missing
 * genotype.</p>
 *
 * <p>Instances of class {@code LowMafGTRec} are immutable.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class LowMafGTRec implements GTRec {

    private final Marker marker;
    private final Samples samples;
    private final int nHaps;
    private final int majorAllele;
    private final int[][] hapIndices;
    private final int[] missingSamples;
    private final boolean isPhased;

    /**
     * Constructs a new {@code LowMafGTRec} representing
     * the specified VCF record's GT format field data.
     * @param listRep the VCF record genotype data
     * @throws NullPointerException if {@code listRep == null}
     */
    public LowMafGTRec(VcfRecGTParser.HapListRep listRep) {
        this.marker = listRep.marker();
        this.samples = listRep.samples();
        this.nHaps = samples.size()<<1;
        this.majorAllele = listRep.majorAllele();
        this.hapIndices = listRep.hapLists(true);
        this.missingSamples = listRep.missingSamples();
        this.isPhased = listRep.isPhased();
    }

    @Override
    public boolean isPhased(int sample) {
        if (sample < 0 || sample >= this.samples().size()) {
            throw new IndexOutOfBoundsException(String.valueOf(sample));
        }
        return isPhased;
    }

    @Override
    public boolean isPhased() {
        return isPhased;
    }

    @Override
    public Samples samples() {
        return samples;
    }


    @Override
    public int size() {
        return nHaps;
    }

    @Override
    public Marker marker() {
        return marker;
    }

    @Override
    public int get(int hap) {
        if (hap < 0 || hap >= nHaps) {
            throw new IndexOutOfBoundsException(String.valueOf(hap));
        }
        for (int j=0; j<hapIndices.length; ++j) {
            if (j != majorAllele) {
                if (Arrays.binarySearch(hapIndices[j], hap) >= 0) {
                    return j;
                }
            }
        }
        return Arrays.binarySearch(missingSamples, (hap>>1))>=0 ? -1 : majorAllele;
    }

    /**
     * Returns the major allele.  If there are two major alleles with the same
     * allele count, the smaller allele is returned.
     * @return the major allele
     */
    public int majorAllele() {
        return majorAllele;
    }

    /**
     * Returns the number of copies of the specified allele.
     * @param allele an allele
     * @return the number of copies of the specified allele
     * @throws IndexOutOfBoundsException if
     * {@code allele < 0 || allele >= this.marker().nAlleles()}
     */
    public int alleleCount(int allele) {
        if (allele==majorAllele) {
            int n = nHaps - (missingSamples.length<<1);
            for (int al=0; al<hapIndices.length; ++al) {
                if (al!=majorAllele) {
                    n -= hapIndices.length;
                }
            }
            return n;
        }
        else {
            return hapIndices[allele].length;
        }
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
