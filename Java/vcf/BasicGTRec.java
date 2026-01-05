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

/**
 * <p>Class {@code BasicGTRec} stores genotypes for a list of samples
 * at a single marker. The phased or unphased status of each genotype is
 * stored.</p>
 *
 * <p>Instances of class {@code BasicGTRec} are immutable.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class BasicGTRec implements GTRec {

    private final Marker marker;
    private final Samples samples;
    private final int[] alleles;
    private final boolean[] isPhased;
    private final boolean allPhased;

    /**
     * Constructs a new {@code BasicGTRec} instance representing
     * the specified VCF record's GT format field data.
     *
     * @param recParser the VCF record genotype data
     * @throws IllegalArgumentException if a format error is detected
     * in the VCF record
     * @throws NullPointerException if {@code recParser == null}
     */
    public BasicGTRec(VcfRecGTParser recParser) {
        int nSamples = recParser.samples().size();
        int nHaps = nSamples<<1;
        int[] alleles0 = new int[nHaps];
        boolean[] isPhased0 = new boolean[nSamples];
        this.allPhased = recParser.storeAlleles(alleles0, isPhased0);
        this.marker = recParser.marker();
        this.samples = recParser.samples();
        this.alleles = alleles0;
        this.isPhased = isPhased0;
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
        return allPhased;
    }

    @Override
    public boolean isPhased(int sample) {
        return isPhased[sample];
    }

    @Override
    public int get(int hap) {
        return alleles[hap];
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
