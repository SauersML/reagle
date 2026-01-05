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

import ints.IntArray;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * <p>Class {@code RefGT} stores a list of samples and a
 * haplotype pair for each sample.
 * </p>
 * <p>Instances of class {@code BasicRefGT} are immutable.<p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class RefGT implements GT {

    private final Markers markers;
    private final Samples samples;
    private final RefGTRec[] recs;

    /**
     * Constructs a new {@code RefGT} instance.
     * @param markers the sequence of markers
     * @param samples the sequence of samples
     * @param refVcfRecs the sequence of per-marker genotype data
     *
     * @throws IllegalArgumentException if
     * {@code markers.nMarkers() != refVcfRecs.length}
     * @throws IllegalArgumentException if
     * {@code refVcfRecs[k].samples().equals(samples) == false} for any
     * {@code k} satisfying {@code 0 <= k  && k < refVcfRecs.length}
     * @throws IllegalArgumentException if
     * {@code refVcfRecs[k].marker().equals(markers.marker(k)) == false}
     * for any {@code k} satisfying {@code 0 <= k && k < refVcfRecs.length}
     * @throws IllegalArgumentException if
     * {@code refVcfRecs[k].isPhased() == false} for any {@code k}
     * satisfying {@code 0 <= k && k < refVcfRecs.length}
     * @throws NullPointerException if
     * {@code markers == null || samples == null || refVcfRecs == null
     * || refVcfRecs[k] == null} for any {@code k} satisfying
     * {@code 0 <= k && k <= refVcfRecs.length}
     */
    public RefGT(Markers markers, Samples samples, RefGTRec[] refVcfRecs) {
        checkData(markers, samples, refVcfRecs);
        this.markers = markers;
        this.samples = samples;
        this.recs = refVcfRecs.clone();
    }

    /**
     * Constructs a new {@code RefHapPairs} instance.
     * @param refVcfRecs the sequence of per-marker genotype data
     *
     * @throws IllegalArgumentException if {@code refVcfRecs.length == 0}
     * @throws IllegalArgumentException if
     * {@code refVcfRecs[k].samples().equals(samples) == false} for any
     * {@code k} satisfying {@code 0 <= k  && k < refVcfRecs.length}
     * @throws IllegalArgumentException if
     * {@code refVcfRecs[k].isPhased() == false} for any {@code k}
     * satisfying {@code 0 <= k && k < refVcfRecs.length}
     * @throws NullPointerException if
     * {@code samples == null || refVcfRecs == null}
     * @throws NullPointerException if
     * {@code (refVcfRecs[k] == null)} for any {@code k} satisfying
     * {@code (0 <= k && k <= refVcfRecs.length)}
     */
    public RefGT(RefGTRec[] refVcfRecs) {
        this.samples = checkData(refVcfRecs);
        Marker[] ma = Arrays.stream(refVcfRecs)
                .parallel()
                .map(rec -> rec.marker())
                .toArray(Marker[]::new);
        this.markers = Markers.create(ma);
        this.recs = refVcfRecs.clone();
    }

    private static Samples checkData(GTRec[] refVcfRecs) {
        if (refVcfRecs.length==0) {
            String s = "Missing data in VCF file";
            throw new IllegalArgumentException(s);
        }
        Samples samples = refVcfRecs[0].samples();
        for (int j=0; j<refVcfRecs.length; ++j) {
            if (refVcfRecs[j].samples().equals(samples)==false) {
                String s = "sample inconsistency at index " + j;
                throw new IllegalArgumentException(s);
            }
            if (refVcfRecs[j].isPhased()==false) {
                String s = "non-reference data at marker index " + j;
                throw new IllegalArgumentException(s);
            }
        }
        return samples;
    }

    private static void checkData(Markers markers, Samples samples,
            GTRec[] refVcfRecs) {
        if (markers.size()!=refVcfRecs.length) {
            String s = "markers.nMarkers()=" + markers.size()
                    + " refVcfRecs.length=" + refVcfRecs.length;
            throw new IllegalArgumentException(s);
        }
        for (int j=0; j<refVcfRecs.length; ++j) {
            if (refVcfRecs[j].samples().equals(samples)==false) {
                String s = "sample inconsistency at index " + j;
                throw new IllegalArgumentException(s);
            }
            if (refVcfRecs[j].marker().equals(markers.marker(j))==false) {
                String s = "marker inconsistency at index " + j;
                throw new IllegalArgumentException(s);
            }
            if (refVcfRecs[j].isPhased()==false) {
                String s = "non-reference data at marker index " + j;
                throw new IllegalArgumentException(s);
            }
        }
    }

    @Override
    public boolean isReversed() {
        return false;
    }

    @Override
    public int nMarkers() {
       return markers.size();
    }

    @Override
    public Marker marker(int marker) {
        return markers.marker(marker);
    }

    @Override
    public Markers markers() {
        return markers;
    }

    @Override
    public int nHaps() {
        return 2*samples.size();
    }

    @Override
    public int nSamples() {
        return samples.size();
    }

    @Override
    public Samples samples() {
        return samples;
    }

    @Override
    public boolean isPhased() {
        return true;
    }

    @Override
    public int allele(int marker, int haplotype) {
        return recs[marker].get(haplotype);
    }

    /**
     * Returns a {@code RefGT} instance restricted to genotype data for
     * the specified markers.
     * @param refGT the {@code RefGT} instance to be restricted
     * @param indices a list of distinct marker indices (from
     * {@code this.markers())} in increasing order
     * @return a {@code RefGT} instance restricted to genotype data for
     * the specified markers
     *
     * @throws IndexOutOfBoundsException if there exists {@code j} such that
     * {@code (0 <= j && j < indices.length)} such that
     * {@code (indices[j] < 0 || indices[j] >= gt.nMarkers())}
     * @throws IllegalArgumentException if there exists {@code j} such that
     * {@code (1 <= j && j < indices.length)} such that
     * {@code (indices[j] <= indice[j - 1])}
     * @throws NullPointerException if
     * {@code gt == null || indices == null}
     */
    public static RefGT restrict(RefGT refGT, int[] indices) {
        RefGTRec[] rra = new RefGTRec[indices.length];
        for (int j=0; j<rra.length; ++j) {
            if (j>0 && indices[j] <= indices[j-1]) {
                throw new IllegalArgumentException(String.valueOf(indices[j]));
            }
            rra[j] = refGT.recs[indices[j]];
        }
        return new RefGT(refGT.markers, refGT.samples, rra);
    }

    @Override
    public RefGT restrict(Markers markers, int[] indices) {
        RefGTRec[] rra = new RefGTRec[indices.length];
        for (int j=0; j<rra.length; ++j) {
            if (j>0 && indices[j] <= indices[j-1]) {
                throw new IllegalArgumentException(String.valueOf(indices[j]));
            }
            rra[j] = this.recs[indices[j]];
        }
        return new RefGT(markers, samples, rra);
    }

    @Override
    public RefGT restrict(int start, int end) {
        Markers restrictMarkers = markers.restrict(start, end);
        RefGTRec[] restrictRecs = IntStream.range(start, end)
                .mapToObj(j -> recs[j])
                .toArray(RefGTRec[]::new);
        return new RefGT(restrictMarkers, samples, restrictRecs);
    }

    /**
     * Returns the {@code RefGTRec} for the specified marker.
     * @param marker the marker index
     * @return the {@code RefGTRec} for the specified marker
     * @throws IndexOutOfBoundsException if
     * {@code marker < 0 || marker >= this.nMarkers()}
     */
    public RefGTRec get(int marker) {
        return recs[marker];
    }
}
