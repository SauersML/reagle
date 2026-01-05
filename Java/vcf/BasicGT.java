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

import blbutil.Const;
import java.util.stream.IntStream;

/**
 * <p>Class {@code BasicGT} represents genotypes for a list of markers and
 * samples.</p>
 *
 * <p>Instances of class {@code BasicGT} are immutable</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class BasicGT implements GT {

    private final Samples samples;
    private final Markers markers;
    private final GTRec[] recs;
    private final boolean isRefData;

    /**
     * Returns the genotype index corresponding to the specified unordered
     * alleles.
     * @param a1 the first allele index of an unordered genotype
     * @param a2 the second allele index of an unordered genotype
     * @return the genotype index corresponding to the specified unordered
     * alleles
     * @throws IllegalArgumentException if {@code a1 < 0 || a2 < 0}
     */
    public static int genotype(int a1, int a2) {
        if (a1<=a2) {
            if (a1 < 0) {
                String s = "allele < 0: " + a1 + " " + a2;
                throw new IllegalArgumentException(s);
            }
            return (a2*(a2+1))/2 + a1;
        }
        else {
            if (a2<0) {
                String s = "allele < 0: " + a1 + " " + a2;
                throw new IllegalArgumentException(s);
            }
            return (a1*(a1+1))/2 + a2;
        }
    }

    /**
     * Constructs a {@code BasicGT} instance from the specified data
     * @param recs genotype records for a list of markers
     * @throws IllegalArgumentException
     * if elements of {@code recs} corresponding to the same chromosome
     * are not contiguous and sorted in chromosome position order
     * @throws IllegalArgumentException if {@code recs.length == 0}
     * @throws IllegalArgumentException if any two elements of
     * {@code recs} correspond to the same genetic marker
     * @throws IllegalArgumentException if
     * {@code (recs[j].samples().equals(recs[k].samples()) == false)} for any
     * {@code j, k} satisfying {@code (0 <= j && j < k && j < recs.length)}
     * @throws NullPointerException if {@code recs == null}
     * @throws NullPointerException if {@code (recs[j] == null)} any {@code j}
     * satisfying {@code (0 <= j && j < recs.length)}
     */
    public BasicGT(GTRec[] recs) {
        this(recs[0].samples(), recs);
    }

    /**
     * Constructs a {@code BasicGT} instance from the specified data
     * @param samples the list of samples with genotype data
     * @param recs genotype records for a list of markers
     * @throws IllegalArgumentException
     * if elements of {@code recs} corresponding to the same chromosome
     * are not contiguous and sorted in chromosome position order
     * @throws IllegalArgumentException if any two elements of
     * {@code recs} correspond to the same genetic marker
     * @throws IllegalArgumentException if
     * {@code (recs[j].samples().equals(samples) == false)} for any {@code j}
     * satisfying {@code (0 <= j && j < recs.length)}
     * @throws NullPointerException if {@code samples == null || recs == null}
     * @throws NullPointerException if {@code (recs[j] == null)} any {@code j}
     * satisfying {@code (0 <= j && j < recs.length)}
     */
    public BasicGT(Samples samples, GTRec[] recs) {
        checkSamples(samples, recs);
        this.markers = markers(recs);
        this.samples = samples;
        this.recs = recs.clone();
        this.isRefData = isRefData(recs);
    }

    /**
     * Constructs a {@code BasicGT} instance from the specified data.
     * @param markers the list of markers with genotype data
     * @param samples the list of samples with genotype data
     * @param recs the genotype data for each marker
     * @throws IllegalArgumentException if
     * {@code (recs[j].marker().equals(markers.marker(j)) == false)} for any
     * {@code j} satisfying {@code (0 <= j && j < recs.length)}
     * @throws IllegalArgumentException if
     * {@code (recs[j].samples().equals(samples) == false)} for any {@code j}
     * satisfying {@code (0 <= j && j < recs.length)}
     * @throws NullPointerException if
     * {@code (markers == null || samples == null || recs == null)}
     * @throws NullPointerException if {@code (recs[j] == null)} any {@code j}
     * satisfying {@code (0 <= j && j < recs.length)}
     */
    public BasicGT(Markers markers, Samples samples, GTRec[] recs) {
        checkMarkersAndSamples(markers, samples, recs);
        this.markers = markers;
        this.samples = samples;
        this.recs = recs.clone();
        this.isRefData = isRefData(recs);
    }

    private static void checkSamples(Samples samples, GTRec[] recs) {
        for (int j=0; j<recs.length; ++j) {
            if (recs[j].samples().equals(samples)==false) {
                throw new IllegalArgumentException("inconsistent samples");
            }
        }
    }

    private static void checkMarkersAndSamples(Markers markers, Samples samples,
            GTRec[] recs) {
        for (int j=0; j<recs.length; ++j) {
            if (recs[j].marker().equals(markers.marker(j))==false) {
                throw new IllegalArgumentException("inconsistent markers");
            }
            if (recs[j].samples().equals(samples)==false) {
                throw new IllegalArgumentException("inconsistent samples");
            }
        }
    }

    private static Markers markers(GTRec[] recs) {
        Marker[] markers = new Marker[recs.length];
        for (int j=0; j<markers.length; ++j) {
            markers[j] = recs[j].marker();
        }
        return Markers.create(markers);
    }

    private static boolean isRefData(GTRec[] recs) {
        boolean isRefData = true;
        for (int j=0; j<recs.length && isRefData==true; ++j) {
            if (recs[j].isPhased()==false) {
                isRefData = false;
            }
        }
        return isRefData;
    }

    @Override
    public boolean isReversed() {
        return false;
    }

    @Override
    public int nMarkers() {
        return recs.length;
    }

    @Override
    public Marker marker(int markerIndex) {
        return markers.marker(markerIndex);
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
        return isRefData;
    }

    @Override
    public int allele(int marker, int hap) {
        return recs[marker].get(hap);
    }

    /**
     * Returns a {@code BasicGT} instance restricted to genotype data for
     * the specified markers.
     * @param gt the {@code BasicGT} instance to be restricted
     * @param indices a list of distinct marker indices (from
     * {@code this.markers())} in increasing order
     * @return a {@code GT} instance restricted to genotype data for
     * the specified markers
     *
     * @throws IndexOutOfBoundsException if there exists {@code j} such that
     * {@code (0 <= j && j < indices.length)} such that
     * {@code (indices[j] < 0 || indices[j] >= gt.nMarkers())}
     * @throws IllegalArgumentException if there exists {@code j} such that
     * {@code (1 <= j && j < indices.length)} such that
     * {@code (indices[j] <= indice[j - 1])}
     * @throws NullPointerException if {@code gt == null || indices == null}
     */
    public static BasicGT restrict(BasicGT gt, int[] indices) {
        Markers restrictMarkers = gt.markers.restrict(indices);
        GTRec[] restrictedRecs = IntStream.range(0, indices.length)
                .mapToObj(j -> gt.recs[indices[j]])
                .toArray(GTRec[]::new);
        return new BasicGT(restrictMarkers, gt.samples, restrictedRecs);
    }

    @Override
    public BasicGT restrict(Markers restrictedMarkers, int[] indices) {
        GTRec[] restrictedRecs = IntStream.range(0, indices.length)
                .mapToObj(j -> recs[indices[j]])
                .toArray(GTRec[]::new);
        return new BasicGT(restrictedMarkers, samples, restrictedRecs);
    }

    @Override
    public BasicGT restrict(int start, int end) {
        Markers restrictMarkers = markers.restrict(start, end);
        GTRec[] restrictRecs = IntStream.range(start, end)
                .mapToObj(j -> recs[j])
                .toArray(GTRec[]::new);
        return new BasicGT(restrictMarkers, samples, restrictRecs);
    }

    /**
     * Returns a string representation of {@code this}.  The exact details
     * of the representation are unspecified and subject to change.
     * @return a string representation of {@code this}
     */
    @Override
    public String toString() {
        StringBuilder sb  = new StringBuilder();
        sb.append("[BasicGT: nMarkers=");
        sb.append(nMarkers());
        sb.append(" nSamples=");
        sb.append(nSamples());
        for (GTRec rec : recs) {
            sb.append(Const.nl);
            sb.append(rec);
        }
        sb.append(']');
        return sb.toString();
    }
}
