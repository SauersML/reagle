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
import ints.IntList;
import ints.WrappedIntArray;
import java.util.Arrays;
import java.util.Optional;
import java.util.stream.IntStream;

/**
 * Class {@code Window} represents a sliding window of target VCF records
 * or a sliding window of reference and target VCF records.
 *
 * <p>Instances of class {@code Window} are immutable.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class Window {

    /**
     * An array that replaces an array of allele counts for a monomorphic
     * marker.
     */
    public static final IntArray ZERO_FREQ_ARRAY = new WrappedIntArray(new int[0]);

    /**
     * An array that replaces an array of allele counts for a marker
     * with high minor allele frequency.
     */
    public static final IntArray HIGH_FREQ_ARRAY = new WrappedIntArray(new int[0]);

    private final GeneticMap genMap;
    private final int windowIndex;
    private final boolean lastWindow;
    private final MarkerIndices indices;

    private final BasicGT targGT;
    private final RefGT refGT;          // null if no reference haplotypes
    private final RefGT restrictRefGT;  // null if no reference haplotypes

    /**
     * Constructs a new {@code Window} instance from the specified data.
     * @param genMap the genetic map
     * @param windowIndex the window index
     * @param lastWindow {@code true} if this window is the last window
     * in the analysis
     * @param markerIndices marker indices of overlap regions and splice points
     * @param targGT the target genotype data
     * @param refGT the reference genotype data or {@code null} if there is
     * no reference genotype data
     * @throws IllegalArgumentException if
     * {@code markerIndices.nTargMarkers() != targGT.nMarkers()}
     * @throws IllegalArgumentException if
     * {@code refGT != null && markerIndices.nMarkers() != refGT.nMarkers()}
     * @throws NullPointerException if
     * {@code genMap == null || markerIndices == null || targGT == null}
     */
    public Window(GeneticMap genMap, int windowIndex, boolean lastWindow,
            MarkerIndices markerIndices, RefGT refGT, BasicGT targGT) {
        if (genMap==null) {
            throw new NullPointerException(GeneticMap.class.toString());
        }
        if (targGT.nMarkers()!=markerIndices.nTargMarkers()) {
            throw new IllegalArgumentException("inconsistent data");
        }
        if (refGT!=null && refGT.nMarkers()!=markerIndices.nMarkers()) {
            throw new IllegalArgumentException("inconsistent data");
        }
        this.genMap = genMap;
        this.windowIndex = windowIndex;
        this.lastWindow = lastWindow;
        this.indices = markerIndices;
        this.targGT = targGT;
        this.refGT = refGT;
        this.restrictRefGT = refGT==null ? null :
                refGT.restrict(targGT.markers(), markerIndices.targMarkerToMarker());
    }

    /**
     * Returns the genetic map.
     * @return the genetic map
     */
    public GeneticMap genMap() {
        return genMap;
    }

    /**
     * Returns the chromosome index for the markers in this window.
     * @return the chromosome index for the markers in this window
     */
    public int chromIndex() {
        return targGT.marker(0).chromIndex();
    }

    /**
     * Returns {@code true} if this window is the last window in the analysis,
     * and returns {@code false} otherwise.
     * @return {@code true} if this window is the last window in the analysis
     */
    public boolean lastWindow() {
        return lastWindow;
    }

    /**
     * Returns the index of this window. The first window has index 1.
     * @return the index of this window.
     */
    public int windowIndex() {
        return windowIndex;
    }

    /**
     * Returns the input genotypes for the target samples
     * in this window.
     * @return the input genotypes for the target samples
     * in this window
     */
    public BasicGT targGT() {
        return targGT;
    }

    /**
     * Returns the optional phased, nonmissing input genotypes for the reference
     * samples in this window.
     * @return the optional phased, nonmissing input genotypes for the reference
     * samples in this window
     */
    public Optional<RefGT> refGT() {
        return refGT==null ? Optional.empty() : Optional.of(refGT);
    }

    /**
     * Returns the optional phased, nonmissing input genotypes for the reference
     * samples restricted to the target data markers in this window.
     * @return the optional phased, nonmissing input genotypes for the reference
     * samples restricted to the target data markers in this window
     */
    public Optional<RefGT> restrictRefGT() {
        return restrictRefGT==null ? Optional.empty()
                : Optional.of(restrictRefGT);
    }

    /**
     * Return a {@code MarkerIndices} instance which stores the overlap
     * and splice points between this window and the preceding
     * and next windows, and the map between reference and target marker
     * indices.
     * @return the {@code MarkerIndices} for this window
     */
    public MarkerIndices indices() {
        return indices;
    }

    /**
     * <p>Returns the indices of the reference and target carriers for each
     * low-frequency allele at the target data markers.  The reference sample
     * indices will be shifted by the number of target samples so that the
     * first reference sample will have an index equal to the number of target
     * samples. An element of the returned array will be empty and equal to
     * {@code Window.ZERO_FREQ_ARRAY} if the allele has no carriers, and the
     * the element will be empty and equal to {@code Window.HIGH_FREQ_ARRAY}
     * if the number of carriers of the allele exceeds the specified
     * maximum number of carriers.</p>
     *
     * <p>The list of carriers for the {@code k}-th allele of the {@code j}-th
     * target marker are stored in entry {@code (j, k)} of the returned array.
     * if the number of carriers is less than or equal to the specified
     * maximum number of carriers.</p>
     * @param maxCarriers the maximum number of carriers in any list
     * of the returned array.
     * @return the indices of the reference and target carriers for each
     * low-frequency allele
     */
    public IntArray[][] carriers(int maxCarriers) {
        return IntStream.range(0, targGT.nMarkers())
                .parallel()
                .mapToObj(j -> carriers(j, maxCarriers))
                .toArray(IntArray[][]::new);
    }

    private IntArray[] carriers(int m, int maxCarriers) {
        int nAlleles = targGT.marker(m).nAlleles();
        IntList[] carriers = IntStream.range(0, nAlleles)
                .mapToObj(i -> new IntList(16))
                .toArray(IntList[]::new);
        int nTargSamples = targGT.nSamples();
        int nRefSamples = (restrictRefGT!=null) ? restrictRefGT.nSamples() : 0;
        for (int s=0; s<nTargSamples; ++s) {
            int hap1 = s << 1;
            int a1 = targGT.allele(m, hap1);
            int a2 = targGT.allele(m, hap1 | 0b1);
            if (a1>=0 && carriers[a1].size()<=maxCarriers) {
                carriers[a1].add(s);
            }
            if (a2>=0 && a2!=a1 && carriers[a2].size()<=maxCarriers) {
                carriers[a2].add(s);
            }
        }
        if (restrictRefGT!=null) {
            for (int s=0; s<nRefSamples; ++s) {
                int hap1 = s << 1;
                int a1 = restrictRefGT.allele(m, hap1);
                int a2 = restrictRefGT.allele(m, hap1 | 0b1);
                if (a1>=0 && carriers[a1].size()<=maxCarriers) {
                    carriers[a1].add(nTargSamples + s);
                }
                if (a2>=0 && a2!=a1 && carriers[a2].size()<=maxCarriers) {
                    carriers[a2].add(nTargSamples + s);
                }
            }
        }
        return Arrays.stream(carriers)
                .map(list -> {
                    if (list.isEmpty()) {
                        return ZERO_FREQ_ARRAY;
                    }
                    else if (list.size() <= maxCarriers) {
                        return new WrappedIntArray(list);
                    }
                    else {
                        return HIGH_FREQ_ARRAY;
                    }
                })
                .toArray(IntArray[]::new);
    }
}
