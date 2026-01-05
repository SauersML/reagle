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

import blbutil.DoubleArray;
import blbutil.FloatArray;
import ints.IntArray;
import ints.IntList;
import ints.WrappedIntArray;
import java.util.stream.IntStream;
import vcf.GT;
import vcf.MarkerMap;

/**
 * <p>Class {@code Ibs2Markers} stores the markers and intervals that are
 * used to detect IBS2 segments.</p>
 *
 * <p>Instances of {@code Ibs2Markers} are immutable.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class Ibs2Markers {

    private static final float MAX_MISS_FREQ = 0.1f;
    private static final float MIN_MINOR_FREQ = 0.1f;

    private static final int MIN_MARKER_CNT = 50;
    private static final float MIN_INTERMARKER_CM = 0.02f;

    private final Boolean[] useMarker;
    private final IntArray stepStarts;

    /**
     * Constructs a new {@code Ibs2Markers} instance from the specified data.
     *
     * @param targGT target genotype data
     * @param map a list whose {@code j}-th element is the genetic map position
     * of the {@code j}-th the marker
     * @param maf a list whose {@code j}-th element is the estimated minor
     * allele frequency of the {@code j}-th the marker
     * @throws IllegalArgumentException if
     * {@code targGT.nMarkers() != map.genPos().size()}
     * @throws IllegalArgumentException if {@code targGT.nMarkers() != maf.size()}
     * @throws NullPointerException if
     * {@code (targGT == null || map == null || maf == null)}
     */
    public Ibs2Markers(GT targGT, MarkerMap map, FloatArray maf) {
        if (map.genPos().size()!=targGT.nMarkers()) {
            throw new IllegalArgumentException(
                    String.valueOf(map.genPos().size()));
        }
        if (maf.size()!=targGT.nMarkers()) {
            throw new IllegalArgumentException(String.valueOf(maf.size()));
        }
        int maxMiss = (int) Math.ceil(MAX_MISS_FREQ*targGT.nHaps());
        Boolean[] useMarker0 = IntStream.range(0, targGT.nMarkers())
                .parallel()
                .mapToObj(m -> useMarker(targGT,  m, maf, maxMiss))
                .toArray(Boolean[]::new);
        this.stepStarts = stepStarts(useMarker0, map);
        this.useMarker = useMarker0;
    }

    private static Boolean useMarker(GT targGT, int m, FloatArray maf,
            int maxMissCnt) {
        int missCnt = 0;
        if (maf.get(m)>=MIN_MINOR_FREQ) {
            for (int h=0, n=targGT.nHaps(); h<n; ++h) {
                if (targGT.allele(m, h)<0) {
                    ++missCnt;
                }
            }
            return missCnt<=maxMissCnt;
        }
        else {
            return false;
        }
    }

    private static IntArray stepStarts(Boolean[] useMarker0, MarkerMap map) {
        DoubleArray genPos = map.genPos();
        int nMarkers = genPos.size();

        IntList indices = new IntList(genPos.size()>>6);
        int lastStart = 0;
        int nextStart = nextStart(genPos, lastStart, useMarker0);
        // following code combines the last two steps
        while (nextStart<nMarkers) {
            indices.add(lastStart);
            lastStart = nextStart;
            nextStart = nextStart(genPos, nextStart, useMarker0);
        }
        return new WrappedIntArray(indices);
    }

    private static int nextStart(DoubleArray genPos, int start,
            Boolean[] useMarker0) {
        double cmPos = genPos.get(start);
        double minCmPos = cmPos + MIN_INTERMARKER_CM;
        int nextStart = start + 1;
        int mkrCnt = 0;
        while (nextStart<useMarker0.length && mkrCnt<MIN_MARKER_CNT) {
            if (useMarker0[nextStart]) {
                cmPos = genPos.get(nextStart);
                if (cmPos<minCmPos) {
                    useMarker0[nextStart] = Boolean.FALSE;
                }
                else {
                    ++mkrCnt;
                    minCmPos = cmPos + MIN_INTERMARKER_CM;
                }
            }
            ++nextStart;
        }
        return nextStart;
    }

    /**
     * Returns the number of markers.
     * @return the number of markers
     */
    public int nMarkers() {
        return useMarker.length;
    }

    /**
     * Returns an increasing list of marker indices that are used to detect
     * IBS2 segments in the specified interval
     * @param start the first marker in the interval (inclusive)
     * @param end the last marker in the interval (exclusive)
     * @return an increasing list of marker indices that are used to detect
     * IBS2 segments in the specified interval
     * @throws IllegalArgumentException if
     * {@code start < 0 || end < start || end > this.nMarkers}
     */
    public int[] markers(int start, int end) {
        if (start<0 || end>useMarker.length) {
            throw new IllegalArgumentException(String.valueOf(start));
        }
        if (end>useMarker.length) {
            throw new IllegalArgumentException(String.valueOf(end));
        }
        IntList markers = new IntList(end - start);
        for (int m=start; m<end; ++m) {
            if (useMarker[m]) {
                markers.add(m);
            }
        }
        return markers.toArray();
    }

    /**
     * Returns the first marker index in each step in increasing order.
     * @return the first marker index in each step
     */
    public IntArray stepStarts() {
        return stepStarts;
    }
}
