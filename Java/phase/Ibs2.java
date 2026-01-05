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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.Predicate;
import java.util.stream.IntStream;
import vcf.GT;
import vcf.MarkerMap;

/**
 * <p>Class {@code Ibs2} stores IBS2 segments that any target sample shares
 * with another target sample.</p>
 *
 * <p>Instances of {@code Ibs2} are immutable.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class Ibs2 {

    private static final float MIN_IBS2_CM = 2.0f;
    private static final float MAX_IBD_GAP_CM = 4.0f;
    private static final Comparator<SampleSeg> sampComparator = SampleSeg.sampleComp();

    private final int nMarkers;
    private final SampleSeg[][] sampleSegs; // [targ sample][segment]

    /**
     * Constructs a new {@code Ibs2} instance from the specified data.
     * @param targGT the target genotype data
     * @param map a list whose {@code j}-th element is the genetic map position
     * of the {@code j}-th the marker
     * @param maf a list whose {@code j}-th element is the estimated
     * minor allele frequency of the {@code j}-th the marker
     * @throws IllegalArgumentException if
     * {@code targGT.nMarkers() != map.genPos().size()}
     * @throws IllegalArgumentException if
     * {@code targGT.nMarkers() != maf.size()}
     * @throws NullPointerException if
     * {@code (targGT == null || map == null || maf == null)}
     */
    public Ibs2(GT targGT, MarkerMap map, FloatArray maf) {
        Ibs2Markers ibs2Markers = new Ibs2Markers(targGT, map, maf);
        Ibs2Sets ibs2Sets = new Ibs2Sets(targGT, ibs2Markers);
        DoubleArray genPos = map.genPos();
        this.nMarkers = targGT.nMarkers();
        this.sampleSegs = IntStream.range(0, targGT.nSamples())
                .parallel()
                .mapToObj(s -> ibs2Segments(targGT, genPos, ibs2Sets, s))
                .toArray(SampleSeg[][]::new);
    }

    private static SampleSeg[] ibs2Segments(GT targGT,  DoubleArray genPos,
            Ibs2Sets ibs2Sets, int sample) {
        SampleSeg[] segList = ibs2Sets.segList(sample);
        Arrays.sort(segList, sampComparator);
        segList = mergeSegments(segList, genPos);
        segList = extendSegments(targGT, sample, segList);
        segList = mergeSegments(segList, genPos);
        return applyLengthFilter(segList, genPos);
    }

    private static SampleSeg[] mergeSegments(SampleSeg[] list, DoubleArray genPos)  {
        if (list.length<2) {
            return list;
        }
        List<SampleSeg> merged = new ArrayList<>();
        SampleSeg prev = list[0];
        for (int j=1; j<list.length; ++j) {
            SampleSeg next = list[j];
            if (prev.sample()==next.sample()
                    && gapCM(prev, next, genPos) <= MAX_IBD_GAP_CM) {
                // Two segments can have identical end markers after extension
                // because homozygous regions are excluded in initial IBS2
                // discovery but are included in extendSegments() method.
                assert prev.inclEnd()<=next.inclEnd();
                prev = new SampleSeg(prev.sample(), prev.start(), next.inclEnd());
            }
            else {
                merged.add(prev);
                prev = next;
            }
        }
        merged.add(prev);
        return merged.toArray(new SampleSeg[0]);
    }

    private static double gapCM(SampleSeg prev, SampleSeg next,
            DoubleArray genPos) {
        return genPos.get(next.start()) - genPos.get(prev.inclEnd());
    }

    private static SampleSeg[] extendSegments(GT targGT, int sample,
            SampleSeg[] list) {
        return Arrays.stream(list)
                .map(ss -> extend(targGT, sample, ss))
                .toArray(SampleSeg[]::new);
        }

    private static SampleSeg extend(GT targGT, int sample, SampleSeg ss) {
        int nMarkers = targGT.nMarkers();
        int sample2 = ss.sample();
        int inclStart = ss.start();
        int exclEnd = ss.inclEnd() + 1;
        while (inclStart>0 && ibs2(targGT, inclStart-1, sample, sample2)) {
            --inclStart;
        }
        while (exclEnd<nMarkers && ibs2(targGT, exclEnd, sample, sample2)) {
            ++exclEnd;
        }
        return new SampleSeg(sample2, inclStart, exclEnd - 1);
    }

    private static boolean ibs2(GT targGT, int m, int s1, int s2) {
        int hap1 = s1 << 1;
        int hap2 = s2 << 1;
        int a1 = targGT.allele(m, hap1);
        int a2 = targGT.allele(m, hap1 | 0b1);
        int b1 = targGT.allele(m, hap2);
        int b2 = targGT.allele(m, hap2 | 0b1);
        return arePhaseConsistent(a1, a2, b1, b2)
                || arePhaseConsistent(a1, a2, b2, b1);
    }

    private static boolean arePhaseConsistent(int a1, int a2, int b1, int b2) {
        return (a1<0 || b1<0 || a1==b1) && (a2<0 || b2<0 || a2==b2);
    }

    private static SampleSeg[] applyLengthFilter(SampleSeg[] list, DoubleArray genPos) {
        Predicate<SampleSeg> predicate =
                ss -> (genPos.get(ss.inclEnd()) - genPos.get(ss.start())) >= MIN_IBS2_CM;
        return Arrays.stream(list)
                .filter(predicate)
                .toArray(SampleSeg[]::new);
    }

    /**
     * Returns the number of markers.
     * @return the number of markers
     */
    public int nMarkers() {
        return nMarkers;
    }

    /**
     * Returns the number of target samples.
     * @return the number of target samples
     */
    public int nTargSamples() {
        return sampleSegs.length;
    }

    /**
     * Returns the number of IBS2 segments for the specified target sample
     * @param targSample a target sample index
     * @return the number of IBS2 segments for the specified target sample
     * @throws IndexOutOfBoundsException if
     * {@code targSample < 0 || targSample >= this.nTargSamples()}
     */
    public int nIbs2Segments(int targSample) {
        return sampleSegs[targSample].length;
    }

    /**
     * Returns {@code true} if {@code (otherSample < this.nTargSamples()}
     * and the specified samples are estimated to be IBS2 at the specified
     * marker, and returns {@code false} otherwise.
     * @param targSample a target sample index
     * @param otherSample a sample index
     * @param marker a marker index
     * @return {@code true} if the specified samples are estimated
     * to be IBD2 at the specified marker
     * @throws IndexOutOfBoundsException if
     * {@code (targSample < 0 || targSample >= this.nTargSamples())}
     * @throws IndexOutOfBoundsException if {@code otherSample < 0}
     * @throws IndexOutOfBoundsException if
     * {@code (marker < 0 || marker >= this.nMarkers())}
     */
    public boolean areIbs2(int targSample, int otherSample, int marker) {
        if (targSample>=sampleSegs.length) {
            throw new IndexOutOfBoundsException(String.valueOf(targSample));
        }
        if (otherSample<0) {
            throw new IndexOutOfBoundsException(String.valueOf(otherSample));
        }
        if (marker<0 || marker >= nMarkers) {
            throw new IndexOutOfBoundsException(String.valueOf(marker));
        }
        if (targSample==otherSample) {
            return true;
        }
        if (sampleSegs[targSample].length>0 && otherSample<sampleSegs.length) {
            for (SampleSeg ss : sampleSegs[targSample]) {
                if (ss.sample()==otherSample) {
                    if (ss.start()<=marker && marker<=ss.inclEnd()) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /**
     * Returns {@code true} if
     * {@code (0 <= otherSample && otherSample < this.nTargSamples())}
     * and the specified samples are estimated to be IBS2 within the
     * specified interval and returns {@code false} otherwise.
     * @param targSample a target sample index
     * @param otherSample a sample index
     * @param start the starting marker index
     * @param inclEnd the ending marker index (inclusive)
     * @return {@code true} if
     * {@code (0 <= otherSample && otherSample < this.nTargSamples())}
     * and the specified samples are estimated to be IBS2 within the
     * specified interval
     * @throws IndexOutOfBoundsException if
     * {@code (targSample < 0 || targSample >= this.nTargSamples())}
     * @throws IndexOutOfBoundsException if
     * {@code start > inclEnd}
     */
    public boolean areIbs2(int targSample, int otherSample, int start, int inclEnd) {
        if (start>inclEnd) {
            throw new IndexOutOfBoundsException(String.valueOf(start));
        }
        final boolean sameSample = targSample==otherSample;
        // "sampleSegs[targSample].length==0" comparison must come first in following
        // conditional statement so that targSample is checked for out-of-bounds error
        if (sampleSegs[targSample].length==0 || sameSample) {
            return sameSample;
        }
        for (SampleSeg ss : sampleSegs[targSample]) {
            if (ss.sample()==otherSample) {
                if (start<=ss.inclEnd() && ss.start()<=inclEnd) {
                    return true;
                }
            }
        }
        return false;
    }
}
