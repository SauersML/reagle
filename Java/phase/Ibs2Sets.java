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

import ints.IntArray;
import ints.IntList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import vcf.GT;

/**
 * <p>Class {@code Ibs2Sets} partitions markers into steps, and stores
 * the sets of samples whose genotypes are consistent with IBS2 in each
 * step.</p>
 *
 * <p>Instances of {@code Ibs2Sets} are immutable.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class Ibs2Sets {

    private static final float MAX_MISS_STEP_FREQ = 0.1f;

    private final int nTargSamples;
    private final int nMarkersM1;
    private final IntArray windowStarts;
    private final int[][][] ibs2Sets; //[window][targ_sample][ibs2_samples]

    /**
     * Constructs a new {@code Ibs2Sets} instance from the specified data.
     * @param targGT the target genotype data
     * @param ibs2Markers the markers and intervals that are used to detect
     * IBS2 segments
     * @throws IllegalArgumentException if
     * {@code targGT.nMarkers() != ibs2Markers.nMarkers()}
     * @throws NullPointerException if
     * {@code (targGT == null || ibs2Markers == null)}
     */
    public Ibs2Sets(GT targGT, Ibs2Markers ibs2Markers) {
        if (targGT.nMarkers()!=ibs2Markers.nMarkers()) {
            throw new IllegalArgumentException(String.valueOf(ibs2Markers.nMarkers()));
        }
        this.nMarkersM1 = targGT.nMarkers() - 1;
        this.nTargSamples = targGT.nSamples();
        this.windowStarts = ibs2Markers.stepStarts();

        WrappedIbs2Sets[] results = IntStream.range(0, windowStarts.size())
                .parallel()
                .mapToObj(w -> ibs2Sets(targGT, ibs2Markers, windowStarts, w))
                .toArray(WrappedIbs2Sets[]::new);
        this.ibs2Sets = Arrays.stream(results)
                .map(wrapper -> wrapper.ibsSets)
                .toArray(int[][][]::new);
    }

    private static WrappedIbs2Sets ibs2Sets(GT targGT, Ibs2Markers ibs2Markers,
            IntArray wStarts, int w) {
        int wP1 = w + 1;
        int start = wStarts.get(w);
        int end = (wP1 < wStarts.size()) ? wStarts.get(wP1) : targGT.nMarkers();
        int[] stepMarkers = ibs2Markers.markers(start, end);
        List<SampClust> partition = new ArrayList<>(1);
        partition.add(initCluster(targGT, stepMarkers));
        for (int m : stepMarkers) {
            int mm = m;
            partition = partition.stream()
                    .flatMap(sampClust -> partition(targGT, sampClust, mm))
                    .collect(Collectors.toCollection(ArrayList::new));
        }
        return results(partition, targGT.nSamples());
    }

    private static SampClust initCluster(GT targGT, int[] stepMarkers) {
        // exclude sample with more than MIX_MISS_FREQ missing genotypes
        int nTargSamples = targGT.nSamples();
        int[] missCnt = new int[nTargSamples];
        for (int m : stepMarkers) {
            for (int s=0; s<nTargSamples; ++s) {
                int hap1 = s <<1;
                if (targGT.allele(m, hap1)==-1 || targGT.allele(m, hap1 | 0b1)==-1) {
                    ++missCnt[s];
                }
            }
        }
        int maxMiss = (int) Math.floor(MAX_MISS_STEP_FREQ*stepMarkers.length);
        int[] initCluster = IntStream.range(0, nTargSamples)
                .filter(s -> missCnt[s]<=maxMiss)
                .toArray();
        boolean isHomozygous = true;
        return new SampClust(initCluster, isHomozygous);
    }

    private static Stream<SampClust> partition(GT targGT, SampClust parent,
            int m) {
        // this method assumes int[] parent.samples is sorted in increasing order
        int nAlleles = targGT.marker(m).nAlleles();
        IntList[] gtToList = new IntList[(nAlleles*(nAlleles+1))>>1];
        boolean[] isHom = isHom(parent.isHomozygous, nAlleles);
        IntList missing = new IntList(32);
        for (int s : parent.samples) {
            int gtIndex = getGT(m, s, targGT);
            if (gtIndex<0) {
                missing.add(s);
                for (int k=0; k<gtToList.length; ++k) {
                    if (gtToList[k]!=null) {
                        gtToList[k].add(s);
                    }
                }
            }
            else {
                if (gtToList[gtIndex]==null) {
                    gtToList[gtIndex] = new IntList();
                    for (int j=0, n=missing.size(); j<n; ++j) {
                        gtToList[gtIndex].add(missing.get(j));
                    }
                }
                gtToList[gtIndex].add(s);
            }
        }
        return IntStream.range(0, gtToList.length)
                .filter(i -> gtToList[i]!=null && gtToList[i].size()>1)
                .mapToObj(i -> new SampClust(gtToList[i].toArray(), isHom[i]));
    }

    private static int getGT(int m, int s, GT targGT) {
        int hap1 = s << 1;
        int a1 = targGT.allele(m, hap1);
        int a2 = targGT.allele(m, hap1 | 0b1);
        if (a1<0 || a2<0) {
            return -1;
        }
        return a1<=a2 ? ((a2*(a2+1))>>1) + a1 : ((a1*(a1+1))>>1) + a2;
    }

    private static boolean[] isHom(boolean prevIsHom, int nAlleles) {
        boolean[] isHom = new boolean[(nAlleles*(nAlleles+1))>>1];
        if (prevIsHom) {
            for (int a=0; a<nAlleles; ++a) {
                isHom[((a*(a+1))>>1) + a] = true;
            }
        }
        return isHom;
    }

    private static WrappedIbs2Sets results(List<SampClust> ibd2Lists,
            int nTargSamples) {
        final int[] EMPTY_ARRAY = new int[0];
        int[][] results = IntStream.range(0, nTargSamples)
                .mapToObj(i -> EMPTY_ARRAY)
                .toArray(int[][]::new);
        for (int j=0, n=ibd2Lists.size(); j<n; ++j) {
            SampClust ibd2List = ibd2Lists.get(j);
            if (ibd2List.isHomozygous==false) {
                int[] ia = ibd2List.samples;
                assert ia.length>1;
                for (int s : ia) {
                    if (results[s]==EMPTY_ARRAY) {
                        results[s] = ia;
                    }
                    else {
                        // sample can be in >1 list due to missing genotypes
                        IntStream is1 = Arrays.stream(results[s]);
                        IntStream is2 = Arrays.stream(ia);
                        results[s] = IntStream.concat(is1, is2)
                                .sorted()
                                .distinct()
                                .toArray();
                    }
                }
            }
        }
        return new WrappedIbs2Sets(results);
    }

    /**
     * Returns the number of target samples.
     * @return the number of target samples
     */
    public int nTargSamples() {
        return nTargSamples;
    }

    /**
     * Returns the detected IBS2 sample segments.
     * @param sample a sample index
     * @return the detected IBS2 sample segments
     * @throws IndexOutOfBoundsException if
     * {@code sample < 0 || sample >= this.nTargSamples()}
     */
    public SampleSeg[] segList(int sample) {
        List<SampleSeg> list = new ArrayList<>();
        for (int w=0; w<ibs2Sets.length; ++w) {
            int[] ia = ibs2Sets[w][sample];
            if (ia.length>0) {
                int wP1 = w + 1;
                int start = windowStarts.get(w);
                int inclEnd = wP1 < windowStarts.size()
                        ? windowStarts.get(wP1) - 1 : nMarkersM1;
                for (int s2 : ia) {
                    if (s2!=sample) {
                        list.add(new SampleSeg(s2, start, inclEnd));
                    }
                }
            }
        }
        return list.toArray(new SampleSeg[0]);
    }

    private static class SampClust {

        private final int[] samples;
        private final boolean isHomozygous;

        private SampClust(int[] samples, boolean areHomozygous) {
            this.samples = samples;
            this.isHomozygous = areHomozygous;
        }

        private SampClust(int nSamples) {
            this.samples = IntStream.range(0, nSamples).toArray();
            this.isHomozygous = true;
        }
    }

    private static class WrappedIbs2Sets {
        // final field ensures visibility of IBS sets across threads
        private final int[][] ibsSets; // [targ_sample][ibs2_samples]
        private WrappedIbs2Sets(int[][] ibsSets) {
            this.ibsSets = ibsSets;
        }
    }
}
