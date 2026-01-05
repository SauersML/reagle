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
import blbutil.Utilities;
import ints.IntList;
import ints.WrappedIntArray;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.stream.IntStream;
import vcf.GT;

/**
 * <p>Class {@code PbwtPhaser} phases input genotype data and imputes
 * missing alleles using the Positional Burrows-Wheeler Transform (PBWT)</p>
 *
 * <p>Instances of class {@code PbwtPhaser} are not thread-safe.</p>
 *
 * <p>Reference: Richard Durbin. (2014) Efficient haplotype matching and storage
 * using the Positional Burrows-Wheeler Transform (PBWT). Bioinformatics
 * 30(9):1266-72.</p>
 *
 * <p>Reference: Olivier Delaneau, Jean-Francois Zagury, Matthew R Robinson,
 * Jonathan Marchini, Emmanouil Dermitzakis. (2019) Accurate, scalable and
 * integrative haplotype estimation. Nature Communications 10(1):5436.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class PbwtPhaser {

    private final int start;
    private final int end;
    private final FwdPbwtPhaser fwdPbwt;

    private PbwtPhaser(FixedPhaseData fpd, int start, int end, long seed) {
        if (start<0 || end>fpd.targGT().nMarkers() || start>=end) {
            throw new IllegalArgumentException(String.valueOf(start));
        }
        this.start = start;
        this.end = end;
        this.fwdPbwt = new FwdPbwtPhaser(fpd, start, end, seed);
    }

    /**
     * Returns an initial phasing for first-stage markers in the target samples.
     * @param fpd the input data for phasing
     * @param seed seed for random number generation
     * @return an initial genotype phasing for the first-stage markers in the
     * target samples
     * @throws NullPointerException if {@code fpd == null}
     */
    public static AtomicReferenceArray<SamplePhase> initPhase(FixedPhaseData fpd,
            long seed) {
        PbwtPhaser[] ppa = pbwtPhasers(fpd, seed);
        int nSamples = fpd.stage1TargGT().nSamples();
        int nThreads = fpd.par().nthreads();
        int maxStepSize = 128;
        int stepSize = Math.min((nSamples + nThreads - 1)/nThreads, maxStepSize);
        int nSteps = (nSamples + (stepSize-1)) / stepSize;
        AtomicReferenceArray<SamplePhase> phase = new AtomicReferenceArray<>(nSamples);
        IntStream.range(0, nSteps)
                .parallel()
                .boxed()
                .forEach(step -> setSamplePhase(fpd, ppa, phase, step, stepSize));
        return phase;
    }

    private static void setSamplePhase(FixedPhaseData fpd, PbwtPhaser[] ppa,
            AtomicReferenceArray<SamplePhase> phase, int step, int stepSize) {
        GT gt = fpd.stage1TargGT();
        int sStart = step*stepSize;
        int sEnd = Math.min(sStart + stepSize, gt.nSamples());
        Indices[] indices = indices(fpd, sStart, sEnd);

        int overlapEnd = 0;
        int[][] haps = new int[(sEnd-sStart)<<1][gt.nMarkers()];
        ppa[0].copyHaps(haps, indices, overlapEnd, sStart, sEnd);
        for (int j=1; j<ppa.length; ++j) {
            overlapEnd = ppa[j-1].end;
            ppa[j].copyHaps(haps, indices, overlapEnd, sStart, sEnd);
        }

        for (int s=sStart; s<sEnd; ++s) {
            int ss = s - sStart;
            int hh1 = ss<<1;
            int hh2 = hh1 | 0b1;
            phase.set(s, new SamplePhase(s, gt.markers(), fpd.stage1Map().genPos(),
                haps[hh1], haps[hh2], indices[ss].hetIndices, indices[ss].missIndices));
        }
    }

    private void copyHaps(int[][] haps, Indices[] indices, int overlapEnd,
            int sStart, int sEnd) {
        int copyStart = (this.start + overlapEnd)>>>1;
        int[][] alignedHaps = haps.clone();
        if (this.start>0) {
            for (int s=sStart; s<sEnd; ++s) {
                int ss = s - sStart;
                int hh1 = ss<<1;
                int hh2 = hh1 | 0b1;
                int alignHet = alignmentHet(indices[ss].hetIndices,
                        start, copyStart, overlapEnd);
                if (alignHet>=0 && switchHapLabels(s, haps[hh1], haps[hh2], alignHet)) {
                    alignedHaps[hh1] = haps[hh2];
                    alignedHaps[hh2] = haps[hh1];
                }
            }
        }
        for (int m=copyStart; m<end; ++m) {
            for (int s=sStart; s<sEnd; ++s) {
                int h1 = s<<1;
                int h2 = h1 | 0b1;
                int hh1 = (s - sStart)<<1;
                int hh2 = hh1 | 0b1;
                alignedHaps[hh1][m] = fwdPbwt.allele(m, h1);
                alignedHaps[hh2][m] = fwdPbwt.allele(m, h2);
            }
        }
    }

    /* Returns -1 if no alignment het exists */
    private static int alignmentHet(WrappedIntArray hetList, int start,
            int copyStart, int overlapEnd) {
        if (hetList.size()==0) {
            return -1;
        }
        int index = insPt(hetList, copyStart);
        if (index==hetList.size() || (hetList.get(index)>=overlapEnd && index>0)) {
            index -= 1;
        }
        int het = hetList.get(index);
        return (start<=het && het<overlapEnd) ? het : -1;
    }

    private boolean switchHapLabels(int sample, int[] hap1, int[] hap2, int alignHet) {
        int h1 = sample<<1;
        int h2 = h1 | 0b1;
        int a1 = hap1[alignHet];
        int a2 = hap2[alignHet];
        int b1 = fwdPbwt.allele(alignHet, h1);
        int b2 = fwdPbwt.allele(alignHet, h2);
        return a1==b2 && a2==b1;
    }

    private static int insPt(WrappedIntArray list, int value) {
        int index = list.binarySearch(value);
        return (index<0) ? -index-1 : index;
    }

    private static PbwtPhaser[] pbwtPhasers(FixedPhaseData fpd, long seed) {
        int[][] windows = hiFreqWindows(fpd);
        return IntStream.range(0, windows.length)
                .parallel()
                .mapToObj(j -> new PbwtPhaser(fpd, windows[j][0], windows[j][1],
                        seed + j))
                .toArray(PbwtPhaser[]::new);
    }

    private static int[][] hiFreqWindows(FixedPhaseData fpd) {
        DoubleArray genPos = fpd.stage1Map().genPos();
        int nMarkers = genPos.size();
        int nThreads = fpd.par().nthreads();
        double totalCM = genPos.get(genPos.size()-1) - genPos.get(0);
        double overlapCM = 0.5;
        double advanceCM = Math.max(4*overlapCM, (totalCM/nThreads));
        List<int[]> windowList = new ArrayList<>(nThreads);
        int from = 0;
        int to = to(genPos, genPos.get(from) + advanceCM);
        while (to<nMarkers) {
            windowList.add(new int[] {from, to});
            from = from(genPos, genPos.get(to) - overlapCM);
            to = to(genPos, genPos.get(to) + advanceCM);
        }
        assert to==nMarkers;
        windowList.add(new int[] {from, to});
        return windowList.toArray(new int[0][]);
    }

    private static int from(DoubleArray genPos, double pos) {
        int insPt = genPos.binarySearch(pos);
        return insPt<0 ? -insPt-1 : insPt;
    }

    private static int to(DoubleArray genPos, double pos) {
        int insPt = genPos.binarySearch(pos);
        return insPt<0 ? -insPt-1 : (insPt+1);  //insPt>=0 implies insPt<genpPos.size()
    }

    private static Indices[] indices(FixedPhaseData fpd, int sStart, int sEnd) {
        GT gt = fpd.stage1TargGT();
        int overlap = fpd.stage1Overlap();
        int nMarkers = gt.nMarkers();
        int len = sEnd - sStart;
        IntList[] missIndices = intLists(len);
        IntList[] hetIndices = intLists(len);
        boolean[] notFirstHet = new boolean[len];
        for (int m=0; m<nMarkers; ++m) {
            for (int s=sStart; s<sEnd; ++s) {
                int ss = s-sStart;
                int hap1 = s << 1;
                int a1 = gt.allele(m, hap1);
                int a2 = gt.allele(m, hap1 | 0b1);
                if (a1<0 || a2<0) {
                    missIndices[ss].add(m);
                }
                else if (a1!=a2) {
                    if (m>=overlap && notFirstHet[ss]) {
                        hetIndices[ss].add(m);
                    }
                    else {
                        notFirstHet[ss] = true;
                    }
                }
            }
        }
        return IntStream.range(0, len)
                .mapToObj(j -> new Indices(missIndices[j], hetIndices[j]))
                .toArray(Indices[]::new);

    }

    private static IntList[] intLists(int length) {
        return IntStream.range(0, length)
                .parallel()
                .mapToObj(j -> new IntList())
                .toArray(IntList[]::new);
    }

    private static class Indices {
        public final WrappedIntArray missIndices;
        public final WrappedIntArray hetIndices;

        public Indices(IntList missIndices, IntList hetIndices) {
            this.missIndices = new WrappedIntArray(missIndices);
            this.hetIndices = new WrappedIntArray(hetIndices);
        }
    }
}
