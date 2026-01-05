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

import vcf.Steps;
import beagleutil.PbwtDivUpdater;
import ints.IndexArray;
import ints.IntArray;
import ints.IntList;
import ints.WrappedIntArray;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;
import vcf.Markers;
import vcf.XRefGT;

/**
 * <p>Class {@code LowFreqPbwtPhaseIbs} uses the Positional Burrows-Wheeler
 * Transform (PBWT) and rare variants to select IBS haplotypes for each
 * sample for each specified genomic interval.</p>
 *
 * <p>Instances of class {@code LowFreqPbwtPhaseIbs} are thread-safe.</p>
 *
 * <p>Reference: Durbin, R. 2014. Bioinformatics 30(9):1266–1272.
 * doi:10.1093/bioinformatics/btu014</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class LowFreqPbwtPhaseIbs {

    private final PhaseData phaseData;
    private final Ibs2 ibs2;
    private final XRefGT allHaps;
    private final WrappedIntArray[] ibsHaps;  //[step][targ hap]

    /**
     * Constructs a new {@code PbwtPhaseIBS} instance from the
     * specified data.
     * @param phaseData the current genotype phase estimates and parameter
     * values
     * @param codedSteps the coded steps
     * @param useBwd {@code true} if last-to-first PBWT should be used
     * @throws IllegalArgumentException if
     * {@code phaseData.fpd().stage1Steps() != codedSteps.steps()}
     * @throws IllegalArgumentException if
     * {@code phaseData.fpd().stage1XRefGT()!=codedSteps.refHaps()}
     * @throws IllegalArgumentException if
     * {@code phaseData.fpd().targGT().samples()!=codedSteps.targSamples()}
     * @throws NullPointerException if
     * {@code phaseData == null || codedSteps == null}
     */
    public LowFreqPbwtPhaseIbs(PhaseData phaseData, CodedSteps codedSteps,
            boolean useBwd) {
        checkConsistency(phaseData, codedSteps);
        this.phaseData = phaseData;
        this.ibs2 = phaseData.fpd().stage1Ibs2();
        this.allHaps = codedSteps.allHaps();
        PbwtIbsData data = new PbwtIbsData(phaseData, codedSteps);
        this.ibsHaps = IntStream.range(0, data.nBatches())
                .parallel()
                .mapToObj(j -> useBwd ? bwdIbsHaps(data, j) : fwdIbsHaps(data, j))
                .flatMap(a -> Arrays.stream(a))
                .toArray(WrappedIntArray[]::new);
    }

    private static void checkConsistency(PhaseData phaseData,
            CodedSteps codedSteps) {
        FixedPhaseData fpd = phaseData.fpd();
        if (fpd.stage1Steps()!=codedSteps.steps()
                || fpd.stage1XRefGT()!=codedSteps.refHaps()
                || fpd.targGT().samples()!=codedSteps.targSamples()) {
            throw new IllegalArgumentException("inconsistent data");
        }
    }

    private WrappedIntArray[] bwdIbsHaps(PbwtIbsData data, int batch) {
        int startStep = data.startStep(batch);
        int endStep = data.endStep(batch);
        int bufferEndStep = data.bufferEndStep(endStep);

        WrappedIntArray[] ibsHaps0 = new WrappedIntArray[endStep - startStep];
        int nHaps = data.nHaps();
        PbwtDivUpdater pbwt = new PbwtDivUpdater(nHaps);
        int[] a = IntStream.range(0, nHaps).toArray();
        int[] d = IntStream.range(0, nHaps+1).map(j -> (bufferEndStep-1)).toArray(); // last entry is sentinal
        int[] aInv  = new int[nHaps];
        int[] iToPrevI = new int[nHaps];
        int[] iToNextI = new int[nHaps];

        for (int j=(bufferEndStep-1); j>=endStep; --j) {
            IndexArray ia = data.codedSteps().get(j);
            pbwt.bwdUpdate(ia, ia.valueSize(), j, a, d);
        }
        for (int j=(endStep-1); j>=startStep; --j) {
            IndexArray ia = data.codedSteps().get(j);
            pbwt.bwdUpdate(ia, ia.valueSize(), j, a, d);
            setInv(a, aInv);
            setIToPrevNextI(data.codedSteps().steps(), j, aInv, iToPrevI, iToNextI);
            ibsHaps0[j-startStep] = getBwdIbsHaps(j, a, d, iToPrevI, iToNextI,
                    data);
        }
        return ibsHaps0;
    }

    private WrappedIntArray[] fwdIbsHaps(PbwtIbsData data, int batch) {
        int startStep = data.startStep(batch);
        int endStep = data.endStep(batch);
        int bufferStartStep = data.bufferStartStep(startStep);

        WrappedIntArray[] ibsHaps0 = new WrappedIntArray[endStep - startStep];
        int nHaps = data.nHaps();
        PbwtDivUpdater pbwt = new PbwtDivUpdater(nHaps);
        int[] a = IntStream.range(0, nHaps).toArray();
        int[] d = IntStream.range(0, nHaps+1).map(j -> bufferStartStep).toArray(); // last entry is sentinal
        int[] aInv  = new int[nHaps];
        int[] iToPrevI = new int[nHaps];
        int[] iToNextI = new int[nHaps];

        for (int j=bufferStartStep; j<startStep; ++j) {
            IndexArray ia = data.codedSteps().get(j);
            pbwt.fwdUpdate(ia, ia.valueSize(), j, a, d);
        }
        for (int j=startStep; j<endStep; ++j) {
            IndexArray ia = data.codedSteps().get(j);
            pbwt.fwdUpdate(ia, ia.valueSize(), j, a, d);
            setInv(a, aInv);
            setIToPrevNextI(data.codedSteps().steps(), j, aInv, iToPrevI, iToNextI);
            ibsHaps0[j-startStep] = getfwdIbsHaps(j, a, d, iToPrevI, iToNextI,
                    data);
        }
        return ibsHaps0;
    }

    private WrappedIntArray getBwdIbsHaps(int step, int[] a, int[] d,
            int[] iToPrevI, int[] iToNextI, PbwtIbsData data) {
        Random rand = new Random(phaseData.seed() + step);
        int mStart = data.codedSteps().steps().start(step);
        int mInclEnd = data.codedSteps().steps().end(step) - 1;
        int[] selectedHaps = new int[data.nTargHaps()];
        d[a.length] = step - 1;  // set sentinal
        for (int i=0; i<a.length; ++i) {
            if (a[i]<data.nTargHaps()) {
                int bestI = bestBwdStage2Index(step, mStart, mInclEnd, i, a, d,
                        iToPrevI, iToNextI, data);
                if (bestI>=0) {
                    selectedHaps[a[i]] = a[bestI];
                }
                else {
                    int u = i;          // inclusive start
                    int v = i + 1;      // exclusive end
                    int uNextMatchEnd = d[u];
                    int vNextMatchEnd = d[v];
                    while ((v - u)<data.nCandidates()
                            && (step<=uNextMatchEnd || step<=vNextMatchEnd)) {
                        if (uNextMatchEnd<=vNextMatchEnd) {
                            vNextMatchEnd = Math.min(d[++v], vNextMatchEnd);
                        }
                        else {
                            uNextMatchEnd = Math.min(d[--u], uNextMatchEnd);
                        }
                    }
                    selectedHaps[a[i]] = getMatch(mStart, mInclEnd, i, u, v, a,
                            rand);
                }
            }
        }
        return new WrappedIntArray(selectedHaps);
    }

    private WrappedIntArray getfwdIbsHaps(int step, int[] a, int[] d,
            int[] iToPrevI, int[] iToNextI, PbwtIbsData data) {
        Random rand = new Random(phaseData.seed() + step);
        int mStart = data.codedSteps().steps().start(step);
        int mInclEnd = data.codedSteps().steps().end(step) - 1;
        int[] selectedHaps = new int[data.nTargHaps()];
        d[a.length] = step + 1;  // set sentinal
        for (int i=0; i<a.length; ++i) {
            if (a[i]<data.nTargHaps()) {
                int bestI = bestFwdStage2Index(step, mStart, mInclEnd, i, a, d,
                        iToPrevI, iToNextI, data);
                if (bestI>=0) {
                    selectedHaps[a[i]] = a[bestI];
                }
                else {
                    int u = i;          // inclusive start
                    int v = i + 1;      // exclusive end
                    int uNextMatchStart = d[u];
                    int vNextMatchStart = d[v];
                    while ((v - u)<data.nCandidates()
                            && (uNextMatchStart<=step || vNextMatchStart<=step)) {
                        if (vNextMatchStart<=uNextMatchStart) {
                            vNextMatchStart = Math.max(d[++v], vNextMatchStart);
                        }
                        else {
                            uNextMatchStart = Math.max(d[--u], uNextMatchStart);
                        }
                    }
                    selectedHaps[a[i]] = getMatch(mStart, mInclEnd, i, u, v, a,
                            rand);
                }
            }
        }
        return new WrappedIntArray(selectedHaps);
    }

    private int bestFwdStage2Index(int step, int mStart, int mInclEnd,
            int i, int[] a, int[] d, int[] iToPrevI, int[] iToNextI, PbwtIbsData tmpData) {
        int bestPrevMatch = -1;
        int bestNextMatch = -1;
        int prevMatchStart = 0;
        int nextMatchStart = 0;

        int minMatchStart = (i+1)<a.length ? Math.min(d[i], d[i+1]) : d[i];
        int dMax = Math.min(minMatchStart + tmpData.maxBackoffSteps(), step);
        int prevI = iToPrevI[i];
        while (prevI>Integer.MIN_VALUE
                && ibs2.areIbs2(a[i]>>1, a[prevI]>>1, mStart, mInclEnd)) {
            prevI = iToPrevI[prevI];
        }
        if (prevI>Integer.MIN_VALUE) {
            assert prevI<i;
            int u = i;
            while ((u-1)!=prevI && d[u]<=dMax) {
                prevMatchStart = Math.max(prevMatchStart, d[u--]);
            }
            if ((u-1)==prevI && d[u]<=dMax) {
                prevMatchStart = Math.max(prevMatchStart, d[u]);
                bestPrevMatch = prevI;
            }
        }
        int nextI = iToNextI[i];
        while (nextI<Integer.MAX_VALUE
                && ibs2.areIbs2(a[i]>>1, a[nextI]>>1, mStart, mInclEnd)) {
            nextI = iToNextI[nextI];
        }
        if (nextI<Integer.MAX_VALUE) {
            assert i<nextI;
            int v = i;
            while ((v+1)!=nextI && d[v+1]<=dMax) {
                nextMatchStart = Math.max(nextMatchStart, d[++v]);
            }
            if ((v+1)==nextI && d[v+1]<=dMax) {
                nextMatchStart = Math.max(nextMatchStart, d[++v]);
                bestNextMatch = nextI;
            }
        }
        if (prevMatchStart<nextMatchStart && bestPrevMatch != -1) {
            return bestPrevMatch;
        }
        else {
            return bestNextMatch;
        }
    }

    private int bestBwdStage2Index(int step, int mStart, int mInclEnd,
            int i, int[] a, int[] d, int[] iToPrevI, int[] iToNextI, PbwtIbsData data) {
        int nStepsM1 = data.codedSteps().steps().size()-1;
        int bestPrevMatch = -1;
        int bestNextMatch = -1;
        int prevMatchInclEnd = nStepsM1;
        int nextMatchInclEnd = nStepsM1;
        int maxMatchStart = (i+1)<a.length ? Math.max(d[i], d[i+1]) : d[i];
        int dMin = Math.max(maxMatchStart - data.maxBackoffSteps(), step);
        int prevI = iToPrevI[i];
        while (prevI>Integer.MIN_VALUE
                && ibs2.areIbs2(a[i]>>1, a[prevI]>>1, mStart, mInclEnd)) {
            prevI = iToPrevI[prevI];
        }
        if (prevI>Integer.MIN_VALUE) {
            assert prevI<i;
            int u = i;
            while ((u-1)!=prevI && d[u]>=dMin) {
                prevMatchInclEnd = Math.min(prevMatchInclEnd, d[u--]);
            }
            if ((u-1)==prevI && d[u]>=dMin) {
                prevMatchInclEnd = Math.min(prevMatchInclEnd, d[u]);
                bestPrevMatch = prevI;
            }
        }
        int nextI = iToNextI[i];
        while (nextI<Integer.MAX_VALUE
                && ibs2.areIbs2(a[i]>>1, a[nextI]>>1, mStart, mInclEnd)) {
            nextI = iToNextI[nextI];
        }
        if (nextI<Integer.MAX_VALUE) {
            assert i<nextI;
            int v = i;
            while ((v+1)!=nextI && d[v+1]>=dMin) {
                nextMatchInclEnd = Math.min(nextMatchInclEnd, d[++v]);
            }
            if ((v+1)==nextI && d[v+1]>=dMin) {
                nextMatchInclEnd = Math.min(nextMatchInclEnd, d[++v]);
                bestNextMatch = nextI;
            }
        }
        if (prevMatchInclEnd>nextMatchInclEnd && bestPrevMatch != -1) {
            return bestPrevMatch;
        }
        else {
            return bestNextMatch;
        }
    }

    private int getMatch(int mStart, int mInclEnd, int i, int iStart,
            int iEnd, int[] a, Random rand) {
        int iLength = iEnd - iStart;
        if (iLength==1) {
            return -1;
        }
        int match = -1;
        int index = iStart + rand.nextInt(iLength);
        for (int j=0; j<iLength && match==-1; ++j) {
            if (ibs2.areIbs2(a[i]>>1, a[index]>>1, mStart, mInclEnd)==false) {
                match = a[index];
            }
            if (++index==iEnd) {
                index = iStart;
            }
        }
        return match;
    }

    /**
     * Returns the current genotype phase estimates and parameter values.
     * @return the current genotype phase estimates and parameter values
     */
    public PhaseData phaseData() {
        return phaseData;
    }

    /**
     * Returns the estimated phased genotypes for the target and reference
     * samples.
     * @return the estimated phased genotypes for the target and reference
     * samples
     */
    public XRefGT allHaps() {
        return allHaps;
    }

    /**
     * Returns the index of a haplotype that is identical by state
     * with the specified target haplotype in the specified genomic interval,
     * or {@code -1} if there is no identical-by-state haplotype.
     * @param hap a target haplotype index
     * @param step an index of a genomic interval
     * @return the index of a haplotype that is identical by state
     * with the specified haplotype int the specified genomic interval
     * @throws IndexOutOfBoundsException if
     * {@code hap < 0 || hap >= this.phaseData().targGT().nHaps()}
     * @throws IndexOutOfBoundsException if
     * {@code step < 0 || step >= this.phaseData().fpd().stage1Steps().size()}
     */
    public int ibsHap(int hap, int step) {
        return ibsHaps[step].get(hap);
    }

    private void setIToPrevNextI(Steps steps, int step, int[] invA,
            int[] iToPrevI, int[] iToNextI) {
        Arrays.fill(iToPrevI, Integer.MIN_VALUE);
        Arrays.fill(iToNextI, Integer.MAX_VALUE);
        ArrayList<IntList> lowFreqHapLists = lowFreqHapLists(steps, step);
        for (int j=0, n=lowFreqHapLists.size(); j<n; ++j) {
            int[] i = sortedAIndices(lowFreqHapLists.get(j), invA);
            for (int k=1; k<i.length; ++k) {
                int i0 = i[k-1];
                int i1 = i[k];
                if (i0>iToPrevI[i1]) {
                    iToPrevI[i1] = i0;
                }
                if (i1<iToNextI[i0]) {
                    iToNextI[i0] = i1;
                }
            }
        }
    }

    private int[] sortedAIndices(IntList haps, int[] invA) {
        return haps.stream()
                .map(h -> invA[h])
                .sorted()
                .toArray();
    }

    private ArrayList<IntList> lowFreqHapLists(Steps steps, int step) {
        FixedPhaseData fpd = phaseData.fpd();
        IntArray hiFreqIndices = fpd.stage1To2();
        int start = step==0 ? 0 : hiFreqIndices.get(steps.start(step));
        int end = (step+1 < steps.size())
                ? hiFreqIndices.get(steps.start(step+1))
                : fpd.targGT().nMarkers();
        return lowFreqHapLists(fpd, start, end);
    }

    private static ArrayList<IntList> lowFreqHapLists(FixedPhaseData fpd,
            int start, int end) {
        ArrayList<IntList> hapLists = new ArrayList<>();
        Markers markers = fpd.targGT().markers();
        for (int m=start; m<end; ++m) {
            int nAlleles = markers.marker(m).nAlleles();
            for (int al=0; al<nAlleles; ++al) {
                IntArray carriers = fpd.carriers(m, al);
                if (carriers.size()>1) {
                    hapLists.add(hapList(carriers));
                }
            }
        }
        return hapLists;
    }

    private static IntList hapList(IntArray carriers) {
        IntList hapList = new IntList(2*carriers.size());
        for (int j=0, n=carriers.size(); j<n; ++j) {
            int sample = carriers.get(j);
            int h1 = sample<<1;
            hapList.add(h1);
            hapList.add(h1|0b1);
        }
        return hapList;
    }

    private static void setInv(int[] a, int[] aInv) {
        for (int j=0; j<a.length; ++j) {
            aInv[a[j]] = j;
        }
    }
}
