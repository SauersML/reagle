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
import ints.WrappedIntArray;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;
import vcf.XRefGT;

/**
 * <p>Class {@code PbwtPhaseIBS} uses the Positional Burrows-Wheeler
 * Transform (PBWT) to find long IBS haplotypes for each sample that
 * contain a specified small genomic interval.</p>
 *
 * <p>Instances of class {@code PbwtPhaseIbs} are thread-safe.</p>
 *
 * <p>Reference: Durbin, R. 2014. Bioinformatics 30(9):1266–1272.
 * doi:10.1093/bioinformatics/btu014</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class PbwtPhaseIbs {

    private final PhaseData phaseData;
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
    public PbwtPhaseIbs(PhaseData phaseData, CodedSteps codedSteps,
            boolean useBwd) {
        checkConsistency(phaseData, codedSteps);
        this.phaseData = phaseData;
        this.allHaps = codedSteps.allHaps();
        PbwtIbsData data = new PbwtIbsData(phaseData, codedSteps);
        if (useBwd) {
            this.ibsHaps = IntStream.range(0, data.nBatches())
                    .parallel()
                    .mapToObj(j -> bwdIbsHaps(data, j))
                    .flatMap(a -> Arrays.stream(a))
                    .toArray(WrappedIntArray[]::new);
        }
        else {
            this.ibsHaps = IntStream.range(0, data.nBatches())
                    .parallel()
                    .mapToObj(j -> fwdIbsHaps(data, j))
                    .flatMap(a -> Arrays.stream(a))
                    .toArray(WrappedIntArray[]::new);
        }
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

        for (int j=(bufferEndStep-1); j>=endStep; --j) {
            IndexArray ia = data.codedSteps().get(j);
            pbwt.bwdUpdate(ia, ia.valueSize(), j, a, d);
        }
        for (int j=(endStep-1); j>=startStep; --j) {
            IndexArray ia = data.codedSteps().get(j);
            pbwt.bwdUpdate(ia, ia.valueSize(), j, a, d);
            ibsHaps0[j-startStep] = getBwdIbsHaps(j, a, d, data);
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

        for (int j=bufferStartStep; j<startStep; ++j) {
            IndexArray ia = data.codedSteps().get(j);
            pbwt.fwdUpdate(ia, ia.valueSize(), j, a, d);
        }
        for (int j=startStep; j<endStep; ++j) {
            IndexArray ia = data.codedSteps().get(j);
            pbwt.fwdUpdate(ia, ia.valueSize(), j, a, d);
            ibsHaps0[j-startStep] = getfwdIbsHaps(j, a, d, data);
        }
        return ibsHaps0;
    }

    private WrappedIntArray getBwdIbsHaps(int step, int[] a, int[] d, PbwtIbsData data) {
        Random rand = new Random(phaseData.seed() + step);
        int mStart = data.codedSteps().steps().start(step);
        int mInclEnd = data.codedSteps().steps().end(step) - 1;
        int[] selectedHaps = new int[data.nTargHaps()];
        Ibs2 ibs2 = phaseData.fpd().stage1Ibs2();
        d[0] = d[a.length] = step - 2;  // set sentinals
        // no need to save and restore old d[0], d[a.length] values
        for (int i=0; i<a.length; ++i) {
            if (a[i]<data.nTargHaps()) {
                int hap = a[i];
                int s1 = hap>>1;
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
                int n = v-u;
                selectedHaps[hap] = -1;
                if (n>1) {
                    int index = u + rand.nextInt(n);
                    for (int j=0; j<n; ++j, ++index) {
                        if (index==v) {
                            index = u;
                        }
                        if (index!=i) {
                            if (ibs2.areIbs2(s1, a[index]>>1, mStart, mInclEnd)==false) {
                                selectedHaps[hap] = a[index];
                                break;
                            }
                        }
                    }
                }
            }
        }
        return new WrappedIntArray(selectedHaps);
    }

    private WrappedIntArray getfwdIbsHaps(int step, int[] a, int[] d,
            PbwtIbsData data) {
        Steps steps = phaseData.fpd().stage1Steps();
        Random rand = new Random(phaseData.seed() + step);
        int nTargHaps = phaseData.fpd().targGT().nHaps();
        int mStart = steps.start(step);
        int mInclEnd = steps.end(step) - 1;
        int[] selectedHaps = new int[nTargHaps];
        Ibs2 ibs2 = phaseData.fpd().stage1Ibs2();
        d[0] = d[a.length] = step + 2;  // set sentinals
        // no need to save and restore old d[0], d[a.length] values
        for (int i=0; i<a.length; ++i) {
            if (a[i]<nTargHaps) {
                int hap = a[i];
                int s1 = hap>>1;
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
                int n = v-u;
                selectedHaps[hap] = -1;
                if (n>1) {
                    int index = u + rand.nextInt(n);
                    for (int j=0; j<n; ++j, ++index) {
                        if (index==v) {
                            index = u;
                        }
                        if (index!=i) {
                            if (ibs2.areIbs2(s1, a[index]>>1, mStart, mInclEnd)==false) {
                                selectedHaps[hap] = a[index];
                                break;
                            }
                        }
                    }
                }
            }
        }
        return new WrappedIntArray(selectedHaps);
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
}
