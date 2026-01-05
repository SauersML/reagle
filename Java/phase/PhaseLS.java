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

import blbutil.Const;
import blbutil.MultiThreadUtils;
import blbutil.Utilities;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import main.Par;

/**
 * <p>Class {@code PhaseLS} contains static methods for estimated genotypes
 * phase using a haploid Li and Stephens hidden Markov model.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class PhaseLS {

    private PhaseLS() {
        // private constructor to prevent instantiation
    }

    /**
     * Updates genotype phase estimates in the target samples.
     * @param pd estimated phased genotypes at stage 1 markers
     * @throws NullPointerException if {@code pd == null}
     */
    public static void runStage1(PhaseData pd) {
        Par par = pd.fpd().par();
        int it = pd.it();
        int nThreads = par.nthreads();
        int nSamples = pd.fpd().targGT().nSamples();
        int nBurninIts = par.burnin();
        PbwtPhaseIbs phaseIbs = pbwtPhaseIbs(pd);
        if (par.em()) {
            Random rand = new Random(pd.seed());
            if (it==0) {
                initializeParameters(phaseIbs, rand);
            }
            else if (it<nBurninIts) {
                updateParameters(phaseIbs, rand);
            }
        }
        AtomicInteger samples = new AtomicInteger(0);
        ExecutorService es = Executors.newFixedThreadPool(nThreads);
        for (int j=0; j<nThreads; ++j) {
            es.submit(() -> {
                try {
                    PhaseBaum2 baum = new PhaseBaum2(phaseIbs);
                    for (int s=samples.getAndIncrement(); s<nSamples;
                            s=samples.getAndIncrement()) {
                        baum.phase(s);
                    }
                }
                catch (Throwable t) {
                    Utilities.exit(t);
                }
            } );
        }
        MultiThreadUtils.shutdownExecService(es);
    }

    private static PbwtPhaseIbs pbwtPhaseIbs(PhaseData pd) {
        boolean useBwd = (pd.it() & 1)==0;
        PbwtPhaseIbs phaseIbs = new PbwtPhaseIbs(pd, pd.codedSteps(), useBwd);
        return phaseIbs;
    }

    private static void initializeParameters(PbwtPhaseIbs phaseIbs, Random rand) {
        PhaseData pd = phaseIbs.phaseData();
        float prevRecInt = pd.recombIntensity();
        int maxInitialIts = 15;
        for (int j=0; j<maxInitialIts; ++j) {
            updateParameters(phaseIbs, rand);
            float recInt = pd.recombIntensity();
            if (Math.abs(recInt - prevRecInt) <= 0.1*prevRecInt) {
                break;
            }
            prevRecInt = recInt;
        }
    }

    private static void updateParameters(PbwtPhaseIbs phaseIbs, Random rand) {
        ParamEstimates paramEst = getParamEst(phaseIbs, rand);
        PhaseData pd = phaseIbs.phaseData();
        float prevPMismatch = pd.pMismatch();
        float pMismatch = paramEst.pMismatch();
        float recombIntensity = paramEst.recombIntensity();
        if (Float.isFinite(pMismatch) && pMismatch>prevPMismatch) {
            pd.updatePMismatch(pMismatch);
        }
        if (Float.isFinite(recombIntensity) && recombIntensity>0f) {
            pd.updateRecombIntensity(recombIntensity);
        }
    }

    private static ParamEstimates getParamEst(PbwtPhaseIbs phaseIbs, Random rand) {
        ParamEstimates paramEst = new ParamEstimates();
        PhaseData pd = phaseIbs.phaseData();
        FixedPhaseData fpd = pd.fpd();
        int[] sampleIndices = samplesToAnalyze(pd, rand);
        int nThreads = Math.min(fpd.par().nthreads(), sampleIndices.length);
        double maxSum = 20000.0/nThreads;
        int minIndices = 50;
        AtomicInteger counter = new AtomicInteger(0);
        ExecutorService es = Executors.newFixedThreadPool(nThreads);
        for (int j=0; j<nThreads; ++j) {
            es.submit(() -> {
                try {
                    HmmParamData hpd = new HmmParamData(phaseIbs);
                    int index = counter.getAndIncrement();
                    while ((hpd.sumSwitchProbs()<maxSum || index<minIndices)
                            && index<sampleIndices.length) {
                        hpd.update(sampleIndices[index]);
                        index = counter.getAndIncrement();
                        hpd.addEstimationData(paramEst);
                    }
                }
                catch (Throwable t) {
                    Utilities.exit(t);
                }
            } );
        }
        MultiThreadUtils.shutdownExecService(es);
        return paramEst;
    }

    private static int[] samplesToAnalyze(PhaseData pd, Random rand) {
        int maxSamplesToAnalyze = 500;
        int nTargSamples = pd.fpd().targGT().nSamples();
        int[] ia = IntStream.range(0, nTargSamples)
                .parallel()
                .toArray();
        if (nTargSamples <= maxSamplesToAnalyze) {
            return ia;
        }
        else {
            Utilities.shuffle(ia, maxSamplesToAnalyze, rand);
            return Arrays.copyOf(ia, maxSamplesToAnalyze);
        }
    }

    /**
     * Returns phased genotypes at all markers.
     * @param pd estimated phased genotypes at stage 1 markers
     * @return estimated phased genotypes at all markers
     * @throws NullPointerException if {@code pd == null}
     */
    public static Stage2Haps runStage2(PhaseData pd) {
        FixedPhaseData fpd = pd.fpd();
        int nThreads = fpd.par().nthreads();
        int nSamples = fpd.targGT().nSamples();
        LowFreqPhaseIbs phaseIbs = new LowFreqPhaseIbs(pd);
        Stage2Haps stage2Haps = new Stage2Haps(pd);
        AtomicInteger samples = new AtomicInteger(0);
        ExecutorService es = Executors.newFixedThreadPool(nThreads);
        for (int j=0; j<nThreads; ++j) {
            es.submit(() -> {
                try {
                    Stage2Baum baum = new Stage2Baum(phaseIbs, stage2Haps);
                    for (int s=samples.getAndIncrement(); s<nSamples;
                            s=samples.getAndIncrement()) {
                        baum.phase(s);
                    }
                }
                catch (Throwable t) {
                    Utilities.exit(t);
                }
            } );
        }
        MultiThreadUtils.shutdownExecService(es);
        return stage2Haps;
    }
}
