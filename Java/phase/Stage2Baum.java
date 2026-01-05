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
import java.util.Optional;
import java.util.Random;
import vcf.GT;
import vcf.RefGT;

/**
 * <p>Class {@code Stage2Baum} applies the forward and backward algorithms
 * for a haploid Li and Stephens hidden Markov model at high-frequency markers,
 * and imputes missing genotypes and heterozygote phase at low-frequency
 * markers.</p>
 *
 * <p>Instances of class {@code Stage2Baum} are not thread-safe.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class Stage2Baum {

    private final FixedPhaseData fpd;
    private final PhaseData phaseData;
    private final HmmStateProbs stateProbs;
    private final int[] nStates = new int[2];
    private final int[][][] states;
    private final float[][][] probs;

    private final GT unphTargGT;
    private final Optional<RefGT> refGT;
    private final int nTargHaps;
    private final int nStage1Markers;
    private final Stage2Haps stage2Haps;
    private final IntArray stage1To2;
    private final Random rand;

    /**
     * Creates a {@code ImputeBaum} instance from the specified data.
     *
     * @param phaseIbs the IBS haplotypes
     * @param stage2Haps an object for storing phased genotypes
     * @throws NullPointerException if
     * {@code phaseIbs == null || stage2Haps == null}
     */
    public Stage2Baum(LowFreqPhaseIbs phaseIbs, Stage2Haps stage2Haps) {
        this.fpd = phaseIbs.phaseData().fpd();
        this.phaseData = phaseIbs.phaseData();
        this.nStage1Markers = fpd.stage1TargGT().nMarkers();
        this.stateProbs =  new HmmStateProbs(phaseIbs);
        this.states = new int[2][nStage1Markers][stateProbs.maxStates()];
        this.probs = new float[2][nStage1Markers][stateProbs.maxStates()];

        this.unphTargGT = fpd.targGT();
        this.refGT = fpd.restrictedRefGT();
        this.nTargHaps = fpd.targGT().nHaps();
        this.stage2Haps = stage2Haps;
        this.stage1To2 = fpd.stage1To2();
        this.rand = new Random(phaseData.seed());
    }

    /**
     * Returns the number of target samples.
     * @return the number of target samples
     */
    public int nTargSamples() {
        return fpd.targGT().nSamples();
    }

    /**
     * Estimates and stores the phased haplotypes for the specified sample.
     * @param targSample a sample index
     * @throws IndexOutOfBoundsException if
     * {@code sample < 0 || sample >= this.nTargSamples()}
     */
    public void phase(int targSample) {
        rand.setSeed(phaseData.seed() + targSample);
        int h1 = (targSample<<1);
        int h2 = h1 | 0b1;
        nStates[0] = stateProbs.run(h1, states[0], probs[0]);
        nStates[1] = stateProbs.run(h2, states[1], probs[1]);

        int start = 0;
        for (int j=0; j<nStage1Markers; ++j) {
            int end = stage1To2.get(j);
            imputeInterval(targSample, start, end);
            start = end + 1;
        }
        imputeInterval(targSample, start, unphTargGT.nMarkers());
    }

    private void imputeInterval(int sample, int start, int end) {
        int hap1 = sample << 1;
        int hap2 = hap1 | 0b1;
        for (int m=start; m<end; ++m) {
            int a1 = unphTargGT.allele(m, hap1);
            int a2 = unphTargGT.allele(m, hap2);
            if (a1>=0 && a2>=0) {
                if (a1!=a2) {
                    float[] alProbs1 = unscaledAlProbs(m, 0, a1, a2);
                    float[] alProbs2 = unscaledAlProbs(m, 1, a1, a2);
                    float p1 = alProbs1[a1]*alProbs2[a2];
                    float p2 = alProbs1[a2]*alProbs2[a1];
                    boolean switchAlleles = (p1<p2 || (p1==p2 && rand.nextBoolean()));
                    if (switchAlleles) {
                        int tmp = a1;
                        a1 = a2;
                        a2 = tmp;
                    }
                }
            }
            else {
                a1 = imputeAllele(m, 0);
                a2 = imputeAllele(m, 1);
            }
            stage2Haps.setPhasedGT(m, sample, a1, a2);
        }
    }

    private float[] unscaledAlProbs(int m, int hapBit, int a1, int a2) {
        float[] alProbs = new float[unphTargGT.marker(m).nAlleles()];
        boolean rare1 = fpd.isLowFreq(m, a1);
        boolean rare2 = fpd.isLowFreq(m, a2);
        int mkrA = fpd.prevStage1Marker(m);
        int mkrB = Math.min(mkrA + 1, nStage1Markers - 1);
        int[] statesA = states[hapBit][mkrA];
        float[] probsA = probs[hapBit][mkrA];
        float[] probsB = probs[hapBit][mkrB];
        for (int j=0, n=nStates[hapBit]; j<n; ++j) {
            int hap = statesA[j];
            int b1 = allele(m, hap);
            int b2 = allele(m, (hap ^ 0b1));
            if (b1>=0 && b2>=0) {
                float wt = fpd.prevStage1Wt(m);
                float prob = wt*probsA[j] + (1.0f - wt)*probsB[j];
                if (b1==b2) {
                    alProbs[b1] += prob;
                }
                else {
                    boolean match1 = rare1 && (a1==b1 || a1==b2);
                    boolean match2 = rare2 && (a2==b1 || a2==b2);
                    if (match1 ^ match2) {
                        if (match1) {
                            alProbs[a1] += prob;
                        }
                        else {
                            alProbs[a2] += prob;
                        }
                    }
                }
            }
        }
        return alProbs;
    }

    private int imputeAllele(int m, int hapBit) {
        float[] alProbs = new float[unphTargGT.marker(m).nAlleles()];
        int mkrA = fpd.prevStage1Marker(m);
        int mkrB = Math.min(mkrA + 1, nStage1Markers - 1);
        int[] statesA = states[hapBit][mkrA];
        float[] stateProbsA = probs[hapBit][mkrA];
        float[] stateProbsB = probs[hapBit][mkrB];
        for (int j=0, n=nStates[hapBit]; j<n; ++j) {
            float wt = fpd.prevStage1Wt(m);
            float prob = wt*stateProbsA[j] + (1.0f - wt)*stateProbsB[j];
            int hap = statesA[j];
            int b1 = allele(m, hap);
            int b2 = allele(m, hap ^ 0b1);
            if (b1>=0 && b2>=0) {
                if (b1==b2 || hap>=nTargHaps) {
                    alProbs[b1] += prob;
                }
                else {
                    boolean isRare1 = fpd.isLowFreq(m, b1);
                    boolean isRare2 = fpd.isLowFreq(m, b2);
                    if (isRare1^isRare2) {
                        if (isRare1) {
                            alProbs[b1] += 0.55*prob;
                            alProbs[b2] += 0.45*prob;
                        }
                        else {
                            alProbs[b1] += 0.45*prob;
                            alProbs[b2] += 0.55*prob;
                        }
                    }
                    else {
                        alProbs[b1] += 0.5*prob;
                        alProbs[b2] += 0.5*prob;
                    }
                }
            }
        }
        return maxIndex(alProbs);
    }

    private int allele(int marker, int hap) {
        if (hap<nTargHaps) {
            return unphTargGT.allele(marker, hap);
        }
        else {
            return refGT.get().allele(marker, hap - nTargHaps);
        }
    }

    private int maxIndex(float[] fa) {
        int maxIndex = 0;
        for (int j=1; j<fa.length; ++j) {
            if (fa[j]>fa[maxIndex]) {
                maxIndex = j;
            }
        }
        return maxIndex;
    }
}
