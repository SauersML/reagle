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

import blbutil.FloatArray;
import java.util.Arrays;
import vcf.GT;

/**
 * <p>Class {@code HmmParamData} generates data for estimating
 * allele mismatch and recombination intensity parameters for a
 * haploid Li and Stephens hidden Markov model.</p>
 *
 * <p>Instances of class {@code HmmParamData} are not thread-safe.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class HmmParamData {

    private final PhaseData phaseData;
    private final GT gt;
    private final int nMarkers;
    private final FloatArray genDist;
    private final FloatArray pRecomb;
    private final byte[][][] alMatch;
    private final BasicPhaseStates states;

    private final float[] fwd;
    private final float[] bwd;
    private final float[][] savedBwd;
    private final float[] emProbs;

    private int mismatchCnt;
    private double sumMismatchProb;
    private double sumGenDist;
    private double sumSwitchProb;

    /**
     * Creates a {@code HmmParamData} instance for the specified data.
     *
     * @param phaseIbs the IBS haplotype segments
     * @throws NullPointerException if {@code phaseIbs == null}
     */
    public HmmParamData(PbwtPhaseIbs phaseIbs) {
        this.phaseData = phaseIbs.phaseData();
        FixedPhaseData fpd = phaseData.fpd();
        this.gt = fpd.stage1TargGT();
        int maxStates = fpd.par().phase_states();
        this.nMarkers = fpd.stage1TargGT().nMarkers();
        this.genDist = fpd.stage1Map().genDist();
        this.alMatch = new byte[2][nMarkers][maxStates];
        this.pRecomb = phaseData.pRecomb();
        this.states = new BasicPhaseStates(phaseIbs, maxStates);

        this.fwd = new float[maxStates];
        this.bwd = new float[maxStates];
        this.savedBwd = new float[nMarkers][maxStates];
        float pMismatch = phaseData.pMismatch();
        this.emProbs = new float[] {1f - pMismatch, pMismatch};

        this.mismatchCnt = 0;
        this.sumMismatchProb = 0.0;
        this.sumGenDist = 0.0;
        this.sumSwitchProb = 0.0;
    }

    /**
     * Returns the number of target samples.
     * @return the number of target samples
     */
    public int nTargSamples() {
        return gt.nSamples();
    }

    /**
     * Adds the generated data for estimating allele mismatch and recombination
     * intensity parameters to the specified object. The generated data is not
     * cleared.
     * @param paramEst the object that estimates model parameters
     * @throws NullPointerException if {@code paramEst == null}
     */
    public void addEstimationData(ParamEstimates paramEst) {
        paramEst.addMismatchData(mismatchCnt, sumMismatchProb);
        paramEst.addSwitchData(sumGenDist, sumSwitchProb);
        mismatchCnt = 0;
        sumMismatchProb = 0.0;
        sumGenDist = 0.0;
        sumSwitchProb = 0.0;
    }

    /**
     * Returns the sum of the probabilities of switching reference
     * haplotypes between consecutive markers obtained from the generated
     * data.
     * @return the sum of the probabilities of switching reference
     * haplotypes between consecutive markers obtained from the generated
     * data
     */
    public double sumSwitchProbs() {
        return sumSwitchProb;
    }

    /**
     * Uses the specified sample to generate data for estimating the
     * allele mismatch and recombination intensity HMM parameters.
     * @param sample a sample index
     * @throws IndexOutOfBoundsException if
     * {@code sample < 0 || sample >= this.nTargSamples()}
     */
    public void update(int sample) {
        int nStates = states.ibsStates(sample, alMatch);
        if (nStates>1) { // otherwise hFactor = nStates/(nStates - 1) is NaN
            getParamData(alMatch[0], nStates);
            getParamData(alMatch[1], nStates);
        }
    }

    private void getParamData(byte[][] alMatch, int nStates) {
        Arrays.fill(bwd, 0, nStates, 1.0f);
        Arrays.fill(savedBwd[nMarkers-1], 0, nStates, 1.0f);
        for (int m=nMarkers-2; m>=0; --m) {
            int mP1 = m + 1;
            HmmUpdater.bwdUpdate(bwd, pRecomb.get(mP1), emProbs,
                    alMatch[mP1], nStates);
            System.arraycopy(bwd, 0, savedBwd[m], 0, nStates);
        }
        float hFactor = nStates / (nStates - 1.0f);
        Arrays.fill(fwd, 0, nStates, 1.0f/nStates);
        float sum = 1.0f;
        for (int m=0; m<nMarkers; ++m) {
            sum = fwdUpdate(m, alMatch[m], nStates, sum, hFactor);
        }
    }

    private float fwdUpdate(int m, byte[] alDiscord, int nStates, float lastSum,
            float hFactor) {
        float pSwitch = pRecomb.get(m);
        float shift = pSwitch/nStates;
        float scale = (1.0f - pSwitch)/lastSum;
        float noSwitchScale = ((1.0f - pSwitch) + shift)/lastSum;
        float jointStateSum = 0.0f;
        float stateSum = 0.0f;

        float[] bwdM = savedBwd[m];
        float fwdSum = 0.0f;
        float mismatchSum = 0f;
        for (int k=0; k<nStates; ++k) {
            float em = emProbs[alDiscord[k]];
            jointStateSum += bwdM[k]*em*noSwitchScale*fwd[k];
            fwd[k] = em*(scale*fwd[k] + shift);
            fwdSum += fwd[k];
            float stateProb = fwd[k]*bwdM[k];
            stateSum += stateProb;
            if (alDiscord[k]>0) {
                mismatchSum += stateProb;
            }
        }
        ++mismatchCnt;
        sumMismatchProb += (mismatchSum/stateSum);
        double switchProb = hFactor*(1.0f - jointStateSum/stateSum);
        if (switchProb>0.0) {
            sumGenDist += genDist.get(m);
            sumSwitchProb += switchProb;
        }
        return fwdSum;
    }
}
