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

/**
 * <p>Class {@code HmmStateProbs} has a method that returns the reference
 * haplotype and probability associated with each HMM state.</p>
 *
 * <p>Instances of class {@code HmmStateProbs} are not thread-safe.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class HmmStateProbs {

    private final PhaseData phaseData;
    private final LowFreqPhaseStates states;
    private final FloatArray pRecomb;
    private final byte[][] mismatch;
    private final float[] bwd;
    private final float[] pMismatch;

    /**
     * Creates a {@code HmmStateProbs} instance from the specified data.
     * @param phaseIbs the IBS haplotypes
     * @throws NullPointerException if {@code phaseIbs == null}
     */
    public HmmStateProbs(LowFreqPhaseIbs phaseIbs) {
        this.phaseData = phaseIbs.phaseData();
        int nMarkers = phaseData.fpd().stage1TargGT().nMarkers();
        int maxStates = phaseData.fpd().par().phase_states()/2;
        this.states = new LowFreqPhaseStates(phaseIbs, maxStates);
        this.pRecomb = phaseData.pRecomb();
        this.mismatch = new byte[nMarkers][maxStates];
        this.bwd = new float[maxStates];
        float pMiss = phaseData.pMismatch();
        this.pMismatch = new float[] {1.0f - pMiss, pMiss};
    }

    /**
     * Stores the HMM reference haplotypes and states probabilities for the
     * specified target haplotype, and returns the number of HMM states
     * per marker.  The contract for this method is undefined
     * if the number of elements in any row of the specified arrays is not
     * greater than or equal to {@code this.maxStates()}.
     *
     * @param targHap a target haplotype index
     * @param refHaps the array in which the reference haplotype corresponding
     * to each hidden state will be stored
     * @param stateProbs the array in which the estimated probability of each
     * hidden state will be stored
     * @return the number of hidden states at each marker
     *
     * @throws IndexOutOfBoundsException if
     * {@code targHap < 0 || targHap >= this.nTargHaps()}
     * @throws IndexOutOfBoundsException if
     * {@code refHaps.length < this.nMarkers()}
     * @throws IndexOutOfBoundsException if
     * {@code stateProbs.length < this.nMarkers()}
     * @throws NullPointerException if
     * {@code refHaps == null || stateProbs == null}
     */
    public int run(int targHap, int[][] refHaps, float[][] stateProbs) {
        int nStates = states.ibsStates(targHap, refHaps, mismatch);
        runFwd(stateProbs, nStates);
        runBwd(stateProbs, nStates);
        return nStates;
    }

    private void runFwd(float[][] probs, int nStates) {
        float lastSum = 0.0f;
        for (int j=0; j<nStates; ++j) {
            probs[0][j] = pMismatch[mismatch[0][j]];
            lastSum += probs[0][j];
        }
        for (int m=1; m<probs.length; ++m) {
            int mM1 = m - 1;
            float pRec = pRecomb.get(m);
            float shift = pRec/nStates;
            float scale = (1.0f - pRec)/lastSum;
            lastSum = 0.0f;
            for (int j=0; j<nStates; ++j) {
                float em = pMismatch[mismatch[m][j]];
                probs[m][j] = em*(scale*probs[mM1][j] + shift);
                lastSum += probs[m][j];
            }
        }
    }

    private void runBwd(float[][] probs, int nStates) {
        int inclEnd = probs.length - 1;
        Arrays.fill(bwd, 0, nStates, 1.0f/nStates);
        for (int m=inclEnd-1; m>=0; --m) {
            int mP1 = m + 1;
            float sum = 0.0f;
            for (int j=0; j<nStates; ++j) {
                bwd[j] *= pMismatch[mismatch[mP1][j]];
                sum += bwd[j];
            }
            float pRec = pRecomb.get(mP1);
            float scale = (1.0f - pRec)/sum;
            float shift = pRec/nStates;
            sum = 0.0f;
            for (int j=0; j<nStates; ++j) {
                bwd[j] = scale*bwd[j] + shift;
                probs[m][j] *= bwd[j];
                sum += probs[m][j];
            }
            for (int j=0; j<nStates; ++j) {
                probs[m][j] /= sum;
            }
        }
    }

    /**
     * Returns the number of markers
     * @return the number of markers
     */
    public int nMarkers() {
        return mismatch.length;
    }

    /**
     * Returns the number of target haplotypes.
     * @return the number of target haplotypes
     */
    public int nTargHaps() {
        return phaseData.fpd().targGT().nHaps();
    }

    /**
     * Returns the maximum number of HMM states.
     * @return the maximum number of HMM states
     */
    public int maxStates() {
        return bwd.length;
    }
}