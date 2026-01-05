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

/**
 * <p>Class {@code HmmUpdater} has static methods for next marker
 * updates of forward and backward HMM values.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class HmmUpdater {

    private HmmUpdater() {
        // private constructor to prevent instantiation
    }

    /**
     * Updates the forward values and returns the sum of the updated forward
     * values.
     * @param fwd the array of forward values that will be updated
     * @param pSwitch an array of probabilities of jumping to a 
     * random HMM state
     * @param fwdSum the sum of forward values in the specified array
     * @param pMismatch two element array with emission probabilities
     * for 0 or 1 mismatches between the observed and reference
     * haplotype alleles
     * @param mismatch the number of allele mismatches (0 or 1) for
     * each HMM state
     * @param nStates the number of states
     * @return the sum of the updated forward values
     * @throws IllegalArgumentException if {@code pMismatch.length != 2}
     * @throws IndexOutOfBoundsException if there exists {@code j} such that
     * {@code (0 <= j && j < nStates)} and
     * {@code (mismatch[j] < 0 || mismatch[j] > 1)}
     * @throws IndexOutOfBoundsException if
     * {@code (fwd.length < nStates || mismatch.length < nStates)}
     * @throws NullPointerException if
     * {@code fwd == null || pMismatch == null || mismatch == null}
     */
    public static float fwdUpdate(float[] fwd, float fwdSum, float pSwitch,
            float[] pMismatch, byte[] mismatch, int nStates) {
        if (pMismatch.length!=2) {
            throw new IllegalArgumentException(String.valueOf(pMismatch.length));
        }
        float shift = pSwitch/nStates;
        float scale = (1.0f - pSwitch)/fwdSum;
        fwdSum = 0.0f;
        for (int k=0; k<nStates; ++k) {
            fwd[k] = pMismatch[mismatch[k]]*(scale*fwd[k] + shift);
            fwdSum += fwd[k];
        }
        return fwdSum;
    }

    /**
     * Updates the backward values.
     * @param bwd the array of backward values that will be updated
     * @param pSwitch an array of probabilities of jumping to a 
     * random HMM state
     * @param pMismatch two element array with emission probabilities
     * for 0 or 1 mismatches between alleles the observed and reference
     * haplotype alleles
     * @param mismatch the number of allele mismatches (0 or 1) for
     * each HMM state
     * @param nStates the number of states
     * @throws IllegalArgumentException if {@code pMismatch.length != 2}
     * @throws IndexOutOfBoundsException if there exists {@code j} such that
     * {@code (0 <= j && j < nStates)} and
     * {@code (mismatch[j] < 0 || mismatch[j] > 1)}
     * @throws IndexOutOfBoundsException if
     * {@code (bwd.length < nStates || mismatch.length < nStates)}
     * @throws NullPointerException if
     * {@code (bwd == null || mismatch == null)}
     * @throws NullPointerException if
     * {@code (bwd == null || pSwitch == null || pMismatch == null || mismatch == null)}
     */
    public static void bwdUpdate(float[] bwd, float pSwitch, float[] pMismatch,
            byte[] mismatch, int nStates) {
        if (pMismatch.length!=2) {
            throw new IllegalArgumentException(String.valueOf(pMismatch.length));
        }
        float sum = 0.0f;
        for (int k=0; k<nStates; ++k) {
            bwd[k] *= pMismatch[mismatch[k]];
            sum += bwd[k];
        }
        float shift = pSwitch/nStates;
        float scale = (1.0f - pSwitch)/sum;
        for (int k=0; k<nStates; ++k) {
            bwd[k] = scale*bwd[k] + shift;
        }
    }
}
