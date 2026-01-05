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

import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * <p>Class {@code ParamEstimates} estimates the allele mismatch probability
 * and the recombination intensity for a haploid Li and Stephens hidden
 * Markov model.</p>
 *
 * <p>Instances of class {@code ParamEstimates} are thread-safe.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class ParamEstimates {

    private final ConcurrentLinkedQueue<RecombData> switchData;
    private final ConcurrentLinkedQueue<MismatchData> mismatchData;

    /**
     * Constructs a new {@code ParamEstimates} instance for the specified
     * data.
     */
    public ParamEstimates() {
        this.switchData = new ConcurrentLinkedQueue<>();
        this.mismatchData = new ConcurrentLinkedQueue<>();
    }

    /**
     * Records the specified allele mismatch data if {@code markerCnt} and
     * {@code pMismatchSum} are finite positive values.
     * @param markerCnt the number of markers with mismatch data
     * @param pMismatchSum the sum of estimated allele mismatch probabilities
     */
    public void addMismatchData(int markerCnt, double pMismatchSum) {
        if (markerCnt>0 && pMismatchSum>0 && Double.isFinite(pMismatchSum)) {
            mismatchData.add(new MismatchData(markerCnt, pMismatchSum));
        }
    }

    /**
     * Records the specified genetic distance and switch probability
     * if {@code genDistances} and {@code switchProbs} are finite positive
     * values.
     * @param genDistances the list of genetic distance
     * @param switchProbs the list of haplotype switch probabilities
     */
    public void addSwitchData(double genDistances, double switchProbs) {
        if (genDistances>0 && switchProbs>0
                && Double.isFinite(genDistances) && Double.isFinite(switchProbs)) {
            switchData.add(new RecombData(genDistances, switchProbs));
        }
    }

    /**
     * Returns the estimated allele mismatch rate. Returns {@code Float.NaN}
     * if there is no data to estimate the allele mismatch rate.  The returned
     * value is not an atomic snapshot. Invocation in the absence of concurrent
     * update will return an accurate result, but concurrent updates that
     * occur white the sum is being calculated might not be incorporated
     * in the returned result.
     * @return the estimated allele mismatch rate
     */
    public float pMismatch() {
        MismatchData[] mda = mismatchData.stream()
                .sorted() // ensures sum of values is repeatable
                .toArray(MismatchData[]::new);
        long sumMarkers = 0;
        double sumPMismatch = 0d;
        for (MismatchData md : mda) {
            sumMarkers += md.markerCnt;
            sumPMismatch += md.pMismatchSum;
        }
        return sumMarkers==0 ? Float.NaN : (float) (sumPMismatch/sumMarkers);
    }

    /**
     * Returns the estimated recombination intensities. Returns {@code Float.NaN}
     * if there is no data from which to estimate recombination intensities.
     * The returned value is NOT an atomic snapshot. An accurate result is
     * guaranteed only if no concurrent updates occur during method
     * invocation.
     * @return the estimated recombination intensities
     */
    public float recombIntensity() {
        RecombData[] rda = switchData.stream()
                .sorted() // ensures sum of values is repeatable
                .toArray(RecombData[]::new);
        double sumSwitches = 0d;
        double sumDistances = 0d;
        for (RecombData rd : rda) {
            sumSwitches += rd.switchProb;
            sumDistances += rd.genDistance;
        }
        return sumDistances==0d ? Float.NaN : (float) (sumSwitches/sumDistances);
    }

    /**
     * Clears all data that has been added via the
     * {@code this.addMismatchData} and {@code this.addSwitchData()}
     * methods.
     */
    public void clear() {
        switchData.clear();
        mismatchData.clear();
    }

    private static class RecombData implements Comparable<RecombData> {

        private final double genDistance;
        private final double switchProb;

        public RecombData(double genDistance, double switchProb) {
            this.genDistance = genDistance;
            this.switchProb = switchProb;
        }

        @Override
        public int compareTo(RecombData o) {
            int val = Double.compare(this.genDistance, o.genDistance);
            return val!=0 ? val : Double.compare(this.switchProb, o.switchProb);
        }
    }

    private static class MismatchData implements Comparable<MismatchData> {

        private final int markerCnt;
        private final double pMismatchSum;

        public MismatchData(int markerCnt, double pMismatchSum) {
            this.markerCnt = markerCnt;
            this.pMismatchSum = pMismatchSum;
        }

        @Override
        public int compareTo(MismatchData o) {
            int val = Double.compare(this.pMismatchSum, o.pMismatchSum);
            return val!=0 ? val : Integer.compare(this.markerCnt, o.markerCnt);
        }
    }
}
