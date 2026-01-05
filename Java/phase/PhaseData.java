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
import java.util.stream.IntStream;
import main.Par;
import vcf.MarkerMap;

/**
 * <p>Class {@code PhaseData} stores the current genotype phase estimates
 * and parameter values.</p>
 *
 * <p>Instances of class {@code PhaseData} are thread-safe.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class PhaseData {

    private final EstPhase estPhase;
    private final float[] leaveUnphProp;
    private final long seed;

    private volatile int it;
    private volatile float lrThreshold;
    private volatile TrProb trProb;
    private volatile float pMismatch;

    /**
     * Constructs a new {@code PhaseData} instance from the specified data.
     * @param fpd the input data for phasing that is the same in each iteration
     * @param seed a seed for generating pseudorandom number
     *
     * @throws NullPointerException if {@code fpd == null}
     */
    public PhaseData(FixedPhaseData fpd, long seed) {
        Par par = fpd.par();
        this.estPhase = new EstPhase(fpd, seed);
        this.leaveUnphProp = leaveUnphasedProp(fpd, estPhase);
        this.seed = seed;
        this.it = 0;
        this.lrThreshold = lrThreshold(par, it);
        float recombIntensity = (float) 0.04f*fpd.par().ne()/fpd.nHaps();
        this.trProb = new TrProb(fpd.stage1Map(), recombIntensity);
        this.pMismatch = Par.liStephensPMismatch(fpd.nHaps());
    }

    static float lrThreshold(Par par, int it) {
        int nBurninIts = par.burnin();
        int nItsM1 = par.iterations() - 1;
        if (it<nBurninIts) {
            return Float.POSITIVE_INFINITY;
        }
        else if (it==(nItsM1 + nBurninIts)) {
            return 1f;
        }
        else {
            double lastVal = 4.0;
            double exp = (double) (nItsM1 - (it-nBurninIts)) / nItsM1;
            double base = par.initial_lr()/lastVal;
            return (float) (lastVal*Math.pow(base, exp));
        }
    }

    private static float[] leaveUnphasedProp(FixedPhaseData fpd,
            EstPhase estPhase) {
        int nIterations = fpd.par().iterations();
        int[] floatBits = IntStream.range(0, fpd.targGT().nSamples())
                .parallel()
                .map(s -> estPhase.get(s).nUnphased())
                .mapToDouble(cnt -> Math.pow(cnt, -1.0/nIterations))
                .mapToInt(p -> Float.floatToRawIntBits((float) p))
                .toArray();
        float[] fa = new float[floatBits.length];
        for (int j=0; j<fa.length; ++j) {
            fa[j] = Float.intBitsToFloat(floatBits[j]);
        }
        return fa;
    }

    /**
     * Returns the recombination intensity.
     * @return the recombination intensity
     */
    public float recombIntensity() {
        return trProb.recombIntensity;
    }

    /**
     * Updates the recombination intensity.
     * @param recombIntensity the updated recombination intensity.
     * @throws IllegalArgumentException if
     * {@code recombIntensity <= 0f || Float.isFinite(recombIntensity) == false}
     */
    public void updateRecombIntensity(float recombIntensity) {
        if (recombIntensity<=0f || Float.isFinite(recombIntensity)==false) {
            throw new IllegalArgumentException(String.valueOf(recombIntensity));
        }
        trProb = new TrProb(estPhase.fpd().stage1Map(), recombIntensity);
    }

    /**
     * Returns the effective population size.
     * @return the effective population size
     */
    public long ne() {
        return ne(trProb.recombIntensity);
    }

    /**
     * Returns the effective population size.
     * @param recombIntensity the recombination intensity
     * @return the effective population size
     */
    private long ne(float recombIntensity) {
        return (long) Math.ceil(25*recombIntensity*estPhase.fpd().nHaps());
    }

    /**
     * Return a {@code FloatArray} of size
     * {@code this.fpd().stage1TargXGT().nMarkers()}
     * whose {@code k}-th element is the probability of transitioning to a
     * random HMM state between the {@code (k-1)}-st and {@code k}-th
     * markers.
     * @return the probability of transitioning to a random HMM state between
     * pair of consecutive markers
     */
    public FloatArray pRecomb() {
        return trProb.pRecomb;
    }

    /**
     * Returns the allele mismatch probability.
     * @return the allele mismatch probability
     */
    public float pMismatch() {
        return pMismatch;
    }

    /**
     * Sets the allele mismatch probability to the specified value.
     * @param pMismatch the allele mismatch probability
     * @throws IllegalArgumentException if
     * {@code pMismatch < 0.0 || pMismatch > 1.0
     *      || Float.isFinite(pMismatch) == false}
     */
    public void updatePMismatch(float pMismatch) {
        if (pMismatch < 0.0 || pMismatch > 1.0
                || Float.isFinite(pMismatch)==false) {
            throw new IllegalArgumentException(String.valueOf(pMismatch));
        }
        this.pMismatch = pMismatch;
    }

    /**
     * Increments the iteration by one.
     */
    public void incrementIt() {
        ++it;
        lrThreshold = lrThreshold(estPhase.fpd().par(), it);
    }

    /**
     * Returns the current iteration.  The initial iteration is 0.
     * @return the current iteration
     */
    public int it() {
        return it;
    }

    /**
     * Sets the iteration equal to the maximum of {@code this.it()}
     * and {@code this.fpd().par().burnin()}.
     */
    public void advanceToFirstPhasingIt() {
        int nBurninIts = estPhase.fpd().par().burnin();
        if (it<nBurninIts) {
            it = nBurninIts;
            lrThreshold = lrThreshold(estPhase.fpd().par(), it);
        }
    }

    /**
     * Returns the input data for phasing that is the same in each iteration.
     * @return the input data for phasing that is the same in each iteration
     */
    public FixedPhaseData fpd() {
        return estPhase.fpd();
    }

    /**
     * Returns the estimated phased genotypes.
     * @return the estimated phased genotypes
     */
    public EstPhase estPhase() {
        return estPhase;
    }

    /**
     * Constructs and returns the coded steps. Since a {@code CodedSteps}
     * object can consume substantial memory, the returned object is not
     * pre-computed, but is constructed upon invocation of this method.
     * @return the coded steps
     */
    public CodedSteps codedSteps() {
        return new CodedSteps(estPhase);
    }

    /**
     * Returns the proportion of unphased heterozygotes at the start of a
     * phasing iteration that will be left unphased at the end of a
     * phasing iteration.
     * @param sample the sample index
     * @return the proportion of unphased heterozygotes at the start of a
     * phasing iteration that will be left unphased at the end of a
     * phasing iteration
     * @throws IndexOutOfBoundsException if
     * {@code sample < 0 || sample >= this.fpd().targGT().nSamples()}
     */
    public float leaveUnphasedProp(int sample) {
        return leaveUnphProp[sample];
    }

    /**
     * Returns the threshold on the phasing likelihood ratio that
     * determines whether a pair of heterozygotes will be marked as phased.
     * @return the threshold on the phasing likelihood ratio that
     * determines whether a pair of heterozygotes will be marked as phased
     */
    public float lrThreshold() {
        return lrThreshold;
    }

    /**
     * Returns an iteration-dependent seed for generating pseudorandom numbers.
     * @return an iteration-dependent seed for generating pseudorandom numbers
     */
    public long seed() {
        return seed + it;
    }

    private static class TrProb {

        private final float recombIntensity;
        private final FloatArray pRecomb;

        public TrProb(MarkerMap map, float recombItensity) {
            this.recombIntensity = recombItensity;
            this.pRecomb = map.pRecomb(recombIntensity);
        }
    }
}
