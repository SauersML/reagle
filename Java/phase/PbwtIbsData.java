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

import main.Par;

/**
 * <p>Class {@code PbwtIbsData} contains parameters and data for finding
 * haplotypes that share an IBS segment with a target haplotype.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class PbwtIbsData {

    private static final int BURNIN_CANDIDATES = 100;
    private static final int MAX_PHASE_CANDIDATES = 90;
    private static final int MIN_PHASE_CANDIDATES = 5;
    private static final int STAGE2_CANDIDATES = 10;
    private static final float MAX_BACKOFF_CM = 0.3f;

    private final CodedSteps codedSteps;
    private final int nHaps;
    private final int nTargHaps;
    private final int nCandidates;
    private final int nSteps;
    private final int nOverlapSteps;
    private final int maxBackoffSteps;
    private final int stepsPerBatch;
    private final int nBatches;

    /**
     * Constructs a new {@code PbwtIbsData} instance from the specified data.
     * @param phaseData the current genotype phase estimates and parameter
     * values
     * @param codedSteps the coded steps
     * @throws IllegalArgumentException if
     * {@code phaseData.fpd().stage1Steps() != codedSteps.steps()}
     * @throws IllegalArgumentException if
     * {@code phaseData.fpd().stage1XRefGT()!=codedSteps.refHaps()}
     * @throws IllegalArgumentException if
     * {@code phaseData.fpd().targGT().samples()!=codedSteps.targSamples()}
     * @throws NullPointerException if
     * {@code phaseData == null || codedSteps == null}
     */
    public PbwtIbsData(PhaseData phaseData, CodedSteps codedSteps) {
        checkConsistency(phaseData, codedSteps);
        FixedPhaseData fpd = phaseData.fpd();
        Par par = fpd.par();
        int nThreads = par.nthreads();
        int nIts = par.burnin() + par.iterations();

        this.codedSteps = codedSteps;
        this.nHaps = fpd.nHaps();
        this.nTargHaps = phaseData.fpd().targGT().nHaps();
        this.nCandidates = phaseData.it()<nIts
                ? nCandidates1(phaseData)
                : Math.min(STAGE2_CANDIDATES, phaseData.fpd().nHaps());
        this.nSteps = codedSteps.steps().size();
        this.nOverlapSteps = (int) Math.rint(par.buffer() / fpd.ibsStep());
        this.maxBackoffSteps = (int) Math.rint(MAX_BACKOFF_CM / fpd.ibsStep());
        this.stepsPerBatch = (nSteps + nThreads - 1) / nThreads;
        this.nBatches = (nSteps + stepsPerBatch - 1) / stepsPerBatch;
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

    private static int nCandidates1(PhaseData phaseData) {
        int nCandidates = BURNIN_CANDIDATES;
        int it = phaseData.it();
        Par par = phaseData.fpd().par();
        if (it>=par.burnin()) {
            double nItsRemaining = par.burnin() + par.iterations() - it;
            double p = (double) nItsRemaining / par.iterations();
            nCandidates = (int) Math.round(p*MAX_PHASE_CANDIDATES);
            nCandidates = Math.max(nCandidates, MIN_PHASE_CANDIDATES);
        }
        return Math.min(nCandidates, phaseData.fpd().nHaps());
    }

    /**
     * Returns the coded steps.
     * @return the codedSteps
     */
    public CodedSteps codedSteps() {
        return codedSteps;
    }

    /**
     * Returns the total number of target and reference haplotypes.
     * @return the total number of target and reference haplotypes
     */
    public int nHaps() {
        return nHaps;
    }

    /**
     * Returns the number of target haplotypes.
     * @return the number of target haplotypes
     */
    public int nTargHaps() {
        return nTargHaps;
    }

    /**
     * Returns the number of candidate haplotypes
     * @return the number of candidate haploytpes
     */
    public int nCandidates() {
        return nCandidates;
    }

    /**
     * Returns the number of steps.
     * @return the number of steps
     */
    public int nSteps() {
        return nSteps;
    }

    /**
     * Returns the number of overlap steps
     * @return the number of overlap steps
     */
    public int nOverlapSteps() {
        return nOverlapSteps;
    }

    /**
     * Returns the number of backoff steps
     * @return the number of backoff steps
     */
    public int maxBackoffSteps() {
        return maxBackoffSteps;
    }

    /**
     * Returns the number of steps per batch
     * @return the number of steps per batch
     */
    public int stepsPerBatch() {
        return stepsPerBatch;
    }

    /**
     * Returns the number of batches.
     * @return the number of batches
     */
    public int nBatches() {
        return nBatches;
    }

    /**
     * Returns the start step (inclusive) for the specified batch:
     * {@code (batch * this.stepsPerbatch())}.
     * @param batch a batch index
     * @throws IndexOutOfBoundsException if
     * {@code (batch < 0 || batch >= this.nBatches()) }
     * @return the start step (inclusive) for the specified batch
     */
    public int startStep(int batch) {
        if (batch < 0 || batch >= nBatches) {
            throw new IndexOutOfBoundsException(String.valueOf(batch));
        }
        return batch*stepsPerBatch;
    }

    /**
     * Returns the end step (exclusive) for the specified batch:
     * {@code Math.min((batch+1)*this.stepsPerBatch(), this.nSteps())}.
     * @param batch a batch index
     * @throws IndexOutOfBoundsException if
     * {@code (batch < 0 || batch >= this.nBatches()) }
     * @return the end step (exclusive) for the specified batch
     */
    public int endStep(int batch) {
        if (batch < 0 || batch >= nBatches) {
            throw new IndexOutOfBoundsException(String.valueOf(batch));
        }
        return Math.min((batch+1)*stepsPerBatch, nSteps);
    }

    /**
     * Returns the start step (inclusive) of the start buffer segment:
     * {@code Math.max((0, startStep - this.nOverlapSteps())}.
     * @param startStep the start step (inclusive) of a segment
     * @throws IndexOutOfBoundsException if
     * {@code (startStep < 0 || startStep >= this.nSteps()) }
     * @return the start step (inclusive) of the start buffer segment
     */
    public int bufferStartStep(int startStep) {
        if (startStep < 0 || startStep >= nSteps) {
            throw new IndexOutOfBoundsException(String.valueOf(startStep));
        }
        return Math.max(0, startStep - nOverlapSteps);
    }

    /**
     * Returns the end step (exclusive) of the end buffer segment:
     * {@code Math.min((endStep + this.nOverlapSteps(), this.nSteps())}.
     * @param endStep the end step (exclusive) of a segment
     * @throws IndexOutOfBoundsException if
     * {@code (endStep <= 0 || endStep > this.nSteps()) }
     * @return the end step (exclusive) of the end buffer segment
     */
    public int bufferEndStep(int endStep) {
        if (endStep <= 0 || endStep > nSteps) {
            throw new IndexOutOfBoundsException(String.valueOf(endStep));
        }
        return Math.min(endStep + nOverlapSteps, nSteps);
    }
}
