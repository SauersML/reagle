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
import ints.IndexArray;
import ints.IntIntMap;
import java.util.Optional;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import vcf.Samples;
import vcf.XRefGT;

/**
 * <p>Class {@code CodedSteps} divides phased genotype data
 * into non-overlapping intervals (the steps), indexes the unique
 * allele sequences in each interval, and stores a map of haplotype
 * index to allele sequence index for each interval.</p>
 *
 * <p>Instances of class {@code CodedSteps} are immutable.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class CodedSteps {

    private final Samples targSamples;
    private final Optional<XRefGT> refHaps;
    private final XRefGT allHaps;
    private final Steps steps;
    private final IndexArray[] codedSteps;

    /**
     * Constructs a new {@code CodedSteps} instance from the specified data.
     * @param estPhase the input data genotype phasing and the current phase
     * estimate for each target sample
     * @throws NullPointerException if {@code estPhase == null}
     */
    public CodedSteps(EstPhase estPhase) {
        FixedPhaseData fpd = estPhase.fpd();
        XRefGT targHaps = estPhase.phasedHaps();
        this.targSamples = targHaps.samples();
        this.refHaps = fpd.stage1XRefGT();
        this.allHaps = refHaps.isPresent()
                ? XRefGT.combine(targHaps, refHaps.get())
                : targHaps;
        this.steps = fpd.stage1Steps();
        this.codedSteps = codedSteps(allHaps, steps, fpd.par().nthreads());
    }

    private static IndexArray[] codedSteps(XRefGT gt, Steps steps, int nThreads) {
        int maxStepsPerBatch0 = 512;
        int nStepsPerBatch0 = (steps.size() + nThreads - 1)/nThreads;
        while (nStepsPerBatch0>maxStepsPerBatch0) {
            nStepsPerBatch0 = (nStepsPerBatch0+1) >> 1;
        }
        int stepsPerBatch = nStepsPerBatch0;
        int nBatches = (steps.size() + (stepsPerBatch-1)) / stepsPerBatch;
        return IntStream.range(0, nBatches)
                .parallel()
                .boxed()
                .flatMap(batch -> codedSteps(gt, steps, batch, stepsPerBatch))
                .toArray(IndexArray[]::new);
    }

    private static Stream<IndexArray> codedSteps(XRefGT gt, Steps steps,
            int batch, int batchSize) {
        int sentinal = -1;
        int startStep = batch*batchSize;
        int endStep = Math.min(startStep + batchSize, steps.size());
        int nSteps = endStep - startStep;
        int[][] hapToSeq = new int[nSteps][gt.nHaps()];
        int[] seqCnt = new int[nSteps];
        IntIntMap[] seqMap = IntStream.range(0, nSteps)
                .mapToObj(j -> new IntIntMap(8))
                .toArray(IntIntMap[]::new);
        for (int h=0, n=gt.nHaps(); h<n; ++h) {
            int mStart = steps.start(startStep);
            for (int j=0; j<nSteps; ++j) {
                int mEnd = steps.end(startStep + j);
                int key = gt.hash(h, mStart, mEnd);
                int seqIndex = seqMap[j].get(key, sentinal);
                if (seqIndex==sentinal) {
                    seqIndex = seqCnt[j]++;
                    seqMap[j].put(key, seqIndex);
                }
                hapToSeq[j][h] = seqIndex;
                mStart = mEnd;
            }
        }
        return IntStream.range(0, nSteps)
                .mapToObj(j -> new IndexArray(hapToSeq[j], seqCnt[j]));
    }

    /**
     * Returns a map from haplotype index to allele sequence index
     * for the specified step
     * @param step a step index
     * @return a map from haplotype index to allele sequence index
     * for the specified step
     * @throws IllegalArgumentException if
     * {@code step < 0 || step >= this.steps().size()}
     */
    public IndexArray get(int step) {
        return codedSteps[step];
    }

    /**
     * Returns the target samples.
     * @return the target samples
     */
    public Samples targSamples() {
        return targSamples;
    }

    /**
     * Returns the reference haplotypes
     * @return the reference haplotypes
     */
    public Optional<XRefGT> refHaps() {
        return refHaps;
    }

    /**
     * Return the phased target and reference genotype data that was used
     * to construct this {@code CodedSteps} instance.  The target haplotypes
     * precede the reference haplotypes.
     * @return the phased target and reference genotype data
     */
    public XRefGT allHaps() {
        return allHaps;
    }

    /**
     * Returns the partition of the markers into non-overlapping intervals.
     * @return the partition of the markers into non-overlapping intervals
     */
    public Steps steps() {
        return steps;
    }
}
