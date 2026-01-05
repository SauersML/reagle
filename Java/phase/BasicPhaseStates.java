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
import blbutil.BitArray;
import blbutil.Utilities;
import ints.IntIntMap;
import java.util.Arrays;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.stream.IntStream;
import beagleutil.CompHapSegment;
import vcf.Markers;
import vcf.XRefGT;

/**
 * <p>Class {@code BasicPhaseStates} has methods for constructing a Li and
 * Stephens HMM for a target haplotype or target sample.
 * </p>
 * <p>Instances of {@code BasicPhaseStates} are not thread-safe.
 * </p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class BasicPhaseStates {

    private static final int NIL = -103;
    private final PbwtPhaseIbs ibsHaps;
    private final PhaseData phaseData;
    private final Steps steps;
    private final XRefGT allHaps;
    private final Markers markers;
    private final int nMarkers;
    private final int maxStates;
    private final int minSteps;

    private final IntIntMap hapToLastIbsStep;
    private final PriorityQueue<CompHapSegment> q;

    private final BitArray[] compHaps;

    /**
     * Constructs a new {@code BasicPhaseStates} object from the specified data.
     * @param ibsHaps the IBS haplotype segments
     * @param maxStates the maximum number of composite reference
     * haplotypes that will be constructed
     * @throws IllegalArgumentException if {@code maxStates < 1}
     * @throws NullPointerException if {@code ibsHaps == null}
     */
    public BasicPhaseStates(PbwtPhaseIbs ibsHaps, int maxStates) {
        if (maxStates < 1) {
            throw new IllegalArgumentException(String.valueOf(maxStates));
        }
        this.ibsHaps = ibsHaps;
        this.phaseData = ibsHaps.phaseData();
        this.steps = phaseData.fpd().stage1Steps();
        this.allHaps = ibsHaps.allHaps();
        this.markers = allHaps.markers();
        this.nMarkers = allHaps.nMarkers();
        this.maxStates = maxStates;
        float phaseStep = phaseData.fpd().ibsStep();
        this.minSteps = Math.max(200, (int) Math.ceil(1.0f/phaseStep)); // 200 steps and 1 cM
        this.hapToLastIbsStep = new IntIntMap(maxStates);
        this.q = new PriorityQueue<>(maxStates);

        int nBits = markers.sumHapBits();
        this.compHaps = IntStream.range(0, maxStates)
                .mapToObj(j -> new BitArray(nBits))
                .toArray(BitArray[]::new);
    }

    /**
     * Returns the number of target samples.
     * @return the number of target samples
     */
    public int nTargSamples() {
        return phaseData.fpd().targGT().nSamples();
    }

    /**
     * Returns the number of markers.
     * @return the number of markers
     */
    public int nMarkers() {
        return phaseData.fpd().targGT().nMarkers();
    }

    /**
     * Returns the maximum number of HMM states at a marker.
     * @return the maximum number of HMM states at a marker
     */
    public int maxStates() {
        return maxStates;
    }

    /**
     * Stores the Li and Stephens HMM for the specified target sample in
     * the specified arrays.The {@code nMismatches} parameter is an array of
     * three two-dimensional arrays: {@code nMismatches[0]} stores
     * stores the allele mismatch data between the reference haplotypes and
     * the haplotype composed of homozygous target genotypes,
     * {@code nMismatches[1]} stores the allele mismatch data between
     * the reference haplotypes and the first target haplotype, and
     * {@code nMismatches[2]} stores the allele mismatch data between
     * the reference haplotypes and the first target haplotype.  Each
     * two-dimensional array must have at least {@code mc.nClusters()}
     * rows, and a column for each HMM state. An element of the
     * two-dimensional array is 0 if the target and reference allele match
     * and is 1 otherwise.
     *
     * @param mc the marker clusters
     * @param refAtMissingGT a list of arrays in which HMM state alleles
     * at markers for which one or both target haplotypes have a missing allele
     * @param nMismatches arrays for storing present or absence of mismatches
     * between target and reference alleles
     * @return the number of state alleles at each marker
     *
     * @throws IndexOutOfBoundsException if
     * {@code sample < 0 || sample >= this.nTargSamples()}
     * @throws IndexOutOfBoundsException if {@code nMismatches.length < 3}
     * @throws IndexOutOfBoundsException if there exists {@code j} such that
     * {@code 0 < j && j < 3 && nMismatches[j].length < mc.nClusters()}
     * @throws IndexOutOfBoundsException if there exists {@code j, m} such that
     * {@code (0 < j && j < 3 && 0 < m && m < mc.nClusters())}, and
     * {@code nMismatches[j][m].length} is less than the number of model
     * states at marker {@code m}
     * @throws IndexOutOfBoundsException if {@code missAlleles.get(j)}
     * is less than the number of model states for any {@code j}
     * that indexes the missing genotypes
     * @throws NullPointerException if
     * {@code (samplePhase == null || mc == null || refAtMissingGT == null)}
     * or if any array is {@code null}
     */
    public int ibsStates(MarkerCluster mc, List<int[]> refAtMissingGT,
            byte[][][] nMismatches) {
        int nCompHaps = setCompRefHaps(mc.samplePhase().sample());
        copyData(mc, nCompHaps, refAtMissingGT, nMismatches);
        return nCompHaps;
    }

    /**
     * Stores the Li and Stephens HMM for the specified target sample in
     * the specified arrays.  The number of allele mismatches (0 or 1)
     * between {@code hap1[m]} and {@code hap2[m]} for the {@code j}-th state
     * are stored in {@code nMismatchs[0][m][j]} and
     * {@code nMismatches[1][m][j]} respectively.
     *
     * @param sample the target sample index
     * @param nMismatches an array of two two-dimensional arrays in which the
     * number of allele mismatches with reference haplotypes for the first
     * haplotype and the second haplotype will be stored
     * @return the number of state alleles at each marker
     *
     * @throws IndexOutOfBoundsException if
     * {@code sample < 0 || sample >= this.nTargSamples()}
     * @throws IndexOutOfBoundsException if {@code nMismatches.length < 2}
     * @throws IndexOutOfBoundsException if there exists {@code j} such that
     * {@code (0 < j && j < 2 && nMismatches[j].length < this.nMarkers())}
     * @throws IndexOutOfBoundsException if there exists {@code j, m} such that
     * {@code (0 < j && j < 2 && 0 < m && m < this.nMarkers())}, and
     * {@code nMismatches[j][m].length} is less than the total number of model
     * states
     * @throws NullPointerException if any array is {@code null}
     */
    public int ibsStates(int sample, byte[][][] nMismatches) {
        int nCompHaps = setCompRefHaps(sample);
        copyData(sample, nCompHaps, nMismatches);
        return nCompHaps;
    }

    private int setCompRefHaps(int sample) {
        int h1 = sample << 1;
        int h2 = (h1 | 0b1);
        q.clear();
        hapToLastIbsStep.clear();
        for (int step=0, n=steps.size(); step<n; ++step) {
            int ibsHap1 = ibsHaps.ibsHap(h1, step);
            if (ibsHap1>=0) {
                addIbsHap(ibsHap1, step);
            }
            int ibsHap2 = ibsHaps.ibsHap(h2, step);
            if (ibsHap2>=0) {
                addIbsHap(ibsHap2, step);
            }
        }
        if (q.isEmpty()) {
            fillQWithRandomHaps(sample);
        }
        int nCompHaps = copyFinalRefSegs();
        return nCompHaps;
    }

    private void addIbsHap(int ibsHap, int step) {
        if (hapToLastIbsStep.get(ibsHap, NIL)==NIL) { // hap not currently in q
            updateHeadOfQ();
            if (q.size()==maxStates
                    || (q.isEmpty()==false && step - q.peek().lastIbsStep() >= minSteps)) {
                CompHapSegment head = q.poll();
                int index = head.compHapIndex();
                int prevHap = head.hap();
                int prevStart = head.startMarker();
                int nextStart = steps.start((head.lastIbsStep() + step) >>> 1);
                hapToLastIbsStep.remove(head.hap());
                allHaps.copyTo(prevHap, prevStart, nextStart, compHaps[index]);

                head.updateSegment(ibsHap, nextStart, step);
                q.offer(head);
            }
            else {
                int index = q.size();
                int start = 0;
                q.offer(new CompHapSegment(ibsHap, start, step, index));
            }
        }
        hapToLastIbsStep.put(ibsHap, step);
    }

    private void updateHeadOfQ() {
        CompHapSegment head = q.peek();
        if (head!=null) {
            int lastIbsStep = hapToLastIbsStep.get(head.hap(), NIL);
            while (head.lastIbsStep()!=lastIbsStep) {
                head = q.poll();
                head.setLastIbsStep(lastIbsStep);
                q.offer(head);
                head = q.peek();
                lastIbsStep = hapToLastIbsStep.get(head.hap(), NIL);
            }
        }
    }

    private int copyFinalRefSegs() {
        int nCompHaps = q.size();
        CompHapSegment head = q.poll();
        while (head!=null) {
            int index = head.compHapIndex();
            int hap = head.hap();
            int startMarker = head.startMarker();
            allHaps.copyTo(hap, startMarker, nMarkers, compHaps[index]);
            head = q.poll();
        }
        return nCompHaps;
    }

    private void copyData(MarkerCluster mc, int nCompHaps,
            List<int[]> refAtMissingGT, byte[][][] nMismatches) {
        SamplePhase phase = mc.samplePhase();
        BitArray hap1 = phase.hap1();
        BitArray hap2 = phase.hap2();
        int missIndex = 0;
        int nClusters = mc.nClusters();
        for (int c=0; c<nClusters; ++c) {
            Arrays.fill(nMismatches[0][c], 0, nCompHaps, (byte) 0);
            Arrays.fill(nMismatches[1][c], 0, nCompHaps, (byte) 0);
            Arrays.fill(nMismatches[2][c], 0, nCompHaps, (byte) 0);
            int mStart = mc.clusterStart(c);
            int mEnd = mc.clusterEnd(c);
            if (mc.isMissingGtOrMaskedHet(c)) {
                assert mEnd-mStart==1;
                int[] refAlleles = refAtMissingGT.get(missIndex++);
                for (int j=0; j<nCompHaps; ++j) {
                    refAlleles[j] = markers.allele(compHaps[j], mStart);
                }
            }
            else {
                int bStart = markers.sumHapBits(mStart);
                int bEnd = markers.sumHapBits(mEnd);
                if (hap1.equal(hap2, bStart, bEnd)) {
                    for (int j=0; j<nCompHaps; ++j) {
                        if (hap1.equal(compHaps[j], bStart, bEnd)==false) {
                            nMismatches[0][c][j] = 1;
                            nMismatches[1][c][j] = 1;
                            nMismatches[2][c][j] = 1;
                        }
                    }
                }
                else {
                    // cluster contains a heterozygote genotype
                    for (int j=0; j<nCompHaps; ++j) {
                        if (hap1.equal(compHaps[j], bStart, bEnd)==false) {
                            nMismatches[1][c][j] = 1;
                        }
                        if (hap2.equal(compHaps[j], bStart, bEnd)==false) {
                            nMismatches[2][c][j] = 1;
                        }
                    }
                }
            }
        }
    }

    private int copyData(int sample, int nCompHaps, byte[][][] nMismatches) {
        byte[][] nMismatches1 = nMismatches[0];
        byte[][] nMismatches2 = nMismatches[1];
        int h1 = sample << 1;
        int h2 = h1 | 0b1;
        for (int m=0; m<nMarkers; ++m) {
            int a1 = allHaps.allele(m, h1);
            int a2 = allHaps.allele(m, h2);
            for (int j=0; j<nCompHaps; ++j) {
                int refAllele = markers.allele(compHaps[j], m);
                nMismatches1[m][j] = (refAllele==a1 ? (byte) 0 : (byte) 1);
                nMismatches2[m][j] = (refAllele==a2 ? (byte) 0 : (byte) 1);
            }
        }
        return nCompHaps;
    }

    private void fillQWithRandomHaps(int sample) {
        assert q.isEmpty();
        int nHaps = allHaps.nHaps();
        int nStates = Math.min(nHaps-2, maxStates);
        if (nStates<=0) {
            Utilities.exit("ERROR: there is only one sample");
        }
        else {
            Random rand = new Random(phaseData.seed() + sample);
            int ibsStep = 0;
            int startMarker = 0;
            int compHapIndex = 0;
            for (int j=0; j<nStates; ++j) {
                int h = rand.nextInt(nHaps);
                while ((h>>1)==sample) {
                    h = rand.nextInt(nHaps);
                }
                if (hapToLastIbsStep.get(h, NIL)==NIL) {
                    q.add(new CompHapSegment(h, startMarker, ibsStep, compHapIndex++));
                    hapToLastIbsStep.put(h, startMarker);
                }
            }
        }
    }
}