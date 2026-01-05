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
package imp;

import ints.IntIntMap;
import ints.IntList;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.stream.IntStream;
import beagleutil.CompHapSegment;

/**
 * <p>Class {@code ImpStates} identifies a list of pseudo-reference haplotypes
 * for a target haplotype. Each pseudo-reference haplotype is a
 * one-dimensional mosaic of reference haplotype segments.
 * </p>
 * <p>Instances of {@code ImpStates} are not thread-safe.
 * </p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class ImpStates {

    private final int NIL = -103;
    private final ImpIbs ibsHaps;
    private final ImpData impData;
    private final int nClusters;
    private final int maxStates;

    private final IntIntMap hapToLastIbsStep;
    private final PriorityQueue<CompHapSegment> q;
    private final IntList[] compHapHap;
    private final IntList[] compHapEnd;

    private final int[] compHapToListIndex;
    private final int[] compHapToHap;
    private final int[] compHapToEnd;

    /**
     * Constructs a new {@code ImpStates} object from the specified data.
     * @param ibsHaps the IBS haplotype segments
     * @throws NullPointerException if {@code ibsHaps == null}
     */
    public ImpStates(ImpIbs ibsHaps) {
        this.ibsHaps = ibsHaps;
        this.impData = ibsHaps.impData();
        this.nClusters = ibsHaps.impData().nClusters();
        this.maxStates = impData.par().imp_states();
        this.hapToLastIbsStep = new IntIntMap(maxStates);
        this.q = new PriorityQueue<>(maxStates);
        this.compHapHap = IntStream.range(0, maxStates)
                .mapToObj(j -> new IntList())
                .toArray(IntList[]::new);
        this.compHapEnd = IntStream.range(0, maxStates)
                .mapToObj(j -> new IntList())
                .toArray(IntList[]::new);
        this.compHapToListIndex = new int[maxStates];
        this.compHapToHap = new int[maxStates];
        this.compHapToEnd = new int[maxStates];
    }

    /**
     * Returns the maximum number of HMM states at a marker.
     * @return the maximum number of HMM states at a marker
     */
    public int maxStates() {
        return maxStates;
    }

    /**
     * Stores the reference haplotype for the {@code j}-th state
     * at the {@code m}-th marker in {@code hapIndices[m][j]}, and stores
     * the equality of the allele carried by the reference haplotype for
     * the {@code j}-th state and the allele carried by the target haplotype
     * at the {@code m}-th marker in {@code alMatch[m][j]}.  The number of
     * HMM states states at each marker is returned.
     * @param targHap the haplotype index
     * @param haps the two-dimensional array in which
     * reference haplotype indices for each HMM state will be stored
     * @param alMatch the two-dimensional array in which allele match status
     * between the target haplotype and HMM state will be stored
     * @return the number of HMM states at each marker
     *
     * @throws IndexOutOfBoundsException if
     * {@code targHap < 0 || targHap >= this.impData().nTargHaps()}
     * @throws IndexOutOfBoundsException if either two-dimensional
     * array is not large enough to contain the rectangular array of
     * HMM states
     * @throws NullPointerException if any array is {@code null}
     */
    public int ibsStates(int targHap, int[][] haps, boolean[][] alMatch) {
        initializeFields();
        for (int j=0, n=ibsHaps.codedSteps().nSteps(); j<n; ++j) {
            int[] ibs = ibsHaps.ibsHaps(targHap, j);
            for (int hap : ibs) {
                updateFields(hap, j);
            }
        }
        if (q.isEmpty()) {
            fillQWithRandomHaps(targHap);
        }
        int numStates = copyData(targHap, haps, alMatch);
        return numStates;
    }

    private void initializeFields() {
        hapToLastIbsStep.clear();
        for (int j=0, n=q.size(); j<n; ++j) {
            compHapHap[j].clear();
            compHapEnd[j].clear();
        }
        q.clear();
    }

    private void updateFields(int hap, int step) {
        if (hapToLastIbsStep.get(hap, NIL)==NIL) { // hap not currently in q
            updateHeadOfQ();
            if (q.size()==maxStates) {
                CompHapSegment head = q.poll();
                int startMarker = ibsHaps.codedSteps().stepStart((head.lastIbsStep() + step) >>> 1);
                hapToLastIbsStep.remove(head.hap());
                compHapHap[head.compHapIndex()].add(hap);         // hap of new segment
                compHapEnd[head.compHapIndex()].add(startMarker); // end of previous segment
                head.updateSegment(hap, startMarker, step);
                q.offer(head);
            }
            else {
                int compHapIndex = q.size();
                int startMarker = 0;
                compHapHap[compHapIndex].add(hap);                // hap of new segment
                q.offer(new CompHapSegment(hap, startMarker, step, compHapIndex));
            }
        }
        hapToLastIbsStep.put(hap, step);
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

    private int copyData(int targHap, int[][] hapIndices, boolean[][] alMatch) {
        int nCompHaps = q.size();
        int shiftedTargHap = impData.nRefHaps() + targHap;
        initializeCopy(nCompHaps);
        for (int m=0; m<nClusters; ++m) {
            int targAllele = impData.allele(m, shiftedTargHap);
            for (int j=0; j<nCompHaps; ++j) {
                if (m==compHapToEnd[j]) {
                    ++compHapToListIndex[j];
                    compHapToHap[j] = compHapHap[j].get(compHapToListIndex[j]);
                    compHapToEnd[j] = compHapEnd[j].get(compHapToListIndex[j]);
                }
                hapIndices[m][j] = compHapToHap[j];
                alMatch[m][j] = impData.allele(m, compHapToHap[j])==targAllele;
            }
        }
        return nCompHaps;
    }

    private void initializeCopy(int nSlots) {
        for (int j=0; j<nSlots; ++j) {
            compHapEnd[j].add(nClusters); // add missing end of last segment
            compHapToListIndex[j] = 0;
            compHapToHap[j] = compHapHap[j].get(0);
            compHapToEnd[j] = compHapEnd[j].get(0);
        }
    }

    private void fillQWithRandomHaps(int hap) {
        assert q.isEmpty();
        int nRefHaps = impData.nRefHaps();
        int nStates = Math.min(nRefHaps, maxStates);
        Random rand = new Random(hap);
        int ibsStep = 0;
        int startMarker = 0;
        for (int j=0; j<nStates; ++j) {
            int h = rand.nextInt(nRefHaps);
            while (h==hap) {
                h = rand.nextInt(nRefHaps);
            }
            compHapHap[j].add(h);            // hap of new segment
            q.add(new CompHapSegment(h, startMarker, ibsStep, j));
        }
    }
}
