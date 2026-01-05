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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import phase.SamplePhase.ClustType;
import vcf.Markers;

/**
 * <p>Interface {@code PhaseBaum2} updates the estimated genotype phase
 * of specified samples.
 * </p>
 * <p>Instances of class {@code PhaseBaum2} are not thread-safe.
 * </p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class PhaseBaum2 implements PhaseBaum {

    private final PhaseData phaseData;
    private final boolean burnin;
    private final float lrThreshold;
    private final boolean maskTrailingHets;
    private final EstPhase estPhase;
    private final Markers markers;
    private final int nMarkers;
    private final List<int[]> refAlleles; // ref panel alleles for each missing genotype
    private final byte[][][] mismatches;  // [3][nMarker][nStates]
    private final float pMismatch;
    private final float[] emProbs;

    private final int maxStates;
    private final BasicPhaseStates states;
    private int nStates;
    private final float[][] fwd;
    private final float[][] bwd;
    private final float[] fwdSums;
    private final List<float[]> bwdMiss1;
    private final List<float[]> bwdMiss2;
    private final List<float[]> bwdHet1;
    private final List<float[]> bwdHet2;

    private boolean swapHaps = false;
    private int nSwaps = 0;

    /**
     * Creates a {@code PhaseLSBaum} instance from the specified data.
     *
     * @param phaseIbs the IBS haplotype segments

     * @throws NullPointerException if {@code phaseIBS == null}
     */
    public PhaseBaum2(PbwtPhaseIbs phaseIbs) {
        this.phaseData = phaseIbs.phaseData();
        this.burnin = phaseData.it() < phaseData.fpd().par().burnin();
        this.lrThreshold = phaseData.lrThreshold();
        float lrMaskThreshold = 50f;
        this.maskTrailingHets = lrThreshold<lrMaskThreshold;
        this.estPhase = phaseData.estPhase();
        this.markers = phaseData.fpd().stage1TargGT().markers();
        this.nMarkers = markers.size();
        this.maxStates = phaseData.fpd().par().phase_states();
        this.states = new BasicPhaseStates(phaseIbs, maxStates);

        this.refAlleles = new ArrayList<>();
        this.mismatches = new byte[3][nMarkers][maxStates];
        this.pMismatch = phaseData.pMismatch();
        this.emProbs = new float[] {1.0f - pMismatch, pMismatch};

        this.fwd = new float[3][maxStates];
        this.bwd = new float[3][maxStates];
        this.fwdSums = new float[3];
        this.bwdMiss1 = new ArrayList<>();
        this.bwdMiss2 = new ArrayList<>();
        this.bwdHet1 = new ArrayList<>();
        this.bwdHet2 = new ArrayList<>();
    }

    /**
     * Returns the number of target samples.
     * @return the number of target samples
     */
    @Override
    public int nTargSamples() {
        return phaseData.fpd().targGT().nSamples();
    }

    /**
     * Estimates and stores the phased haplotypes for the specified sample
     * @param sample a sample index
     * @throws IndexOutOfBoundsException if
     * {@code sample < 0 || sample >= this.nTargSamples()}
     */
    @Override
    public void phase(int sample) {
        SamplePhase samplePhase = estPhase.get(sample);
        if (maskTrailingHets) {
            samplePhase.maskTrailingUnphasedHets();
        }
        int nUnphHets = samplePhase.nUnphased();
        int nMaskedHets = samplePhase.nMasked();
        int nMissingOrMasked = samplePhase.nMissing() + nMaskedHets;
        if (nMissingOrMasked>0 || nUnphHets>0) {
            nSwaps = 0;
            swapHaps = false;
            MarkerCluster mc = new MarkerCluster(phaseData, sample);
            ensureCapacity(nUnphHets, nMissingOrMasked);
            nStates = states.ibsStates(mc, refAlleles, mismatches);
            bwdAlg(mc);
            fwdAlg(mc);
            estPhase.set(sample, samplePhase);
            SwapRate.increment(nUnphHets, nSwaps);
        }
    }

    private void ensureCapacity(int nUnph, int nMiss) {
        if (refAlleles.size()<nMiss) {
            for (int j=refAlleles.size(); j<nMiss; ++j) {
                refAlleles.add(new int[maxStates]);
                bwdMiss1.add(new float[maxStates]);
                bwdMiss2.add(new float[maxStates]);
            }
        }
        if (bwdHet1.size()<nUnph) {
            for (int j=bwdHet1.size(); j<nUnph; ++j) {
                bwdHet1.add(new float[maxStates]);
                bwdHet2.add(new float[maxStates]);
            }
        }
    }

    private void bwdAlg(MarkerCluster mc) {
        SamplePhase samplePhase = mc.samplePhase();
        int missIndex = samplePhase.nMissing() + samplePhase.nMasked() - 1;
        int unphIndex = samplePhase.nUnphased() - 1;
        initializeBwdFields();
        int lastCluster = mc.nClusters()-1;
        if (mc.isMissingGtOrMaskedHet(lastCluster)) {
            System.arraycopy(bwd[0], 0, bwdMiss1.get(missIndex), 0, nStates);
            System.arraycopy(bwd[0], 0, bwdMiss2.get(missIndex), 0, nStates);
            --missIndex;
        }
        for (int c=(lastCluster-1); c>=0; --c) {
            bwdStep(mc, c);
            if (mc.isMissingGtOrMaskedHet(c)) {
                System.arraycopy(bwd[1], 0, bwdMiss1.get(missIndex), 0, nStates);
                System.arraycopy(bwd[2], 0, bwdMiss2.get(missIndex), 0, nStates);
                --missIndex;
            }
            if (mc.isUnphasedHet(c+1)) {
                System.arraycopy(bwd[1], 0, bwdHet1.get(unphIndex), 0, nStates);
                System.arraycopy(bwd[2], 0, bwdHet2.get(unphIndex), 0, nStates);
                System.arraycopy(bwd[0], 0, bwd[1], 0, nStates);
                System.arraycopy(bwd[0], 0, bwd[2], 0, nStates);
                --unphIndex;
            }
        }
        assert missIndex == -1;
        assert unphIndex == -1;
    }

    private void initializeBwdFields() {
        Arrays.fill(bwd[0], 0, nStates, 1.0f/nStates);
        System.arraycopy(bwd[0], 0, bwd[1], 0, nStates);
        System.arraycopy(bwd[0], 0, bwd[2], 0, nStates);
    }

    private void bwdStep(MarkerCluster mc, int cluster) {
        int cP1 = cluster + 1;
        float pRec = mc.pRecomb().get(cP1);
        float clustEm = (mc.clusterEnd(cP1) - mc.clusterStart(cP1))*pMismatch;
        if (clustEm>=0.5f) {
            clustEm = 0.5f;
        }
        emProbs[1] = clustEm;
        emProbs[0] = 1f - clustEm;
        HmmUpdater.bwdUpdate(bwd[0], pRec, emProbs, mismatches[0][cP1], nStates);
        HmmUpdater.bwdUpdate(bwd[1], pRec, emProbs, mismatches[1][cP1], nStates);
        HmmUpdater.bwdUpdate(bwd[2], pRec, emProbs, mismatches[2][cP1], nStates);
    }

    private void fwdAlg(MarkerCluster mc) {
        int missIndex = 0;
        int unphHetIndex = 0;
        initializeFwdFields();
        IntArray unphClusters = mc.unphasedHetClusters();
        for (int c=0, n=mc.nClusters(); c<n; ++c) {
            if (mc.isUnphasedHet(c)) {
                phaseHet(mc.samplePhase(), unphHetIndex, c);
                ++unphHetIndex;
                if (swapHaps) {
                    int swapEnd = unphHetIndex < unphClusters.size()
                            ? unphClusters.get(unphHetIndex)
                            : mc.nClusters();
                        swapHaps(mc, c, swapEnd);
                }
                System.arraycopy(fwd[0], 0, fwd[1], 0, nStates);
                System.arraycopy(fwd[0], 0, fwd[2], 0, nStates);
                fwdSums[1] = fwdSums[2] = fwdSums[0];
            }
            fwdStep(mc, c);
            if (mc.isMissingGtOrMaskedHet(c)) {
                imputeAlleles(mc, c, missIndex++);
            }
        }
    }

    private void initializeFwdFields() {
        Arrays.fill(fwd[0], 0, nStates, 1.0f/nStates);
        System.arraycopy(fwd[0], 0, fwd[1], 0, nStates);
        System.arraycopy(fwd[0], 0, fwd[2], 0, nStates);
        fwdSums[2] = fwdSums[1] = fwdSums[0] = 1.0f;
    }

    private void fwdStep(MarkerCluster mc, int cluster) {
        float pRec = mc.pRecomb().get(cluster);
        float clustEm = (mc.clusterEnd(cluster) - mc.clusterStart(cluster))*pMismatch;
        if (clustEm>=0.5f) {
            clustEm = 0.5f;
        }
        emProbs[1] = clustEm;
        emProbs[0] = 1.0f - clustEm;
        fwdSums[0] = HmmUpdater.fwdUpdate(fwd[0], fwdSums[0], pRec, emProbs, mismatches[0][cluster], nStates);
        fwdSums[1] = HmmUpdater.fwdUpdate(fwd[1], fwdSums[1], pRec, emProbs, mismatches[1][cluster], nStates);
        fwdSums[2] = HmmUpdater.fwdUpdate(fwd[2], fwdSums[2], pRec, emProbs, mismatches[2][cluster], nStates);
    }

    private void swapHaps(MarkerCluster mc, int startClust, int endClust) {
        for (int c=startClust; c<endClust; ++c) {
            byte[] tmpMatch = mismatches[1][c];
            mismatches[1][c] = mismatches[2][c];
            mismatches[2][c] = tmpMatch;
        }
        mc.samplePhase().swapHaps(mc.clusterStart(startClust), mc.clusterEnd(endClust-1));
    }

    private void imputeAlleles(MarkerCluster mc, int cluster, int missIndex) {
        assert (mc.clusterEnd(cluster) - mc.clusterStart(cluster))==1;
        assert mc.isMissingGtOrMaskedHet(cluster);
        float[] stateProbs1 = bwdMiss1.get(missIndex);
        float[] stateProbs2 = bwdMiss2.get(missIndex);
        if (swapHaps) {
            float[] tmp = stateProbs1;
            stateProbs1 = stateProbs2;
            stateProbs2 = tmp;
        }
        int[] refAl = refAlleles.get(missIndex);
        for (int k=0; k<nStates; ++k) {
            stateProbs1[k] *= fwd[1][k];
            stateProbs2[k] *= fwd[2][k];
        }
        int marker = mc.clusterStart(cluster);
        int nAlleles = markers.marker(marker).nAlleles();
        float[] alFreq1 = new float[nAlleles];
        float[] alFreq2 = new float[nAlleles];
        for (int k=0; k<nStates; ++k) {
            alFreq1[refAl[k]] += stateProbs1[k];
            alFreq2[refAl[k]] += stateProbs2[k];
        }
        SamplePhase samplePhase = mc.samplePhase();
        ClustType clustType = samplePhase.clustType(cluster);
        if (clustType==ClustType.MISSING_GT) {
            imputeMissingGT(samplePhase, marker, alFreq1, alFreq2);
        }
        else if (clustType==ClustType.MASKED_HET) {
            imputeMaskedHet(samplePhase, cluster, marker, alFreq1, alFreq2);
        }
    }

    private void imputeMissingGT(SamplePhase samplePhase, int marker,
            float[] alFreq1, float[] alFreq2) {
        int a1 = 0;
        int a2 = 0;
        for (int j=1; j<alFreq1.length; ++j) {
            if (alFreq1[j]>alFreq1[a1]) {
                a1 = j;
            }
            if (alFreq2[j]>alFreq2[a2]) {
                a2 = j;
            }
        }
        samplePhase.setAllele1(marker, a1);
        samplePhase.setAllele2(marker, a2);
    }

    private void imputeMaskedHet(SamplePhase samplePhase, int cluster,
            int marker, float[] alFreq1, float[] alFreq2) {
        int a1 = samplePhase.allele1(marker);
        int a2 = samplePhase.allele2(marker);
        assert a1!=a2;
        float pNoSwitch = alFreq1[a1]*alFreq2[a2];
        float pSwitch = alFreq1[a2]*alFreq2[a1];
        if (pSwitch > pNoSwitch) {
            samplePhase.setAllele1(marker, a2);
            samplePhase.setAllele2(marker, a1);
            if (pSwitch>=(lrThreshold*pNoSwitch)) {
                samplePhase.markMaskedHetClusterAsPhased(cluster);
            }
        }
        else if (pNoSwitch>=(lrThreshold*pSwitch)) {
            samplePhase.markMaskedHetClusterAsPhased(cluster);
        }
    }

    private void phaseHet(SamplePhase samplePhase, int unphHetIndex, int cluster) {
        float[] b1 = bwdHet1.get(unphHetIndex);
        float[] b2 = bwdHet2.get(unphHetIndex);
        float p11 = 0.0f;
        float p12 = 0.0f;
        float p21 = 0.0f;
        float p22 = 0.0f;
        for (int k=0; k<nStates; ++k) {
            p11 += fwd[1][k]*b1[k];
            p12 += fwd[1][k]*b2[k];
            p21 += fwd[2][k]*b1[k];
            p22 += fwd[2][k]*b2[k];
        }
        float num = (p11*p22);
        float den = (p12*p21);
        boolean lastSwapHaps = swapHaps;
        swapHaps = num < den;
        if (swapHaps!=lastSwapHaps) {
            ++nSwaps;
        }
        if (burnin==false) {
            if (num>=den*lrThreshold || (swapHaps && den>=num*lrThreshold) ) {
                samplePhase.markUnphasedHetClusterAsPhased(cluster);
            }
        }
    }
}
