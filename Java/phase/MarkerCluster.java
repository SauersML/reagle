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
import ints.IntArray;
import ints.WrappedIntArray;
import phase.SamplePhase.ClustType;

/**
 * <p>Class {@code MarkerCluster} represents a partition of markers into
 * contiguous marker clusters.</p>
 *
 * <p>Instances of class {@code MarkerCluster} are immutable.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class MarkerCluster {

    private final SamplePhase samplePhase;
    private final int[] clusterToEnd;
    private final IntArray unphHetClusters;
    private final FloatArray pRecomb;

    /**
     * Constructs a new {@code MarkerCluster} instance from the specified data.
     *
     * @param phaseData the input data for the next genotype phasing iteration
     * @param sample a sample index
     * @throws IndexOutOfBoundsException if
     * {@code sample < 0 || sample >= phaseData.targGT().nSamples()}
     * @throws NullPointerException if {@code phaseData == null}
     */
    public MarkerCluster(PhaseData phaseData, int sample) {
        this.samplePhase = phaseData.estPhase().get(sample);
        this.clusterToEnd = samplePhase.clustEnds();
        this.unphHetClusters = unphHetClusters(samplePhase);
        this.pRecomb = pClustRecomb(phaseData.pRecomb(), clusterToEnd);
    }

    private static IntArray unphHetClusters(SamplePhase samplePhase) {
        int nUnph = samplePhase.nUnphased();
        int nClusters = samplePhase.nClusters();
        int[] unphHetClusters = new int[nUnph];
        int index = 0;
        for (int c=0; c<nClusters; ++c) {
            if (samplePhase.clustType(c)==ClustType.UNPHASED_HET) {
                unphHetClusters[index++] = c;
            }
        }
        return new WrappedIntArray(unphHetClusters);
    }

    private static FloatArray pClustRecomb(FloatArray pRecomb,
            int[] cluster2End) {
        int nClusters = cluster2End.length;
        float[] pClustRecomb = new float[nClusters];
        int start = cluster2End[0];
        for (int j=1; j<nClusters; ++j) {
            int end = cluster2End[j];
            float pNoRecomb = 1.0f;
            for (int k=start; k<end; ++k) {
                pNoRecomb *= (1.0f - pRecomb.get(k));
            }
            pClustRecomb[j] = 1.0f - pNoRecomb;
            start = end;
        }
        return new FloatArray(pClustRecomb);
    }

    /**
     * Return the estimated haplotypes.
     * @return the estimated haplotypes
     */
    public SamplePhase samplePhase() {
        return samplePhase;
    }

    /**
     * Returns the number of clusters
     * @return the number of clusters
     */
    public int nClusters() {
        return clusterToEnd.length;
    }

    /**
     * Returns the inclusive start marker for the cluster.
     * @param index a cluster index
     * @return the inclusive start marker for the cluster
     * @throws IndexOutOfBoundsException if
     * {@code index < 0 || index >= this.nClusteres()}
     */
    public int clusterStart(int index) {
        return index==0 ? 0 : clusterToEnd[index-1];
    }

    /**
     * Returns the exclusive end marker for the cluster.
     * @param index a cluster index
     * @return the exclusive marker for the cluster
     * @throws IndexOutOfBoundsException if
     * {@code index < 0 || index >= this.nClusteres()}
     */
    public int clusterEnd(int index) {
        return clusterToEnd[index];
    }

    /**
     * Return a {@code FloatArray} of size {@code this.nClusters()}
     * whose {@code k}-th element is the probability of transitioning to a
     * random HMM state between the {@code k}-th cluster and the
     * previous cluster.
     * @return a {@code FloatArray} of size {@code this.nClusters()}
     * whose {@code k}-th element is the probability of transitioning to a
     * random HMM state between the {@code k}-th cluster and the
     * previous cluster
     */
    public FloatArray pRecomb() {
        return pRecomb;
    }

    /**
     * Returns a sorted list of cluster indices in increasing order for which
     * the cluster contains an unphased heterozygote.
     * @return a sorted list of cluster indices in increasing order for which
     * the cluster contains an unphased heterozygote
     */
    public IntArray unphasedHetClusters() {
        return unphHetClusters;
    }

    /**
     * Returns {@code true} if the cluster has an unphased heterozygous genotype,
     * and returns {@code false} otherwise.
     * @param cluster a cluster index
     * @return {@code true} if the cluster has an unphased heterozygous genotype,
     * and returns {@code false} otherwise
     * @throws IndexOutOfBoundsException if
     * {@code cluster < 0 || cluster >= this.nClusters()}
     */
    public boolean isUnphasedHet(int cluster) {
        return samplePhase.clustType(cluster)==ClustType.UNPHASED_HET;
    }

    /**
     * Returns {@code true} if the cluster has a phased heterozygous genotype,
     * and returns {@code false} otherwise.
     * @param cluster a cluster index
     * @return {@code true} if the cluster has a phased heterozygous genotype,
     * and returns {@code false} otherwise
     * @throws IndexOutOfBoundsException if
     * {@code cluster < 0 || cluster >= this.nClusters()}
     */
    public boolean isPhasedHet(int cluster) {
        return samplePhase.clustType(cluster)==ClustType.PHASED_HET;
    }

    /**
     * Returns {@code true} if the cluster is a heterozygous genotype,
     * and returns {@code false} otherwise.
     * @param cluster a cluster index
     * @return {@code true} if the cluster is a heterozygous genotype,
     * and {@code false} otherwise
     * @throws IndexOutOfBoundsException if
     * {@code cluster < 0 || cluster >= this.nClusters()}
     */
    public boolean isHet(int cluster) {
        ClustType clustType = samplePhase.clustType(cluster);
        return clustType==ClustType.UNPHASED_HET
                || clustType==ClustType.PHASED_HET;
    }

        /**
     * Returns {@code true} if the cluster has a missing genotype, and
     * returns {@code false} otherwise.
     * @param cluster a cluster index
     * @return {@code true} if the cluster has a missing genotype
     * @throws IndexOutOfBoundsException if
     * {@code cluster < 0 || cluster >= this.nClusters()}
     */
    public boolean isMissingGT(int cluster) {
        return samplePhase.clustType(cluster) == ClustType.MISSING_GT;
    }

    /**
     * Returns {@code true} if the cluster is a masked heterozygote genotype,
     * and returns {@code false} otherwise.
     * @param cluster a cluster index
     * @return {@code true} if the cluster is a masked heterozygote genotype
     * @throws IndexOutOfBoundsException ift
     * {@code cluster < 0 || cluster >= this.nClusters()}
     */
    public boolean isMaskedHet(int cluster) {
        return samplePhase.clustType(cluster) == ClustType.MASKED_HET;
    }

    /**
     * Returns {@code true} if the cluster has a missing genotype or a
     * masked heterozygote genotype, and returns {@code false} otherwise.
     * @param cluster a cluster index
     * @return {@code true} if the cluster has a missing genotype or a
     * masked heterozygote genotype, and returns {@code false} otherwise
     * @throws IndexOutOfBoundsException if
     * {@code cluster < 0 || cluster >= this.nClusters()}
     */
    public boolean isMissingGtOrMaskedHet(int cluster) {
        ClustType clustType = samplePhase.clustType(cluster);
        return clustType == ClustType.MISSING_GT
                || clustType==ClustType.MASKED_HET;
    }
}
