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

import blbutil.BitArray;
import blbutil.DoubleArray;
import ints.IntArray;
import ints.IntList;
import java.util.Arrays;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import vcf.Markers;

/**
 * <p>Each instance of class {@code SamplePhase} stores an estimated haplotype
 * pair for a sample.
 * </p>
 * <p>Instances of class {@code SamplePhase} are not thread-safe.
 * </p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class SamplePhase {

    public static enum ClustType {
        MISSING_GT,
        MASKED_HET,
        HOMOZYGOUS_GT,
        PHASED_HET,
        UNPHASED_HET
    }

    private static final ClustType[] clustTypes = ClustType.values();

    private final int sample;
    private final Markers markers;
    private final BitArray hap1;
    private final BitArray hap2;
    private final byte[] clustSize;
    private final byte[] clustType;
    private final int[] clustTypeCnt = new int[clustTypes.length];


    /**
     * Constructs a new {@code SamplePhase} instance from the specified data.
     * @param sample the sample index
     * @param markers the list of markers
     * @param genPos the genetic positions of the specifed markers
     * @param hap1 the list of alleles on the first haplotype
     * @param hap2 the list of alleles on the second haplotype
     * @param unphasedHets the indices of markers whose genotype phase with respect
     * to the preceding heterozygote is unknown
     * @param missingGTs the indices of markers whose genotype is missingGT
     * @throws IllegalArgumentException if {@code sample < 0}
     * @throws IllegalArgumentException if
     * {@code genPos.size() != markers.nMarkers()}
     * @throws IllegalArgumentException if
     * {@code hap1.length != markers.nMarkers() || hap2.length != markers.nMarkers()}
     * @throws IllegalArgumentException if the specified {@code unphasedHet} or
     * {@code missingGT} list is not a strictly increasing list of
     * marker indices between 0 (inclusive) and {@code markers.nMarkers()}
     * (exclusive)
     * @throws NullPointerException if any argument is {@code null}
     */
    public SamplePhase(int sample, Markers markers, DoubleArray genPos,
            int[] hap1, int[] hap2, IntArray unphasedHets, IntArray missingGTs) {
        if (sample < 0) {
            throw new IllegalArgumentException(String.valueOf(sample));
        }
        this.sample = sample;
        int nMarkers = markers.size();
        if (nMarkers!=genPos.size()) {
            throw new IllegalArgumentException(String.valueOf(genPos.size()));
        }
        if (hap1.length!=nMarkers) {
            throw new IllegalArgumentException(String.valueOf(hap1.length));
        }
        if (hap2.length!=nMarkers) {
            throw new IllegalArgumentException(String.valueOf(hap2.length));
        }
        checkIncreasing(unphasedHets, nMarkers);
        checkIncreasing(missingGTs, nMarkers);
        this.markers = markers;
        this.hap1 = new BitArray(markers.sumHapBits());
        this.hap2 = new BitArray(markers.sumHapBits());
        markers.allelesToBits(hap1, this.hap1);
        markers.allelesToBits(hap2, this.hap2);
        float maxClusterCM = 0.005f;
        IntList clustTypeList = new IntList();
        IntList clustSizeList = new IntList();
        setClusters(hap1, hap2, missingGTs, unphasedHets, genPos, maxClusterCM,
                clustTypeList, clustTypeCnt, clustSizeList);
        this.clustType = toByteArray(clustTypeList);
        this.clustSize = toByteArray(clustSizeList);
        assert clustSize.length==clustType.length;
    }

    private static void checkIncreasing(IntArray ia, int nMarkers) {
        int last = -1;
        for (int j=0, n=ia.size(); j<n; ++j) {
            if (ia.get(j)<=last) {
                throw new IllegalArgumentException(ia.toString());
            }
            last = ia.get(j);
        }
        if (last>=nMarkers) {
            throw new IllegalArgumentException(ia.toString());
        }
    }

    private static void setClusters(int[] hap1, int[] hap2, IntArray missingGT,
            IntArray unphHets, DoubleArray genPos, float maxCM,
            IntList clustType, int[] clustTypeCnt, IntList clustSizeList)  {
        int nMarkers = genPos.size();
        double maxClustEnd = genPos.get(0) + maxCM;
        boolean prevIsMissingOrHet = false;
        int lastEnd = 0;
        int missIndex = 0;
        int unphIndex = 0;
        int nextMiss = missIndex<missingGT.size() ? missingGT.get(missIndex++) : -1;
        int nextUnph = unphIndex<unphHets.size() ? unphHets.get(unphIndex++) : -1;
        ClustType prevType = ClustType.HOMOZYGOUS_GT;
        for (int m=0; m<nMarkers; ++m) {
            int size = m - lastEnd;
            ClustType type = clustType(m==nextMiss, m==nextUnph, hap1[m], hap2[m]);
            if (type==ClustType.MISSING_GT) {
                nextMiss = missIndex<missingGT.size() ? missingGT.get(missIndex++) : -1;
            }
            else if (type==ClustType.UNPHASED_HET) {
                nextUnph = unphIndex<unphHets.size() ? unphHets.get(unphIndex++) : -1;
            }
            boolean isMissingOrHet = type==ClustType.MISSING_GT
                    || type==ClustType.UNPHASED_HET
                    || type==ClustType.PHASED_HET;
            if (isMissingOrHet || prevIsMissingOrHet
                    || genPos.get(m)>maxClustEnd || size==255) {
                if (m>0) {
                    clustType.add(prevType.ordinal());
                    ++clustTypeCnt[prevType.ordinal()];
                    clustSizeList.add(size);
                    maxClustEnd = genPos.get(m) + maxCM;
                    lastEnd = m;
                }
                prevType = type;
            }
            prevIsMissingOrHet = isMissingOrHet;
        }
        clustType.add(prevType.ordinal());
        ++clustTypeCnt[prevType.ordinal()];
        clustSizeList.add(nMarkers - lastEnd);
    }

    private static ClustType clustType(boolean isMissing, boolean isUnphased,
            int a1, int a2) {
        if (isMissing) {
            return ClustType.MISSING_GT;
        }
        else if (a1==a2) {
            return ClustType.HOMOZYGOUS_GT;
        }
        else if (isUnphased) {
            return ClustType.UNPHASED_HET;
        }
        else {
           return ClustType.PHASED_HET;
        }
    }

    private static byte[] toByteArray(IntList intList) {
        byte[] ba = new byte[intList.size()];
        for (int j=0; j<ba.length; ++j) {
            ba[j] = (byte) intList.get(j);
        }
        return ba;
    }

    /**
     * Returns the sample index.
     * @return the sample index
     */
    public int sample() {
        return sample;
    }

    /**
     * Masks the trailing unphased heterozygote or heterozygotes in any maximal
     * sequence of consecutive unphased heterozygotes if the maximal sequence
     * has size two or three and spans less than 3000 base pairs.
     */
    public void maskTrailingUnphasedHets() {
        int maxUnphHetClusters = 3;
        int maxMaskedBasePairs = 3000;
        IntList unphHetMarkers = new IntList();
        IntList unphHetClusters = new IntList();
        int startMarker = 0;
        for (int c=0; c<clustType.length; ++c) {
            ClustType ct = clustType(c);
            if (ct==ClustType.PHASED_HET) {
                if (2<=unphHetClusters.size() && unphHetClusters.size()<=maxUnphHetClusters) {
                    maskTrailingUnphasedHets(unphHetClusters, unphHetMarkers, maxMaskedBasePairs);
                }
                unphHetMarkers.clear();
                unphHetClusters.clear();
            }
            else if (ct==ClustType.UNPHASED_HET) {
                unphHetMarkers.add(startMarker);
                unphHetClusters.add(c);
            }
            startMarker += clustSize[c] & 0xff;
        }
        if (2<=unphHetClusters.size() && unphHetClusters.size()<=maxUnphHetClusters) {
            maskTrailingUnphasedHets(unphHetClusters, unphHetMarkers, maxMaskedBasePairs);
        }
        assert startMarker==markers.size();
    }

    private void maskTrailingUnphasedHets(IntList unphHetClusters,
            IntList unphHetMarkers, int maxMaskedBasePairs) {
        int lastMaskedIndex = unphHetClusters.size()-2;
        if (lastMaskedIndex==0) {
            maskHetCluster(unphHetClusters.get(lastMaskedIndex));
        }
        else if (lastMaskedIndex>0) {
            int startPos = markers.marker(unphHetMarkers.get(0)).pos();
            int endPos = markers.marker(unphHetMarkers.get(lastMaskedIndex)).pos();
            if ((endPos - startPos)<=maxMaskedBasePairs) {
                for (int j=0; j<=lastMaskedIndex; ++j) {
                    maskHetCluster(unphHetClusters.get(j));
                }
            }
        }
    }

    /**
     * Masks the unphased heterozygote genotype in the specified cluster.
     * @param cluster a cluster index
     * @throws IllegalArgumentException if
     * {@code this.clustType(cluster) != ClustType.UNPHASED_HET}
     * @throws IndexOutOfBoundsException if
     * {@code (cluster < 0 || cluster >= this.nClusters())}
     */
    public void maskHetCluster(int cluster) {
        if (clustType[cluster]!=ClustType.UNPHASED_HET.ordinal()) {
            throw new IllegalArgumentException(String.valueOf(clustType[cluster]));
        }
        clustType[cluster] = (byte) ClustType.MASKED_HET.ordinal();
        --clustTypeCnt[ClustType.UNPHASED_HET.ordinal()];
        ++clustTypeCnt[ClustType.MASKED_HET.ordinal()];
    }

    /**
     * Marks the specified unphased heterozygote genotype as phased.
     * @param cluster a cluster index
     * @throws IllegalArgumentException if
     * {@code this.clustType(cluster) != ClustType.UNPHASED_HET}
     * @throws IndexOutOfBoundsException if
     * {@code (cluster < 0 || cluster >= this.nClusters())}
     */
    public void markUnphasedHetClusterAsPhased(int cluster) {
        if (clustType[cluster]!=ClustType.UNPHASED_HET.ordinal()) {
            throw new IllegalArgumentException(String.valueOf(clustType[cluster]));
        }
        clustType[cluster] = (byte) ClustType.PHASED_HET.ordinal();
        --clustTypeCnt[ClustType.UNPHASED_HET.ordinal()];
        ++clustTypeCnt[ClustType.PHASED_HET.ordinal()];
    }

    /**
     * Marks the specified masked heterozygote genotype as phased.
     * @param cluster a cluster index
     * @throws IllegalArgumentException if
     * {@code this.clustType(cluster) != ClustType.MASKED_HET}
     * @throws IndexOutOfBoundsException if
     * {@code (cluster < 0 || cluster >= this.nClusters())}
     */
    public void markMaskedHetClusterAsPhased(int cluster) {
        if (clustType[cluster]!=ClustType.MASKED_HET.ordinal()) {
            throw new IllegalArgumentException(String.valueOf(clustType[cluster]));
        }
        clustType[cluster] = (byte) ClustType.PHASED_HET.ordinal();
        --clustTypeCnt[ClustType.MASKED_HET.ordinal()];
        ++clustTypeCnt[ClustType.PHASED_HET.ordinal()];
    }

    /**
     * Returns the (exclusive) end marker indices of each marker cluster.
     * The returned list is sorted in increasing order.
     * @return the (exclusive) end marker indices of each marker cluster
     */
    public int[] clustEnds() {
        int[] clustEnds = new int[clustSize.length];
        int cumSum = 0;
        for (int j=0; j<clustSize.length; ++j) {
            cumSum += (clustSize[j] & 0xff); // convert unsigned byte to integer
            clustEnds[j] = cumSum;
        }
        return clustEnds;
    }

    /**
     * Returns the list of markers.
     * @return the list of markers
     */
    public Markers markers() {
        return markers;
    }

    /**
     * Returns the number of genotype clusters
     * @return the number of genotype clusters
     */
    public int nClusters() {
        return clustSize.length;
    }

    /**
     * Returns the size of the specified cluster
     * @param cluster a cluster index
     * @return the size of the specified cluster
     * @throws IllegalArgumentException if
     * {@code (cluster < 0 || cluster >= this.nClusters())}
     */
    public int clustSize(int cluster) {
        return clustSize[cluster] & 0xff;
    }

    /**
     * Returns the cluster type
     * @param cluster a cluster index
     * @return the cluster type
     * @throws IndexOutOfBoundsException if
     * {@code (cluster < 0 || cluster >= this.clustEnds().length())}
     */
    public ClustType clustType(int cluster) {
        return clustTypes[clustType[cluster]];
    }

    /**
     * Returns the number of unphased, non-masked heterozygotes.
     * @return the number of unphased, non-masked heterozygotes
     */
    public int nUnphased() {
        return clustTypeCnt[ClustType.UNPHASED_HET.ordinal()];
    }

    /**
     * Returns the number of phased, non-masked heterozygotes.
     * @return the number of phased, non-masked heterozygotes
     */
    public int nPhased() {
        return clustTypeCnt[ClustType.PHASED_HET.ordinal()];
    }

    /**
     * Returns the number of masked heterozygotes.
     * @return the number of masked heterozygotes
     */
    public int nMasked() {
        return clustTypeCnt[ClustType.MASKED_HET.ordinal()];
    }

    /**
     * Returns the number of missing genotypes.
     * @return the number of missing genotypes
     */
    public int nMissing() {
        return clustTypeCnt[ClustType.MISSING_GT.ordinal()];
    }

    /**
     * Returns the number of homozygote clusters.
     * @return the number of homozygote clusters
     */
    public int nHomClusters() {
        return clustTypeCnt[ClustType.HOMOZYGOUS_GT.ordinal()];
    }

    /**
     * Copies the stored haplotypes to the specified {@code BitList} objects
     * @param hap1 a {@code BitList} in which the sample's first haplotype's
     * alleles will be  stored
     * @param hap2 a {@code BitList} in which the sample's second haplotype's
     * alleles will be  stored
     * @throws IllegalArgumentException if
     * {@code hap1.size() != this.markers().sumHaplotypeBits()}
     * @throws IllegalArgumentException if
     * {@code hap2.size()!= this.markers().sumHaplotypeBits()}
     * @throws NullPointerException if {@code hap1 == null || hap2 == null}
     */
    public void getHaps(BitArray hap1, BitArray hap2) {
        int nBits = markers.sumHapBits();
        if (hap1.size() != nBits || hap2.size() != nBits) {
            throw new IllegalArgumentException("inconsistent data");
        }
        hap1.copyFrom(this.hap1, 0, this.hap1.size());
        hap2.copyFrom(this.hap2, 0, this.hap2.size());
    }


    /**
     * Returns the allele on the first haplotype for the specified marker.
     * @param marker the marker index
     * @return the allele on the first haplotype for the specified marker
     * @throws IndexOutOfBoundsException if
     * {@code marker < 0 || marker >= this.markers().nMarkers()}
     */
    public int allele1(int marker) {
       return markers.allele(hap1, marker);
    }

    /**
     * Returns the allele on the second haplotype for the specified marker.
     * @param marker the marker index
     * @return the allele on the second haplotype for the specified marker
     * @throws IndexOutOfBoundsException if
     * {@code marker < 0 || marker >= this.markers().nMarkers()}
     */
    public int allele2(int marker) {
        return markers.allele(hap2, marker);
    }

    /**
     * Sets the allele on the first haplotype for the specified marker
     * to the specified allele
     * @param marker the marker index
     * @param allele the allele
     * @throws IndexOutOfBoundsException if
     * {@code marker < 0 || marker >= this.markers().nMarkers()}
     * @throws IndexOutOfBoundsException if
     * {@code allele < 0 || allele >= this.markers().marker(marker).nAlleles()}
     */
    public void setAllele1(int marker, int allele) {
        markers.setAllele(marker, allele, hap1);
    }

    /**
     * Sets the allele on the second haplotype for the specified marker
     * to the specified allele
     * @param marker the marker index
     * @param allele the allele
     * @throws IndexOutOfBoundsException if
     * {@code marker < 0 || marker >= this.markers().nMarkers()}
     * @throws IndexOutOfBoundsException if
     * {@code allele < 0 || allele >= this.markers().marker(marker).nAlleles()}
     */
    public void setAllele2(int marker, int allele) {
        markers.setAllele(marker, allele, hap2);
    }

    /**
     * Swaps the alleles of the two haplotypes in the specified range of
     * markers.
     * @param start the start marker index (inclusive)
     * @param end the end marker index (exclusive)
     * @throws IndexOutOfBoundsException if
     * {@code start < 0 || start > end || start >= this.markers().nMarkers()}
     */
    public void swapHaps(int start, int end) {
        int startBit = markers.sumHapBits(start);
        int endBit = markers.sumHapBits(end);
        BitArray.swapBits(hap1, hap2, startBit, endBit);
    }

    /**
     * Returns the first haplotype.  The haplotype is encoded with the
     * {@code this.markers().allelesToBits()} method.
     * @return the first haplotype
     */
    public BitArray hap1() {
        return new BitArray(this.hap1);
    }

    /**
     * Returns the second haplotype.  The haplotype is encoded with the
     * {@code this.markers().allelesToBits()} method.
     * @return the second haplotype
     */
    public BitArray hap2() {
        return new BitArray(this.hap2);
    }

    /**
     * Returns the current estimated phased genotypes.  This method converts
     * column-major data into row-major data.
     * @param estPhase the current estimated phased genotypes for each target
     * sample
     * @return the current estimated phased genotypes
     * @throws NullPointerException if {@code estPhase == null}
     */
    public static BitArray[] toBitLists(EstPhase estPhase) {
        int nThreads = estPhase.fpd().par().nthreads();
        int nMarkers = estPhase.fpd().stage1TargGT().nMarkers();
        int nRecsPerBatch = (nMarkers + nThreads - 1)/nThreads;
        while (nRecsPerBatch>4096) {
            nRecsPerBatch = (nRecsPerBatch+1) >> 1;
        }
        int stepSize = nRecsPerBatch;
        int nSteps = (nMarkers + (stepSize-1)) / stepSize;
        return IntStream.range(0, nSteps)
                .parallel()
                .boxed()
                .flatMap(step -> bitLists(estPhase, step, stepSize))
                .toArray(BitArray[]::new);
    }

    private static Stream<BitArray> bitLists(EstPhase estPhase, int step, int stepSize) {
        int nSamples = estPhase.fpd().targGT().nSamples();
        int nHaps = nSamples<<1;
        Markers markers = estPhase.fpd().stage1TargGT().markers();
        int mStart = step*stepSize;
        int mEnd = Math.min(mStart + stepSize, markers.size());
        BitArray[] bitLists = IntStream.range(mStart, mEnd)
                .mapToObj(j -> new BitArray(nHaps*markers.marker(j).bitsPerAllele()))
                .toArray(BitArray[]::new);
        int[] bitsPerAllele = IntStream.range(mStart, mEnd)
                .map(m -> markers.marker(m).bitsPerAllele())
                .toArray();
        for (int s=0; s<nSamples; ++s) {
            SamplePhase sampPhase = estPhase.get(s);
            int h1 = s<<1;
            int h2 = h1 | 0b1;
            int inBit1 = markers.sumHapBits(mStart);
            int inBit2 = markers.sumHapBits(mStart);
            for (int m=mStart; m<mEnd; ++m) {
                int mOffset = m - mStart;
                int nBits = bitsPerAllele[mOffset];
                int startOutBit1 = h1*nBits;
                int startOutBit2 = h2*nBits;
                for (int i=0; i<nBits; ++i) {
                    if (sampPhase.hap1.get(inBit1++)) {
                        bitLists[mOffset].set(startOutBit1 + i);
                    }
                    if (sampPhase.hap2.get(inBit2++)) {
                        bitLists[mOffset].set(startOutBit2 + i);
                    }
                }
            }
        }
        return Arrays.stream(bitLists);
    }
}
