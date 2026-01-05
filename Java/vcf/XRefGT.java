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
package vcf;

import blbutil.BitArray;
import blbutil.Const;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import phase.SamplePhase;

/**
 * <p>Class {@code XRefGT} represents phased, non-missing genotypes for a list
 * of samples that are stored in column-major (i.e. haplotype-major) order.
 * </p>
 *
 * <p>Instances of class {@code XRefGT} are immutable.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class XRefGT implements GT {

    private final Samples samples;
    private final Markers markers;
    private final BitArray[] haps;

    private XRefGT(Markers markers, Samples samples, BitArray[] haps) {
        /* Callers of this private constructor must perform parameter checking
           and ensure that no references to {@code haps} escape */
        this.markers = markers;
        this.samples = samples;
        this.haps = haps;
    }

    /**
     * Returns a new {@code XRefGT} instance from the specified data.
     * The order of samples and haplotypes is preserved. Samples
     * in the first {@code XRefGT} parameter are placed before samples
     * in the second {@code XRefGT} parameter in the returned {@code XRefGT}
     * instance.
     * @param first phased genotype data for a list of samples
     * @param second phased genotype data for a list of samples
     * @return a new {@code XRefGT} instance
     * @throws NullPointerException if
     * {@code first == null || second == null}
     * @throws IllegalArgumentException if the lists of samples in the two
     * specified {@code XRefGT} parameters are not disjoint
     * @throws IllegalArgumentException if
     * {@code first.markers().equals(second.markers()) == false}
     */
    public static XRefGT combine(XRefGT first, XRefGT second) {
        if (first.markers().equals(second.markers())==false) {
            throw new IllegalArgumentException("inconsisent data");
        }
        Samples samples = Samples.combine(first.samples(), second.samples());
        Stream<BitArray> hapStream1 = Arrays.stream(first.haps);
        Stream<BitArray> hapStream2 = Arrays.stream(second.haps);
        BitArray[] haps = Stream.concat(hapStream1, hapStream2)
                .parallel()
                .toArray(BitArray[]::new);
        return new XRefGT(first.markers, samples, haps);
    }

    /**
     * Returns a new {@code XRefGT} instance from the specified data.
     *
     * @param samples the list of samples
     * @param phase the phased genotypes
     * @return a new {@code XRefGT} instance
     *
     * @throws IllegalArgumentException if
     * {@code phase.length()==0 || samples.size() != phase.length()}
     * @throws IllegalArgumentException if
     * if there exists {@code j} such that {@code (0 <= j && j < phase.length()
     * && phase.get(j).markers().equals(phase.get(j).markers())==false}
     * @throws NullPointerException if {@code samples == null || phase == null}
     * @throws NullPointerException if there exists {@code j} such that
     * {@code (0 <= j && j < phase.length() && phase.get(j) == null)}
     */
    public static XRefGT from(Samples samples,
            AtomicReferenceArray<SamplePhase> phase) {
        int nSamples = phase.length();
        if (nSamples==0 || samples.size()!=nSamples) {
            throw new IllegalArgumentException(String.valueOf(nSamples));
        }
        Markers markers = phase.get(0).markers();
        BitArray[] haps = IntStream.range(0, nSamples<<1)
                .parallel()
                .mapToObj(h -> {
                    SamplePhase sampPhase = phase.get(h>>1);
                    if (sampPhase.markers().equals(markers)==false) {
                        throw new IllegalArgumentException("inconsistent data");
                    }
                    return (h & 0b1)==0 ? sampPhase.hap1() : sampPhase.hap2();
                })
                .toArray(BitArray[]::new);
        return new XRefGT(markers, samples, haps);
    }

//    NB: The toBitLists() method is commented-out because it is
//        not currently used
//    ToDo: decide whether to delete toBitLists() method after
//          XRefGT amd BrefGT code stabilizes.
//
//    /**
//     * Returns the phased, non-missing genotypes as a {@code BitArray[]}.
//     * @param nThreads the maximum number of computational threads for object
//     * construction
//     * @return the phased, non-missing genotypes as a {@code BitArray[]}
//     * @throws IllegalArgumentException if {@code nThreads < 1}
//     */
//    public BitArray[] toBitLists(int nThreads) {
//        if (nThreads<1) {
//            throw new IllegalArgumentException(String.valueOf(nThreads));
//        }
//        int nRecsPerBatch = (markers.size() + nThreads - 1)/nThreads;
//        while (nRecsPerBatch>4096) {
//            nRecsPerBatch = (nRecsPerBatch+1) >> 1;
//        }
//        int stepSize = nRecsPerBatch;
//        int nSteps = (markers.size() + (stepSize-1)) / stepSize;
//        return IntStream.range(0, nSteps)
//                .parallel()
//                .boxed()
//                .flatMap(step -> bitLists(step, stepSize))
//                .toArray(BitArray[]::new);
//    }
//
//    private Stream<BitArray> bitLists(int step, int stepSize) {
//        int mStart = step*stepSize;
//        int mEnd = Math.min(mStart + stepSize, markers.size());
//        BitArray[] bitLists = IntStream.range(mStart, mEnd)
//                .mapToObj(j -> new BitArray(haps.length*markers.marker(j).bitsPerAllele()))
//                .toArray(BitArray[]::new);
//        int[] bitsPerAllele = IntStream.range(mStart, mEnd)
//                .map(m -> markers.marker(m).bitsPerAllele())
//                .toArray();
//        for (int h=0; h<haps.length; ++h) {
//            int inBit = markers.sumHapBits(mStart);
//            for (int m=mStart; m<mEnd; ++m) {
//                int mOffset = m - mStart;
//                int nBits = bitsPerAllele[mOffset];
//                int startOutBit = h*nBits;
//                for (int i=0; i<nBits; ++i) {
//                    if (haps[h].get(inBit++)) {
//                        bitLists[mOffset].set(startOutBit + i);
//                    }
//                }
//            }
//        }
//        return Arrays.stream(bitLists);
//    }

    /**
     * Returns a new {@code XRefGT} instance from the specified data. The
     * returned {@code XRefGT} instance will represent the same genotypes,
     * the same list of markers, and same list of samples as the specified
     * genotype data,
     *
     * @param gt phased, nonmissing genotype data
     * @param nThreads the maximum number of computational threads for object
     * construction
     * @return an {@code XRefGT} instance
     *
     * @throws IllegalArgumentException if {@code gt.isPhased() == false}
     * @throws IllegalArgumentException if {@code nThreads < 1}
     * @throws NullPointerException if {@code refGT == null}
     */
    public static XRefGT fromPhasedGT(GT gt, int nThreads) {
        if (nThreads<1) {
            throw new IllegalArgumentException(String.valueOf(nThreads));
        }
        BitArray[] haps = hapData(gt, nThreads);
        return new XRefGT(gt.markers(), gt.samples(), haps);
    }

    private static BitArray[] hapData(GT gt, int nThreads) {
        if (gt.isPhased()==false) {
            throw new IllegalArgumentException(String.valueOf(gt));
        }
        int nHapsPerBatch = (gt.nHaps() + nThreads - 1)/nThreads;
        while (nHapsPerBatch>4096) {
            nHapsPerBatch = (nHapsPerBatch+1) >> 1;
        }
        int stepSize = nHapsPerBatch;
        int nSteps = (gt.nHaps() + (stepSize-1)) / stepSize;
        return IntStream.range(0, nSteps)
                .parallel()
                .boxed()
                .flatMap(step -> hapData(gt, step, stepSize))
                .toArray(BitArray[]::new);
    }

    private static Stream<BitArray> hapData(GT phasedGT, int step, int stepSize) {
        int nMarkers = phasedGT.nMarkers();
        int nHapBits = phasedGT.markers().sumHapBits();
        int start = step*stepSize;
        int end = Math.min(start + stepSize, phasedGT.nHaps());
        BitArray[] haps = IntStream.range(0, end-start)
                .mapToObj(j -> new BitArray(nHapBits))
                .toArray(BitArray[]::new);

        for (int m=0; m<nMarkers; ++m) {
            setHapData(phasedGT, m, haps, start);
        }
        return Arrays.stream(haps);
    }

    private static void setHapData(GT phasedGT, int m, BitArray[] haps,
            int hapOffset) {
        assert phasedGT.isPhased();
        Markers markers = phasedGT.markers();
        int startBit = markers.sumHapBits(m);
        int endBit = markers.sumHapBits(m+1);
        for (int j=0; j<haps.length; ++j) {
            int allele = phasedGT.allele(m, hapOffset + j);
            long mask = 1;
            for (int i=startBit; i<endBit; ++i) {
                if ((allele & mask)==mask) {
                    assert i<haps[j].size();
                    haps[j].set(i);
                }
                mask <<= 1;
            }
        }
    }

    /**
     * Returns a hash code for the specified alleles.
     * @param hap a haplotype
     * @param start the first marker (inclusive)
     * @param end the last marker (exclusive)
     * @return a hash code for the specified alleles
     * @throws IndexOutOfBoundsException if
     * {@code hap < 0 || hap >= this.nHaps()}
     * @throws IndexOutOfBoundsException if
     * {@code start < 0 || start > to || end >= this.nMarkers()}
     */
    public int hash(int hap, int start, int end) {
        int startBit = markers.sumHapBits(start);
        int endBit = markers.sumHapBits(end);
        return haps[hap].hash(startBit, endBit);
    }

    @Override
    public boolean isReversed() {
        return false;
    }

    @Override
    public int nMarkers() {
        return markers.size();
    }

    @Override
    public Marker marker(int markerIndex) {
        return markers.marker(markerIndex);
    }

    @Override
    public Markers markers() {
        return markers;
    }

    @Override
    public int nHaps() {
        return haps.length;
    }

    @Override
    public int nSamples() {
        return samples.size();
    }

    @Override
    public Samples samples() {
        return samples;
    }

    @Override
    public boolean isPhased() {
        return true;
    }

    @Override
    public int allele(int marker, int hap) {
        return markers.allele(haps[hap], marker);
    }

    @Override
    public GT restrict(Markers markers, int[] indices) {
        return new RestrictedGT(this, markers, indices);
    }

    @Override
    public XRefGT restrict(int start, int end) {
        Markers restrictMarkers = markers.restrict(start, end);
        int startBit = markers.sumHapBits(start);
        int endBit = markers.sumHapBits(end);
        BitArray[] restrictHaps = IntStream.range(0, haps.length)
                .parallel()
                .mapToObj(h -> haps[h].restrict(startBit, endBit))
                .toArray(BitArray[]::new);
        return new XRefGT(restrictMarkers, samples, restrictHaps);
    }

    /**
     * Copies the specified bit sequence to the specified {@code bitList}
     * @param hap the haplotype index
     * @param start the start marker
     * @param end the end marker
     * @param bitList the destination {@code bitList}
     * @throws IllegalArgumentException if {@code start > end}
     * @throws IndexOutOfBoundsException if {@code hap < 0 || hap >= this.nHaps()}
     * @throws IndexOutOfBoundsException if
     * {@code start < 0 || end > this.markers()}
     * @throws IndexOutOfBoundsException if {@code end <= this.markers() &&
     * bitList.size() < this.markers().sumHaplotypeBits(end)}
     * @throws NullPointerException if {@code bitList == null}
     */
    public void copyTo(int hap, int start, int end, BitArray bitList) {
        int startBit = markers.sumHapBits(start);
        int endBit = markers.sumHapBits(end);
        bitList.copyFrom(haps[hap], startBit, endBit);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(this.getClass().toString());
        sb.append(" nMarkers=");
        sb.append(nMarkers());
        sb.append(" nSamples=");
        sb.append(nSamples());
        sb.append(Const.nl);
        int nMarkers = markers.size();
        int nSamples = samples.size();
        for (int m=0; m<nMarkers; ++m) {
            sb.append(markers.marker(m));
            sb.append(Const.nl);
            sb.append(Const.MISSING_DATA_CHAR);     // QUAL
            sb.append(Const.tab);
            sb.append("PASS");                      // FILTER
            sb.append(Const.tab);
            sb.append(Const.MISSING_DATA_CHAR);     // INFO
            sb.append(Const.tab);
            sb.append("GT");                        // FORMAT
            for (int s=0; s<nSamples; ++s) {
                int hap1 = s << 1;
                sb.append(Const.tab);
                sb.append(allele(m, hap1));
                sb.append(Const.phasedSep);
                sb.append(allele(m, hap1 | 0b1));
            }
        }
        sb.append(Const.nl);
        sb.append(']');
        return sb.toString();
    }
}
