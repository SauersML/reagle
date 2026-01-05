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
import blbutil.DoubleArray;
import blbutil.FloatArray;
import blbutil.Utilities;
import ints.IntArray;
import ints.WrappedIntArray;
import java.util.Arrays;
import java.util.Optional;
import java.util.Random;
import java.util.stream.IntStream;
import main.Par;
import main.Pedigree;
import vcf.GT;
import vcf.GeneticMap;
import vcf.MarkerMap;
import vcf.Markers;
import vcf.RefGT;
import vcf.SplicedGT;
import vcf.Window;
import vcf.XRefGT;

/**
 * <p>Class {@code FixedPhaseData} stores immutable data for a
 * marker window.  The definition of low-frequency markers is
 * determined by the {@code Par.rare()} method and
 * {@code FixedPhaseData.MAX_HIFREQ_PROP} field.</p>
 *
 * <p>Instances of class {@code FixedPhaseData} are immutable.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class FixedPhaseData {

    private static final float MAX_HIFREQ_PROP = 0.75f;

    private final Par par;
    private final Pedigree ped;
    private final int window;

    private final MarkerMap map;
    private final Steps stage1Steps;
    private final GT targGT;
    private final Optional<RefGT> restrictedRefGT;
    private final int overlap;

    private final MarkerMap stage1Map;
    private final float ibsStep;
    private final GT stage1TargGT;
    private final Optional<RefGT> stage1RefGT;
    private final Optional<XRefGT> stage1XRefGT;
    private final FloatArray stage1Maf;
    private final int stage1Overlap;
    private final Ibs2 stage1Ibs2;

    private final int nHaps;
    private final IntArray[][] carriers;

    private final IntArray stage1To2;
    private final int[] prevStage1Marker;
    private final float[] prevStage1Wt;   // interpolation weight

    /**
     * Constructs a new {@code FixedPhaseData} instance from the
     * specified data.
     *
     * @param par the analysis parameters
     * @param ped the pedigree data for the target samples
     * @param window input data for the next marker window
     * @param phasedOverlap initial phased target genotypes due to
     * overlap with the previous window or {@code null} if there are
     * no initial phased target genotypes
     *
     * @throws IllegalArgumentException if
     * {@code (phasedOverlap != null && phasedOverlap.isPhased() == false)}
     * @throws IllegalArgumentException if
     * {@code (phasedOverlap != null
     * && data.targGT().samples().equals(phasedOverlap.samples()) == false)}
     * @throws IllegalArgumentException if
     * {@code (phasedOverlap != null
     * && data.targGT().nMarkers() < phasedOverlap.nMarkers())}
     * @throws IllegalArgumentException if
     * {@code (phasedOverlap != null &&
     * phasedOverlap.marker(j).equals(data.targGT().marker(j) == false)}
     * for some {@code j} satisfying
     * {@code (0 <= j && j <= overlapHaps.nMarkers())}
     * @throws NullPointerException if
     * {@code (par == null || genMap == null || data == null)}
     */
    public FixedPhaseData(Par par, Pedigree ped, Window window, GT phasedOverlap) {
        checkData(window, phasedOverlap);
        int nTargMarkers = window.targGT().nMarkers();
        this.par = par;
        this.ped = ped;
        this.window = window.windowIndex();

        this.map = markerMap(window.genMap(), window.targGT().markers());
        this.targGT = phasedOverlap==null ? window.targGT() :
                new SplicedGT(phasedOverlap, window.targGT());
        this.restrictedRefGT = window.restrictRefGT();
        this.overlap = phasedOverlap==null ? 0 : phasedOverlap.nMarkers();
        this.nHaps = nHaps(window);

        IntArray[][] rareCarriers = carriers(par, window);
        int[] hiFreqInd = hiFreqIndices(rareCarriers);
        if (hiFreqInd.length<2
                || hiFreqInd.length > MAX_HIFREQ_PROP*nTargMarkers) {
            hiFreqInd = IntStream.range(0, nTargMarkers).toArray();
            ignoreLowFreqCarriers(rareCarriers);
            this.carriers = rareCarriers;
            this.stage1Map = map;
            this.ibsStep = par.step_scale()*medianDiff(stage1Map.genPos());
            this.stage1Steps = new Steps(stage1Map, ibsStep);
            this.stage1TargGT = targGT;
            this.stage1RefGT = restrictedRefGT;
            this.stage1XRefGT = stage1RefGT.isPresent()
                    ? Optional.of(XRefGT.fromPhasedGT(stage1RefGT.get(), par.nthreads()))
                    : Optional.empty();
            this.stage1Overlap = overlap;
            this.stage1To2 = new WrappedIntArray(hiFreqInd);
            this.prevStage1Marker = IntStream.range(0, targGT.nMarkers())
                    .parallel()
                    .toArray();
            float[] fa = new float[targGT.nMarkers()];
            Arrays.fill(fa, 1.0f);
            this.prevStage1Wt = fa;
        }
        else {
            Markers hiFreqMarkers = targGT.markers().restrict(hiFreqInd);
            this.carriers = rareCarriers;
            this.stage1Map = map.restrict(hiFreqInd);
            this.ibsStep = par.step_scale()*medianDiff(stage1Map.genPos());
            this.stage1Steps = new Steps(stage1Map, ibsStep);
            this.stage1TargGT = targGT.restrict(hiFreqMarkers, hiFreqInd);
            this.stage1RefGT = restrict(restrictedRefGT, hiFreqMarkers, hiFreqInd);
            this.stage1XRefGT = stage1RefGT.isPresent()
                    ? Optional.of(XRefGT.fromPhasedGT(stage1RefGT.get(), par.nthreads()))
                    : Optional.empty();
            this.stage1Overlap = stage1TargOverlap(phasedOverlap, hiFreqInd);
            this.stage1To2 = new WrappedIntArray(hiFreqInd);
            this.prevStage1Marker = prevStage1Marker(targGT.nMarkers(), stage1To2);
            this.prevStage1Wt = prevWt(map, stage1To2);
        }
        int maxMafHaps = 10000;
        this.stage1Maf = maf(stage1RefGT, stage1TargGT, maxMafHaps, par.seed());
        this.stage1Ibs2 = new Ibs2(stage1TargGT, stage1Map, stage1Maf);
    }

    private static MarkerMap markerMap(GeneticMap genMap, Markers markers) {
        double meanGenDiff = MarkerMap.meanSingleBaseGenDist(genMap, markers);
        return MarkerMap.create(genMap, meanGenDiff, markers);
    }

    private static Optional<RefGT> restrict(Optional<RefGT> refGT,
            Markers hiFreqMarkers, int[] hiFreqIndices) {
        if(refGT.isPresent()) {
            return Optional.of(refGT.get().restrict(hiFreqMarkers, hiFreqIndices));
        }
        else {
            return refGT;
        }
    }

    private static void ignoreLowFreqCarriers(IntArray[][] carriers) {
        for (int j=0; j<carriers.length; ++j) {
            Arrays.fill(carriers[j], Window.HIGH_FREQ_ARRAY);
        }
    }

    private static float medianDiff(DoubleArray da) {
        double[] diffs = IntStream.range(1, da.size())
                .parallel()
                .mapToDouble(j -> (da.get(j) - da.get(j-1)))
                .sorted()
                .toArray();
        int n = diffs.length;
        return (0.5f) * (float) (diffs[(n-1)>>1] + diffs[n>>1]);
    }

    private static void checkData(Window window, GT phasedOverlap) {
        if (phasedOverlap!=null) {
            GT targ = window.targGT();
            if (phasedOverlap.isPhased()==false) {
                throw new IllegalArgumentException("unphased");
            }
            if (targ.samples().equals(phasedOverlap.samples())==false) {
                throw new IllegalArgumentException("inconsistent data");
            }
            if (phasedOverlap.nMarkers() > targ.nMarkers()) {
                throw new IllegalArgumentException("inconsistent data");
            }
            for (int j=0, n=phasedOverlap.nMarkers(); j<n; ++j) {
                if (phasedOverlap.marker(j).equals(targ.marker(j))==false) {
                    throw new IllegalArgumentException("inconsistent data");
                }
            }
        }
    }

    private static int nHaps(Window window) {
        Optional<RefGT> refGT = window.refGT();
        int nRefHaps = refGT.isPresent() ? refGT.get().nHaps() : 0;
        return window.targGT().nHaps() + nRefHaps;
    }

    private static IntArray[][] carriers(Par par, Window window) {
        Optional<RefGT> refGT = window.refGT();
        int nRefSamples = refGT.isPresent() ? refGT.get().nSamples() : 0;
        int nSamples = window.targGT().nSamples() + nRefSamples;
        int maxCarriers = Math.max(3, (int) Math.floor(nSamples*par.rare()));
        return window.carriers(maxCarriers);
    }

    private static int[] hiFreqIndices(IntArray[][] carriers) {
        return IntStream.range(0, carriers.length)
                .parallel()
                .filter(m -> 1 < Arrays.stream(carriers[m])
                                    .filter(ia -> ia==Window.HIGH_FREQ_ARRAY)
                                    .count())
                .toArray();
    }

    private static FloatArray maf(Optional<RefGT> optRefGT, GT targGT,
            int maxHaps, long seed) {
        Random rand = new Random(seed);
        int[] targHaps = randHaps(targGT, maxHaps, rand);
        int[] refHaps;
        if (targHaps.length<maxHaps && optRefGT.isPresent()) {
            int maxRefHaps = maxHaps - targHaps.length;
            refHaps = randHaps(optRefGT.get(), maxRefHaps, rand);
        }
        else {
            refHaps = new int[0];
        }
        double[] maf = IntStream.range(0, targGT.nMarkers())
                .parallel()
                .mapToDouble(m -> maf(targGT, optRefGT, targHaps, refHaps, m))
                .toArray();
        return new FloatArray(maf);
    }

    private static int[] randHaps(GT gt, int maxHaps, Random rand) {
        int nHaps = gt.nHaps();
        int[] ia = IntStream.range(0, nHaps)
                    .parallel()
                    .toArray();
        if (nHaps>maxHaps) {
            Utilities.shuffle(ia, maxHaps, rand);
            ia = Arrays.copyOf(ia, maxHaps);
            Arrays.sort(ia);
        }
        return ia;
    }

    private static double maf(GT gt, Optional<RefGT> optRefGT, int[] targHaps,
            int[] refHaps, int m) {
        int[] modCnts = new int[gt.marker(m).nAlleles()+1];
        for (int h : targHaps) {
            ++modCnts[gt.allele(m, h) + 1];
        }
        if (optRefGT.isPresent() && refHaps.length>0) {
            RefGT refGT = optRefGT.get();
            for (int h : refHaps) {
                ++modCnts[refGT.allele(m, h) + 1];
            }
        }
        modCnts[0] = 0; // zero-out missing count;
        Arrays.sort(modCnts);
        int den = 0;
        for (int j=1; j<modCnts.length; ++j) {
            den += modCnts[j];
        }
        return den==0 ? 0.0 : (double) modCnts[modCnts.length-2]/den;
    }

    private static int stage1TargOverlap(GT phasedOverlap, int[] hiFreqMkrs) {
        if (phasedOverlap==null) {
            return 0;
        }
        int insPt = Arrays.binarySearch(hiFreqMkrs, phasedOverlap.nMarkers());
        return (insPt<0) ? (-insPt - 1) : insPt;
    }

    private static int[] prevStage1Marker(int nMarkers, IntArray stage1Indices) {
        int[] mkrA = new int[nMarkers];
        int nHiFreq = stage1Indices.size();
        int start = stage1Indices.get(1);
        for (int j=2; j<nHiFreq; ++j) {
            int end = stage1Indices.get(j);
            Arrays.fill(mkrA, start, end, j-1);
            start = end;
        }
        Arrays.fill(mkrA, start, nMarkers, nHiFreq-1);
        return mkrA;
    }

    private static float[] prevWt(MarkerMap map, IntArray markerIndices) {
        DoubleArray genPos = map.genPos();
        float[] prevWt = new float[genPos.size()];
        Arrays.fill(prevWt, 0, markerIndices.get(0), 1.0f);
        int start = markerIndices.get(0);
        for (int j=1, n=markerIndices.size(); j<n; ++j) {
            int end = markerIndices.get(j);
            double posA = genPos.get(start);
            double posB = genPos.get(end);
            double d = posB - posA;
            prevWt[start] = 1.0f;
            for (int m=start+1; m<end; ++m) {
                prevWt[m] = (float) ((posB - genPos.get(m))/d);
            }
            start = end;
        }
        Arrays.fill(prevWt, start, genPos.size(), 1.0f);
        return prevWt;
    }

    /**
     * Return the analysis parameters.
     * @return the analysis parameters
     */
    public Par par() {
        return par;
    }

    /**
     * Returns the index of the marker window.
     * @return the index of the marker window
     */
    public int window() {
        return window;
    }

    /**
     * Returns the parent-offspring relationships.
     * @return the parent-offspring relationships
     */
    public Pedigree ped() {
        return ped;
    }

    /**
     * Returns the genetic map for the markers.
     * @return the genetic map for the markers
     */
    public MarkerMap map() {
        return map;
    }

    /**
     * Returns the optional phased, nonmissing reference genotypes.
     * @return the optional phased, nonmissing reference genotypes
     */
    public Optional<RefGT> restrictedRefGT() {
        return restrictedRefGT;
    }

    /**
     * Returns the input target genotypes. The returned allele data is
     * stored in marker-major order.
     * @return the input target genotypes
     */
    public GT targGT() {
        return targGT;
    }

    /**
     * Returns the number of initial markers that have phased target
     * genotypes due to overlap with the previous marker window.
     * @return the number of initial markers that have phased target
     * genotypes due to overlap with the previous marker window
     */
    public int overlap() {
        return overlap;
    }

    /**
     * Returns the genetic map for the stage1 markers.
     * @return the genetic map for the stage1 markers
     */
    public MarkerMap stage1Map() {
        return stage1Map;
    }

    /**
     * Returns the minimum cM step length for composite reference
     * haplotype construction.
     * @return the minimum cM step length for composite reference
     * haplotype construction
     */
    public float ibsStep() {
        return ibsStep;
    }

    /**
     * Returns a partition of the stage1 markers into a sequence of sets
     * of consecutive markers (the steps).
     * @return a partition of the stage1 markers into a sequence of sets
     * of consecutive markers (the steps)
     */
    public Steps stage1Steps() {
        return stage1Steps;
    }

    /**
     * Returns the optional phased, nonmissing reference genotypes for the
     * stage1 markers.  The returned allele data is stored in marker-major
     * order.
     * @return the optional phased, nonmissing reference genotypes for the
     * stage1 markers
     */
    public Optional<RefGT> stage1RefGT() {
        return stage1RefGT;
    }

    /**
     * Returns the optional phased, nonmissing reference genotypes for the
     * stage1 markers. The returned allele data is stored in haplotype-major
     * order.
     * @return the optional phased, nonmissing reference genotypes for the
     * stage1 markers
     */
    public Optional<XRefGT> stage1XRefGT() {
        return stage1XRefGT;
    }

    /**
     * Returns the input target genotypes at the stage1 markers.
     * The returned allele data is stored in marker-major order.
     * @return the input target genotypes at the stage1 markers
     */
    public GT stage1TargGT() {
        return stage1TargGT;
    }

    /**
     * Returns a list whose {@code j}-th element is the estimated
     * minor allele frequency of the {@code j}-th the stage1 marker.
     * @return the estimated stage1 minor allele frequencies
     */
    public FloatArray stage1Maf() {
        return stage1Maf;
    }

    /**
     * Returns the number of stage1 markers that have phased target genotypes
     * due to overlap with the previous window.
     * @return the number of stage1 markers that have phased target genotypes
     * due to overlap with the previous window
     */
    public int stage1Overlap() {
        return stage1Overlap;
    }

    /**
     * Return the sum of the number of reference and target haplotypes.
     * @return the sum of the number of reference and target haplotypes
     */
    public int nHaps() {
        return nHaps;
    }

    /**
     * Returns a map from stage1 marker index to marker index
     * @return a map from stage1 marker index to marker index
     */
    public IntArray stage1To2() {
        return stage1To2;
    }

    /**
     * Returns the IBS2 data for the stage1 markers.
     * @return the IBS2 data for the stage1 markers
     */
    public Ibs2 stage1Ibs2() {
        return stage1Ibs2;
    }

    /**
     * Returns the indices of the reference and target samples for the
     * specified low-frequency allele.  The reference sample indices will be
     * shifted by the number of target samples. so that the first reference
     * sample will have an index equal to the number of target samples.
     * The returned list will be sorted in order of increasing sample index.
     * The returned array will be empty and equal to
     * {@code vcf.Data.ZERO_FREQ_ARRAY} if the allele has no carriers, and the
     * returned array will be empty and equal to
     * {@code vcf.Data.HIGH_FREQ_ARRAY} if the allele is not a low-frequency
     * allele.
     * @param marker a marker index
     * @param allele an allele index for the specified marker
     * @return the indices of the reference and target samples that the
     * specified low-frequency allele
     * @throws IndexOutOfBoundsException if
     * {@code marker < 0 || marker >= this.targGT().nMarkers()}
     * @throws IndexOutOfBoundsException if
     * {@code allele < 0 || allele >= this.targGT().marker(marker).nAlleles()}
     */
    public IntArray carriers(int marker, int allele) {
        return carriers[marker][allele];
    }

    /**
     * Returns {@code true} if the specified allele is a low-frequency allele,
     * and returns {@code false} otherwise.
     * @param marker a marker index
     * @param allele an allele index for the specified marker
     * @return {@code true} if the specified allele is a low-frequency allele
     * @throws IndexOutOfBoundsException if
     * {@code marker < 0 || marker >= this.targGT().nMarkers()}
     * @throws IndexOutOfBoundsException if
     * {@code allele < 0 || allele >= this.targGT().marker(marker).nAlleles()}
     */
    public boolean isLowFreq(int marker, int allele) {
        return carriers[marker][allele]!=Window.HIGH_FREQ_ARRAY;
    }

    /**
     * Returns the index of the closest stage1 marker (in the list of
     * stage1 markers) with position less than or equal to the
     * position of the specified marker, or 0 if no such stage1 marker exists.
     * @param marker a marker index
     * @return the index of the closest preceding stage1 marker
     * @throws IndexOutOfBoundsException if
     * {@code marker < 0 || marker >= this.targGT().nMarkers()}
     */
    public int prevStage1Marker(int marker) {
        return prevStage1Marker[marker];
    }

    /**
     * Returns the linear interpolation weight associated with the
     * preceding stage1 marker (see {@code this.prevStage1Marker(marker)}).
     * @param marker a marker index
     * @return the linear interpolation weight associated with the
     * preceding stage1 marker
     * @throws IndexOutOfBoundsException if
     * {@code marker < 0 || marker >= this.targGT().nMarkers()}
     */
    public float prevStage1Wt(int marker) {
        return prevStage1Wt[marker];
    }
}
