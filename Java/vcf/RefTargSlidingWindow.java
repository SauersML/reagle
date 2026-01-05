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

import beagleutil.ChromIds;
import blbutil.Const;
import blbutil.FileIt;
import blbutil.Filter;
import blbutil.InputIt;
import blbutil.SampleFileIt;
import blbutil.Utilities;
import bref.Bref3It;
import java.util.ArrayList;
import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import main.Par;
import main.Pedigree;

/**
 * <p>Class {@code RefTargSlidingWindow} represents a sliding window of
 * reference and target VCF records.</p>
 *
 * <p>Instances of class {@code RefTargSlidingWindow} are not thread-safe.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class RefTargSlidingWindow implements SlidingWindow {

    private final SampleFileIt<GTRec> targIt;
    private final SampleFileIt<RefGTRec> refIt;
    private final Pedigree ped;
    private final GeneticMap genMap;
    private final BlockingQueue<Window> q;
    private final Reader reader;

    private int cumTargMarkers;
    private int cumRefMarkers;
    private boolean noMoreWindows;

    /**
     * Constructs and returns a new {@code RefTargSlidingWindow} instance for
     * the specified reference and target data.  An exception will be thrown
     * if the command line parameters are incorrectly specified.
     *
     * @param par the command line parameters
     * @return a new {@code RefTargSlidingWindow} instance for the specified
     * data
     *
     * @throws IllegalArgumentException if a VCF record format error is detected
     * or there is no data remaining after sample and marker filtering
     * @throws NullPointerException if {@code par == null}
     */
    public static RefTargSlidingWindow instance(Par par) {
        RefTargSlidingWindow windowIt = new RefTargSlidingWindow(par);
        Thread t = new Thread(windowIt.reader);
        t.setDaemon(true);
        t.start();
        return windowIt;
    }

    private RefTargSlidingWindow(Par par) {
        Filter<String> sFilter = FilterUtil.sampleFilter(par.excludesamples());
        Filter<Marker> mFilter = FilterUtil.markerFilter(par.excludemarkers());
        this.targIt = targIt(par, sFilter, mFilter);
        this.refIt = refIt(par, sFilter, mFilter);
        this.ped = new Pedigree(targIt.samples(), par.ped());
        this.genMap = GeneticMap.geneticMap(par.map(), par.chromInt());
        this.q = new ArrayBlockingQueue<>(1);
        this.reader = new Reader(par, genMap, refIt, targIt, q);
        this.cumTargMarkers = 0;
        this.cumRefMarkers = 0;
        this.noMoreWindows = false;
    }

    private static SampleFileIt<GTRec> targIt(Par par, Filter<String> sFilter,
            Filter<Marker> mFilter) {
        int nBufferedBlocks = par.nthreads() << 2;
        FileIt<String> it = InputIt.fromBGZipFile(par.gt(), nBufferedBlocks);
        SampleFileIt<GTRec> targIt = VcfIt.create(it, sFilter, mFilter,
                VcfIt.TO_LOWMEM_GT_REC);
        if (par.chromInt() != null) {
            targIt = new IntervalVcfIt<>(targIt, par.chromInt());
        }
        return targIt;
    }

    private static SampleFileIt<RefGTRec> refIt(Par par,
            Filter<String> sampleFilter, Filter<Marker> markerFilter) {
        SampleFileIt<RefGTRec> refIt;
        String filename = par.ref().toString();
        if (filename.endsWith(".bref")) {
            String s = Const.nl + "ERROR: bref format (.bref) is not supported"
                    + Const.nl + "       Reference files should be in bref3 format (.bref3)";
            Utilities.exit(s);
        }
        if (filename.endsWith(".bref3")) {
            refIt = new Bref3It(par.ref(), sampleFilter, markerFilter);
            if (par.chromInt() != null) {
                refIt = new IntervalVcfIt<>(refIt, par.chromInt());
            }
        } else {
            if (filename.endsWith(".vcf") == false
                    && filename.endsWith(".vcf.gz") == false
                    && filename.endsWith(".vcf.bgz") == false) {
                System.err.println(Const.nl
                        + "ERROR: unrecognized reference filename extension: "
                        + Const.nl
                        + "       expected \".bref3\", \".bref4\", \".vcf\", \".vcf.gz\", or \".vcf.bgz\""
                        + Const.nl);
            }
            int nBufferedBlocks = par.nthreads() << 2;
            FileIt<String> it = InputIt.fromBGZipFile(par.ref(), nBufferedBlocks);
            refIt = RefIt.create(it, sampleFilter, markerFilter);
            if (par.chromInt() != null) {
                refIt = new IntervalVcfIt<>(refIt, par.chromInt());
            }
        }
        return refIt;
    }

    @Override
    public Samples targSamples() {
        return targIt.samples();
    }

    @Override
    public Pedigree ped() {
        return ped;
    }

    @Override
    public GeneticMap genMap() {
        return genMap;
    }

    @Override
    public int cumTargMarkers() {
        return cumTargMarkers;
    }

    @Override
    public int cumMarkers() {
        return cumRefMarkers;
    }

    @Override
    public Optional<Window> nextWindow() {
        if (noMoreWindows) {
            return Optional.empty();
        } else {
            Window window = SlidingWindow.takeFromQ(q);
            if (window.lastWindow()) {
                noMoreWindows = true;
            }
            MarkerIndices indices = window.indices();
            cumTargMarkers += (indices.nTargMarkers() - indices.targOverlapEnd());
            cumRefMarkers += (indices.nMarkers() - indices.overlapEnd());
            return Optional.of(window);
        }
    }

    @Override
    public void close() {
        noMoreWindows = true;
        targIt.close();
        refIt.close();
    }

    /**
     * Returns a string representation of {@code this}. The exact details of the
     * representation are unspecified and subject to change.
     *
     * @return a string representation of {@code this}.
     */
    @Override
    public String toString() {
        return RefTargSlidingWindow.class.toString();
    }

    private static class Reader implements Runnable {

        private final GeneticMap genMap;
        private final float windowCM;
        private final int windowMarkers;
        private final float overlapCM;
        private final int overlapMarkers;
        private final boolean impute;
        private final SampleFileIt<RefGTRec> refIt;
        private final SampleFileIt<GTRec> targIt;
        private final BlockingQueue<Window> q;

        private final ArrayList<GTRec> targOverlap;
        private final ArrayList<RefGTRec> refOverlap;
        private final ArrayList<Boolean> inTargOverlap;
        private final ArrayList<GTRec> targRecs;
        private final ArrayList<RefGTRec> refRecs;
        private final ArrayList<Boolean> inTarg;
        private GTRec nextTargRec;
        private RefGTRec nextRefRec;

        public Reader(Par par, GeneticMap genMap,
                SampleFileIt<RefGTRec> refIt, SampleFileIt<GTRec> targIt,
                BlockingQueue<Window> q) {
            this.genMap = genMap;
            this.windowCM = par.window();
            this.windowMarkers = par.window_markers();
            this.overlapCM = par.overlap();
            this.overlapMarkers = windowMarkers >> 2;
            this.impute = par.impute();
            this.targIt = targIt;
            this.refIt = refIt;
            this.q = q;

            this.targOverlap = new ArrayList<>();
            this.refOverlap = new ArrayList<>();
            this.inTargOverlap = new ArrayList<>();
            this.targRecs = new ArrayList<>(10000);
            this.refRecs = new ArrayList<>(10000);
            this.inTarg = new ArrayList<>(10000);
        }

        @Override
        public void run() {
            try {
                if (targIt.hasNext()==false || refIt.hasNext()==false) {
                    throw new IllegalArgumentException("no genotype data");
                }
                nextTargRec = targIt.next();
                nextRefRec = refIt.next();
                int windowIndex = 0;
                while (nextTargRec!=null && nextRefRec!=null) {
                    int chromIndex = nextTargRec.marker().chromIndex();
                    advanceRefItToChrom(refIt, chromIndex);
                    double endCm = nextEndCm(nextRefRec.marker());
                    int endPos = genMap.basePos(chromIndex, endCm);
                    Window window = readWindow(chromIndex, endPos, ++windowIndex);
                    SlidingWindow.addToQ(q, window);
                    int targOverlapStart = window.indices().targOverlapStart();
                    int refOverlapStart = window.indices().overlapStart();
                    targOverlap.addAll(targRecs.subList(targOverlapStart, targRecs.size()));
                    refOverlap.addAll(refRecs.subList(refOverlapStart, refRecs.size()));
                    inTargOverlap.addAll(inTarg.subList(refOverlapStart, inTarg.size()));
                }
            } catch (Throwable e) {
                Utilities.exit(e);
            }
        }

        private double nextEndCm(Marker nextRefMarker) {
            double endCm = genMap.genPos(nextRefMarker);
            if (refOverlap.isEmpty()) {
                endCm += windowCM;
            } else {
                endCm += (windowCM - overlapCM);
            }
            return endCm;
        }

        private Window readWindow(int chromIndex, int endPos, int windowIndex) {
            int refOverlapEnd = refOverlap.size();
            resetLists();
            while (nextTargRec != null
                    && nextTargRec.marker().chromIndex() == chromIndex
                    && nextTargRec.marker().pos() < endPos
                    && refRecs.size()<windowMarkers) {
                Marker targMarker = nextTargRec.marker();
                int targPos = targMarker.pos();
                while (nextRefRec != null
                        && nextRefRec.marker().chromIndex()==chromIndex
                        && (nextRefRec.marker().pos() < targPos
                            || (nextRefRec.marker().pos() == targPos && targMarker.equals(nextRefRec.marker()) == false))) {
                    if (impute) {
                        refRecs.add(nextRefRec);
                        inTarg.add(Boolean.FALSE);
                    }
                    nextRefRec = refIt.hasNext() ? refIt.next() : null;
                }
                if (nextRefRec!=null && nextRefRec.marker().equals(targMarker)) {
                    targRecs.add(nextTargRec);
                    refRecs.add(nextRefRec);
                    inTarg.add(Boolean.TRUE);
                    nextRefRec = refIt.hasNext() ? refIt.next() : null;
                }
                nextTargRec = targIt.hasNext() ? targIt.next() : null;
            }
            if (impute) {
                while (nextRefRec!=null
                        && nextRefRec.marker().chromIndex()==chromIndex
                        && nextRefRec.marker().pos()<endPos
                        && refRecs.size()<windowMarkers) {
                    refRecs.add(nextRefRec);
                    inTarg.add(Boolean.FALSE);
                    nextRefRec = refIt.hasNext() ? refIt.next() : null;
                }
            }
            return window(refOverlapEnd, windowIndex, chromIndex, endPos);
        }

        private void resetLists() {
            targRecs.clear();
            refRecs.clear();
            inTarg.clear();
            targRecs.addAll(targOverlap);
            refRecs.addAll(refOverlap);
            inTarg.addAll(inTargOverlap);
            targOverlap.clear();
            refOverlap.clear();
            inTargOverlap.clear();
        }

        private Window window(int refOverlapEnd, int windowIndex, int chromIndex,
                int endPos) {
            if (targRecs.isEmpty() || refRecs.isEmpty()) {
                throw new IllegalArgumentException(
                        emptyWindowErrorMessage(chromIndex, endPos));
            }
            RefGT refGT = new RefGT(refRecs.toArray(new RefGTRec[0]));
            BasicGT targGT = new BasicGT(targRecs.toArray(new GTRec[0]));
            boolean lastWindow = (nextTargRec==null || nextRefRec==null);
            MarkerIndices markerIndices = markerIndices(refGT, lastWindow,
                    refOverlapEnd, endPos);
            return new Window(genMap, windowIndex, lastWindow,
                    markerIndices, refGT, targGT);
        }

        private String emptyWindowErrorMessage(int chromIndex, int endPos) {
            assert refRecs.isEmpty() || targRecs.isEmpty();
            if (refRecs.isEmpty()) {
                return "The window ending at "
                        + ChromIds.instance().id(chromIndex) + ":" + endPos
                        + Const.nl + "contains no reference markers"
                        + Const.nl + "Do the reference and target VCF files contain the same"
                        + Const.nl + "chromosomes in the same order?"
                        + Const.nl;
            }
            else {
                assert targRecs.isEmpty();
                return "The reference and target VCF files contain no markers in common in the window: "
                        + Const.nl + ChromIds.instance().id(chromIndex)
                        + ":" + refRecs.get(0).marker().pos()
                        + "-" + endPos
                        + Const.nl + "Do both VCF files share any markers in this window?"
                        + Const.nl + "Do both VCF files contain the same chromosomes in the same order?"
                        + Const.nl;
            }
        }

        private MarkerIndices markerIndices(RefGT refGT, boolean lastWindow,
                int refOverlapEnd, int endPos) {
            boolean chromEnd =  lastWindow
                    || (refRecs.get(0).marker().chromIndex() != nextRefRec.marker().chromIndex());
            int refOverlapStart = overlapStart(refGT, chromEnd, endPos);
            boolean[] inTarget = new boolean[inTarg.size()];
            for (int j=0; j<inTarget.length; ++j) {
                inTarget[j] = inTarg.get(j);
            }
            return new MarkerIndices(inTarget, refOverlapEnd, refOverlapStart);
        }

        private int overlapStart(RefGT refGT, boolean chromEnd, int endPos) {
            if (chromEnd) {
                return refGT.nMarkers();
            } else {
                int nMarkersM1 = refGT.nMarkers() - 1;
                int chromIndex = refGT.marker(nMarkersM1).chromIndex();
                double endGenPos = genMap.genPos(chromIndex, endPos-1);
                double startGenPos = endGenPos - overlapCM;
                int key = genMap.basePos(chromIndex, startGenPos);
                int low = Math.max(0, refGT.nMarkers()-overlapMarkers);
                int high = nMarkersM1;
                while (low <= high) {
                    int mid = (low + high) >>> 1;
                    int midPos = refGT.marker(mid).pos();
                    if (midPos < key) {
                        low = mid + 1;
                    } else if (midPos > key) {
                        high = mid - 1;
                    } else {
                        return firstIndexWithPos(refGT, mid);
                    }
                }
                assert high < low;
                return firstIndexWithPos(refGT, Math.max(0, high));
            }
        }

        private int firstIndexWithPos(RefGT refGT, int index) {
            int pos = refGT.marker(index).pos();
            while (index > 0 && refGT.marker(index - 1).pos() == pos) {
                --index;
            }
            return index;
        }

        private void advanceRefItToChrom(SampleFileIt<RefGTRec> refIt, int chromIndex) {
            while (nextRefRec!=null
                    && nextRefRec.marker().chromIndex() != chromIndex
                    && refIt.hasNext()) {
                nextRefRec = refIt.next();
            }
        }
    }
}
