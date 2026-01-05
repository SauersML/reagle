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

import blbutil.FileIt;
import blbutil.Filter;
import blbutil.InputIt;
import blbutil.SampleFileIt;
import blbutil.Utilities;
import java.util.ArrayList;
import java.util.Optional;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import main.Par;
import main.Pedigree;

/**
 * <p>Class {@code TargSlidingWindow} represents a sliding window of
 * target VCF records.</p>
 *
 * <p>Instances of class {@code TargSlidingWindow} are not thread-safe.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class TargSlidingWindow implements SlidingWindow {

    private final SampleFileIt<GTRec> targIt;
    private final Pedigree ped;
    private final GeneticMap genMap;
    private final BlockingQueue<Window> q;
    private final Reader reader;

    private int cumTargMarkers;
    private boolean noMoreWindows;

    /**
     * Constructs and returns a new {@code TargSlidingWindow} instance
     * for the specified target data.
     *
     * @param par the command line parameters
     * @return a new {@code TargSlidingWindow} instance for the specified
     * target data
     *
     * @throws IllegalArgumentException if {@code par.ref() != null}
     * @throws IllegalArgumentException if a VCF record format error is detected
     * or if there is no data remaining after sample and marker filtering
     * @throws NullPointerException if {@code par == null}
     */
    public static TargSlidingWindow instance(Par par) {
        TargSlidingWindow windowIt = new TargSlidingWindow(par);
        Thread t = new Thread(windowIt.reader);
        t.setDaemon(true);
        t.start();
        return windowIt;
    }

    private TargSlidingWindow(Par par) {
        if (par.ref()!=null) {
            throw new IllegalStateException("par.ref()=" + par.ref());
        }
        this.targIt = targIt(par);
        this.ped = new Pedigree(targIt.samples(), par.ped());
        this.genMap = GeneticMap.geneticMap(par.map(), par.chromInt());
        this.q = new ArrayBlockingQueue<>(1);
        this.reader = new Reader(par, genMap, targIt, q);
        this.cumTargMarkers = 0;
        this.noMoreWindows = false;
    }

    private static SampleFileIt<GTRec> targIt(Par par) {
        int nBufferedBlocks = par.nthreads() << 2;
        FileIt<String> it = InputIt.fromBGZipFile(par.gt(), nBufferedBlocks);
        Filter<String> sFilter = FilterUtil.sampleFilter(par.excludesamples());
        Filter<Marker> mFilter = FilterUtil.markerFilter(par.excludemarkers());
        SampleFileIt<GTRec> targIt = VcfIt.create(it, sFilter, mFilter,
                VcfIt.TO_LOWMEM_GT_REC);
        if (par.chromInt()!=null) {
            targIt = new IntervalVcfIt<>(targIt, par.chromInt());
        }
        return targIt;
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
        return cumTargMarkers;
    }

    @Override
    public Optional<Window> nextWindow() {
        if (noMoreWindows) {
            return Optional.empty();
        }
        else {
            Window window = SlidingWindow.takeFromQ(q);
            if (window.lastWindow()) {
                noMoreWindows = true;
            }
            MarkerIndices indices = window.indices();
            cumTargMarkers += (indices.nTargMarkers() - indices.targOverlapEnd());
            return Optional.of(window);
        }
    }

    @Override
    public void close() {
        noMoreWindows = true;
        targIt.close();
    }

    /**
     * Returns a string representation of {@code this}.  The exact
     * details of the representation are unspecified and subject to change.
     * @return a string representation of {@code this}.
     */
    @Override
    public String toString() {
        return TargSlidingWindow.class.toString();
    }

    private static class Reader implements Runnable {

        private final GeneticMap genMap;
        private final float windowCM;
        private final int windowMarkers;
        private final float overlapCM;
        private final int overlapMarkers;
        private final SampleFileIt<GTRec> targIt;
        private final BlockingQueue<Window> q;

        private final ArrayList<GTRec> overlap;
        private final ArrayList<GTRec> recs;
        private GTRec nextRec;

        public Reader(Par par, GeneticMap genMap, SampleFileIt<GTRec> it,
                BlockingQueue<Window> q) {
            this.genMap = genMap;
            this.windowCM = par.window();
            this.windowMarkers = par.window_markers();
            this.overlapCM = par.overlap();
            this.overlapMarkers = windowMarkers >> 2;
            this.targIt = it;
            this.q = q;

            this.overlap = new ArrayList<>();
            this.recs = new ArrayList<>(10000);
        }

        @Override
        public void run() {
            try {
                if (targIt.hasNext()==false) {
                    throw new IllegalArgumentException("Error: no genotype data");
                }
                nextRec = targIt.next();
                int windowIndex = 0;
                while (nextRec!=null) {
                    int chromIndex = nextRec.marker().chromIndex();
                    double nextEndCm = nextEndCm(nextRec.marker());
                    int endPos = genMap.basePos(chromIndex, nextEndCm);
                    Window window = readWindow(chromIndex, endPos, ++windowIndex);
                    SlidingWindow.addToQ(q, window);
                    int overlapStart = window.indices().overlapStart();
                    overlap.addAll(recs.subList(overlapStart, recs.size()));
                }
            }
            catch (Throwable e) {
                Utilities.exit(e);
            }
        }

        private double nextEndCm(Marker nextMarker) {
            double endCm = genMap.genPos(nextMarker);
            if (overlap.isEmpty()) {
                endCm += windowCM;
            } else {
                endCm += (windowCM - overlapCM);
            }
            return endCm;
        }

        private Window readWindow(int chromIndex, int endPos, int windowIndex) {
            int overlapEnd = overlap.size();
            recs.clear();
            recs.addAll(overlap);
            overlap.clear();
            while (nextRec!=null
                    && nextRec.marker().chromIndex()==chromIndex
                    && nextRec.marker().pos()<endPos
                    && recs.size()<windowMarkers) {
                recs.add(nextRec);
                nextRec = targIt.hasNext() ? targIt.next() : null;
            }
            return window(overlapEnd, chromIndex, windowIndex);
        }

        private Window window(int overlapEnd, int chromIndex, int windowIndex) {
            BasicGT targGT = new BasicGT(recs.toArray(new GTRec[0]));
            boolean lastWindow = (nextRec==null);
            boolean chromEnd = nextRec==null
                    || nextRec.marker().chromIndex()!=chromIndex;
            int overlapStart = targOverlapStart(targGT, chromEnd);
            MarkerIndices markerIndices = new MarkerIndices(overlapEnd,
                    overlapStart, targGT.nMarkers());
            return new Window(genMap, windowIndex, lastWindow, markerIndices,
                    null, targGT);
        }

        private int targOverlapStart(BasicGT targGT, boolean chromEnd) {
            if (chromEnd) {
                return targGT.nMarkers();
            }
            else {
                int nMarkersM1 = targGT.nMarkers()-1;
                Marker marker = targGT.marker(nMarkersM1);
                double endGenPos = genMap.genPos(marker);
                double startGenPos = endGenPos - overlapCM;
                int key = genMap.basePos(marker.chromIndex(), startGenPos);
                int low = Math.max(0, targGT.nMarkers()-overlapMarkers);
                int high = nMarkersM1;
                while (low <= high) {
                    int mid = (low + high) >>> 1;
                    int midPos = targGT.marker(mid).pos();
                    if (midPos < key) {
                        low = mid + 1;
                    }
                    else if (midPos > key) {
                        high = mid - 1;
                    }
                    else {
                        return firstIndexWithPos(targGT, mid);
                    }
                }
                assert high < low;
                return firstIndexWithPos(targGT, Math.max(0, high));
            }
        }

        private int firstIndexWithPos(BasicGT targGT, int index) {
            int pos = targGT.marker(index).pos();
            while (index>0 && targGT.marker(index-1).pos()==pos) {
                --index;
            }
            return index;
        }
    }
}
