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

import blbutil.DoubleArray;
import ints.IntList;

/**
 * <p>Class {@code Steps} represents a partition of a list of markers into
 * a sequence of sets of consecutive markers (the steps).</p>
 *
 * <p>Instances of class {@code Steps} are immutable.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class Steps {

    private final MarkerMap map;
    private final int[] stepEnds;

    /**
     * Constructs a new {@code Steps} instance from the specified data.
     * @param map the marker map
     * @param minStep the minimum distance between the first markers in
     * consecutive steps
     *
     * @throws IllegalArgumentException if
     * {@code minStep <= 0f || Float.isFinite(minStep) == false}
     * @throws NullPointerException if {@code map == null}
     */
    public Steps(MarkerMap map, float minStep) {
        if (minStep <= 0f || Float.isFinite(minStep)==false) {
            throw new IllegalArgumentException(String.valueOf(minStep));
        }
        this.map = map;
        this.stepEnds = stepEnds(map, minStep);
    }

     private static int[] stepEnds(MarkerMap map, double minStep) {
        DoubleArray genPos = map.genPos();
        int nMarkers = genPos.size();
        IntList indices = new IntList(nMarkers>>1);
        int end = 0;
        while (end<nMarkers) {
            double minGenPos = genPos.get(end) + minStep;
            ++end;
            while (end<genPos.size() && genPos.get(end)<minGenPos) {
                ++end;
            }
            indices.add(end);
        }
        return indices.toArray();
    }

    /**
     * Returns the number of steps.
     * @return the number of steps
     */
    public int size() {
        return stepEnds.length;
    }

    /**
     * Returns the index of the first marker in the specified step.
     * @param step a step index
     * @return the index of the first marker in the specified step
     * @throws IllegalArgumentException if
     * {@code step < 0 || step >= this.nSteps()}
     */
    public int start(int step) {
        return step==0 ? 0 : stepEnds[step-1];
    }

    /**
     * Returns the index of the last marker (exclusive) in the specified step.
     * @param step a step index
     * @return the index of the last marker (exclusive) in the specified step
     * @throws IllegalArgumentException if
     * {@code step < 0 || step >= this.nSteps()}
     */
    public int end(int step) {
        return stepEnds[step];
    }

    /**
     * Return the marker map.
     * @return the marker map
     */
    public MarkerMap map() {
        return map;
    }
}
