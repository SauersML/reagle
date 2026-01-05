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

import java.util.concurrent.atomic.AtomicLong;

/**
 * <p>Class {@code SwapRate} stores the proportion of unphased heterozygotes
 * whose phase with respect to the previous heteroygote has been reversed.
 * </p>
 * <p>Instances of class {@code SwapRate} are thread-safe.
 * </p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class SwapRate {

    private static final AtomicLong nSwaps = new AtomicLong(0);
    private static final AtomicLong nUnphHets = new AtomicLong(0);

    private SwapRate() {
        // private constructor to prevent instantiation
    }
    
    /**
     * Returns the proportion of unphased heterozygotes whose phase
     * relative to the previous heterozygote has been reversed.
     * The counters for the number of heterozygotes whose phase has been
     * reversed and for the total number of unphased heterozygotes are then
     * set to 0.
     * @return the proportion of unphased heterozygotes whose phase
     * has been changed
     */
    public static double getAndResetSwapRate() {
        double rate = (double) nSwaps.get() / nUnphHets.get();
        nSwaps.set(0);
        nUnphHets.set(0);
        return rate;
    }

    /**
     * Increments the number of unphased heterozygotes and the number of
     * unphased heterozygotes whose phase with respect to the preceding 
     * heterozygote has been reversed.
     * @param nUnphHets the value that will be added to number of 
     * unphased heterozygotes
     * @param nSwaps the value that will be added to number of unphased 
     * heterozygotes whose phase has been reversed
     */
    public static void increment(int nUnphHets, int nSwaps) {
        SwapRate.nSwaps.addAndGet(nSwaps);
        SwapRate.nUnphHets.addAndGet(nUnphHets);
    }
}
