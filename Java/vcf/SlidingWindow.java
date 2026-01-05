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

import java.io.Closeable;
import java.util.Optional;
import java.util.concurrent.BlockingQueue;
import main.Pedigree;

/**
 * <p>Interface {@code SlidingWindow} represents a sliding window of VCF
 * records.</p>
 *
 * <p>Instances of class {@code SlidingWindow} are not thread-safe.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public interface SlidingWindow extends Closeable {

    /**
     * Returns the target samples.
     * @return the target samples
     */
    Samples targSamples();

    /**
     * Returns the target sample pedigree data.
     * @return the target sample pedigree data
     */
    Pedigree ped();

    /**
     * Returns the genetic map.
     * @return the genetic map
     */
    GeneticMap genMap();

    /**
     * Returns the number of distinct target markers returned in
     * preceding {@code this.nextWindow()} method calls.
     * @return the number of distinct target markers returned in
     * preceding {@code this.nextWindow()} method calls
     */
    int cumTargMarkers();

    /**
     * Returns the number of distinct markers returned in
     * preceding {@code this.nextWindow()} method calls.
     * @return the number of distinct markers returned in
     * preceding {@code this.nextWindow()} method calls
     */
    int cumMarkers();

    /**
     * Returns the next sliding window of VCF records.
     *
     * @return the next sliding window of VCF records or
     * <code>Optional.empty()</code> if there are no additional windows.
     *
     * @throws IllegalArgumentException if a format error in the input data
     * is detected
     */
    public Optional<Window> nextWindow();

    /**
     * Releases any I/O resources controlled by this object.
     */
    @Override
    void close();

    /**
     * Takes and returns an element from the specified queue.
     * @param <E> the element type
     * @param q the queue
     * @return an element from the specified queue
     * @throws NullPointerException if {@code q == null}
     * @throws RuntimeException if an {@code InterruptedException} is
     * thrown while waiting to take an element
     */
    public static <E> E takeFromQ(BlockingQueue<E> q) {
        E e = null;
        try {
            e = q.take();
        } catch (InterruptedException ex) {
            throw new RuntimeException(ex);
        }
        return e;
    }

    /**
     * Adds the specified element to the specified queue.
     * @param <E> the element type
     * @param q the queue
     * @param e the element to be added
     * @throws NullPointerException if {@code q == null}
     * @throws RuntimeException if an {@code InterruptedException} is
     * thrown while waiting to add the element
     */
    public static <E> void addToQ(BlockingQueue<E> q, E e) {
        try {
            q.put(e);
        } catch (InterruptedException ex) {
            throw new RuntimeException(ex);
        }
    }
}
