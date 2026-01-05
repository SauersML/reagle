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
package blbutil;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Class {@code Utilities} contains miscellaneous static utility methods
 * for multi-threaded programming.
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class MultiThreadUtils {

    private MultiThreadUtils() {
        // private constructor to prevent instantiation
    }

    /**
     * Inserts the specified element at the tail of the specified blocking
     * queue, waiting for space to become available if the queue is full.
     * The Java Virtual Machine is terminated if an {@code InterruptedException}
     * is thrown while waiting for space to be come available in the queue.
     * @param <E> the element type
     * @param q a blocking queue
     * @param e the element to add
     * @throws NullPointerException if {@code q == null || e == null}
     */
    public static <E> void putInBlockingQ(BlockingQueue<E> q, E e) {
        try {
            q.put(e);
        } catch (InterruptedException ex) {
            Utilities.exit(ex, "ERROR: ");
        }
    }

    /**
     * Inserts the specified element at the tail of the specified blocking
     * queue, waiting up to the specified time for space to become available
     * if the queue is full.
     * The Java Virtual Machine is terminated if an {@code InterruptedException}
     * is thrown while waiting for space to be come available in the queue.
     * @param <E> the element type
     * @param q a blocking queue
     * @param e the element to add
     * @param timeout the number of time units to wait before giving up
     * @param unit the time unit
     * @return {@code true} if element was added to the queue, and
     * false otherwise
     * @throws NullPointerException if
     * {@code q == null || e == null || unit == null}
     */
    public static <E> boolean putInBlockingQ(BlockingQueue<E> q, E e,
            long timeout, TimeUnit unit) {
        try {
            return q.offer(e, timeout, unit);
        } catch (InterruptedException ex) {
            Utilities.exit(ex, "ERROR: ");
        }
        return false;
    }

    /**
     * Removes and returns the element at the head of the specified blocking
     * queue, waiting if necessary for an element to become available.
     * The Java Virtual Machine is terminated if an {@code InterruptedException}
     * is thrown while waiting for space to be come available in the queue.
     * @param <E> the element type
     * @param q a blocking queue
     * @return the element at the head of the queue
     */
    public static <E> E takeFromBlockingQ(BlockingQueue<E> q) {
        try {
            return q.take();
        } catch (InterruptedException ex) {
            Utilities.exit(ex, "ERROR: ");
        }
        assert false;
        return null;
    }

    /**
     * Blocks the current thread until the specified {@code CountDownLatch}
     * has counted down to 0. The Java Virtual Machine is terminated if an
     * {@code InterruptedException} is thrown while waiting for for the
     * {@code CountDownLatch} to count down to 0.
     * @param latch the count down latch
     * @throws NullPointerException if {@code latch == null}
     */
    public static void await(CountDownLatch latch) {
        try {
            latch.await();
        }
        catch (InterruptedException e) {
            Utilities.exit(e, "ERROR");
        }
    }

    /**
     * Shuts down and awaits termination of the specified
     * {@code ExecutorService}. The Java Virtual Machine is terminated if an
     * {@code InterruptedException} is thrown while awaiting termination
     * of the executor service.
     * @param es the executor service to be shut down
     * @throws NullPointerException if {@code es == null}
     */
    public static void shutdownExecService(ExecutorService es) {
        try {
            es.shutdown();
            es.awaitTermination(Long.MAX_VALUE, TimeUnit.DAYS);
        }
        catch (InterruptedException e) {
            Utilities.exit(e, "ERROR");
        }
    }

}

