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
package ints;

import java.util.Arrays;

/**
 * <p>Class {@code SynchedIntList} represents a list of integers.
 * Class {@code SynchedIntList} supports a {@code clear()} method, 
 * but it does not support a {@code remove()} method.</p>
 *
 * <p>Instances of class {@code SynchedIntList} are thread-safe.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class SynchedIntList {

    /**
     * The default initial capacity of an {@code SynchedIntList},
     * which is 16.
     */
    public static final int DEFAULT_INIT_CAPACITY = 16;

    private int size;
    private int[] values;

    /**
     * Constructs an {@code SynchedIntList} object with the default
     * initial capacity.
     *
     * @see #DEFAULT_INIT_CAPACITY
     */
    public SynchedIntList() {
        this(DEFAULT_INIT_CAPACITY);
    }

    /**
     * Constructs an {@code SynchedIntList} object with the specified
     * initial capacity.
     *
     * @param initCapacity the initial capacity of this list
     * @throws IllegalArgumentException if {@code initCapacity < 0}
     */
    public SynchedIntList(int initCapacity) {
        if (initCapacity < 0) {
            throw new IllegalArgumentException(String.valueOf(initCapacity));
        }
        this.size = 0;
        this.values = new int[initCapacity];
    }

    /**
     * Constructs an {@code SynchedIntList} by cloning the specified array.
     *
     * @param ia a list of integer values
     * @throws NullPointerException if {@code ia == null}
     */
    public SynchedIntList(int[] ia) {
        this.size = ia.length;
        this.values = ia.clone();
    }

    /**
     * Constructs an {@code SynchedIntList} by copying the specified
     * {@code SynchedIntList}.
     *
     * @param intList a list of integer values
     * @throws NullPointerException if {@code intList == null}
     */
    public SynchedIntList(SynchedIntList intList) {
        this.size = intList.size();
        this.values = Arrays.copyOf(intList.values, intList.size());
    }

    /**
     * Adds the specified integer to the end of this list.
     *
     * @param value the integer to be added to the end of this list
     */
    public synchronized void add(int value) {
        if (size==values.length) {
            int newCapacity = (values.length * 3)/2 + 1;
            this.values = Arrays.copyOf(this.values, newCapacity);
        }
        this.values[size++] = value;
    }

    /**
     * Returns the element at the specified position in this list.
     * @param index the index of the element to be returned
     * @return the element at the specified position in this list
     * @throws IndexOutOfBoundsException if
     * {@code index < 0 || index >= this.size()}
     */
    public synchronized int get(int index) {
        if (index >= size) {
            throw new IndexOutOfBoundsException(String.valueOf(index));
        }
        return values[index];
    }

    /**
     * Replaces the element at the specified position in this list with the
     * specified element.
     * @param index the index of the element to be replaced
     * @param value the value to be stored at the specified position
     * in this list
     * @return the previous value at the specified position in this list
     * @throws IndexOutOfBoundsException if
     * {@code index < 0 || index >= this.size()}
     */
    public synchronized int set(int index, int value) {
        if (index >= size) {
            throw new IndexOutOfBoundsException(String.valueOf(index));
        }
        int oldValue = values[index];
        values[index] = value;
        return oldValue;
    }

    /**
     * Returns the number of elements in this list.
     * @return the number of elements in this list
     */
    public synchronized int size() {
        return size;
    }

    /**
     * Returns an integer array containing the sequence of elements in this
     * list.
     * @return an integer array containing the sequence of elements in this
     * list
     */
    public synchronized int[] toArray() {
        return Arrays.copyOf(values, size);
    }

    /**
     * Removes all elements from this list.
     */
    public synchronized void clear() {
        this.size = 0;
    }

    /**
     * Returns {@code java.util.Arrays.toString(this.toArray())}
     *
     * @return {@code java.util.Arrays.toString(this.toArray())}
     */
    @Override
    public synchronized String toString() {
        return Arrays.toString(toArray());
    }
}
