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

import java.util.Arrays;

/**
 * Class {@code FloatArray} represents an immutable list of float floating
 * point values.
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class FloatArray {

    private final float[] values;

    /**
     * Constructs an {@code FloatArray} object with the specified
     * values.
     * @param values the list of floating point values
     * @throws NullPointerException if {@code values == null}
     */
    public FloatArray(float[] values) {
        this.values = values.clone();
    }

    /**
     * Constructs an {@code FloatArray} object with the specified
     * values.
     * @param values the list of floating point values
     * @throws NullPointerException if {@code values == null}
     */
    public FloatArray(double[] values) {
        this.values = new float[values.length];
        for (int j=0; j<values.length; ++j) {
            this.values[j] = (float) values[j];
        }
    }

    /**
     * Constructs and returns a FloatArray from the specified list of bit
     * representations. Each {@code integer} is transformed into a float
     * using the {@code Float.intBitsToFloat()} method.
     * @param bits a list of bit
     * @return a {@code FloatArray}
     */
    public static FloatArray fromIntBits(int[] bits) {
        return new FloatArray(bits);
    }

    private FloatArray(int[] values) {
        this.values = new float[values.length];
        for (int j=0; j<values.length; ++j) {
            this.values[j] = Float.intBitsToFloat(values[j]);
        }
    }

    /**
     * Returns the float at the specified position in this list.
     * @param index the index of the returned float
     * @return the float at the specified position in this list
     * @throws IndexOutOfBoundsException if
     * {@code index < 0 || index >= size}
     */
    public float get(int index) {
        return values[index];
    }

    /**
     * Returns the number of elements in this list.
     * @return the number of elements in this list
     */
    public int size() {
        return values.length;
    }

    /**
     * Returns {@code true} if this list has no elements, and returns
     * {@code false} otherwise.
     * @return {@code true} if this list has no elements, and returns
     * {@code false} otherwise
     */
    public boolean isEmpty() {
        return values.length==0;
    }

    /**
     * Searches {@code this} for the specified value using the binary search
     * algorithm. This list must be sorted (as by the
     * {@code java.util.Arrays.sort(float[])} method) prior to making
     * this call. If it is not sorted, the results are undefined.
     * If the list contains multiple elements with the specified
     * value, there is no guarantee which one will be found. This method
     * considers all NaN values to be equivalent and equal.
     *
     * @param key the value to be searched for
     *
     * @return index of the search key, if it is contained in the list;
     * otherwise, {@code (-(insertion point) - 1)}. The insertion point is
     * defined as the point at which the key would be inserted into the list:
     * the index of the first element greater than the key, or
     * {@code this.size()} if all elements in the list are less than the
     * specified key. Note that this guarantees that the return value will
     * be {@code >= 0} if and only if the key is found.
     */
    public int binarySearch(float key) {
        return Arrays.binarySearch(values, key);
    }

    /**
     * Searches the specified range of {@code this} for the specified value
     * using the binary search algorithm. This range must be sorted (as by the
     * {@code java.util.Arrays.sort(float[])} method) prior to making
     * this call. If it is not sorted, the results are undefined.
     * If the range contains multiple elements with the specified
     * value, there is no guarantee which one will be found. This method
     * considers all NaN values to be equivalent and equal.
     *
     * @param fromIndex the index of the first element (inclusive) to be searched
     * @param toIndex the index of the last element (exclusive) to be searched
     * @param key the value to be searched for
     *
     * @return index of the search key, if it is contained in the list;
     * otherwise, {@code (-(insertion point) - 1)}. The insertion point is
     * defined as the point at which the key would be inserted into the list:
     * the index of the first element greater than the key, or
     * {@code this.size()} if all elements in the list are less than the
     * specified key. Note that this guarantees that the return value will
     * be {@code >= 0} if and only if the key is found.
     *
     * @throws IllegalArgumentException if {@code fromIndex > toIndex}
     * @throws ArrayIndexOutOfBoundsException if
     * {@code fromIndex < 0 || toIndex > this.size()}
     */
    public int binarySearch(int fromIndex, int toIndex, float key) {
        return Arrays.binarySearch(values, fromIndex, toIndex, key);
    }

    /**
     * Returns an integer array containing the sequence of elements in this
     * list.
     * @return an integer array containing the sequence of elements in this
     * list
     */
    public float[] toArray() {
        return values.clone();
    }

    /**
     * Returns a string representation of this list that is
     * obtained by calling {@code java.util.Arrays.toString(this.toArray())}.
     *
     * @return a string representation of this list
     */
    @Override
    public String toString() {
        return Arrays.toString(values);
    }
}
