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

import beagleutil.IntInterval;
import blbutil.Const;
import java.util.Comparator;

/**
 * <p>Class {@code SampleSeg} represents a segment of genotype data in
 * a sample.</p>
 *
 * <p>Instances of class {@code SampleSeg} are immutable</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class SampleSeg implements Comparable<SampleSeg>, IntInterval {

    private final int sample;
    private final int start;
    private final int inclEnd;

    /**
     * Constructs a new {@code SampleSeg} instance from the specified data.
     * @param sample the sample index
     * @param start the start marker index (inclusive)
     * @param inclEnd the end marker index (inclusive)
     * @throws IllegalArgumentException if {@code start > inclEnd}
     */
    public SampleSeg(int sample, int start, int inclEnd) {
        if (start > inclEnd) {
            throw new IllegalArgumentException(String.valueOf(start));
        }
        this.sample = sample;
        this.start = start;
        this.inclEnd = inclEnd;
    }

    /**
     * Returns the sample index.
     * @return the sample index
     */
    public int sample() {
        return sample;
    }

    /**
     * Returns the start marker index (inclusive).
     * @return the start marker index (inclusive)
     */
    @Override
    public int start() {
        return start;
    }

    /**
     * Returns the end marker index (inclusive).
     * @return the end marker index (inclusive)
     */
    @Override
    public int inclEnd() {
        return inclEnd;
    }

    /**
     * Returns a string representation of {@code this}.  The exact
     * details of the representation are unspecified and subject to change.
     * @return a string representation of {@code this}
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(70);
        sb.append("[");
        sb.append(sample);
        sb.append(": ");
        sb.append(start);
        sb.append(Const.hyphen);
        sb.append(inclEnd);
        sb.append("]");
        return sb.toString();
    }

    /**
     * <p>Returns the hash code value for this object. The hash code is defined
     * by the following calculation:
     * </p>
     * <pre>
     *  int hash = 5;
     *  hash = 89 * hash + this.hap();
     *  hash = 89 * hash + this.start();
     *  hash = 89 * hash + this.inclEnd();
     </pre>
     * @return the hash code value for this object
     */
    @Override
    public int hashCode() {
        int hash = 5;
        hash = 89*hash + this.sample;
        hash = 89*hash + this.start;
        hash = 89*hash + this.inclEnd;
        return hash;
    }

    /**
     * Compares the specified object with this {@code SampleSeg} for
     * equality. Returns {@code true} if the specified object is a
     * {@code SampleSeg} instance and if this {@code SampleSeg} is
     * equal to the specified {@code SampleSeg}, and returns
     * {@code false}  otherwise.  Two {@code SampleSeg}  instances
     * are equal if they have equal sample indices,
     * equal start marker indices, and equal end marker indices.
     * @param o the object to be compared with this {@code SampleSeg}
     * for equality
     * @return {@code true} if the specified object is equal to {@code this}
     */
    @Override
    public boolean equals(Object o) {
        if (o==null) {
            return false;
        }
        if (getClass()!=o.getClass()) {
            return false;
        }
        final SampleSeg other=(SampleSeg) o;
        return (this.sample==other.sample && this.start==other.start
                && this.inclEnd==other.inclEnd);
    }

    /**
     * Compares this {@code SampleSeg} with the specified {@code SampleSeg}
     * for order.  Returns a negative integer, zero, or a positive integer
     * as this object is less than, equal to, or greater than the specified
     * object. {@code SampleSeg} instances are ordered first by
     * {@code this.start()}, then by {@code this.end()}, and finally by
     * t{@code this.sample()}.
     * @param ss the {@code SampleSeg} to be compared with this {@code SampleSeg}
     * @return a negative integer, zero, or a positive integer as this
     * {@code SampleSeg} is less than, equal to, or greater than the
     * specified {@code SampleSeg}
     * @throws NullPointerException if {@code ss == null}
     */
    @Override
    public int compareTo(SampleSeg ss) {
        if (this.start != ss.start) {
            return (this.start < ss.start) ? -1 : 1;
        }
        else if (this.inclEnd != ss.inclEnd) {
            return (this.inclEnd < ss.inclEnd) ? -1 : 1;
        }
        if (this.sample != ss.sample) {
            return (this.sample < ss.sample) ? -1 : 1;
        }
        return 0;
    }

    /**
     * Returns a comparator that orders first by {@code this.sample()}, then
     * by {@code this.start()}, and finally by {@code this.end()}.
     * @return a comparator that orders first by {@code this.sample()}, then
     * by {@code this.start()}, and finally by {@code this.end()}
     */
    public static Comparator<SampleSeg> sampleComp() {
        return (SampleSeg t1, SampleSeg t2) -> {
            if (t1.sample != t2.sample) {
                return (t1.sample < t2.sample) ? -1 : 1;
            }
            if (t1.start != t2.start) {
                return (t1.start < t2.start) ? -1 : 1;
            }
            else if (t1.inclEnd != t2.inclEnd) {
                return (t1.inclEnd < t2.inclEnd) ? -1 : 1;
            }
            return 0;
        } ;
    }
}
