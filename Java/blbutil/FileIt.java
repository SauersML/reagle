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

import java.io.Closeable;
import java.io.File;

/**
 * <p>An iterator for data elements in a file.  If an {@code IOException}
 * is thrown while reading a file, the {@code IOException} is trapped,
 * an appropriate error message is written to standard out, and the
 * Java Virtual Machine is terminated.  The {@code Iterator.remove()} method
 * is unsupported and throws an {@code UnsupportedOperationException}.</p>
 *
 * <p>When the {@code FileIt} object is no longer needed, the {@code close()}
 * method should be invoked to release any system resources controlled
 * by the object.</p>
 *
 * @param <E> the type of the elements returned by this iterator's
 * {@code next()} method.
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public interface FileIt<E> extends java.util.Iterator<E>, Closeable {

    /**
     * Returns the file from which the data are read, or
     * {@code null} if the data are read from standard input or if the
     * data source is unknown.
     * @return the file from which the data are read, or
     * {@code null} if the data are read from standard input or if the
     * data source is unknown
     */
    File file();

    /**
     * Closes the input stream and releases any system resources that are
     * associated with it. If the input stream is already closed then
     * invoking this method has no effect.
     */
    @Override
    public void close();

    /**
     * Returns a string representation of {@code this}.  The exact
     * details of the representation are unspecified and subject to change.
     *
     * @return a string representation of {@code this}
     */
    @Override
    public String toString();
}
