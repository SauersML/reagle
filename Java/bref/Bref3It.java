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
package bref;

import blbutil.FileUtil;
import blbutil.Filter;
import blbutil.SampleFileIt;
import blbutil.Utilities;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.NoSuchElementException;
import vcf.Marker;
import vcf.RefGTRec;
import vcf.Samples;

/**
 * <p>Class {@code Bref3It} represents  an iterator whose {@code next()} which
 * returns records from a bref version 3 file.
 * </p>
 * <p>Instances of class {@code Bref3It} are not thread-safe.
 * </p>
 * <p>Methods of this class will terminate the Java Virtual Machine with
 * an error message if an I/O error or file format error is detected.
 * </p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class Bref3It implements SampleFileIt<RefGTRec> {

    private final File brefFile;
    private final DataInputStream dataIn;
    private final Bref3Reader bref3Reader;
    private final Deque<RefGTRec> buffer;

    /**
     * Constructs a new {@code Bref3It} instance.
     * @param brefFile a bref version 3 file or {@code null} if the
     * bref version 3 file is to be read from standard input
     *
     * @throws IllegalArgumentException if a format error is detected in a
     * line of the specified bref file
     */
    public Bref3It(File brefFile) {
        this(brefFile, Filter.acceptAllFilter(), Filter.acceptAllFilter());
    }

    /**
     * Constructs a new {@code Bref4It} instance.
     * @param brefFile a bref v3 file or {@code null} if the bref3 file
     * is to be read from stdin
     * @param sampleFilter a sample filter
     * @param markerFilter a marker filter
     *
     * @throws IllegalArgumentException if a format error is detected in
     * the specified bref v3 file
     * @throws NullPointerException if
     * {@code (sampleFilter == null) || (markerFilter == null)}
     */
    public Bref3It(File brefFile, Filter<String> sampleFilter,
            Filter<Marker> markerFilter) {
        InputStream dis;
        if (brefFile==null) {
            dis = new BufferedInputStream(System.in);
        }
        else {
            dis = FileUtil.bufferedInputStream(brefFile);
        }
        this.brefFile = brefFile;
        this.dataIn = new DataInputStream(dis);
        this.bref3Reader = new Bref3Reader(brefFile, dataIn, sampleFilter, markerFilter);
        this.buffer = new ArrayDeque<>(500);
        bref3Reader.readBlock(dataIn, buffer);
    }

    /**
     * Returns {@code true} if the iteration has more elements, and returns
     * {@code false} otherwise.
     * @return {@code true} if the iteration has more elements
     */
    @Override
    public boolean hasNext() {
        return !buffer.isEmpty();
    }

    /**
     * Returns the next element in the iteration.
     * @return the next element in the iteration
     * @throws NoSuchElementException if the iteration has no more elements
     */
    @Override
    public RefGTRec next() {
        if (hasNext()==false) {
            throw new NoSuchElementException();
        }
        RefGTRec rec = buffer.removeFirst();
        if (buffer.isEmpty()) {
            bref3Reader.readBlock(dataIn, buffer);
        }
        return rec;
    }

    @Override
    public void close() {
        try {
            dataIn.close();
        } catch (IOException ex) {
            Utilities.exit(ex, "Error closing file");
        }
        buffer.clear();
    }

    @Override
    public File file() {
        return brefFile;
    }

    @Override
    public Samples samples() {
        return bref3Reader.samples();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(80);
        sb.append(this.getClass().toString());
        sb.append(" : ");
        sb.append(brefFile);
        return sb.toString();
    }
}
