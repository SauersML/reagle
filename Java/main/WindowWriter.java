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
package main;

import blbutil.BGZIPOutputStream;
import blbutil.Utilities;
import imp.ImpData;
import imp.ImputedVcfWriter;
import imp.StateProbs;
import ints.IntList;
import ints.UnsignedByteArray;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.Closeable;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.stream.IntStream;
import phase.Stage2Haps;
import vcf.GT;
import vcf.GTRec;
import vcf.Samples;
import vcf.VcfWriter;

/**
 * <p>Class {@code WindowWriter} writes VCF and IBD output data.
 * </p>
 * <p>Instances of class {@code WindowWriter} are not thread-safe.
 * </p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class WindowWriter implements Closeable {

    private final Samples samples;
    private final String outPrefix;
    private final File vcfOutFile;

    /**
     * Constructs a new {@code WindowWriter} object.
     * @param par the analysis parameters
     * @param samples the sample whose data will be printed
     *
     * @throws IllegalArgumentException if {@code outPrefix.length() == 0}
     * @throws NullPointerException if
     * {@code par == null || samples == null}
     */
    public WindowWriter(Par par, Samples samples) {
        if (samples==null) {
            throw new NullPointerException("samples==null");
        }
        this.outPrefix = par.out();
        this.vcfOutFile = new File(outPrefix + ".vcf.gz");
        this.samples = samples;

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (PrintWriter vcfOut=new PrintWriter(
                new BGZIPOutputStream(baos, false))) {
            boolean ds = true;
            boolean ap = par.ap();
            boolean gp = par.gp();
            boolean gl = false;
            VcfWriter.writeMetaLines(samples.ids(), Main.PROGRAM,
                    ds, ap, gp, gl, vcfOut);
        }
        try {
            try (FileOutputStream fos=new FileOutputStream(vcfOutFile)) {
                fos.write(baos.toByteArray());
            }
        } catch (IOException e) {
            fileOutputError(vcfOutFile, e);
        }
    }

    /**
     * Returns the output file prefix.
     * @return the output file prefix
     */
    public String outPrefix() {
        return outPrefix;
    }

    /**
     * Returns the samples whose data is written by {@code this}.
     * @return the samples whose data is written by {@code this}
     */
    public Samples samples() {
        return samples;
    }

    /**
     * Prints the data in {@code alProbs} for markers
     * with index between {@code refStart} (inclusive) and
     * {@code refEnd} (exclusive) to the output
     * VCF file: {@code this.outPrefix() + ".vcf.gz"}.
     *
     * @param impData the input data for genotype imputation
     * @param stateProbs the imputed state probabilities
     * @param start the starting reference marker index (inclusive)
     * @param end the ending reference marker index (exclusive)
     *
     * @throws IllegalArgumentException if
     * {@code stateProbs.size() != impData.nTargHaps()}
     * @throws IndexOutOfBoundsException if
     * {@code start < 0 || end > impData.refGT().nMarkers()}
     * @throws NullPointerException if {@code impData==null || stateProbs==null}
     * @throws NullPointerException if any element of {@code stateProbs} is
     * {@code null}
     */
    public void printImputed(ImpData impData, int start, int end,
            AtomicReferenceArray<StateProbs> stateProbs) {
        checkInterval(start, end, impData.refGT().nMarkers());
        if (stateProbs.length() != impData.nTargHaps()) {
            throw new IllegalArgumentException("inconsistent data:");
        }
        UnsignedByteArray[] output = IntStream.range(0, impData.nClusters())
                .parallel()
                .mapToObj(c -> toByteArray(impData, start, end, stateProbs, c))
                .toArray(UnsignedByteArray[]::new);
        append(output, vcfOutFile);
    }

    private static UnsignedByteArray toByteArray(ImpData impData,
            int refStart, int refEnd,
            AtomicReferenceArray<StateProbs> stateProbs, int m) {
        ImputedVcfWriter ivw = new ImputedVcfWriter(impData, refStart, refEnd, m);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (PrintWriter out = new PrintWriter(
                new BGZIPOutputStream(baos, false))) {
            ivw.appendRecords(stateProbs, out);
        }
        return new UnsignedByteArray(baos);
    }

    /**
     * Appends the data in phased genotypes for the specified markers
     * to the output VCF file: {@code this.outPrefix() + ".vcf.gz"}.
     *
     * @param phasedTarg the estimated target haplotypes
     * @param start the starting marker index (inclusive)
     * @param end the ending marker index (exclusive)
     * @throws IllegalArgumentException if {@code phasedTarg.isPhased() == false}
     * @throws IndexOutOfBoundsException if
     * {@code start < 0 || end > phasedTarg.nMarkers() || start > end}
     * @throws NullPointerException if {@code phasedTarg == null}
     */
    public void printPhased(GT phasedTarg, int start, int end) {
        checkInterval(start, end, phasedTarg.nMarkers());
        long t0 = System.nanoTime();
        int blockSize = 50000;
        int stepSize = 100;
        int[] blockEnds = ends(start, end, blockSize);
        for (int j=1; j<blockEnds.length; ++j) {
            int[] stepEnds = ends(blockEnds[j-1], blockEnds[j], stepSize);
            UnsignedByteArray[] output = IntStream.range(1, stepEnds.length)
                    .parallel()
                    .mapToObj(i -> toByteArray(phasedTarg, stepEnds[i-1], stepEnds[i]))
                    .toArray(UnsignedByteArray[]::new);
            append(output, vcfOutFile);
        }
        long t1 = System.nanoTime();
        System.out.println("WindWriter tot: " + Utilities.elapsedNanos(t1-t0));
    }

    public void printPhased(Stage2Haps stage2Haps, int start, int end) {
        GT targGT = stage2Haps.fpd().targGT();
        checkInterval(start, end, targGT.nMarkers());
        long t0 = System.nanoTime();
        int blockSize = 20000;
        int stepSize = 50;
        int[] blockEnds = ends(start, end, blockSize);
        for (int j=1; j<blockEnds.length; ++j) {
            int blockStart = blockEnds[j-1];
            int blockEnd = blockEnds[j];
            GTRec[] recBlock = stage2Haps.toGTRecs(blockStart, blockEnd);
            int[] stepEnds = ends(blockStart, blockEnd, stepSize);
            UnsignedByteArray[] output = IntStream.range(1, stepEnds.length)
                    .parallel()
                    .mapToObj(i -> toByteArray(samples, Arrays.copyOfRange(recBlock,
                            (stepEnds[i-1]-blockStart), (stepEnds[i]-blockStart))))
                    .toArray(UnsignedByteArray[]::new);
            append(output, vcfOutFile);
        }
        long t1 = System.nanoTime();
        System.out.println("WindWriter tot: " + Utilities.elapsedNanos(t1-t0));
    }

    private static void checkInterval(int start, int end, int nMarkers) {
        if (start<0 || end>nMarkers || end<start) {
            throw new IllegalArgumentException("start=" + start + " end=" + end
                    + " nMarkers=" + nMarkers);
        }
    }

    private static int[] ends(int start, int end, int step) {
        IntList starts = new IntList(2 + ((end - start) / step));
        for (int m=start; m<end; m+=step) {
            starts.add(m);
        }
        starts.add(end);
        return starts.toArray();
    }

    private static UnsignedByteArray toByteArray(GT phasedTarg, int start, int end) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (PrintWriter vcfOut=new PrintWriter(
                new BGZIPOutputStream(baos, false))) {
            VcfWriter.appendRecords(phasedTarg, start, end, vcfOut);
        }
        return new UnsignedByteArray(baos);
    }

    private static UnsignedByteArray toByteArray(Samples samples, GTRec[] phasedRecs) {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (PrintWriter vcfOut=new PrintWriter(
                new BGZIPOutputStream(baos, false))) {
            VcfWriter.appendRecords(samples, phasedRecs, vcfOut);
        }
        return new UnsignedByteArray(baos);
    }

    private static void append(UnsignedByteArray[] output, File outFile)  {
        boolean append = true;
        try {
            try (OutputStream fos = new BufferedOutputStream(
                    new FileOutputStream(outFile, append))) {
                for (UnsignedByteArray uba : output) {
                    uba.write(fos);
                }
            }
        } catch (IOException e) {
            fileOutputError(outFile, e);
        }
    }

    @Override
    public void close() {
        boolean append = true;
        try {
            try (FileOutputStream fos = new FileOutputStream(vcfOutFile, append);
                    BufferedOutputStream bos = new BufferedOutputStream(fos);
                    BGZIPOutputStream bgzip = new BGZIPOutputStream(bos, true)) {
                // write empty BGZIP block to bgzip by closing bgzip
            }
        } catch (IOException e) {
            Utilities.exit(e, "Error closing file: " + vcfOutFile);
        }
    }

    private static void fileOutputError(File file, Exception e) {
        Utilities.exit(e, "Error writing to file: " + file);
    }
}
