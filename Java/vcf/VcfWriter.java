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

import blbutil.Const;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.Calendar;
import java.util.stream.IntStream;

/**
 * <p>Class {@code VcfWriter} contains static methods for writing data in
 * VCF 4.2 format.
 * </p>
 * <p>Instances of class {@code VcfWriter} are not thread-safe.
 * </p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class VcfWriter {

    private static final String FILE_FORMAT = "##fileformat=VCFv4.2";

    private static final String AF_INFO = "##INFO=<ID=AF,Number=A,Type=Float,"
            + "Description=\"Estimated ALT Allele Frequencies\">";
    private static final String DR2_INFO = "##INFO=<ID=DR2,Number=A,Type=Float,"
            + "Description=\"Dosage R-Squared: estimated squared correlation between "
            + "estimated REF dose [P(RA) + 2*P(RR)] and true REF dose\">";
    private static final String IMP_INFO = "##INFO=<ID=IMP,Number=0,Type=Flag,"
            + "Description=\"Imputed marker\">";

    private static final String GT_FORMAT = "##FORMAT=<ID=GT,Number=1,Type=String,"
            + "Description=\"Genotype\">";
    private static final String DS_FORMAT = "##FORMAT=<ID=DS,Number=A,Type=Float,"
            +"Description=\"estimated ALT dose [P(RA) + 2*P(AA)]\">";
    private static final String AP1_FORMAT = "##FORMAT=<ID=AP1,Number=A,Type=Float,"
            +"Description=\"estimated ALT dose on first haplotype\">";
    private static final String AP2_FORMAT = "##FORMAT=<ID=AP2,Number=A,Type=Float,"
            +"Description=\"estimated ALT dose on second haplotype\">";
    private static final String GL_FORMAT = "##FORMAT=<ID=GL,Number=G,Type=Float,"
            + "Description=\"Log10-scaled Genotype Likelihood\">";
    private static final String GP_FORMAT = "##FORMAT=<ID=GP,Number=G,Type=Float,"
            + "Description=\"Estimated Genotype Probability\">";

    private static final String SHORT_CHROM_PREFIX= "#CHROM" + Const.tab + "POS"
            + Const.tab + "ID" + Const.tab + "REF" + Const.tab + "ALT"
            + Const.tab + "QUAL" + Const.tab + "FILTER" + Const.tab + "INFO";

    private static final String LONG_CHROM_PREFIX =
            SHORT_CHROM_PREFIX + Const.tab + "FORMAT";


    private VcfWriter() {
        // private constructor prevents instantiation
    }

    /**
     * Writes VCF meta-information lines and header line to the specified
     * {@code PrintWriter}. Only one FORMAT subfield, the GT subfield,
     * is described in the meta-information lines.
     * @param sampleIds the sample identifiers
     * @param source a description of the data source, or {@code null} if
     * no description is to be printed
     * @param out the {@code PrintWriter} to which VCF meta-information
     * lines will be written
     * @throws NullPointerException if {@code out == null}
     * @throws NullPointerException if
     * {@code sampleIds == null}, or if {@code sampleIds[j] == null} for any
     * {@code j} satisfying {@code (0 <= j && j < <sampleIds.length)}
     */
    public static void writeMetaLinesGT(String[] sampleIds, String source,
            PrintWriter out) {
        boolean ds = false;
        boolean ap = false;
        boolean gp = false;
        boolean gl = false;
        writeMetaLines(sampleIds, source, ds, ap, gp, gl, out);
    }

    /**
     * Writes VCF meta-information lines and header line to the specified
     * {@code PrintWriter}.
     * @param sampleIds the sample identifiers
     * @param source a description of the data source, or {@code null} if
     * no description is to be printed
     * @param ds{ @code true} if the meta-information lines
     * will describe the DS FORMAT subfield and {@code false} otherwise
     * @param ap {@code true} if the meta-information lines
     * will describe the AP1 and AP2 FORMAT subfields and {@code false} otherwise
     * @param gp {@code true} if the meta-information lines
     * will describe the GP FORMAT subfield and {@code false} otherwise
     * @param gl {@code true} if the meta-information lines
     * will describe the GL FORMAT subfield and {@code false} otherwise
     * @param out the {@code PrintWriter} to which VCF meta-information lines
     * will be written.
     * @throws NullPointerException if {@code out == null}
     * @throws NullPointerException if
     * {@code sampleIds == null}, or if {@code sampleIds[j] == null} for any
     * {@code j} satisfying {@code (0 <= j && j < sampleIds.length)}
     */
    public static void writeMetaLines(String[] sampleIds, String source,
            boolean ds, boolean ap, boolean gp, boolean gl,
            PrintWriter out) {
        out.print(FILE_FORMAT);
        out.print(Const.nl);
        out.print("##filedate=");
        out.print(now());
        out.print(Const.nl);
        if (source != null) {
            out.print("##source=\"");
            out.print(source);
            out.println("\"");
        }
        if (ds) {
            out.println(AF_INFO);
            out.println(DR2_INFO);
            out.println(IMP_INFO);
        }
        out.println(GT_FORMAT);
        if (ds) {
            out.println(DS_FORMAT);
        }
        if (ap) {
            out.println(AP1_FORMAT);
            out.println(AP2_FORMAT);
        }
        if (gp) {
            out.println(GP_FORMAT);
        }
        if (gl) {
            out.println(GL_FORMAT);
        }
        out.print(LONG_CHROM_PREFIX);
        for (String id : sampleIds) {
            if (id==null) {
                throw new NullPointerException("id==null");
            }
            out.print(Const.tab);
            out.print(id);
        }
        out.println();
    }

    private static String now() {
        String dateFormat = "yyyyMMdd";
        Calendar cal = Calendar.getInstance();
        SimpleDateFormat sdf = new SimpleDateFormat(dateFormat);
        return sdf.format(cal.getTime());
    }

    /**
     * Writes the data in phased genotypes for the specified markers
     * to the specified {@code PrintWriter}.
     * @param phasedTarg the estimated haplotype allele probabilities
     * @param start the starting marker index (inclusive)
     * @param end the ending marker index (exclusive)
     * @param out the {@code PrintWriter} to which VCF records will be written
     * @throws IllegalArgumentException if
     * {@code phasedTarg.isPhased() == false}
     * @throws IndexOutOfBoundsException if
     * {@code (start < 0 || start > end || end > phasedTarg.nMarkers())}
     * @throws NullPointerException if
     * {@code phasedTarg == null || out == null}
     */
    public static void appendRecords(GT phasedTarg, int start, int end,
            PrintWriter out) {
        if (start > end) {
            throw new IllegalArgumentException("start=" + start + " end=" + end);
        }
        if (phasedTarg.isPhased()==false) {
            throw new IllegalArgumentException("unphased genotypes");
        }
        Samples samples = phasedTarg.samples();
        Markers markers = phasedTarg.markers();
        VcfRecBuilder[] recBuilders = recBuilders(markers, samples.size(), start, end);
        for (int s=0, n=phasedTarg.nSamples(); s<n; ++s) {
            if (samples.isDiploid(s)) {
                int h1 = s<<1;
                int h2 = h1 | 0b1;
                for (int m=start; m<end; ++m) {
                    int a1 = phasedTarg.allele(m, h1);
                    int a2 = phasedTarg.allele(m, h2);
                    recBuilders[m-start].addSampleData(a1, a2);
                }
            }
            else {
                for (int m=start; m<end; ++m) {
                    int a1 = phasedTarg.allele(m, (s<<1));
                    recBuilders[m-start].addSampleData(a1);
                }
            }
        }
        for (VcfRecBuilder vrb : recBuilders) {
            vrb.writeRec(out);
        }
    }

    /**
     * Writes the data in phased genotypes for the specified markers
     * to the specified {@code PrintWriter}.
     * @param samples the list of samples
     * @param recs the estimated haplotype allele probabilities
     * @param out the {@code PrintWriter} to which VCF records will be written
     * @throws IllegalArgumentException if
     * {@code phasedTarg.isPhased() == false}
     * @throws IndexOutOfBoundsException if
     * {@code (start < 0 || start > end || end > phasedTarg.nMarkers())}
     * @throws NullPointerException if
     * {@code phasedTarg == null || out == null}
     */
    public static void appendRecords(Samples samples, GTRec[] recs,
            PrintWriter out) {
        VcfRecBuilder[] recBuilders = recBuilders(samples, recs);
        int nSamples = samples.size();
        for (int s=0; s<nSamples; ++s) {
            if (samples.isDiploid(s)) {
                int h1 = s<<1;
                int h2 = h1 | 0b1;
                for (int j=0; j<recBuilders.length; ++j) {
                    int a1 = recs[j].get(h1);
                    int a2 = recs[j].get(h2);
                    recBuilders[j].addSampleData(a1, a2);
                }
            }
            else {
                for (int j=0; j<recBuilders.length; ++j) {
                    int a1 = recs[j].get(s<<1);
                    recBuilders[j].addSampleData(a1);
                }
            }
        }
        for (VcfRecBuilder vrb : recBuilders) {
            vrb.writeRec(out);
        }
    }

    private static VcfRecBuilder[] recBuilders(Samples samples, GTRec[] recs) {
        int nSamples = samples.size();
        VcfRecBuilder[] builders = new VcfRecBuilder[recs.length];
        for (int j=0; j<recs.length; ++j) {
            if (recs[j].isPhased()==false) {
                throw new IllegalArgumentException("unphased genotypes");
            }
            else if (recs[j].samples().equals(samples)==false) {
                throw new IllegalArgumentException("inconsistent samples");
            }
            builders[j] = new VcfRecBuilder(recs[j].marker(), nSamples);
        }
        return builders;
    }

    private static VcfRecBuilder[] recBuilders(Markers markers, int nSamples,
            int start, int end) {
        return IntStream.range(start, end)
                .mapToObj(m -> new VcfRecBuilder(markers.marker(m), nSamples))
                .toArray(VcfRecBuilder[]::new);
    }

    /**
     * Prints the first 9 VCF record fields for the specified marker to
     * the specified {@code PrintWriter}.  Only one VCF FORMAT subfield,
     * the GT subfield, is printed.
     *
     * @param marker a marker
     * @param out the {@code PrintWriter} to which the first 9 VCF record
     * fields will be written
     *
     * @throws NullPointerException if {@code marker == null || out == null}
     */
    public static void printFixedFieldsGT(Marker marker, PrintWriter out) {
        out.print(marker);
        out.print(Const.tab);
        out.print(Const.MISSING_DATA_CHAR); // QUAL
        out.print(Const.tab);
        out.print("PASS");                  // FILTER
        out.print(Const.tab);
        out.print(Const.MISSING_DATA_CHAR); // INFO
        out.print(Const.tab);
        out.print("GT");
    }
}
