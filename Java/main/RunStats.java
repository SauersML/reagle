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

import blbutil.Const;
import blbutil.FileUtil;
import blbutil.Utilities;
import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Optional;
import phase.FixedPhaseData;
import phase.PhaseData;
import vcf.Marker;
import vcf.Markers;
import vcf.RefGT;
import vcf.Window;

/**
 * <p>Class {@code RunStats} contains methods for storing and printing
 * statistics describing a Beagle analysis.</p>
 *
 * <p>Instances of class {@code RunStats} are not thread-safe.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class RunStats {

    private final Par par;
    private final PrintWriter log;
    private final long startNanos;

    private long totalPhaseNanos = 0;

    private long imputeNanos = 0;
    private long totalImputeNanos = 0;

    /**
     * Constructs a new {@code RunStats} instance.
     * @param par the analysis parameters
     * @throws NullPointerException if {@code par == null}
     */
    RunStats(Par par) {
        this.startNanos = System.nanoTime();
        this.par = par;
        this.log = FileUtil.printWriter(new File(par.out() + ".log"));
    }

    /**
     * Prints initial information about the analysis to a log
     * file and to standard output.
     */
    public void printStartInfo() {
        String[] argList = par.args();
        if (par.noNThreads()) {
            // add nthreads parameter to list of command line parameters
            int oldLength = argList.length;
            argList = Arrays.copyOf(argList, argList.length+1);
            argList[oldLength] = "nthreads=" + par.nthreads();
        }
        Utilities.duoPrint(log, Main.SHORT_HELP + Const.nl);
        Utilities.duoPrintln(log, "Start time: " + Utilities.timeStamp());
        Utilities.duoPrint(log, Utilities.commandLine(Main.PROGRAM, argList));
        if (par.ped() != null) {
            String s = Const.nl + "WARNING: This version will not model"
                    + " duos or trios in the pedigree file";
            Utilities.duoPrintln(log, s);
        }
        if (par.map() == null) {
            String s = Const.nl + "No genetic map is specified: using 1 cM = 1 Mb";
            Utilities.duoPrintln(log, s);
        }
        log.flush();
    }

    /**
     * Returns the analysis parameters.
     * @return the analysis parameters
     */
    public Par par() {
        return par;
    }

   /**
     * Prints information about the samples to a log
     * file and to standard output.
     * @param ped the pedigree data for the target samples
     * @param window the input data for the current marker window
     * @throws NullPointerException if
     * {@code this.par().ped() != null && ped == null}
     */
    public void printSampleSummary(Pedigree ped, Window window) {
        Optional<RefGT> optRefGT = window.refGT();
        int nRefSamples = optRefGT.isPresent() ? optRefGT.get().nSamples() : 0;
        Utilities.duoPrint(log, Const.nl);
        Utilities.duoPrint(log, String.format("Reference samples: %,20d%n",
                nRefSamples));
        Utilities.duoPrint(log, String.format("Study     samples: %,20d%n",
                 window.targGT().nSamples()));
        if (par.ped() != null) {
            Utilities.duoPrint(log, "  ");
            Utilities.duoPrint(log, String.valueOf(ped.nSingles()));
            Utilities.duoPrintln(log, " singles");
            Utilities.duoPrint(log, "  ");
            Utilities.duoPrint(log, String.valueOf(ped.nDuos()));
            Utilities.duoPrintln(log, " duos");
            Utilities.duoPrint(log, "  ");
            Utilities.duoPrint(log, String.valueOf(ped.nTrios()));
            Utilities.duoPrintln(log, " trios");
        }
        log.flush();
    }

   /**
     * Prints information about the marker window to a log
     * file and to standard output.
     * @param window input data for the next marker window
     * @param fpd the input data for phasing that is the same in each iteration
     * @throws NullPointerException if {@code (window == null) || (fpd == null)}
     */
    public void printWindowUpdate(Window window, FixedPhaseData fpd) {
        Markers targMarkers = fpd.targGT().markers();
        Markers stage1TargMarkers = fpd.stage1TargGT().markers();
        Markers markers = window.refGT().map(refGT -> refGT.markers()).orElse(targMarkers);
        Marker first = markers.marker(0);
        Marker last = markers.marker(markers.size() - 1);
        StringBuilder sb = new StringBuilder(30);
        sb.append(Const.nl);
        sb.append("Window ");
        sb.append(fpd.window());
        sb.append(" [");
        String chr = first.chrom();
        if (chr.equals(Const.MISSING_DATA_STRING)==false) {
            sb.append(chr);
            sb.append(Const.colon);
        }
        sb.append(first.pos());
        sb.append(Const.hyphen);
        if (chr.equals(last.chrom())==false) {
            sb.append(last.chrom());
            sb.append(Const.colon);
        }
        sb.append(last.pos());
        sb.append(']');
        sb.append(Const.nl);
        if (window.refGT().isPresent()) {
            sb.append(String.format("Reference markers: %,20d%n", markers.size()));
        }
        sb.append(String.format("Study     markers: %,20d%n", targMarkers.size()));
        if (stage1TargMarkers.size() != targMarkers.size()) {
            sb.append(String.format("Stage 1   markers: %,20d%n", stage1TargMarkers.size()));
        }
        Utilities.duoPrint(log, sb.toString());
        log.flush();
    }

    /**
     * Prints information about the complete analysis to a log
     * file and to standard output, and closes the log file.
     * @param nTargetMarkers the total number of target markers analyzed
     * @param nMarkers the total number of markers analyzed
     */
    public void printSummaryAndClose(int nTargetMarkers, int nMarkers) {
        long totalTime = System.nanoTime() - startNanos;
        Utilities.duoPrint(log, Const.nl);
        Utilities.duoPrintln(log, "Cumulative Statistics:" + Const.nl);
        if (nTargetMarkers != nMarkers) {
            Utilities.duoPrint(log,
                    String.format("Reference markers: %,20d%n", nMarkers));
        }
        Utilities.duoPrint(log,
                String.format("Study     markers: %,20d%n%n", nTargetMarkers));

        if (totalPhaseNanos > 1000) {
            duoPrintNanos("Haplotype phasing time:        ", totalPhaseNanos);
        }
        if (totalImputeNanos > 0) {
            duoPrintNanos("Imputation time:               ", totalImputeNanos);
        }
        duoPrintNanos(    "Total time:                    ", totalTime);
        Utilities.duoPrintln(log, Const.nl + "End time: "
                + Utilities.timeStamp());
        Utilities.duoPrintln(log, Main.PROGRAM + " finished");
        log.close();
    }

    /**
     * Increases the cumulative phasing time by the specified number of
     * nanoseconds.
     * @param nanos the elapsed nanoseconds for updating the haplotype
     * estimates
     */
    public void phaseNanos(long nanos) {
        totalPhaseNanos += nanos;
    }

    /**
     * Stores the time for imputing ungenotyped marker and increases
     * the cumulative imputation time by the specified number
     * of nanoseconds.
     * @param nanos the nanoseconds required to impute ungenotyped
     * markers
     */
    public void imputationNanos(long nanos) {
        imputeNanos = nanos;
        totalImputeNanos += nanos;
    }

    /**
     * Prints run time for most recent imputation to a log file
     * and to standard output.
     */
    public void printImputationUpdate() {
        Utilities.duoPrint(log, Const.nl);
        duoPrintNanos("Imputation time:               ", imputeNanos);
        log.flush();
    }

    /**
     * Prints the specified string to the log file and to standard out.
     * @param msg the message to be printed
     */
    public void println(String msg) {
        Utilities.duoPrintln(log, msg);
        log.flush();
    }

    /**
     * Prints information about the specified iteration, and adds the
     * specified elapsed nanoseconds to the total phasing time.
     * @param pd estimated phased genotypes at stage 1 markers
     * @param elapsedNanos the elapsed nanoseconds for the iteration
     */
    public void printStage1Info(PhaseData pd, long elapsedNanos) {
        if (pd.it()==par.burnin() && par.em()) {
            printEstimatedParameters(pd.ne(), pd.pMismatch());
        }
        phaseNanos(elapsedNanos);
        String msg;
        int it = pd.it();
        if (it < par.burnin()) {
            if (it==0) {
               println("");
            }
            msg = "Burnin  iteration " + (it+1) + ":"; // count from 1
        }
        else {
            it -= par.burnin();
            if (it==0) {
                println("");
            }
            msg = "Phasing iteration " + (it+1) + ":";  // count from 1
        }
        duoPrintNanos(String.format("%1$-31s", msg), elapsedNanos);
    }

    /**
     * Prints the specified elapsed nanoseconds for stage 2 phasing.
     * @param elapsedNanos the elapsed nanoseconds for stage 2 phasing
     */
    public void printStage2Info(long elapsedNanos) {
        phaseNanos(elapsedNanos);
        String msg = "Low frequency phasing:";
        duoPrintNanos(String.format("%1$-31s", msg), elapsedNanos);
    }

    /**
     * Prints the specified estimated effective population size.
     * @param ne the estimated effective population size
     * @param pMismatch the estimated allele mismatch parameter
     */
    public void printEstimatedParameters(long ne, float pMismatch) {
        Utilities.duoPrintln(log, "");
        Utilities.duoPrint(log, String.format("%1$-31s", "Estimated ne:"));
        Utilities.duoPrintln(log, String.valueOf(ne));
        Utilities.duoPrint(log, String.format("%1$-31s", "Estimated err:"));
        Utilities.duoPrintln(log, String.format("%1$7.1e", pMismatch));
    }

    /**
     * Print the specified message followed by the human
     * elapsed time as formatted by
     * {@code blbutil.Utilities.elapsedNanos(nanos)}
     * @param message the message to be printed
     * @param nanos the elapsed time in nanoseconds
     */
    public void duoPrintNanos(String message, long nanos) {
        Utilities.duoPrint(log, message);
        Utilities.duoPrintln(log, Utilities.elapsedNanos(nanos));
    }
}
