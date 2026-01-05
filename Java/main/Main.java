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
import blbutil.Utilities;
import imp.ImpData;
import imp.ImpLS;
import imp.StateProbs;
import java.io.File;
import java.util.Locale;
import java.util.Optional;
import java.util.Random;
import java.util.concurrent.atomic.AtomicReferenceArray;
import phase.FixedPhaseData;
import phase.PhaseData;
import phase.PhaseLS;
import phase.Stage2Haps;
import phase.SwapRate;
import vcf.BasicGT;
import vcf.GT;
import vcf.MarkerIndices;
import vcf.RefTargSlidingWindow;
import vcf.SlidingWindow;
import vcf.TargSlidingWindow;
import vcf.Window;
import vcf.XRefGT;

/**
 * Class {@code Main} is the entry class for the Beagle program. See
 * {@code Par.usage()} and online program documentation for usage instructions.
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class Main {

    /**
     * The program name and version.
     */
    private static final String VERSION = "(version 5.5)";

    /**
     * The program name and commit version.
     */
    public static final String PROGRAM = "beagle.27Feb25.75f.jar";

    /**
     * The command to invoke the program.
     */
    public static final String COMMAND = "java -jar beagle.27Feb25.75f.jar";

    /**
     * The copyright string.
     */
    public static final String COPYRIGHT = "Copyright (C) 2014-2024 Brian L. Browning";

    /**
     * The program name and a brief help message.
     */
    public static final String SHORT_HELP = Main.PROGRAM + " " + VERSION
            + Const.nl + Main.COPYRIGHT
            + Const.nl + "Enter \"" + COMMAND
            + "\" to list command line argument";

    private final Par par;
    private final SlidingWindow slidingWind;
    private final RunStats runStats;
    private final WindowWriter windowWriter;
    private final Random rand;  // generates distinct seed for each window

    /**
     * Entry point to Beagle program. See {@code Parameters.usage()} and program
     * documentation description of valid arguments.
     *
     * @param args command line arguments
     */
    public static void main(String[] args) {
        Locale.setDefault(Locale.US);
        if (args.length == 0) {
            System.out.println(PROGRAM + " " + VERSION);
            System.out.println(COPYRIGHT);
            System.out.println(Par.usage());
            System.exit(0);
        }
        Par par = parameters(args);
        System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism",
                String.valueOf(par.nthreads()));
        RunStats runStats = new RunStats(par);
        runStats.printStartInfo();

        try (SlidingWindow slidingWind = slidingWindow(par);
                WindowWriter winOut = new WindowWriter(par, slidingWind.targSamples())) {
            Main main = new Main(par, slidingWind, winOut, runStats);
            main.phaseAndImpute();
            runStats.printSummaryAndClose(slidingWind.cumTargMarkers(),
                    slidingWind.cumMarkers());
        }
    }

    private Main(Par par, SlidingWindow slidingWindow, WindowWriter windWriter,
            RunStats runStats) {
        this.par = par;
        this.slidingWind = slidingWindow;
        this.runStats = runStats;
        this.windowWriter = windWriter;
        this.rand = new Random(par.seed());
    }

    private static SlidingWindow slidingWindow(Par par) {
        if (par.ref() == null) {
            return TargSlidingWindow.instance(par);
        } else {
            return RefTargSlidingWindow.instance(par);
        }
    }

    private void phaseAndImpute() {
        Optional<Window> optWindow = slidingWind.nextWindow();
        printSampleSummary(optWindow);
        GT overlap = null;
        while (optWindow.isPresent()) {
            Window window = optWindow.get();
            FixedPhaseData fpd = new FixedPhaseData(par, slidingWind.ped(),
                window, overlap);
            runStats.printWindowUpdate(window, fpd);
            PhaseData pd = new PhaseData(fpd, rand.nextLong());
            if (fpd.targGT().isPhased()) {
                XRefGT phasedTarg = XRefGT.fromPhasedGT(fpd.targGT(), par.nthreads());
                overlap = printWindow(window, phasedTarg);
            }
            else {
                phaseStage1Variants(pd);
                if (fpd.stage1TargGT().nMarkers()==fpd.targGT().nMarkers()) {
                    XRefGT phasedTarg = pd.estPhase().phasedHaps();
                    overlap = printWindow(window, phasedTarg);
                }
                else {
                    Stage2Haps stage2Haps = phaseStage2Variants(pd);
                    overlap = printWindow(window, stage2Haps);
                }
            }
            optWindow = slidingWind.nextWindow();
        }
        slidingWind.close();
    }

    private void printSampleSummary(Optional<Window> optWindow) {
        if (optWindow.isPresent()) {
            runStats.printSampleSummary(slidingWind.ped(), optWindow.get());
        }
    }

    private void phaseStage1Variants(PhaseData pd) {
        int nIts = par.burnin() + par.iterations();
        double maxBurninSwapRate = 0.01;
        while (pd.it()<nIts) {
            long t0 = System.nanoTime();
            PhaseLS.runStage1(pd);
            runStats.printStage1Info(pd, (System.nanoTime() - t0));
            pd.incrementIt();
            double swapRate = SwapRate.getAndResetSwapRate();
            if (pd.it()<par.burnin() && swapRate<=maxBurninSwapRate) {
                pd.advanceToFirstPhasingIt();
            }
        }
    }

    private Stage2Haps phaseStage2Variants(PhaseData pd) {
        long t0 = System.nanoTime();
        Stage2Haps stage2Haps = PhaseLS.runStage2(pd);
        runStats.printStage2Info(System.nanoTime() - t0);
        return stage2Haps;
    }

    private XRefGT printWindow(Window window, XRefGT phasedTarg) {
        boolean impute = window.indices().nMarkers() != window.indices().nTargMarkers();
        if (impute==false) {
            int mStart = window.indices().prevTargSplice();
            int mEnd = window.indices().nextTargSplice();
            windowWriter.printPhased(phasedTarg, mStart, mEnd);
            return phasedOverlap(window, phasedTarg);
        }
        else {
            long t0 = System.nanoTime();
            ImpData impData = new ImpData(par, window, phasedTarg, window.genMap());
            AtomicReferenceArray<StateProbs> stateProbs = ImpLS.stateProbs(impData);
            int mStart = window.indices().prevSplice();
            int mEnd = window.indices().nextSplice();
            windowWriter.printImputed(impData, mStart, mEnd, stateProbs);
            runStats.imputationNanos(System.nanoTime() - t0);
            runStats.printImputationUpdate();
            return phasedOverlap(window, phasedTarg);
        }
    }

    private GT printWindow(Window window, Stage2Haps stage2Haps) {
        boolean impute = window.indices().nMarkers() != window.indices().nTargMarkers();
        if (impute==false) {
            int mStart = window.indices().prevTargSplice();
            int mEnd = window.indices().nextTargSplice();
            windowWriter.printPhased(stage2Haps, mStart, mEnd);
            return stage2Haps.toBasicGT(window.indices().targOverlapStart(), mEnd);
        } else {
            long t0 = System.nanoTime();
            BasicGT phasedHapMajor = stage2Haps.toBasicGT(0, window.targGT().nMarkers());
            XRefGT phasedTarg = XRefGT.fromPhasedGT(phasedHapMajor, par.nthreads());
            ImpData impData = new ImpData(par, window, phasedTarg, window.genMap());
            AtomicReferenceArray<StateProbs> stateProbs = ImpLS.stateProbs(impData);
            int mStart = window.indices().prevSplice();
            int mEnd = window.indices().nextSplice();
            windowWriter.printImputed(impData, mStart, mEnd, stateProbs);
            runStats.imputationNanos(System.nanoTime() - t0);
            runStats.printImputationUpdate();
            return phasedOverlap(window, phasedTarg);
        }
    }

    private XRefGT phasedOverlap(Window window, XRefGT phasedTarg) {
        assert phasedTarg.isPhased();
        MarkerIndices markerIndices = window.indices();
        int nextOverlap = markerIndices.targOverlapStart();
        int nextSplice = markerIndices.nextTargSplice();
        int nMarkers = nextSplice - nextOverlap;
        return nMarkers==0 ? null : phasedTarg.restrict(nextOverlap, nextSplice);
    }

    /*
     * Checks that certain parameters are consistent, and prints error
     * message and exits if parameters are inconsistent.
     *
     * @param args the command line arguments.
     */
    private static Par parameters(String[] args) {
        // warnings are printed in RunStats.startInfo() method
        Par par = new Par(args);
        checkOutputPrefix(par);
        if (par.window() < 1.1 * par.overlap()) {
            String s = SHORT_HELP + Const.nl
                    + Const.nl + "ERROR: The \"window\" parameter must be at least "
                    + "1.1 times the \"overlap\" parameter"
                    + Const.nl + "Exiting program.";
            Utilities.exit(s);
        }
        return par;
    }

    private static void checkOutputPrefix(Par par) {
        File outPrefix = new File(par.out());
        if (outPrefix.isDirectory()) {
            String s = "ERROR: \"out\" parameter cannot be a directory: \""
                    + par.out() + "\"";
            Utilities.exit(Par.usage() + s);
        }

        File vcfOut = new File(par.out() + ".vcf.gz");
        if (vcfOut.equals(par.ref())) {
            String s = "ERROR: VCF output file equals input file: " + par.ref();
            Utilities.exit(Par.usage() + s);
        }
        if (vcfOut.equals(par.gt())) {
            String s = "ERROR: VCF output file equals input file: " + par.gt();
            Utilities.exit(Par.usage() + s);
        }
    }
}
