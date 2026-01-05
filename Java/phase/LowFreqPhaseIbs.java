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

import blbutil.Utilities;
import vcf.XRefGT;

/**
 * <p>Class {@code LowFreqPhaseIbs} identifies haplotypes that share a long
 * IBS segment or a low frequency variant with a specified haplotype
 * in a specified genomic interval.</p>
 *
 * <p>Instances of {@code LowFreqPhaseIbs} are immutable.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class LowFreqPhaseIbs {

    private final PhaseData phaseData;

    private final XRefGT allHaps;
    private final LowFreqPbwtPhaseIbs fwdPhaseIbs;
    private final LowFreqPbwtPhaseIbs bwdPhaseIbs;

    /**
     * Constructs a new {@code LowFreqPhaseIbs} object from the specified data.
     *
     * @param phaseData the current genotype phase estimates and parameter
     * values
     *
     * @throws NullPointerException if {@code phaseData == null}
     */
    public LowFreqPhaseIbs(PhaseData phaseData) {
        CodedSteps codedSteps = phaseData.codedSteps();
        this.phaseData = phaseData;
        this.allHaps = codedSteps.allHaps();
        this.fwdPhaseIbs = new LowFreqPbwtPhaseIbs(phaseData, codedSteps, false);
        this.bwdPhaseIbs = new LowFreqPbwtPhaseIbs(phaseData, codedSteps, true);
    }

    /**
     * Returns the input data for the next phase update.
     * @return the input data for the next phase update
     */
    public PhaseData phaseData() {
        return phaseData;
    }

    /**
     * Returns the estimated phased genotypes for the target and reference
     * samples.
     * @return the estimated phased genotypes for the target and reference
     * samples
     */
    public XRefGT allHaps() {
        return allHaps;
    }

    /**
     * Returns the IBS haplotype index for the specified haplotype in
     * the specified genomic interval, when the PBWT is performed in
     * order of increasing marker index.An index of {@code -1} is stored
     * in an entry if no IBS haplotype is available.
     * @param targHap a target haplotype index
     * @param step an index of a genomic interval
     * @return the IBS haplotype index from a forward analysis
     * @throws IndexOutOfBoundsException if
     * {@code targHap < 0 || targHap >= this.phaseData().targGT().nHaps()}
     * @throws IndexOutOfBoundsException if
     * {@code step < 0 || step >= this.phaseData().fpd().stage1Steps().size()}
     */
    public int fwdIbsHap(int targHap, int step) {
        return fwdPhaseIbs.ibsHap(targHap, step);
    }

    /**
     * Returns the IBS haplotype index for the specified haplotype in
     * the specified genomic interval, when the PBWT is performed in
     * order of decreasing marker index. An index of {@code -1} is stored
     * in an entry if no IBS haplotype is available.
     * @param targHap a target haplotype index
     * @param step an index of a genomic interval
     * @return the IBS haplotype index from a backward analysis
     * @throws IndexOutOfBoundsException if
     * {@code targHap < 0 || targHap >= this.phaseData().targGT().nHaps()}
     * @throws IndexOutOfBoundsException if
     * {@code step < 0 || step >= this.phaseData().fpd().stage1Steps().size()}
     */
    public int bwdIbsHap(int targHap, int step) {
        return bwdPhaseIbs.ibsHap(targHap, step);
    }
}
