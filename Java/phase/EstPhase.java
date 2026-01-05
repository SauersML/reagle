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

import java.util.concurrent.atomic.AtomicReferenceArray;
import vcf.BitArrayRefGTRec;
import vcf.XRefGT;

/**
 * <p>Class {@code EstPhase} stores input genotype data and the
 * current estimated phased genotypes for each target sample.</p>
 *
 * <p>Instances of class {@code EstPhase} are thread-safe.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class EstPhase {

    private final FixedPhaseData fpd;
    private final AtomicReferenceArray<SamplePhase> phase;

    /**
     * Constructs a new {@code EstPhase} instance from the specified data.
     * @param fpd the input data for phasing
     * @param seed the seed for random number generation
     * @throws NullPointerException if {@code fpd == null}
     */
    public EstPhase(FixedPhaseData fpd, long seed) {
        this.fpd = fpd;
        this.phase = PbwtPhaser.initPhase(fpd, seed);
    }

    /**
     * Returns the input data for phasing that is the same in each iteration.
     * @return the input data for phasing that is the same in each iteration
     */
    public FixedPhaseData fpd() {
        return fpd;
    }

    /**
     * Sets the specified phased genotypes for the specified sample.
     * @param  sample the sample index
     * @param samplePhase the estimated phased genotypes
     * @throws IndexOutOfBoundsException if
     * {@code sample < 0 || sample >= this.fpd().stage1TargGT().nSamples()}
     * @throws NullPointerException if {@code samplePhase == null}
     */
    public void set(int sample, SamplePhase samplePhase) {
        if (samplePhase==null) {
            throw new NullPointerException(SamplePhase.class.toString());
        }
        phase.set(sample, samplePhase);
    }

    /**
     * Return the estimated phase for the specified sample.
     * @param sample a sample index
     * @return the estimated phase for the specified sample
     * @throws IndexOutOfBoundsException if
     * {@code sample < 0 || sample >= this.fpd().targGT().nSamples()}
     */
    public SamplePhase get(int sample) {
        return phase.get(sample);
    }

    /**
     * Returns the current estimated phased genotypes for the target samples.
     * @return the current estimated phased genotypes for the target samples
     */
    public XRefGT phasedHaps() {
        return XRefGT.from(fpd.stage1TargGT().samples(), phase);
    }

    /**
     * Returns the current estimated phased genotypes for the target samples.
     * @return the current estimated phased genotypes for the target samples
     */
    public BitArrayRefGTRec[] toGTRecs() {
        return BitArrayRefGTRec.toBitArrayRefGTRecs(this);
    }
}
