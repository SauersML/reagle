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
package beagleutil;

/**
 * <p>Class {@code CompHapSegment} represents a copied haplotype segment
 * in a composite reference haplotype.</p>
 *
 * <p>Instances of class {@code CompHapSegment} are not thread-safe.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class CompHapSegment implements Comparable<CompHapSegment> {

    private int hap;
    private int startMarker;
    private int lastIbsStep;
    private final int compHapIndex;

    /**
     * Constructs a new {@code CompHapSegment} from the specified data.
     * @param hap the haplotype
     * @param startMarker the index of the first marker in the haplotype segment
     * @param ibsStep the last recorded IBS step
     * @param compHapIndex the composite haplotype index
     */
    public CompHapSegment(int hap, int startMarker, int ibsStep, int compHapIndex) {
        this.hap = hap;
        this.startMarker = startMarker;
        this.lastIbsStep = ibsStep;
        this.compHapIndex = compHapIndex;
    }

    /**
     * Update the haplotype, the first marker in the haplotype segment,
     * and the last recorded IBS step.
     * @param hap the haplotype
     * @param startMarker the first marker in the haplotype segment
     * @param lastIbsStep the last recorded IBS step
     */
    public void updateSegment(int hap, int startMarker, int lastIbsStep) {
        this.hap = hap;
        this.startMarker = startMarker;
        this.lastIbsStep = lastIbsStep;
    }

    /**
     * Updates the last recorded IBS step to the specified value
     * @param ibsStep the last recorded IBS Step
     */
    public void setLastIbsStep(int ibsStep) {
        this.lastIbsStep = ibsStep;
    }

    /**
     * Returns the haplotype.
     * @return the haplotype
     */
    public int hap() {
        return hap;
    }

    /**
     * Returns the first marker in the haplotype segment
     * @return the first marker in the haplotype segment
     */
    public int startMarker() {
        return startMarker;
    }

    /**
     * Returns the last recorded IBS step for {@code this.hap()}.
     * @return the last recorded IBS step for {@code this.hap()}
     */
    public int lastIbsStep() {
        return lastIbsStep;
    }

    /**
     * Returns the composite haplotype index.
     * @return the composite haplotype index
     */
    public int compHapIndex() {
        return compHapIndex;
    }

    /**
     * Compares the specified segment to {@code this} for order.  Returns
     * -1, 0, or 1 according to whether {@code this.end()} is less than,
     * equal, or greater than {@code seg.end()}.
     * @param seg the object to be compared
     * @return -1, 0, or 1 according to whether {@code this.end()} is less
     * than, equal, or greater than {@code seg.end()}
     */
    @Override
    public int compareTo(CompHapSegment seg) {
        if (this.lastIbsStep!=seg.lastIbsStep) {
            return this.lastIbsStep<seg.lastIbsStep ? -1 : 1;
        } else {
            return 0;
        }
    }

}
