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

/**
 * <p>Interface {@code PhaseBaum} updates the estimated genotype phase 
 * of specified samples.
 * </p>
 * <p>Instances of classes that implement {@code PhaseBaum} are not 
 * required to be thread-safe.
 * </p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public interface PhaseBaum {

    /**
     * Returns the number of target samples.
     * @return the number of target samples
     */
    int nTargSamples();

    /**
     * Estimates and stores the phased haplotypes for the specified sample
     * @param sample a sample index
     * @throws IndexOutOfBoundsException if
     * {@code sample < 0 || sample >= this.nTargSamples()}
     */
    void phase(int sample);
    
}
