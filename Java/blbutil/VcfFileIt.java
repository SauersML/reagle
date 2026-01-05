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

import vcf.VcfHeader;

/**
 * <p>An iterator for records in a VCF file.  Each record contains
 * data for the same set of samples.
 *</p>
 * Instances of class {@code VcfFileIt} are not thread-safe.
 *
 * @param <E> the type of the elements returned by this iterator's
 * {@code next()} method.
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public interface VcfFileIt<E> extends SampleFileIt<E> {

    /**
     * Returns the VCF meta-information lines and header line
     * @return the VCF meta-information lines and header line
     */
    VcfHeader vcfHeader();
}
