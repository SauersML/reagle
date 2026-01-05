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

import beagleutil.ChromIds;

/**
 * <p>Class {@code PositionMap} represents a genetic map obtained by
 * multiplying chromosome position by a scale factor.
 * </p>
 * <p>Instances of class {@code PositionMap} are immutable.
 * </p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class PositionMap implements GeneticMap {

    private final double scaleFactor;
    private final double invScaleFactor;

    /**
     * Returns the scale factor that is multiplied by the chromosome position
     * to obtain the corresponding genetic map position
     * @return the scale factor.
     */
    public double scaleFactor() {
        return scaleFactor;
    }

    /**
     * Constructs a new {@code PositionMap} instance.
     * @param scaleFactor the factor that is multiplied by
     * a base position to obtain the corresponding genetic map position
     * @throws IllegalArgumentException if
     * {@code scaleFactor <= 0d || Double.isFinite(scaleFactor) == false}
     */
    public PositionMap(double scaleFactor) {
        if (Double.isFinite(scaleFactor) == false || scaleFactor <= 0d) {
            throw new IllegalArgumentException(String.valueOf(scaleFactor));
        }
        this.scaleFactor = scaleFactor;
        this.invScaleFactor = 1.0/scaleFactor;
    }

    @Override
    public int basePos(int chrom, double geneticPosition) {
        if (chrom < 0 || chrom >= ChromIds.instance().size()) {
            throw new IndexOutOfBoundsException(String.valueOf(chrom));
        }
        long pos = Math.round(geneticPosition*invScaleFactor);
        if (pos>Integer.MAX_VALUE) {
            String s = "An estimated base position exceeds the maximum integer value"
                    + blbutil.Const.nl
                    + "Is the window parameter in cM units?";
            throw new IllegalArgumentException(s);
        }
        return (int) pos;
    }

    @Override
    public double genPos(Marker marker) {
        return scaleFactor*marker.pos();
    }

    @Override
    public double genPos(int chrom, int basePosition) {
        if (chrom < 0 || chrom >= ChromIds.instance().size()) {
            throw new IndexOutOfBoundsException(String.valueOf(chrom));
        }
        return scaleFactor*basePosition;
    }
}
