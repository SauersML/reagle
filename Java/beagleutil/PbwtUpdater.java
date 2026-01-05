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

import ints.IntArray;
import ints.IntList;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * <p>Class {@code PbwtDivUpdater} updates prefix arrays using the positional
 * Burrows-Wheeler transform (PBWT).</p>
 *
 * <p>Instances of {@code PbwtDivUpdater} are not thread-safe.</p>
 *
 * <p>Reference: Durbin, Richard (2014) Efficient haplotype matching and storage
 *    using the positional Burrows-Wheeler transform (PBWT).
 *    Bioinformatics 30(9):166-1272. doi: 10.1093/bioinformatics/btu014</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public class PbwtUpdater {

    private final int nHaps;
    private IntList[] a; // data for updated prefix array

    /**
     * Constructs a new {@code PbwtUpdater} instance for the specified data.
     * @param nHaps the number of haplotypes at each position
     * @throws IllegalArgumentException if {@code nHaps < 0}
     */
    public PbwtUpdater(int nHaps) {
        if (nHaps<0) {
            throw new IllegalArgumentException(String.valueOf(nHaps));
        }
        int initNumAlleles = 4;
        this.nHaps = nHaps;
        this.a = IntStream.range(0, initNumAlleles)
                .mapToObj(i -> new IntList())
                .toArray(IntList[]::new);
    }

    /**
     * Returns the number of haplotypes.
     * @return the number of haplotypes
     */
    public int nHaps() {
        return nHaps;
    }

    /**
     * Update the specified prefix and divergence arrays using the forward
     * Positional Burrows-Wheeler Transform. The contract for this method is
     * undefined if the specified {@code prefix} array is not a permutation of
     * {@code 0, 1, 2, ..., (nHaps - 1)}.
     *
     * @param rec the haplotype alleles
     * @param nAlleles the number of alleles
     * @param prefix the prefix array
     *
     * @throws IllegalArgumentException if {@code nAlleles < 1}
     * @throws IllegalArgumentException if
     * {@code rec.size() != this.nHaps() || prefix.length != this.nHaps()}
     * @throws IndexOutOfBoundsException if
     * {@code (prefix[j] < 0 || prefix[j] >= rec.size()}
     * for any {@code j} satisfying {@code (0 <= j && j < prefix.length)}
     * @throws IndexOutOfBoundsException if
     * {@code (rec.get[j] < 0 || rec.get(j) >= nAlleles)}
     * for any {@code j} satisfying {@code (0 <= j && j < this.nHaps())}
     * @throws NullPointerException if {@code rec == null || prefix == null}
     */
    public void update(IntArray rec, int nAlleles, int[] prefix) {
        if (rec.size()!=nHaps) {
            throw new IllegalArgumentException(String.valueOf(rec.size()));
        }
        if (prefix.length!=nHaps) {
            throw new IllegalArgumentException(String.valueOf(prefix.length));
        }
        initializeArrays(nAlleles);
        for (int h : prefix) {
            int allele = rec.get(h);
            if (allele>=nAlleles) {
                throw new IndexOutOfBoundsException(String.valueOf(allele));
            }
            a[allele].add(h);
        }
        updatePrefix(nAlleles, prefix);
    }

    /**
     * Update the specified prefix and divergence arrays using the forward
     * Positional Burrows-Wheeler Transform. The contract for this method is
     * undefined if the specified {@code prefix} array is not a permutation of
     * {@code 0, 1, 2, ..., (nHaps - 1)}.
     *
     * @param alleles the haplotype alleles
     * @param nAlleles the number of alleles
     * @param prefix the prefix array
     *
     * @throws IllegalArgumentException if {@code nAlleles < 1}
     * @throws IllegalArgumentException if
     * {@code alleles.length != this.nHaps() || prefix.length != this.nHaps()}
     * @throws IndexOutOfBoundsException if
     * {@code (prefix[j] < 0 || prefix[j] >= rec.size()}
     * for any {@code j} satisfying {@code (0 <= j && j < prefix.length)}
     * @throws IndexOutOfBoundsException if
     * {@code (alleles[j] < 0 || alleles[j] >= nAlleles)}
     * for any {@code j} satisfying {@code (0 <= j && j < this.nHaps())}
     * @throws NullPointerException if {@code alleles == null || prefix == null}
     */
    public void update(int[] alleles, int nAlleles, int[] prefix) {
        if (alleles.length!=nHaps) {
            throw new IllegalArgumentException(String.valueOf(alleles.length));
        }
        if (prefix.length!=nHaps) {
            throw new IllegalArgumentException(String.valueOf(prefix.length));
        }
        initializeArrays(nAlleles);
        for (int h : prefix) {
            int allele = alleles[h];
            if (allele>=nAlleles) {
                throw new IndexOutOfBoundsException(String.valueOf(allele));
            }
            a[allele].add(h);
        }
        updatePrefix(nAlleles, prefix);
    }

    private void updatePrefix(int nAlleles, int[] prefix) {
        int start = 0;
        for (int al=0; al<nAlleles; ++al) {
            int size = a[al].size();
            System.arraycopy(a[al].toArray(), 0, prefix, start, size);
            start += size;
            a[al].clear();
        }
        assert start == nHaps;
    }

    private void initializeArrays(int nAlleles) {
        if (nAlleles<1) {
            throw new IllegalArgumentException(String.valueOf(nAlleles));
        }
        ensureArrayCapacity(nAlleles);
    }

    private void ensureArrayCapacity(int nAlleles) {
        if (nAlleles>a.length) {
            int oldLength = a.length;
            a = Arrays.copyOf(a, nAlleles);
            for (int j = oldLength; j<a.length; ++j) {
                a[j] = new IntList();
            }
        }
    }
}
