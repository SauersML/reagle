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
import blbutil.Const;
import ints.IntList;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;
import java.util.stream.IntStream;

/**
 * <p>Class {@code Marker} represents a VCF record's CHROM, POS, ID, REF,
 * ALT, QUAL, FILTER, and INFO fields. The number of alleles in the VCF
 * record must be less than or equal to {@code Marker.MAX_N_ALLELES}.</p>
 *
 * <p>Instances of class {@code Marker} are immutable.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class Marker implements Comparable<Marker> {

    static final short STORED_N_ALLELES_MASK = 0xff;
    private static final short INDEXED_N_ALLELES_MASK = 0b111;
    private static final short SNV_INDEX_MASK = 0x7f;

    static final String[] SNV_PERMS = MarkerUtils.snvPerms();

    static final short ID_STORED = (short) (1<<15);
    static final short ALLELES_STORED = (short) (1<<14);
    static final short QUAL_STORED = (short) (1<<13);
    static final short FILTER_STORED = (short) (1<<12);
    static final short INFO_STORED = (short) (1<<11);   // indicates complete INFO field is stored
    static final short END_STORED = (short) (1<<10);    // indicates INFO/END field is stored
    static final short FLAGS =
            ID_STORED | ALLELES_STORED | QUAL_STORED | FILTER_STORED | INFO_STORED  | END_STORED;

    private final short chromIndex;
    private final int pos;
    private final short fieldInfo;
    private final String fields;

    private Marker(short chromIndex, int pos, short fieldInfo, String fields) {
        this.chromIndex = chromIndex;
        this.pos = pos;
        this.fieldInfo = fieldInfo;
        this.fields = fields;
    }

    /**
     * Constructs a new {@code Marker} instance from the data. If the
     * specified VCF record contains an INFO/END field, the INFO/END field
     * will be stored.
     * @param rec a VCF record
     * @param markerParser an object that filters and parses a VCF record's ID,
     * QUAL, FILTER, and INFO subfields
     * @return a new {@code Marker} instance
     *
     * @throws IllegalArgumentException if the specified VCF record does not
     * contain at least 8 tab characters
     * @throws IllegalArgumentException if the VCF CHROM field contains
     * whitespace
     * @throws IllegalArgumentException if the specified VCF record has more
     * than 255 alleles
     * @throws IndexOutOfBoundsException if the index of the VCF CHROM field
     * exceeds {@code Short.MAX_VALUE}
     * @throws NullPointerException if
     * {@code (vcfRecord == null) || (vcfFields==null)}
     * @throws NumberFormatException if the VCF record POS field is not a
     * parsable integer
     */
    public static Marker instance(String rec, MarkerParser markerParser) {
        IntList tabs = MarkerUtils.first8TabIndices(rec);
        StringBuilder sb = new StringBuilder();
        short chromIndex = MarkerUtils.chromIndex(rec, rec.substring(0, tabs.get(0)));
        int pos = Integer.parseInt(rec.substring(tabs.get(0)+1, tabs.get(1)));
        short fieldInfo = markerParser.storeMarkerFields(rec, sb, tabs);
        String fields = sb.length()==0 ? null : sb.toString();
        return new Marker(chromIndex, pos, fieldInfo, fields);
    }

    /**
     * Returns the VCF CHROM field.
     * @return the VCF CHROM field
     */
    public String chrom() {
        return ChromIds.instance().id(chromIndex);
    }

    /**
     * Returns the chromosome index.
     * @return the chromosome index
     */
    public int chromIndex() {
        return chromIndex;
    }

    /**
     * Returns the VCF POS field
     * @return the VCF POS field
     */
    public int pos() {
        return pos;
    }

    /**
     * Returns {@code true} if the VCF ID field has non-missing
     * data, and returns {@code false} otherwise.
     *
     * @return {@code true} if the VCF ID field has non-missing
     * data, and returns {@code false} otherwise
     */
    public boolean hasIdData() {
        return (fieldInfo & ID_STORED)==ID_STORED;
    }

    /**
     * Returns {@code true} if the VCF QUAL field has non-missing
     * data, and returns {@code false} otherwise.
     *
     * @return {@code true} if the VCF QUAL field has non-missing
     * data, and returns {@code false} otherwise
     */
    public boolean hasQualData() {
        return (fieldInfo & QUAL_STORED)==QUAL_STORED;
    }

    /**
     * Returns {@code true} if the VCF FILTER field has non-missing
     * data, and returns {@code false} otherwise.
     *
     * @return {@code true} if the VCF FILTER field has non-missing
     * data, and returns {@code false} otherwise
     */
    public boolean hasFilterData() {
        return (fieldInfo & FILTER_STORED)==FILTER_STORED;
    }

    /**
     * Returns {@code true} if the VCF INFO field has non-missing
     * data, and returns {@code false} otherwise.
     *
     * @return {@code true} if the VCF INFO field has non-missing
     * data, and returns {@code false} otherwise
     */
    public boolean hasInfoData() {
        return (fieldInfo & INFO_STORED)==INFO_STORED;
    }

    /**
     * Returns {@code true} if the VCF INFO/END field is defined,
     * and returns {@code false} otherwise.
     *
     * @return {@code true} if the VCF INFO/END field has is defined
     */
    public boolean hasEndValue() {
        return (fieldInfo & END_STORED)==END_STORED;
    }

    /**
     * Returns the VCF ID field.
     * @return the VCF ID field
     */
    public String id() {
        if ((fieldInfo & ID_STORED)==ID_STORED) {
            int endIndex = fields.indexOf(Const.tab);
            return endIndex<0 ? fields : fields.substring(0, endIndex);
        }
        else {
            return Const.MISSING_DATA_STRING;
        }
    }

    /**
     * Returns the tab-separated VCF REF and ALT fields
     * @return the tab-separated VCF REF and ALT fields
     */
    public String alleles() {
        if ((fieldInfo & ALLELES_STORED)==ALLELES_STORED) {
            int start = 0;
            if ((fieldInfo & ID_STORED)==ID_STORED) {
                start = fields.indexOf(Const.tab) + 1;      // start of REF
            }
            int end = fields.indexOf(Const.tab, start);     // end of REF
            end = fields.indexOf(Const.tab, end+1);         // end of ALT
            return end<0 ? fields.substring(start) : fields.substring(start, end);
        }
        else {
            int snvIndex = (fieldInfo>>>3) & SNV_INDEX_MASK;  // SNV_PERMS.length==100 && 100 < 0x7f
            int nAlleles = fieldInfo & INDEXED_N_ALLELES_MASK;
            if (nAlleles==1) {
                return SNV_PERMS[snvIndex];
            }
            else {
                int length =(nAlleles<<1) - 1;
                return SNV_PERMS[snvIndex].substring(0, length);
            }
        }
    }

    /**
     * Returns the number of nucleotides in the reference allele.
     * @return the number of nucleotides in the reference allele
     */
    public int nRefBases() {
        if ((fieldInfo & ALLELES_STORED)==ALLELES_STORED) {
            int start = 0;
            if ((fieldInfo & ID_STORED)==ID_STORED) {
                start = fields.indexOf(Const.tab) + 1;      // start of REF
            }
            int end = fields.indexOf(Const.tab, start);     // end of REF
            return end - start;
        }
        else {
            return 1;
        }
    }

    /**
     * Returns the number of alleles for the marker, including the REF
     * allele.
     * @return the number of alleles for the marker, including the REF
     * allele
     */
    public int nAlleles() {
        if ((fieldInfo & ALLELES_STORED)==ALLELES_STORED) {
            return (fieldInfo & STORED_N_ALLELES_MASK);
        }
        else {
            return fieldInfo & INDEXED_N_ALLELES_MASK;
        }
    }

    /**
     * Returns the minimum number of bits required to store a non-missing
     * allele.
     * @return the minimum number of bits required to store a non-missing
     * allele
     */
    public int bitsPerAllele() {
        return Integer.SIZE - Integer.numberOfLeadingZeros(nAlleles()-1);
    }

    private int qualStartIndex() {
        int start = 0;
        if ((fieldInfo & ID_STORED)==ID_STORED) {
            start = fields.indexOf(Const.tab) + 1;         // skip ID field
        }
        if ((fieldInfo & ALLELES_STORED)==ALLELES_STORED) {
            start = fields.indexOf(Const.tab, start) + 1;  // skip REF field
            start = fields.indexOf(Const.tab, start) + 1;  // skip ALT field
        }
        return start;
    }

    private String extractField(int start, char endDelimiter) {
        int end = fields.indexOf(endDelimiter, start);
        return end<0 ? fields.substring(start) : fields.substring(start, end);
    }

    /**
     * Returns the VCF QUAL field.
     * @return the VCF QUAL field
     */
    public String qual() {
        if ((fieldInfo & QUAL_STORED)==QUAL_STORED) {
            return extractField(qualStartIndex(), Const.tab);
        }
        else {
            return Const.MISSING_DATA_STRING;
        }
    }

    /**
     * Returns the VCF FILTER field.
     * @return the VCF FILTER field.
     */
    public String filter() {
        if ((fieldInfo & FILTER_STORED)==FILTER_STORED) {
            int start = qualStartIndex();
            if ((fieldInfo & QUAL_STORED)==QUAL_STORED) {
                start = fields.indexOf(Const.tab, start) + 1; // skip QUAL field
            }
            return extractField(start, Const.tab);
        }
        else {
            return Const.MISSING_DATA_STRING;
        }
    }

    /**
     * Returns the VCF INFO field.
     * @return the VCF INFO field.
     */
    public String info() {
        if ((fieldInfo & INFO_STORED)==INFO_STORED || (fieldInfo & END_STORED)==END_STORED) {
            int start = qualStartIndex();
            if ((fieldInfo & QUAL_STORED)==QUAL_STORED) {
                start = fields.indexOf(Const.tab, start) + 1;  // skip QUAL field
            }
            if ((fieldInfo & FILTER_STORED)==FILTER_STORED) {
                start = fields.indexOf(Const.tab, start) + 1;  // skip FILTER field
            }
            return extractField(start, Const.tab);
        }
        else {
            return Const.MISSING_DATA_STRING;
        }
    }

    /**
     * Returns the INFO/END value.  Returns "" if the INFO/END value is not
     * defined.
     * @return the INFO/END value
     */
    public String endValue() {
        if ((fieldInfo & END_STORED)==END_STORED) {
            int startValue = qualStartIndex();
            if ((fieldInfo & QUAL_STORED)==QUAL_STORED) {
                startValue = fields.indexOf(Const.tab, startValue) + 1;  // skip QUAL field
            }
            if ((fieldInfo & FILTER_STORED)==FILTER_STORED) {
                startValue = fields.indexOf(Const.tab, startValue) + 1;  // skip FILTER field
            }
            int startKey = fields.indexOf("END=", startValue);
            assert startKey >= 0;
            return extractField((startKey + 4), Const.semicolon);
        }
        else {
            return "";
        }
    }

    /**
     * <p>Returns the hash code value for this object.
     * The hash code is defined by the following calculation:
     * </p>
     *   <pre>
     *   int hash = 5;
     *   hash = 29 * hash + this.chromIndex();
     *   hash = 29 * hash + this.pos();
     *   hash = 29 * hash + this.alleles().hashCode();
     *   hash = 29 * hash + this.endValue().hashCode();
    *   </pre>
     *
     * @return the hash code value for this marker
     */
    @Override
    public int hashCode() {
        int hash = 5;
        hash = 29 * hash + chromIndex;
        hash = 29 * hash + pos;
        hash = 29 * hash + alleles().hashCode();
        hash = 29 * hash + endValue().hashCode();
        return hash;
    }

    /**
     * Returns {@code true} if the specified object is a
     * {@code Marker} with the same chromosome, position, alleles, and
     * INFO/END value, and returns {@code false} otherwise.
     *
     * @param obj object to be compared with {@code this} for equality
     *
     * @return {@code true} if the specified object is a {@code Marker} with
     * the same chromosome, position, alleles, and INFO/END value
     */
    @Override
    public boolean equals(Object obj) {
        if (this==obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final Marker other = (Marker) obj;
        if (this.chromIndex != other.chromIndex) {
            return false;
        }
        if (this.pos != other.pos) {
            return false;
        }
        if (this.alleles().equals(other.alleles()) == false) {
            return false;
        }
        return this.endValue().equals(other.endValue());
    }

    /**
     * Compares this marker with the specified marker
     * for order, and returns a negative integer, 0, or a positive integer
     * depending on whether this marker is less than, equal to,
     * or greater than the specified marker. Markers are compared
     * using the values returned by the {@code chromIndex()}, {@code pos()},
     * {@code alleles()}, and {@code endValue()} methods. The returned
     * value is defined by the following calculation:
     *   <pre>
     *   if (this.chromIndex() != other.chromIndex()) {
     *       return (this.chromIndex &lt; other.chromIndex()) ? -1 : 1;
     *   }
     *   if (this.pos() != other.pos()) {
     *       return (this.pos &lt; other.pos()) ? -1 : 1;
     *   }
     *   int value = this.alleles().compareTo(other.alleles());
     *   if (value!=0) {
     *       return value;
     *   }
     *   return this.endValue().compareTo(other.endValue());
     *   </pre>
     *
     * @param other the {@code Marker} to be compared
     * @return a negative integer, 0, or a positive integer
     * depending on whether this marker is less than, equal,
     * or greater than the specified marker
     */
    @Override
    public int compareTo(Marker other) {
        if (this.chromIndex != other.chromIndex) {
            return (this.chromIndex < other.chromIndex) ? -1 : 1;
        }
        if (this.pos != other.pos) {
            return (this.pos < other.pos) ? -1 : 1;
        }
        int value = this.alleles().compareTo(other.alleles());
        if (value!=0) {
            return value;
        }
        return this.endValue().compareTo(other.endValue());
    }

    /**
     * Writes a representation of the VCF record ID, REF, ALT, QUAL, FILTER,
     * and INFO fields to the specified output. The exact details of the
     * representation are unspecified and subject to change. The written data
     * can be read with the {@code Marker.readNonPosFields()} method.
     * @param out the output destination
     * @throws IOException if an I/O error occurs
     * @throws NullPointerException if {@code out == null}
     */
    public void writeNonPosFields(DataOutput out) throws IOException {
        out.writeShort(fieldInfo);
        if ((fieldInfo & FLAGS)!=0) {
            out.writeUTF(fields);
        }
    }

    /**
     * Reads the VCF record ID, REF, ALT, QUAL, FILTER, and INFO fields and
     * returns a marker with these fields and the specified CHROM and POS
     * fields.  The contract for this method is unspecified if
     * {@code chromIndex} is not a valid chromosome index.
     * @param chromIndex the chromosome index
     * @param pos the chromosome position
     * @param in the input source
     * @return a {@code Marker}
     * @throws IOException if an I/O error occurs
     * @throws NullPointerException if {@code in == null}
     */
    public static Marker readNonPosFields(short chromIndex, int pos,
            DataInput in) throws IOException {
        short fieldInfo = (short) in.readShort();
        String fields = (fieldInfo & Marker.FLAGS) == 0 ? null : in.readUTF();
        return new Marker(chromIndex, pos, fieldInfo, fields);
    }

     /**
      * Returns a string equal to the first five tab-delimited fields
      * of a VCF record corresponding to this marker (the CHROM, POS, ID,
      * REF, and ALT fields).
      *
      * @return a string equal to the first five tab-delimited fields
      * of a VCF record corresponding to this marker
      */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(50);
        sb.append(chrom());
        sb.append(Const.tab);
        sb.append(pos);
        sb.append(Const.tab);
        sb.append(id());
        sb.append(Const.tab);
        sb.append(alleles());
        return sb.toString();
    }

    /**
     * Appends the first eight tab-delimited fields of a VCF record for this
     * marker (the CHROM, POS, ID, REF, ALT, QUAL, FILTER, and INFO fields)
     * to the specified {@code StringBuilder}.
     * @param sb the {@code StringBuilder} to be appended
     * @throws NullPointerException if {@code (sb == null)}
     */
    public void appendFirst8Fields(StringBuilder sb) {
        appendFirst7FieldsAndTab(sb);
        sb.append(info());
    }

    /**
     * Appends the first eight tab-delimited fields of a VCF record for this
     * marker (the CHROM, POS, ID, REF, ALT, QUAL, FILTER, and INFO fields)
     * and add the specified INFO/AN and INFO/AC fields.  If the INFO/AN
     * or iNFO/AC fields exist in {@code this.info()}, the fields will
     * be replaced with the specified INFO/AN and INFO/AC fields.
     *
     * @param sb the {@code StringBuilder} to be appended
     * @param an the total number of alleles in called genotypes
     * @param alleleCounts an array of length {@code this.nAlleles()} whose
     * {@code k-th} entry is the allele count in called genotypes
     * for the {@code k}-th allele
     * @throws IllegalArgumentException if
     * {@code (this.nAlleles() != alleleCounts.length)}
     * @throws NullPointerException if
     * {@code ((sb == null) || (alleleCounts == null))}
     */
    public void appendFirst8Fields(StringBuilder sb, int an, int[] alleleCounts) {
        appendFirst7FieldsAndTab(sb);
        appendInfo(sb, an, alleleCounts);
    }

    private void appendFirst7FieldsAndTab(StringBuilder sb) {
        sb.append(chrom());
        sb.append(Const.tab);
        sb.append(pos);
        sb.append(Const.tab);
        sb.append(id());
        sb.append(Const.tab);
        sb.append(alleles());
        sb.append(Const.tab);
        sb.append(qual());
        sb.append(Const.tab);
        sb.append(filter());
        sb.append(Const.tab);
    }

    private void appendInfo(StringBuilder sb, int an, int[] alleleCounts) {
        if (nAlleles()!=alleleCounts.length) {
            throw new IllegalArgumentException(Arrays.toString(alleleCounts));
        }
        appendCounts(sb, an, alleleCounts);
        String info = info();
        int start = 0;
        while (start<info.length()) {
            int end = info.indexOf(';', start);
            if (end==-1) {
                end = info.length();
            }
            if (start<end) {
                String field = info.substring(start, end).trim();
                if (field.length()>0
                        && field.startsWith("AN=")==false
                        && field.startsWith("AC=")==false) {
                    sb.append(';');
                    sb.append(field);
                }
            }
            start = end + 1;
        }
    }

    private void appendCounts(StringBuilder sb, int an, int[] alleleCounts) {
        sb.append("AN=");
        sb.append(an);
        sb.append(";AC=");
        for (int j=1; j<alleleCounts.length; ++j) {   // begin with first ALT allele
            if (j>1){
                sb.append(',');
            }
            sb.append(alleleCounts[j]);
        }
    }

    // [July 2, 2024] The following code for determining how to map targ
    // alleles to ref alleles is UNDER CONSTRUCTION.
    private static int[] targToRefAllele(Marker ref, Marker targ) {
        if (isStrandKnownSNV(targ)==false) {
            return null;
        }
        int refOffset = targ.pos() - ref.pos();
        String refAlleles = ref.alleles();
        String targAlleles = targ.alleles();
        boolean flipStrand = (refAlleles.charAt(refOffset) == flipStrand(targAlleles.charAt(0)));
        if (flipStrand==false && refAlleles.charAt(refOffset)==targAlleles.charAt(0)==false) {
            return null;
        }
        int[] refAlleleIndex = refAlleleIndex(refAlleles, refOffset, ref.nAlleles());
        if (refAlleleIndex==null) {
            return null;
        }
        int[] targToRefAllele = new int[targ.nAlleles()];
        Arrays.fill(targToRefAllele, 1, targToRefAllele.length, -1);
        for (int j=1; j<targToRefAllele.length; ++j) {
            char targAllele = targAlleles.charAt(j<<1);
            if (flipStrand) {
                targAllele = flipStrand(targAllele);
            }
            int refAlIndex = refAlleleIndex[alleleIndex(targAllele)];
            if (refAlIndex>=0) {
                targToRefAllele[j] = refAlIndex;
            }
            else {
                return null;
            }
        }
        return targToRefAllele;
    }

    private static boolean isStrandKnownSNV(Marker marker) {
        if ((marker.fieldInfo & ALLELES_STORED)==ALLELES_STORED) {
            return false;
        }
        String targAlleles = marker.alleles();
        if (targAlleles.charAt(2)==Const.MISSING_DATA_CHAR) {
            return false;
        }
        char refAllele = targAlleles.charAt(0);
        for (int j=2, n=targAlleles.length(); j<n; j+=2) {
            if (refAllele==flipStrand(targAlleles.charAt(j))) {
                return false;
            }
        }
        return true;
    }

    private static int[] refAlleleIndex(String refAlleles, int refOffset,
            int nRefAlleles) {
        int nRefBases = refAlleles.indexOf(Const.tab);
        assert nRefBases>=0;
        if (refOffset < 0 || refOffset >= nRefBases
                || allCharsAreNucleotides(refAlleles, 0, nRefBases)==false) {
            return null;
        }
        int[] refAlleleIndex = IntStream.range(0, 4).map(j -> -2).toArray();
        refAlleleIndex[alleleIndex(refAlleles.charAt(refOffset))] = 0;
        int start = nRefBases + 1;
        for (int j=1; j<nRefAlleles; ++j) {
            assert start < refAlleles.length();
            int end = refAlleles.indexOf(Const.comma, start);
            if (end<0) {
                end = refAlleles.length();
            }
            if ((end-start)==nRefBases) {
                int alleleIndex = alleleIndex(refAlleles.charAt(start + refOffset));
                if (alleleIndex == -1 || refAlleleIndex[alleleIndex] != -2) {
                    return null;
                }
                refAlleleIndex[alleleIndex] = j;
            }
            start = end + 1;
        }
        return refAlleleIndex;
    }

    private static boolean allCharsAreNucleotides(String s, int start, int end) {
        for (int j=start; j<end; ++j) {
            char c = s.charAt(j);
            if (c!='A' && c!='C' && c!='T' && c!='G') {
                return false;
            }
        }
        return true;
    }

    private static int alleleIndex(char c) {
        switch (c) {
            case 'A' : return 0;
            case 'C' : return 1;
            case 'G' : return 2;
            case 'T' : return 3;
            default : return -1;
        }
    }

    private static char flipStrand(char c) {
        switch (c) {
            case 'A' : return 'T';
            case 'C' : return 'G';
            case 'G' : return 'C';
            case 'T' : return 'A';
            case 'N' : return 'N';
            case '*' : return '*';
            default: throw new IllegalArgumentException(String.valueOf(c));
        }
    }
}
