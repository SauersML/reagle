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

import blbutil.Const;
import blbutil.StringUtil;
import ints.IntList;
import java.util.Arrays;
import static vcf.Marker.ALLELES_STORED;
import static vcf.Marker.FILTER_STORED;
import static vcf.Marker.ID_STORED;
import static vcf.Marker.QUAL_STORED;
import static vcf.Marker.SNV_PERMS;

/**
 * <p>Class {@code MarkerParser} is a parser and filter for a VCF record's
 * ID, REF, ALT, QUAL, FILTER, and INFO subfields.</p>
 *
 * <p>Instances of class {@code MarkerParser} are immutable.</p>
 *
 * @author Brian L. Browning {@code <browning@uw.edu>}
 */
public final class MarkerParser {

    private final boolean storeId;
    private final boolean storeQual;
    private final boolean storeFilter;
    private final boolean storeInfo;

    /**
     * Constructs a new {@code MarkerParser} instance from the specified
     * data.
     * @param storeId {@code true} if a non-missing VCF ID field will be stored
     * @param storeQual {@code true} if non-missing VCF QUAL field will be stored
     * @param storeFilter {@code true} if a non-missing VCF FILTER field will be stored
     * @param storeInfo {@code true} if a non-missing VCF INFO field will be stored
     */
    public MarkerParser(boolean storeId, boolean storeQual,
            boolean storeFilter, boolean storeInfo) {
        this.storeId = storeId;
        this.storeQual = storeQual;
        this.storeFilter = storeFilter;
        this.storeInfo = storeInfo;
    }

    /**
     * Filter and parses a VCF record's ID, REF, ALT, QUAL, FILTER,
     * and INFO subfields.
     * If the specified VCF record contains an INFO/END field, the INFO/END
     * field will be stored, even if {@code (this.storeInfo() == false)}.
     * The contract for this method is undefined if {@code tabs} does not
     * contain the indices of the first 8 tabs of the specified VCF record
     * in ascending order.
     * @param rec a VCF record
     * @param sb the {@code StringBuilder} in which fields will be stored
     * @param tabs the indices of the first 8 tabs in the specified VCF record
     * @return a {@code short} encoding information about the stored fields.
     *
     * @throws IllegalArgumentException if the specified VCF record has more
     * than 255 alleles
     * @throws IndexOutOfBoundsException if {@code tabs.size() < 8}
     * @throws NullPointerException if
     * {@code (rec == null) || (sb == null) || (tabs == null)}
     */
    short storeMarkerFields(String rec, StringBuilder sb, IntList tabs) {
        short info = (short) 0;
        info = storeField(rec, tabs.get(1)+1, tabs.get(2), info, sb, storeId, ID_STORED);
        info = storeAlleles(rec, tabs.get(2)+1, tabs.get(4), info, sb);
        info = storeField(rec, tabs.get(4)+1, tabs.get(5), info, sb, storeQual, QUAL_STORED);
        info = storeField(rec, tabs.get(5)+1, tabs.get(6), info, sb, storeFilter, FILTER_STORED);
        info = storeInfo(rec, tabs.get(6)+1, tabs.get(7), info, sb);
        return info;
    }

    private static short storeAlleles(String vcfRec, int start, int end,
            short fieldInfo, StringBuilder sb) {
        String refAltAlleles = new String(vcfRec.substring(start, end));
        int snvIndex = snvIndex(refAltAlleles);
        if (snvIndex>=0) {
            int nAlleles = refAltAlleles.endsWith("\t.") ? 1 : ((end - start + 1) >> 1);
            fieldInfo |= (snvIndex<<3);
            fieldInfo |= nAlleles;
        }
        else {
            int tabIndex = refAltAlleles.indexOf(Const.tab);
            if ((tabIndex+1)==refAltAlleles.length()) {
                String s = "ERROR: missing ALT field: "
                        + MarkerUtils.truncate(vcfRec, 80);
                throw new IllegalArgumentException(s);
            }
            int nAlleles = nAlleles(refAltAlleles, tabIndex);
            if (nAlleles > Marker.STORED_N_ALLELES_MASK) {
                throw new IndexOutOfBoundsException(String.valueOf(nAlleles)
                    + " alleles: " + vcfRec.substring(0, 80));
            }
            if (sb.length()>0) {
                sb.append(Const.tab);
            }
            sb.append(refAltAlleles);
            fieldInfo |= nAlleles;
            fieldInfo |= ALLELES_STORED;
        }
        return fieldInfo;
    }

    private static int snvIndex(String refAndAlt) {
        int index = Arrays.binarySearch(SNV_PERMS, refAndAlt);
        if (index<0) {
            index = -index-1;
        }
        if (index==SNV_PERMS.length) {
            return -1;
        }
        else {
            return SNV_PERMS[index].startsWith(refAndAlt) ? index : -1;
        }
    }

    private static int nAlleles(String refAltAlleles, int tabIndex) {
        int startAllele = tabIndex + 1;
        if (startAllele==(refAltAlleles.length()-1)
                && refAltAlleles.charAt(startAllele)==Const.MISSING_DATA_CHAR) {
            return 1;
        }
        else {
            int nAlleles = 2;
            startAllele = refAltAlleles.indexOf(Const.comma, startAllele) + 1;
            while (startAllele>0) {
                ++nAlleles;
                startAllele = refAltAlleles.indexOf(Const.comma, startAllele) + 1;
            }
            return nAlleles;
        }
    }

    /**
     * Appends the VCF field to the specified {@code StringBuilder} if
     * {@code (isStored == true)} and the VCF field is not MISSING ('.').
     * A tab will be appended before the VCF field if {@code sb.length()>0}.
     * @param vcfRec a VCF record
     * @param start the index (inclusive) of the first character in the VCF QUAL
     * field
     * @param end the index (exclusive) of the last character in the VCF QUAL
     * field
     * @param flags the current VCF record flags
     * @param sb the {@code StringBuilder} to which the VCF field may be
     * appended
     * @param isStored {@code true} if a non-missing VCF field should be
     * appended to the {@code StringBuilder}
     * @param flag the flag bits that will be set if the VCF field is
     * non-missing and appended to {@code sb}
     * @return {@code (flags | flag)} if the VCF field is non-missing and
     * appended to the {@code StringBuilder} and {@code flags} otherwise.
     *
     * @throws IllegalArgumentException if {@code (start < 0)},
     * or {@code (start > end)}, or {@code (end > vcfRec.length())}
     * @throws NullPointerException if {@code (vcfRec == null) || (sb == null)}
     */
    private short storeField(String vcfRec, int start, int end, short flags,
            StringBuilder sb, boolean isStored, short flag) {
        if (start<0 || start>end || end>vcfRec.length()) {
            throw new IllegalArgumentException("start=" + start + " end=" + end);
        }
        if (sb == null) {
            throw new IllegalArgumentException(StringBuilder.class.toString());
        }
        if (isStored &&
                ((end-start)!=1 || vcfRec.charAt(start)!=Const.MISSING_DATA_CHAR)) {
            if (sb.length()>0) {
                sb.append(Const.tab);
            }
            sb.append(vcfRec, start, end);
            flags |= flag;
        }
        return flags;
    }

    /**
     * Appends the VCF INFO field to the specified {@code StringBuilder} if
     * {@code (this.storeInof() == true)} and the VCF INFO field is not
     * MISSING ('.'). A tab will be appended before the VCF INFO field if
     * {@code sb.length()>0}.
     * @param vcfRec a VCF record
     * @param start the index (inclusive) of the first character in the VCF
     * INFO field
     * @param end the index (exclusive) of the last character in the VCF INFO
     * field
     * @param flags the current marker flags
     * @param sb the {@code StringBuilder} to which the VCF INFO field may be
     * appended
     * @return {@code (flags | Marker.QUAL_FILTER)} if the VCF INFO field
     * was stored and {@code flags} otherwise.
     *
     * @throws IllegalArgumentException if {@code (start < 0)},
     * or {@code (start > end)}, or {@code (end > vcfRec.length())}
     * @throws NullPointerException if {@code (vcfRec == null) || (sb == null)}
     */
    private short storeInfo(String vcfRec, int start, int end, short flags,
            StringBuilder sb) {
        if (start<0 || start>end || end>vcfRec.length()) {
            throw new IllegalArgumentException("start=" + start + " end=" + end);
        }
        if (sb == null) {
            throw new IllegalArgumentException(StringBuilder.class.toString());
        }
        if ((end-start)!=1 || vcfRec.charAt(start)!=Const.MISSING_DATA_CHAR) {
            String endSubfield = endSubfield(vcfRec, start, end);
            if (storeInfo || endSubfield!=null) {
                if (sb.length()>0) {
                    sb.append(Const.tab);
                }
                if (storeInfo) {
                    sb.append(vcfRec, start, end);
                    flags |= Marker.INFO_STORED;
                    if (endSubfield != null) {
                        flags |= Marker.END_STORED;
                    }
                }
                else {
                    assert endSubfield!=null;
                    sb.append(endSubfield);
                    flags |= Marker.END_STORED;

                }
            }
        }
        return flags;
    }

    /*
     * Returns the first INFO/END subfield, or {@code null} if there
     * is no INFO/END subfield.
     */
    private static String endSubfield(String vcfRec, int infoStart, int infoEnd) {
        String infoField = vcfRec.substring(infoStart, infoEnd);
        String[] fields = StringUtil.getFields(infoField, Const.semicolon);
        for (String field : fields) {
            if (field.startsWith("END=")) {
                return field;
            }
        }
        return null;
    }

    /**
     * Returns a string description of {@code this}.  The exact details of
     * the description of are unspecified and subject to change.
     *
     * @return a string description of {@code this}
     */
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(50);
        sb.append("ID=");
        sb.append(storeId());
        sb.append(" QUAL=");
        sb.append(storeQual());
        sb.append(" FILTER=");
        sb.append(storeFilter());
        sb.append(" INFO=");
        sb.append(storeInfo());
        return sb.toString();
    }

    /**
     * Returns {@code true} if the VCF ID field will be stored
     * @return  {@code true} if the VCF ID field will be stored
     */
    public boolean storeId() {
        return storeId;
    }

    /**
     * Returns {@code true} if the VCF QUAL field will be stored
     * @return  {@code true} if the VCF QUAL field will be stored
     */
    public boolean storeQual() {
        return storeQual;
    }

    /**
     * Returns {@code true} if the VCF FILTER field will be stored
     * @return  {@code true} if the VCF FILTER field will be stored
     */
    public boolean storeFilter() {
        return storeFilter;
    }

    /**
     * Returns {@code true} if the VCF INFO field will be stored
     * @return  {@code true} if the VCF INFO field will be stored
     */
    public boolean storeInfo() {
        return storeInfo;
    }
}