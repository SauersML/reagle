
#[cfg(test)]
mod debug_test {
    use super::*;
    use bitvec::prelude::*;

    #[test]
    fn test_dense_column_bit_setting() {
        let alleles = vec![0u8, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0];
        let col = DenseColumn::from_alleles(alleles.iter().copied(), 2);
        let bits = col.bits_raw();
        println!("Bits: {:?}", bits);
        
        let word = bits[0];
        // Check bit 6
        assert_eq!((word >> 6) & 1, 1, "Bit 6 should be set");
        // Check bit 10
        assert_eq!((word >> 10) & 1, 1, "Bit 10 should be set");
    }
}
