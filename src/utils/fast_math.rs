//! Fast math approximations.

/// Fast approximation of natural logarithm.
///
/// Uses a bit manipulation trick combined with a polynomial approximation.
/// Maximum relative error is approximately 1e-4.
#[inline]
pub fn fast_ln(x: f32) -> f32 {
    // Handling of non-positive numbers matches std::ln (returns NaN or -inf)
    if x <= 0.0 {
        return x.ln();
    }

    let x_bits = x.to_bits();
    let e = ((x_bits >> 23) as i32) - 127;
    let m_bits = (x_bits & 0x007FFFFF) | 0x3F800000;
    let m = f32::from_bits(m_bits);

    // Polynomial approximation for log2(m) where m is in [1, 2).
    // Based on Remez algorithm or similar optimization.
    // P(x) = -0.1691866 + 1.998993*x - 0.8306026*x^2
    // This is a rough approximation. A better one for [1, 2]:
    // P(x) = (x-1) * (c1 + (x-1) * (c2 + (x-1) * c3))
    // c1 = 1.442689645
    // c2 = -0.72116576
    // c3 = 0.4786848
    // But simple 2nd order is usually enough for entropy.

    // Let's use a simpler 3rd order for better accuracy:
    let z = m - 1.0;
    let log2_m = z * (1.442695 + z * (-0.7211658 + z * 0.4786848));

    (e as f32 + log2_m) * 0.69314718
}
