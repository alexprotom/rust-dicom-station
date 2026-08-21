//! IEEE-754 binary16 ↔ binary32 conversion.
//!
//! Checkpoints and weight caches store half-precision tensors; every
//! computation upcasts to `f32`, so these two functions are the whole of the
//! crate's half-precision support. Both are exact: `f16_to_f32` is lossless
//! (every binary16 value is representable in binary32) and `f32_to_f16`
//! rounds to nearest, ties to even — the same rule PyTorch and numpy use.

/// Widen a binary16 bit pattern to `f32`. Subnormals, infinities and NaNs
/// are all preserved.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x3ff) as u32;
    let f = match (exp, frac) {
        (0, 0) => sign,
        (0, f) => {
            // subnormal: renormalize into the f32 exponent range
            let mut e = 127 - 15 + 1;
            let mut f = f;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((f & 0x3ff) << 13)
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, f) => sign | 0x7f80_0000 | (f << 13),
        (e, f) => sign | ((e + 127 - 15) << 23) | (f << 13),
    };
    f32::from_bits(f)
}

/// Narrow an `f32` to a binary16 bit pattern, rounding to nearest with ties
/// to even. Values too large for binary16 saturate to infinity; values too
/// small become subnormal and then zero.
pub fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;

    if exp == 0xff {
        // Inf / NaN. Keep NaN non-zero in the mantissa so it stays a NaN.
        return sign
            | 0x7c00
            | if frac != 0 {
                (frac >> 13) as u16 | 0x0200
            } else {
                0
            };
    }
    let unbiased = exp - 127;
    if unbiased > 15 {
        return sign | 0x7c00; // overflow → infinity
    }
    if unbiased >= -14 {
        // normal in binary16
        let mant = frac >> 13;
        let round = frac & 0x1fff;
        let half = 0x1000;
        let mut out = (((unbiased + 15) as u16) << 10) | mant as u16;
        if round > half || (round == half && out & 1 == 1) {
            out += 1; // may carry into the exponent, which is correct
        }
        return sign | out;
    }
    if unbiased < -25 {
        return sign; // underflow → signed zero
    }
    // subnormal: shift the implicit leading 1 into the mantissa
    let mant = frac | 0x0080_0000;
    let shift = (-unbiased - 14) as u32 + 13;
    let out = mant >> shift;
    let round = mant & ((1 << shift) - 1);
    let half = 1u32 << (shift - 1);
    let mut out = out as u16;
    if round > half || (round == half && out & 1 == 1) {
        out += 1;
    }
    sign | out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn widening_known_values() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!((f16_to_f32(0x3555) - 0.333).abs() < 1e-3);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert_eq!(f16_to_f32(0xfc00), f32::NEG_INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        // smallest positive subnormal: 2^-24
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        // largest finite binary16
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
    }

    #[test]
    fn narrowing_known_values() {
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(-2.0), 0xc000);
        assert_eq!(f32_to_f16(0.0), 0x0000);
        assert_eq!(f32_to_f16(-0.0), 0x8000);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(65520.0), 0x7c00); // rounds up past the max → inf
        assert_eq!(f32_to_f16(f32::INFINITY), 0x7c00);
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
        assert_eq!(f32_to_f16(2f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_f16(2f32.powi(-25)), 0x0000); // ties-to-even → zero
        assert_eq!(f32_to_f16(2f32.powi(-26)), 0x0000);
    }

    #[test]
    fn round_trips_every_binary16_value() {
        // f16 -> f32 is lossless, so narrowing must return the original bits
        // for every one of the 65 536 patterns (NaNs compare by payload, so
        // they are checked separately above).
        for bits in 0u32..=0xffff {
            let bits = bits as u16;
            let exp = (bits >> 10) & 0x1f;
            if exp == 0x1f {
                continue; // Inf / NaN handled above
            }
            assert_eq!(f32_to_f16(f16_to_f32(bits)), bits, "bits {bits:#06x}");
        }
    }

    #[test]
    fn narrowing_rounds_to_nearest_even() {
        // 1 + 2^-11 sits exactly halfway between 1.0 (even mantissa) and the
        // next binary16 value, so it must round down to 1.0.
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11)), 0x3c00);
        // 1 + 3·2^-11 is halfway between 0x3c01 (odd) and 0x3c02 → rounds up.
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2f32.powi(-11)), 0x3c02);
        // just over halfway always rounds away
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-11) * 1.001), 0x3c01);
    }
}
