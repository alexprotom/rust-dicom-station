//! The cubic B-spline Parzen window of Mattes mutual information, shared by
//! the 3D engine ([`super::plastimatch`]) and the 2D measures
//! ([`super::metric2d`]).

/// Cubic B-spline `β³(t)`: support `(-2, 2)`, sums to one over integer
/// shifts.
#[inline]
pub(crate) fn beta3(t: f64) -> f64 {
    let a = t.abs();
    if a < 1.0 {
        2.0 / 3.0 - a * a + a * a * a * 0.5
    } else if a < 2.0 {
        let d = 2.0 - a;
        d * d * d / 6.0
    } else {
        0.0
    }
}

/// `dβ³/dt`.
#[inline]
pub(crate) fn dbeta3(t: f64) -> f64 {
    let a = t.abs();
    if a < 1.0 {
        -2.0 * t + 1.5 * t * a
    } else if a < 2.0 {
        let d = 2.0 - a;
        -t.signum() * d * d * 0.5
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_window_is_a_partition_of_unity() {
        for i in 0..20 {
            let t = i as f64 / 20.0;
            let s: f64 = (-2..=2).map(|d| beta3(t - d as f64)).sum();
            assert!((s - 1.0).abs() < 1e-12, "t {t}: {s}");
        }
    }
}
