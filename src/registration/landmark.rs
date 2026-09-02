//! Landmark-based deformable registration — plastimatch's `landmark_warp`.
//!
//! No image intensity is read at any point: the deformation is a radial
//! basis interpolation of displacements the user measured by hand, which is
//! exactly what is wanted when the two images have nothing an intensity
//! metric can lock onto (CT against MR, a post-operative cavity, a study
//! whose anatomy genuinely changed) or when a physicist wants the alignment
//! to honour specific anatomical points and nothing else.
//!
//! Three kernels, the ones plastimatch offers:
//!
//! | kernel | φ(r) | support | exact at the landmarks |
//! |---|---|---|---|
//! | Thin-plate spline | `r` | global, plus an affine term | yes (λ = 0) |
//! | Gaussian | `exp(−r² / 2R²)` | global but decaying | yes (λ = 0) |
//! | Wendland ψ₃,₁ | `(1 − r/R)⁴ (4r/R + 1)` | compact, zero beyond `R` | yes (λ = 0) |
//!
//! The thin-plate spline is the classic choice: it minimizes bending energy
//! over the whole domain and carries an affine term, so a global shift or
//! rotation implied by the landmarks is represented exactly. The two radial
//! kernels have no affine term, so the displacement decays back to zero away
//! from the landmarks — which is the point of the compactly supported
//! Wendland kernel: a local correction that provably leaves distant anatomy
//! untouched.
//!
//! `stiffness` is plastimatch's regularization: it is added to the diagonal
//! of the interpolation matrix, which trades exactness at the landmarks for
//! a smoother field (and rescues a system made singular by two landmarks
//! placed on top of each other).

use anyhow::{bail, Result};

use super::*;

/// Which radial basis interpolates the landmark displacements.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LandmarkKernel {
    /// Thin-plate spline: `φ(r) = r`, with an affine term. Global, minimizes
    /// bending energy, reproduces an affine field exactly.
    #[default]
    ThinPlate,
    /// `φ(r) = exp(−r² / 2R²)`. Global but decaying; smooth everywhere.
    Gaussian,
    /// Wendland ψ₃,₁: `φ(r) = (1 − r/R)⁴ (4r/R + 1)`, zero beyond `R`.
    /// The only kernel that provably leaves distant anatomy untouched.
    Wendland,
}

impl LandmarkKernel {
    pub const ALL: [LandmarkKernel; 3] = [
        LandmarkKernel::ThinPlate,
        LandmarkKernel::Gaussian,
        LandmarkKernel::Wendland,
    ];

    pub fn label(self) -> &'static str {
        match self {
            LandmarkKernel::ThinPlate => "Thin-plate spline",
            LandmarkKernel::Gaussian => "Gaussian RBF",
            LandmarkKernel::Wendland => "Wendland RBF",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            LandmarkKernel::ThinPlate => {
                "φ(r) = r with an affine term: the classic minimum-bending-energy warp. \
                 Global - every landmark influences the whole volume - and it reproduces \
                 a global shift or rotation exactly. Needs at least four landmarks that \
                 are not coplanar."
            }
            LandmarkKernel::Gaussian => {
                "φ(r) = exp(−r²/2R²). Smooth and global, but the displacement decays \
                 with distance, so anatomy far from every landmark is left almost alone. \
                 The radius sets how far a landmark reaches."
            }
            LandmarkKernel::Wendland => {
                "φ(r) = (1 − r/R)⁴(4r/R + 1), exactly zero beyond the radius: a strictly \
                 local correction. Nothing outside the radius of some landmark moves at \
                 all, which is what a local fix-up of one structure should do."
            }
        }
    }

    /// True when the kernel has a finite reach the user sets.
    pub fn uses_radius(self) -> bool {
        self != LandmarkKernel::ThinPlate
    }

    /// φ(r) with the given radius (ignored by the thin-plate spline).
    #[inline]
    fn phi(self, r: f64, radius: f64) -> f64 {
        match self {
            LandmarkKernel::ThinPlate => r,
            LandmarkKernel::Gaussian => {
                let s = (r / radius.max(1e-6)).powi(2);
                (-0.5 * s).exp()
            }
            LandmarkKernel::Wendland => {
                let t = r / radius.max(1e-6);
                if t >= 1.0 {
                    0.0
                } else {
                    (1.0 - t).powi(4) * (4.0 * t + 1.0)
                }
            }
        }
    }

    /// Does the system carry the four-term affine block?
    fn has_affine(self) -> bool {
        self == LandmarkKernel::ThinPlate
    }
}

/// Kernel, stiffness and reach of a landmark warp.
#[derive(Clone, Copy, Debug)]
pub struct LandmarkParams {
    pub kernel: LandmarkKernel,
    /// Added to the diagonal of the interpolation matrix (plastimatch's
    /// regularization): 0 interpolates the landmarks exactly, larger values
    /// smooth the field and tolerate inconsistent pairs.
    pub stiffness: f64,
    /// Reach of the Gaussian and Wendland kernels, mm.
    pub radius_mm: f64,
}

impl Default for LandmarkParams {
    fn default() -> Self {
        LandmarkParams {
            kernel: LandmarkKernel::ThinPlate,
            stiffness: 0.0,
            radius_mm: 50.0,
        }
    }
}

/// One paired point: where it is in the fixed image, and where the same
/// anatomy is in the moving image.
#[derive(Clone, Debug)]
pub struct LandmarkPair {
    pub name: String,
    pub fixed: Vec3,
    pub moving: Vec3,
}

impl LandmarkPair {
    pub fn new(name: impl Into<String>, fixed: Vec3, moving: Vec3) -> Self {
        LandmarkPair {
            name: name.into(),
            fixed,
            moving,
        }
    }

    /// The displacement the pair asks for, mm.
    pub fn displacement(&self) -> Vec3 {
        self.moving - self.fixed
    }
}

/// A solved radial-basis deformation.
#[derive(Clone, Debug)]
pub struct RbfWarp {
    pub kernel: LandmarkKernel,
    pub radius: f64,
    /// Fixed-image positions of the landmarks.
    pub centers: Vec<Vec3>,
    /// The displacement each landmark asked for (for the residual readout).
    pub targets: Vec<Vec3>,
    /// One weight vector per centre.
    weights: Vec<Vec3>,
    /// `[a0, ax, ay, az]` — the affine term, empty for the radial kernels.
    affine: Vec<Vec3>,
}

impl RbfWarp {
    /// Displacement at a fixed-image point.
    #[inline]
    pub fn displacement(&self, p: Vec3) -> Vec3 {
        let mut d = Vec3::ZERO;
        for (c, w) in self.centers.iter().zip(&self.weights) {
            let phi = self.kernel.phi((p - *c).length(), self.radius);
            if phi != 0.0 {
                d = d + *w * phi;
            }
        }
        if self.affine.len() == 4 {
            d = d
                + self.affine[0]
                + self.affine[1] * p.x
                + self.affine[2] * p.y
                + self.affine[3] * p.z;
        }
        d
    }

    /// How far each landmark ends up from where it was asked to go, mm.
    pub fn residuals(&self) -> Vec<f64> {
        self.centers
            .iter()
            .zip(&self.targets)
            .map(|(c, t)| (self.displacement(*c) - *t).length())
            .collect()
    }

    /// Root-mean-square landmark residual, mm.
    pub fn rms_residual(&self) -> f64 {
        let r = self.residuals();
        if r.is_empty() {
            return 0.0;
        }
        (r.iter().map(|v| v * v).sum::<f64>() / r.len() as f64).sqrt()
    }

    /// One line for the result panel.
    pub fn describe(&self) -> String {
        match self.kernel {
            LandmarkKernel::ThinPlate => format!(
                "{} through {} landmark(s)",
                self.kernel.label(),
                self.centers.len()
            ),
            _ => format!(
                "{} through {} landmark(s), reach {:.0} mm",
                self.kernel.label(),
                self.centers.len(),
                self.radius
            ),
        }
    }
}

/// Solve `a · x = b` in place for `m` right-hand sides, by Gaussian
/// elimination with partial pivoting.
///
/// `a` is `n × n` row-major and `b` is `n × m` row-major; both are consumed.
/// The systems here are tens to a few hundred unknowns, so a dense direct
/// solve is both the simplest and the fastest thing available — and it is
/// exact, which an iterative solver on an ill-conditioned thin-plate matrix
/// would not be.
fn solve_in_place(a: &mut [f64], b: &mut [f64], n: usize, m: usize) -> Result<()> {
    for col in 0..n {
        // Partial pivot.
        let mut piv = col;
        let mut best = a[col * n + col].abs();
        for r in col + 1..n {
            let v = a[r * n + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-12 {
            bail!(
                "the landmark system is singular - landmarks are coincident or coplanar; \
                 move one, add one, or raise the stiffness"
            );
        }
        if piv != col {
            for c in 0..n {
                a.swap(col * n + c, piv * n + c);
            }
            for c in 0..m {
                b.swap(col * m + c, piv * m + c);
            }
        }
        let d = a[col * n + col];
        for r in col + 1..n {
            let f = a[r * n + col] / d;
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                a[r * n + c] -= f * a[col * n + c];
            }
            for c in 0..m {
                b[r * m + c] -= f * b[col * m + c];
            }
        }
    }
    // Back substitution.
    for col in (0..n).rev() {
        for c in 0..m {
            let mut acc = b[col * m + c];
            for k in col + 1..n {
                acc -= a[col * n + k] * b[k * m + c];
            }
            b[col * m + c] = acc / a[col * n + col];
        }
    }
    Ok(())
}

/// Solve the interpolation system for a set of landmark pairs.
pub fn solve(pairs: &[LandmarkPair], p: &LandmarkParams) -> Result<RbfWarp> {
    let n = pairs.len();
    if n == 0 {
        bail!("place at least one landmark pair before running a landmark warp");
    }
    if p.kernel.has_affine() && n < 4 {
        bail!(
            "the thin-plate spline needs at least 4 landmark pairs (it also solves for an \
             affine term); {n} placed - add more, or switch to the Gaussian or Wendland kernel"
        );
    }
    let extra = if p.kernel.has_affine() { 4 } else { 0 };
    let dim = n + extra;
    let mut a = vec![0.0f64; dim * dim];
    let mut b = vec![0.0f64; dim * 3];

    for i in 0..n {
        for j in 0..n {
            a[i * dim + j] = p
                .kernel
                .phi((pairs[i].fixed - pairs[j].fixed).length(), p.radius_mm);
        }
        a[i * dim + i] += p.stiffness;
        if extra == 4 {
            let q = pairs[i].fixed;
            let row = [1.0, q.x, q.y, q.z];
            for (k, v) in row.iter().enumerate() {
                a[i * dim + n + k] = *v;
                a[(n + k) * dim + i] = *v;
            }
        }
        let d = pairs[i].displacement();
        b[i * 3] = d.x;
        b[i * 3 + 1] = d.y;
        b[i * 3 + 2] = d.z;
    }

    solve_in_place(&mut a, &mut b, dim, 3)?;

    let weights: Vec<Vec3> = (0..n)
        .map(|i| Vec3::new(b[i * 3], b[i * 3 + 1], b[i * 3 + 2]))
        .collect();
    let affine: Vec<Vec3> = (n..dim)
        .map(|i| Vec3::new(b[i * 3], b[i * 3 + 1], b[i * 3 + 2]))
        .collect();

    Ok(RbfWarp {
        kernel: p.kernel,
        radius: p.radius_mm,
        centers: pairs.iter().map(|l| l.fixed).collect(),
        targets: pairs.iter().map(|l| l.displacement()).collect(),
        weights,
        affine,
    })
}

/// The landmark engine: solve the system, wrap it as a transform.
pub(super) fn run(params: &RegParams) -> Result<EngineOutput> {
    let warp = solve(&params.landmarks, &params.landmark)?;
    let residual = warp.rms_residual();
    // The identity rigid part keeps the composition rule the same for every
    // method; the affine content of a thin-plate solution lives in the warp.
    let center = params.landmarks.iter().fold(Vec3::ZERO, |a, l| a + l.fixed)
        * (1.0 / params.landmarks.len() as f64);
    Ok(EngineOutput {
        transform: Transform3 {
            rigid: RigidTransform::identity(center),
            warp: Warp::Rbf(warp),
        },
        iterations: params.landmarks.len(),
        final_metric: residual,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs_from(shift: Vec3) -> Vec<LandmarkPair> {
        let corners = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(100.0, 0.0, 0.0),
            Vec3::new(0.0, 100.0, 0.0),
            Vec3::new(0.0, 0.0, 100.0),
            Vec3::new(100.0, 100.0, 100.0),
        ];
        corners
            .iter()
            .enumerate()
            .map(|(i, c)| LandmarkPair::new(format!("L{i}"), *c, *c + shift))
            .collect()
    }

    #[test]
    fn every_kernel_interpolates_its_landmarks_exactly() {
        for kernel in LandmarkKernel::ALL {
            let pairs = pairs_from(Vec3::new(3.0, -2.0, 1.0));
            let p = LandmarkParams {
                kernel,
                stiffness: 0.0,
                radius_mm: 200.0,
            };
            let w = solve(&pairs, &p).unwrap();
            let res = w.residuals();
            assert_eq!(res.len(), pairs.len());
            for (i, r) in res.iter().enumerate() {
                assert!(*r < 1e-6, "{}: landmark {i} off by {r} mm", kernel.label());
            }
            assert!(w.rms_residual() < 1e-6);
            assert!(w.describe().contains("landmark"));
        }
    }

    #[test]
    fn the_thin_plate_spline_reproduces_a_global_shift_everywhere() {
        let shift = Vec3::new(4.0, -5.0, 6.0);
        let w = solve(
            &pairs_from(shift),
            &LandmarkParams {
                kernel: LandmarkKernel::ThinPlate,
                ..Default::default()
            },
        )
        .unwrap();
        // Far from every landmark the affine term must still carry the shift.
        for p in [
            Vec3::new(50.0, 50.0, 50.0),
            Vec3::new(-40.0, 130.0, 20.0),
            Vec3::new(400.0, 400.0, 400.0),
        ] {
            assert!(
                (w.displacement(p) - shift).length() < 1e-6,
                "at {p:?} got {:?}",
                w.displacement(p)
            );
        }
    }

    #[test]
    fn the_wendland_kernel_leaves_distant_anatomy_alone() {
        let w = solve(
            &pairs_from(Vec3::new(5.0, 0.0, 0.0)),
            &LandmarkParams {
                kernel: LandmarkKernel::Wendland,
                stiffness: 0.0,
                radius_mm: 30.0,
            },
        )
        .unwrap();
        let far = Vec3::new(1000.0, 1000.0, 1000.0);
        assert_eq!(w.displacement(far), Vec3::ZERO);
        // …while a point right beside a landmark still moves.
        assert!(w.displacement(Vec3::new(2.0, 0.0, 0.0)).length() > 1.0);
    }

    #[test]
    fn stiffness_trades_exactness_for_smoothness() {
        let mut pairs = pairs_from(Vec3::new(3.0, 0.0, 0.0));
        // One inconsistent pair: the same place, a different displacement.
        pairs.push(LandmarkPair::new(
            "odd",
            Vec3::new(0.5, 0.0, 0.0),
            Vec3::new(0.5 + 30.0, 0.0, 0.0),
        ));
        let stiff = solve(
            &pairs,
            &LandmarkParams {
                kernel: LandmarkKernel::Gaussian,
                stiffness: 1.0,
                radius_mm: 60.0,
            },
        )
        .unwrap();
        assert!(stiff.rms_residual() > 1e-3, "a stiff fit is not exact");
        let exact = solve(
            &pairs,
            &LandmarkParams {
                kernel: LandmarkKernel::Gaussian,
                stiffness: 0.0,
                radius_mm: 60.0,
            },
        )
        .unwrap();
        assert!(exact.rms_residual() < stiff.rms_residual());
    }

    #[test]
    fn too_few_or_degenerate_landmarks_are_reported_not_guessed() {
        let params = LandmarkParams::default();
        assert!(solve(&[], &params).is_err());
        let three: Vec<LandmarkPair> = pairs_from(Vec3::ZERO).into_iter().take(3).collect();
        let e = solve(&three, &params).unwrap_err().to_string();
        assert!(e.contains("at least 4"), "{e}");
        // Two landmarks in the same place, different answers: singular.
        let dup = vec![
            LandmarkPair::new("a", Vec3::ZERO, Vec3::new(1.0, 0.0, 0.0)),
            LandmarkPair::new("b", Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0)),
        ];
        assert!(solve(
            &dup,
            &LandmarkParams {
                kernel: LandmarkKernel::Gaussian,
                stiffness: 0.0,
                radius_mm: 10.0,
            }
        )
        .is_err());
    }
}
