//! Stationary velocity fields on a lattice: the exponential by scaling and
//! squaring, scaling in time, composition, the logarithm of a displacement
//! field, the Jacobian.
//!
//! A stationary velocity field `v` defines a deformation for every time,
//! the flow `phi_t = exp(t v)`: `phi_0` is the identity, `phi_1` is the
//! fitted transform, and every `phi_t` is invertible with inverse
//! `exp(-t v)`. That is the property the posture path is built on (Arsigny,
//! Commowick, Pennec, Ayache, *A log-Euclidean framework for statistics on
//! diffeomorphisms*, MICCAI 2006; Vercauteren et al., *Symmetric log-domain
//! diffeomorphic registration*, MICCAI 2008).
//!
//! **Representation.** A [`Lattice`] is a regular grid of nodes with its
//! own axes (those of the first state's image, so its distance maps are the
//! image's), a [`Field`] a displacement per node, an [`Svf`] a *twist* per
//! node ([`super::se3::Twist`]): the velocity at a point `p` near node `n`
//! is `xi_n(p) = w_n x p + v_n`. A plain vector field is the special case
//! `w = 0`; the poly-rigid layer fills `w` with the blended rotations.
//!
//! **Exponential.** Scaling and squaring: the velocity is divided by `2^N`,
//! the first small step is taken *exactly* at every node, as the rigid
//! motion `exp(xi_n / 2^N)` of that node's own twist, and the step is
//! composed with itself `N` times, `phi <- phi o phi`, each composition
//! reading the field trilinearly. Where the twist is the same over a region
//! (inside a rigid body) every step is affine there, trilinear
//! interpolation reproduces an affine field exactly, and the region moves
//! by exactly its rigid motion at every `t`; elsewhere the first-order
//! error of the usual `x + v / 2^N` start is gone and only the
//! interpolation's remains. `N` is chosen so the first step moves no node
//! further than a quarter of the lattice spacing.
//!
//! **Exact regions.** Where the field is a single twist over a region
//! (inside a rigid body's region of [`super::polyrigid`]), a node whose path
//! stays in the region moves along that twist's screw exactly, and the
//! scaling and squaring's answer for it - right to the interpolation error
//! where the path crosses cells whose corners are not all in the region -
//! is replaced by the exact rigid motion ([`ExactRegion`]).
//!
//! **Outside the lattice** a field is extended linearly from its edge cell
//! for two cells, then held constant: points carried past the lattice by a
//! large motion keep the motion of the edge rather than stopping dead.
//!
//! **Logarithm.** The velocity of a given displacement `d` is found by the
//! fixed-point iteration `v <- v + (d - exp(v))`, started at `v = d`, until
//! the largest residual is below a hundredth of a millimetre (a handful of
//! rounds for the smooth fields a B-spline registration returns). This is
//! the plain form of Arsigny's inverse scaling and squaring; it converges
//! for deformations whose gradient stays well below one, which a
//! regularised residual deformation is.
//!
//! **Jacobian.** `det(I + grad d)` at each node by central differences
//! along the lattice axes (one-sided at the faces), turned into patient
//! derivatives through the lattice axes.
//!
//! Compositions are per node (`par_iter` over nodes, no sums), so a field
//! is the same on any thread count; the maxima are taken in node order.

use rayon::prelude::*;

use super::se3::{self, Twist, M3};
use crate::geometry::Vec3;
use crate::registration::VectorField;
use crate::volume::Grid;

/// A regular lattice of nodes in patient space.
#[derive(Clone, Debug, PartialEq)]
pub struct Lattice {
    pub dims: [usize; 3],
    /// Node spacing along each axis, mm.
    pub spacing: [f64; 3],
    /// Patient position of node (0, 0, 0).
    pub origin: Vec3,
    /// Unit directions of the three lattice axes.
    pub axes: [Vec3; 3],
}

impl Lattice {
    /// A lattice at about `step_mm` over the voxel centres of `grid`, one
    /// node beyond them on every side, aligned with its axes.
    pub fn over(grid: &Grid, step_mm: f64) -> Lattice {
        let step = step_mm.max(0.5);
        let axes = [grid.row_dir, grid.col_dir, grid.normal];
        let dims: [usize; 3] = std::array::from_fn(|a| {
            let extent = (grid.dims[a].max(1) - 1) as f64 * grid.spacing[a];
            (extent / step).ceil() as usize + 3
        });
        let origin = grid.origin - axes[0] * step - axes[1] * step - axes[2] * step;
        Lattice {
            dims,
            spacing: [step; 3],
            origin,
            axes,
        }
    }

    pub fn len(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        i + self.dims[0] * (j + self.dims[1] * k)
    }

    /// `(i, j, k)` of node `n`.
    #[inline]
    pub fn ijk(&self, n: usize) -> [usize; 3] {
        let [nx, ny, _] = self.dims;
        [n % nx, (n / nx) % ny, n / (nx * ny)]
    }

    #[inline]
    pub fn node(&self, n: usize) -> Vec3 {
        let [i, j, k] = self.ijk(n);
        self.origin
            + self.axes[0] * (i as f64 * self.spacing[0])
            + self.axes[1] * (j as f64 * self.spacing[1])
            + self.axes[2] * (k as f64 * self.spacing[2])
    }

    /// Continuous node coordinates of a patient point.
    #[inline]
    pub fn coords(&self, p: Vec3) -> [f64; 3] {
        let d = p - self.origin;
        [
            d.dot(self.axes[0]) / self.spacing[0],
            d.dot(self.axes[1]) / self.spacing[1],
            d.dot(self.axes[2]) / self.spacing[2],
        ]
    }

    pub fn min_spacing(&self) -> f64 {
        self.spacing.iter().cloned().fold(f64::INFINITY, f64::min)
    }

    /// The lattice as an image grid (for distance maps and resampling).
    pub fn grid(&self) -> Grid {
        Grid {
            dims: self.dims,
            spacing: self.spacing,
            origin: self.origin,
            row_dir: self.axes[0],
            col_dir: self.axes[1],
            normal: self.axes[2],
            frame_of_reference_uid: String::new(),
        }
    }

    /// Is `p` within the nodes (no extension needed)?
    pub fn contains(&self, p: Vec3) -> bool {
        let u = self.coords(p);
        (0..3).all(|a| u[a] >= 0.0 && u[a] <= (self.dims[a] - 1) as f64)
    }

    /// Trilinear cell and weights of `p`, extended linearly up to two cells
    /// outside and constant beyond.
    #[inline]
    fn cell(&self, p: Vec3) -> ([usize; 3], [f64; 3]) {
        let u = self.coords(p);
        let mut base = [0usize; 3];
        let mut f = [0.0f64; 3];
        for a in 0..3 {
            let n = self.dims[a];
            if n < 2 {
                continue;
            }
            let top = (n - 2) as f64;
            let lo = u[a].floor().clamp(0.0, top);
            base[a] = lo as usize;
            f[a] = (u[a] - lo).clamp(-2.0, 3.0);
        }
        (base, f)
    }
}

/// Trilinear blend of eight corner values with (possibly extrapolating)
/// weights.
#[inline]
fn trilinear(c: [Vec3; 8], f: [f64; 3]) -> Vec3 {
    let l = |a: Vec3, b: Vec3, t: f64| a + (b - a) * t;
    let x00 = l(c[0], c[1], f[0]);
    let x10 = l(c[2], c[3], f[0]);
    let x01 = l(c[4], c[5], f[0]);
    let x11 = l(c[6], c[7], f[0]);
    l(l(x00, x10, f[1]), l(x01, x11, f[1]), f[2])
}

/// A displacement per node: the map `p -> p + d(p)`.
#[derive(Clone, Debug)]
pub struct Field {
    pub lattice: Lattice,
    pub d: Vec<Vec3>,
}

impl Field {
    /// The identity on `lattice`.
    pub fn zero(lattice: &Lattice) -> Field {
        Field {
            lattice: lattice.clone(),
            d: vec![Vec3::ZERO; lattice.len()],
        }
    }

    /// `f(p) - p` sampled at every node.
    pub fn from_map(lattice: &Lattice, f: impl Fn(Vec3) -> Vec3 + Sync) -> Field {
        let d = (0..lattice.len())
            .into_par_iter()
            .map(|n| {
                let p = lattice.node(n);
                f(p) - p
            })
            .collect();
        Field {
            lattice: lattice.clone(),
            d,
        }
    }

    /// Displacement at a patient point.
    #[inline]
    pub fn sample(&self, p: Vec3) -> Vec3 {
        let lat = &self.lattice;
        let (b, f) = lat.cell(p);
        let [nx, ny, nz] = lat.dims;
        let i1 = (b[0] + 1).min(nx - 1);
        let j1 = (b[1] + 1).min(ny - 1);
        let k1 = (b[2] + 1).min(nz - 1);
        let at = |i: usize, j: usize, k: usize| self.d[i + nx * (j + ny * k)];
        trilinear(
            [
                at(b[0], b[1], b[2]),
                at(i1, b[1], b[2]),
                at(b[0], j1, b[2]),
                at(i1, j1, b[2]),
                at(b[0], b[1], k1),
                at(i1, b[1], k1),
                at(b[0], j1, k1),
                at(i1, j1, k1),
            ],
            f,
        )
    }

    /// Where the field takes `p`.
    #[inline]
    pub fn map(&self, p: Vec3) -> Vec3 {
        p + self.sample(p)
    }

    /// `after o before`, on `before`'s lattice: `p -> after(before(p))`.
    pub fn compose(after: &Field, before: &Field) -> Field {
        let lat = &before.lattice;
        let d = (0..lat.len())
            .into_par_iter()
            .map(|n| {
                let p = lat.node(n);
                let q = p + before.d[n];
                q + after.sample(q) - p
            })
            .collect();
        Field {
            lattice: lat.clone(),
            d,
        }
    }

    /// The displacement scaled by `s`: the linear path `p + s d(p)`.
    pub fn scaled(&self, s: f64) -> Field {
        Field {
            lattice: self.lattice.clone(),
            d: self.d.par_iter().map(|v| *v * s).collect(),
        }
    }

    /// The inverse map, by the fixed-point iteration `x <- y - d(x)` at
    /// every node `y` (exact for a smooth field whose gradient is below
    /// one; where it is not, the iteration stops after `iterations` with
    /// what it has - which is what a non-invertible path deserves).
    pub fn inverse(&self, iterations: usize) -> Field {
        let lat = &self.lattice;
        let d = (0..lat.len())
            .into_par_iter()
            .map(|n| {
                let y = lat.node(n);
                let mut x = y - self.d[n];
                for _ in 0..iterations {
                    let next = y - self.sample(x);
                    let moved = (next - x).length();
                    x = next;
                    if moved < 1e-6 {
                        break;
                    }
                }
                x - y
            })
            .collect();
        Field {
            lattice: lat.clone(),
            d,
        }
    }

    /// The deformation gradient `I + grad d` at node `n`, in patient axes.
    pub fn gradient_at(&self, n: usize) -> M3 {
        let lat = &self.lattice;
        let ijk = lat.ijk(n);
        let mut g = se3::I3;
        for a in 0..3 {
            let na = lat.dims[a];
            if na < 2 {
                continue;
            }
            let mut lo = ijk;
            let mut hi = ijk;
            if ijk[a] > 0 {
                lo[a] -= 1;
            }
            if ijk[a] + 1 < na {
                hi[a] += 1;
            }
            let steps = (hi[a] - lo[a]) as f64;
            let da = (self.d[lat.index(hi[0], hi[1], hi[2])]
                - self.d[lat.index(lo[0], lo[1], lo[2])])
                * (1.0 / (steps * lat.spacing[a]));
            let ax = lat.axes[a];
            let dv = [da.x, da.y, da.z];
            let av = [ax.x, ax.y, ax.z];
            for r in 0..3 {
                for c in 0..3 {
                    g[r][c] += dv[r] * av[c];
                }
            }
        }
        g
    }

    /// `det(I + grad d)` at every node.
    pub fn jacobians(&self) -> Vec<f64> {
        (0..self.d.len())
            .into_par_iter()
            .map(|n| se3::det(&self.gradient_at(n)))
            .collect()
    }

    /// The deformation gradient of the map at a patient point, by central
    /// differences of [`Self::map`] `h` mm either side along the patient
    /// axes: what the path does to the tissue at that point, read through
    /// the same interpolation that moves it (a point inside a rigid body,
    /// whose cell's corners all move with the body, reads a rotation).
    pub fn gradient_at_point(&self, p: Vec3, h: f64) -> M3 {
        let mut g = [[0.0; 3]; 3];
        let axes = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        for (c, e) in axes.iter().enumerate() {
            let d = (self.map(p + *e * h) - self.map(p - *e * h)) * (0.5 / h);
            g[0][c] = d.x;
            g[1][c] = d.y;
            g[2][c] = d.z;
        }
        g
    }

    /// The deformation gradient at a patient point: that of the nearest
    /// node.
    pub fn gradient_near(&self, p: Vec3) -> M3 {
        let u = self.lattice.coords(p);
        let mut ijk = [0usize; 3];
        for a in 0..3 {
            ijk[a] = u[a].round().clamp(0.0, (self.lattice.dims[a] - 1) as f64) as usize;
        }
        self.gradient_at(self.lattice.index(ijk[0], ijk[1], ijk[2]))
    }

    /// The largest displacement, mm.
    pub fn max_norm(&self) -> f64 {
        self.d.iter().map(|v| v.length()).fold(0.0, f64::max)
    }

    /// The field in the registration module's form, for its glyphs, its
    /// deformed grid and a DICOM Deformable Spatial Registration.
    pub fn to_vector_field(&self, region: Option<String>) -> VectorField {
        VectorField {
            dims: self.lattice.dims,
            spacing: self.lattice.spacing,
            origin: self.lattice.origin,
            axes: self.lattice.axes,
            data: self.d.clone(),
            max_mag: self.max_norm(),
            region,
        }
    }
}

/// Nodes where the field is one twist and nothing else (inside a rigid
/// body's region of [`super::polyrigid`]): a node whose whole path stays
/// among them moves by that twist's rigid motion exactly, and
/// [`Svf::exp`] gives it that motion rather than the scaling and squaring's
/// approximation of it.
#[derive(Clone, Debug)]
pub struct ExactRegion {
    pub xi: Twist,
    /// Node indices, ascending.
    pub nodes: Vec<u32>,
}

/// A stationary velocity field: a twist per node.
#[derive(Clone, Debug)]
pub struct Svf {
    pub lattice: Lattice,
    pub xi: Vec<Twist>,
    /// Regions of a single twist (empty for a plain vector field).
    pub exact: Vec<ExactRegion>,
}

impl Svf {
    pub fn zero(lattice: &Lattice) -> Svf {
        Svf {
            lattice: lattice.clone(),
            xi: vec![Twist::ZERO; lattice.len()],
            exact: Vec::new(),
        }
    }

    /// A plain vector field (no turning) as a velocity field.
    pub fn from_vectors(lattice: &Lattice, v: Vec<Vec3>) -> Svf {
        Svf {
            lattice: lattice.clone(),
            xi: v.into_iter().map(Twist::translation).collect(),
            exact: Vec::new(),
        }
    }

    /// Velocity at node `n`.
    #[inline]
    pub fn velocity(&self, n: usize) -> Vec3 {
        self.xi[n].velocity(self.lattice.node(n))
    }

    /// The fastest node, mm per unit time.
    pub fn max_speed(&self) -> f64 {
        (0..self.xi.len())
            .map(|n| self.velocity(n).length())
            .fold(0.0, f64::max)
    }

    /// Is it zero everywhere?
    pub fn is_zero(&self) -> bool {
        self.xi.iter().all(|x| *x == Twist::ZERO)
    }

    /// How many squarings `exp(t v)` takes: enough that the first step
    /// moves no node further than a quarter of the spacing.
    pub fn squarings(&self, t: f64) -> usize {
        let reach = self.max_speed() * t.abs();
        let quarter = 0.25 * self.lattice.min_spacing();
        if reach <= quarter || reach == 0.0 {
            return 0;
        }
        ((reach / quarter).log2().ceil() as usize).min(24)
    }

    /// `exp(t v)` as a displacement field on the same lattice.
    pub fn exp(&self, t: f64) -> Field {
        let lat = &self.lattice;
        let n_sq = self.squarings(t);
        let s = t / (1u64 << n_sq) as f64;
        // The first step, exact per node.
        let d = (0..lat.len())
            .into_par_iter()
            .map(|n| {
                let p = lat.node(n);
                se3::exp(&self.xi[n].scaled(s)).map(p) - p
            })
            .collect();
        let mut field = Field {
            lattice: lat.clone(),
            d,
        };
        for _ in 0..n_sq {
            field = Field::compose(&field, &field);
        }
        self.overwrite_exact(&mut field, t);
        field
    }

    /// Give the nodes whose path to time `t` stays inside one region of a
    /// single twist the exact motion of that twist. The path is checked at
    /// sixteen times, each point by its nearest node.
    fn overwrite_exact(&self, field: &mut Field, t: f64) {
        const STEPS: usize = 16;
        let lat = &self.lattice;
        let [nx, ny, nz] = lat.dims;
        for region in &self.exact {
            if region.nodes.is_empty() {
                continue;
            }
            let mut member = vec![false; lat.len()];
            for &n in &region.nodes {
                member[n as usize] = true;
            }
            let steps: Vec<se3::Rigid> = (1..=STEPS)
                .map(|s| se3::exp(&region.xi.scaled(t * s as f64 / STEPS as f64)))
                .collect();
            let end = se3::exp(&region.xi.scaled(t));
            let exact: Vec<(usize, Vec3)> = region
                .nodes
                .par_iter()
                .filter_map(|&n| {
                    let n = n as usize;
                    let p = lat.node(n);
                    let stays = steps.iter().all(|g| {
                        let u = lat.coords(g.map(p));
                        let (i, j, k) = (u[0].round(), u[1].round(), u[2].round());
                        if i < 0.0 || j < 0.0 || k < 0.0 {
                            return false;
                        }
                        let (i, j, k) = (i as usize, j as usize, k as usize);
                        i < nx && j < ny && k < nz && member[i + nx * (j + ny * k)]
                    });
                    stays.then(|| (n, end.map(p) - p))
                })
                .collect();
            for (n, d) in exact {
                field.d[n] = d;
            }
        }
    }

    /// `s v`.
    pub fn scaled(&self, s: f64) -> Svf {
        Svf {
            lattice: self.lattice.clone(),
            xi: self.xi.iter().map(|x| x.scaled(s)).collect(),
            exact: self
                .exact
                .iter()
                .map(|r| ExactRegion {
                    xi: r.xi.scaled(s),
                    nodes: r.nodes.clone(),
                })
                .collect(),
        }
    }
}

/// What the logarithm of a displacement field reports besides the field.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct LogFit {
    /// Rounds the iteration took.
    pub rounds: usize,
    /// Largest `|exp(v) - d|` at the end, mm.
    pub residual_mm: f64,
}

/// The stationary velocity field whose exponential is `field` (see the
/// module doc).
pub fn log(field: &Field, max_rounds: usize) -> (Svf, LogFit) {
    const TOL_MM: f64 = 0.01;
    let lat = &field.lattice;
    let mut v = field.d.clone();
    let mut fit = LogFit::default();
    for round in 0..max_rounds.max(1) {
        let e = Svf::from_vectors(lat, v.clone()).exp(1.0);
        let r: Vec<Vec3> = field
            .d
            .par_iter()
            .zip(e.d.par_iter())
            .map(|(a, b)| *a - *b)
            .collect();
        let worst = r.iter().map(|x| x.length()).fold(0.0, f64::max);
        fit = LogFit {
            rounds: round + 1,
            residual_mm: worst,
        };
        if worst < TOL_MM {
            break;
        }
        v.par_iter_mut()
            .zip(r.par_iter())
            .for_each(|(a, b)| *a = *a + *b);
    }
    (Svf::from_vectors(lat, v), fit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posture::se3::{exp as se3_exp, log as se3_log, Rigid};

    fn lattice() -> Lattice {
        Lattice {
            dims: [24, 20, 18],
            spacing: [5.0, 5.0, 5.0],
            origin: Vec3::new(-60.0, -50.0, -40.0),
            axes: [
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
        }
    }

    fn rigid(angle_deg: f64) -> Rigid {
        let r = se3_exp(&Twist {
            w: Vec3::new(1.0, 0.2, -0.1).normalized() * angle_deg.to_radians(),
            v: Vec3::ZERO,
        })
        .r;
        Rigid {
            r,
            t: Vec3::new(4.0, -3.0, 6.0),
        }
    }

    #[test]
    fn one_rigid_twist_flows_to_its_motion_at_every_time() {
        let lat = lattice();
        let g = rigid(50.0);
        let xi = se3_log(&g);
        let v = Svf {
            lattice: lat.clone(),
            xi: vec![xi; lat.len()],
            exact: Vec::new(),
        };
        for t in [0.0, 0.25, 0.5, 1.0] {
            let f = v.exp(t);
            let want = se3_exp(&xi.scaled(t));
            // The nodes whose path stays on the lattice: past its faces the
            // extension is held after two cells, so exactness ends there.
            for n in 0..lat.len() {
                let p = lat.node(n);
                if p.length() > 15.0 {
                    continue;
                }
                let err = (f.map(p) - want.map(p)).length();
                assert!(err < 1e-9, "t {t} node {n}: {err}");
            }
        }
    }

    #[test]
    fn the_flow_backwards_undoes_the_flow_forwards() {
        let lat = lattice();
        // A smooth, turning, non-uniform field.
        let xi: Vec<Twist> = (0..lat.len())
            .map(|n| {
                let p = lat.node(n);
                let s = (-(p.length() / 40.0).powi(2)).exp();
                Twist {
                    w: Vec3::new(0.0, 0.0, 0.4 * s),
                    v: Vec3::new(3.0 * s, -2.0 * s, 1.0),
                }
            })
            .collect();
        let v = Svf {
            lattice: lat.clone(),
            xi,
            exact: Vec::new(),
        };
        for t in [0.3, 1.0] {
            let fwd = v.exp(t);
            let back = v.scaled(-1.0).exp(t);
            let round = Field::compose(&back, &fwd);
            // Inside, away from the faces where the extension is a guess.
            let mut worst: f64 = 0.0;
            for (n, d) in round.d.iter().enumerate() {
                let [i, j, k] = lat.ijk(n);
                if i < 3
                    || j < 3
                    || k < 3
                    || i + 4 > lat.dims[0]
                    || j + 4 > lat.dims[1]
                    || k + 4 > lat.dims[2]
                {
                    continue;
                }
                worst = worst.max(d.length());
            }
            // Interpolation on a 5 mm lattice: a few per cent of a step.
            assert!(worst < 0.3, "t {t}: {worst} mm");
            assert!(fwd.jacobians().iter().all(|&j| j > 0.0));
        }
    }

    #[test]
    fn the_logarithm_of_an_exponential_comes_back() {
        let lat = lattice();
        let vel: Vec<Vec3> = (0..lat.len())
            .map(|n| {
                let p = lat.node(n);
                let s = (-(p.length() / 35.0).powi(2)).exp();
                Vec3::new(4.0 * s, 1.5 * s, -2.0 * s)
            })
            .collect();
        let v = Svf::from_vectors(&lat, vel.clone());
        let d = v.exp(1.0);
        let (back, fit) = log(&d, 20);
        assert!(fit.residual_mm < 0.01, "{fit:?}");
        let worst = (0..lat.len())
            .map(|n| (back.velocity(n) - vel[n]).length())
            .fold(0.0, f64::max);
        assert!(worst < 0.05, "{worst}");
    }

    #[test]
    fn an_affine_field_is_sampled_exactly_inside_and_just_outside() {
        let lat = lattice();
        let g = rigid(20.0);
        let f = Field::from_map(&lat, |p| g.map(p));
        for p in [
            Vec3::new(0.3, 2.1, -7.7),
            Vec3::new(-61.0, 0.0, 0.0),
            Vec3::new(56.0, 47.0, 46.0),
        ] {
            assert!((f.map(p) - g.map(p)).length() < 1e-9, "{p:?}");
        }
        // det of a rigid motion is one, and its inverse is its inverse.
        assert!(f.jacobians().iter().all(|j| (j - 1.0).abs() < 1e-9));
        let inv = f.inverse(60);
        let p = Vec3::new(5.0, -4.0, 3.0);
        assert!((inv.map(p) - g.inverse().map(p)).length() < 1e-6);
    }
}
