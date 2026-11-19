//! The posture model: a continuous, invertible deformation between studies
//! of one patient in different positions, which can be evaluated at any
//! fraction of the way, played, and measured along.
//!
//! One patient is often imaged in more than one position - lying and
//! seated, prone and supine, arms up and down, a full and an empty bladder.
//! A registration says where every point of one scan went in the other;
//! this says *how it got there*: each study is a **state**, and between two
//! neighbouring states the model is a path `Phi_t`, `t` in `[0, 1]`, with
//! `Phi_0` the identity and `Phi_1` the fitted transform. It is an
//! interpolation of anatomy that was observed at the end points, not a
//! biomechanical simulation: a state halfway is "the smoothest invertible
//! path between what was scanned", which is the right thing to look at and
//! to measure along, and the wrong thing to plan a treatment on.
//!
//! **The anchor frame.** Two studies in two positions have two frames of
//! reference, and "how far did the bladder move" has no answer until
//! something is held still. Every state is first brought into the frame of
//! the first by the rigid transform that lays its **anchor** - one
//! structure (the sacrum, a vertebra, the target), or the whole image, or
//! nothing when the scanner's frame is to be kept - onto the first state's
//! ([`Anchor`]). Everything after that happens in the first state's patient
//! coordinates (LPS millimetres): the images and structures of the other
//! states are resampled into it, the anchor stands still in it, and every
//! number of [`measure`] is in it.
//!
//! **Three layers** ([`fit`]), because bones move as rigid bodies, often
//! through large angles, while soft tissue follows them and deforms on top:
//!
//! 1. *Rigid bodies*: each structure marked rigid (by default the bones
//!    found on every state, [`fit::is_bone`]) gets its own rigid motion from
//!    the surface registration of [`crate::registration::shape`], seeded by
//!    the principal axes when the local search alone cannot reach a large
//!    turn.
//! 2. *Poly-rigid blend* ([`polyrigid`]): the bodies' motions as twists
//!    (matrix logarithms, [`se3`]) mixed with weights that are one in each
//!    body's swept region and fall off with the distance to it, a
//!    stationary velocity field `v` whose flow ([`svf`]) moves each body by
//!    exactly its screw and the space between them smoothly and invertibly.
//! 3. *Residual*: what is left after the bodies (organ filling, gravity on
//!    the abdomen) by the existing B-spline engines, faded to nothing inside
//!    the bodies, as a second velocity field `u`.
//!
//! **The path.** For a stationary velocity field the deformation at time
//! `t` is its flow for time `t`, so the whole path is
//! `Phi_t = exp(t v) o exp(t u)`: invertible at every `t` with inverse
//! `exp(-t u) o exp(-t v)`, symmetric (the way back is the way there walked
//! backwards), and a rigid body travels on a screw. The residual acts first,
//! in the first state's configuration where the bodies' fading is exact;
//! the plan this module was built from wrote the two in the other order,
//! which would let the residual bend a bone that had already moved. The
//! **linear** mode, `x + t d(x)` on the end-to-end displacement `d` of any
//! transform the station holds ([`Mode::Linear`]), is kept as the cheap
//! alternative: indistinguishable for small changes, it cuts the chord of a
//! turning bone instead of following the arc (the bone shrinks halfway),
//! and the table says when its Jacobian goes below zero on the way.
//!
//! **More than two states** are a chain: a model per neighbouring pair, `t`
//! running over `[0, n]`, each state optionally carrying a posture
//! parameter (an angle, a day, a filling) that the time axis can be read in
//! ([`PostureModel::param_at`]). Within a segment a body turns at a constant
//! rate; across a chain each body is put on a cubic curve through every
//! state instead ([`chain`]), so its velocity does not jump at a state.
//!
//! **What it cannot represent**: sliding (bowel against the abdominal wall,
//! lung against the chest wall: a smooth field shears where anatomy slips);
//! content that changes (gas, stool, a filling bladder are not a deformation
//! of the same material: the boundary moves and the grey values inside are
//! a cross-fade); anything outside both scans (the model is the overlap of
//! the fields of view); and the couch or the chair, which are not anatomy.
//!
//! **Where everything is.** [`se3`], [`svf`] and [`polyrigid`] are the
//! mathematics, with no DICOM and no interface, tested to machine
//! precision; [`fit`] the three layers; [`chain`] the bodies' curves
//! through the states of a chain; [`apply`] the path applied to
//! points, meshes, structure sets, volumes and doses, the key frames and
//! the three slices on screen ([`gpu`] the same slices on the GPU);
//! [`measure`] the four tables, the checks and the CSV; [`file`] the model
//! as a file; [`film`] the path as an animated GIF, rendered frame by
//! frame; [`crate::workflow::posture`] the one entry the viewer, the CLI
//! (`examples/posture_cli.rs`), the MCP server and the workflow step call.

pub mod apply;
pub mod chain;
pub mod file;
pub mod film;
pub mod fit;
pub mod gpu;
pub mod measure;
pub mod polyrigid;
pub mod se3;
pub mod svf;
pub mod template;

use std::sync::Arc;

use crate::geometry::Vec3;
use crate::volume::Grid;
use se3::Rigid;
use svf::{Field, Lattice, Svf};

/// How the path between two states is walked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Mode {
    /// The three layers, flowed: invertible at every `t`, rigid bodies on
    /// their screws.
    #[default]
    Diffeomorphic,
    /// `x + t d(x)` on the end-to-end displacement: no fitting of its own,
    /// right for small changes, wrong for a turning bone.
    Linear,
}

impl Mode {
    pub const ALL: [Mode; 2] = [Mode::Diffeomorphic, Mode::Linear];

    pub fn label(self) -> &'static str {
        match self {
            Mode::Diffeomorphic => "Diffeomorphic",
            Mode::Linear => "Linear",
        }
    }

    pub fn key(self) -> &'static str {
        match self {
            Mode::Diffeomorphic => "diffeomorphic",
            Mode::Linear => "linear",
        }
    }

    pub fn from_key(s: &str) -> Option<Mode> {
        Mode::ALL.into_iter().find(|m| {
            m.key().eq_ignore_ascii_case(s.trim()) || m.label().eq_ignore_ascii_case(s.trim())
        })
    }

    pub fn hint(self) -> &'static str {
        match self {
            Mode::Diffeomorphic => {
                "Rigid bodies moved on their own screws, the space between them \
                 blended smoothly, a residual deformation on top - invertible at \
                 every point of the path."
            }
            Mode::Linear => {
                "Every point walks the straight line to where the end-to-end \
                 transform puts it. Fine for a few millimetres; a turning bone \
                 cuts the chord instead of following the arc and shrinks halfway."
            }
        }
    }
}

/// What is held still.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Anchor {
    /// One structure contoured on every state, laid onto itself by a rigid
    /// surface fit.
    Structure(String),
    /// The whole image, by a rigid intensity registration.
    Image,
    /// Nothing: the states' own patient coordinates, for scans that share
    /// a frame of reference.
    Scanner,
    /// Chosen when the model is built ([`crate::workflow::posture::auto_anchor`]):
    /// the sacrum when every scan sees all of it, else the middle one of the
    /// vertebrae every scan sees whole, else the largest such bone, else the
    /// whole image. A request only: the model records what was chosen.
    Auto,
}

impl Anchor {
    pub fn label(&self) -> String {
        match self {
            Anchor::Structure(s) => s.clone(),
            Anchor::Image => "the whole image (rigid)".into(),
            Anchor::Scanner => "the scanner's frame (none)".into(),
            Anchor::Auto => "chosen when built".into(),
        }
    }

    /// `structure:NAME`, `image`, `scanner` or `auto`, the way the CLI and
    /// the MCP tool take it.
    pub fn parse(s: &str) -> Anchor {
        let t = s.trim();
        match t.to_ascii_lowercase().as_str() {
            "image" | "whole image" => Anchor::Image,
            "scanner" | "none" => Anchor::Scanner,
            "auto" | "" => Anchor::Auto,
            _ => Anchor::Structure(t.strip_prefix("structure:").unwrap_or(t).trim().to_string()),
        }
    }

    pub fn key(&self) -> String {
        match self {
            Anchor::Structure(s) => format!("structure:{s}"),
            Anchor::Image => "image".into(),
            Anchor::Scanner => "scanner".into(),
            Anchor::Auto => "auto".into(),
        }
    }
}

/// What the model knows about one state, besides its image.
#[derive(Clone, Default)]
pub struct StateInfo {
    pub label: String,
    pub date: String,
    pub series: String,
    /// The patient position the header gives (HFS, FFS, sitting, ...).
    pub position: String,
    /// The posture parameter typed for it (an angle, a day, a filling).
    pub posture: Option<f64>,
    /// The state's own image grid, in its own patient coordinates.
    pub grid: Option<Grid>,
}

/// The model between two neighbouring states.
#[derive(Clone, Debug)]
pub struct Segment {
    pub mode: Mode,
    /// Layer 2: the poly-rigid velocity field.
    pub velocity: Svf,
    /// Layer 3: the residual velocity field, applied first.
    pub residual: Option<Svf>,
    /// The linear mode's end-to-end displacement.
    pub linear: Option<Field>,
    /// Layer 1, with each body's fit.
    pub bodies: Vec<fit::FittedBody>,
    /// In a chain: the bodies on curves through the states ([`chain`]);
    /// `None`, every body on its stationary screw.
    pub curves: Option<Arc<chain::SegmentCurves>>,
    /// What went into it, one line each.
    pub notes: Vec<String>,
}

impl Segment {
    /// The poly-rigid layer at `s`: `exp(s v)`, or with curves the flow of
    /// [`chain::SegmentCurves::velocity_at`] for unit time; `sign` -1 for
    /// its inverse.
    fn blend_at(&self, s: f64, sign: f64) -> Field {
        match &self.curves {
            Some(c) if !c.is_stationary() => c.velocity_at(&self.velocity, s).scaled(sign).exp(1.0),
            _ => self.velocity.exp(sign * s),
        }
    }

    /// A body's motion from the segment's start to `s`, as the path moves
    /// it (its curve, or its screw).
    pub fn body_motion(&self, body: usize, s: f64) -> Option<Rigid> {
        let b = self.bodies.get(body)?;
        Some(
            match self.curves.as_ref().and_then(|c| c.bodies.get(body)) {
                Some(c) => c.pose(s),
                None => se3::exp(&b.xi.scaled(s.clamp(0.0, 1.0))),
            },
        )
    }

    /// A point of the start state at `s` in `[0, 1]` of the way: `Phi_s`.
    pub fn forward(&self, s: f64) -> Field {
        match self.mode {
            Mode::Diffeomorphic => {
                let v = self.blend_at(s, 1.0);
                match &self.residual {
                    Some(u) => Field::compose(&v, &u.exp(s)),
                    None => v,
                }
            }
            Mode::Linear => match &self.linear {
                Some(d) => d.scaled(s),
                None => Field::zero(&self.velocity.lattice),
            },
        }
    }

    /// The point of the start state that is at `y` at `s`: `Phi_s^-1`.
    pub fn backward(&self, s: f64) -> Field {
        match self.mode {
            Mode::Diffeomorphic => {
                let v = self.blend_at(s, -1.0);
                match &self.residual {
                    Some(u) => Field::compose(&u.exp(-s), &v),
                    None => v,
                }
            }
            Mode::Linear => self.forward(s).inverse(40),
        }
    }
}

/// One moment of the path: everything needed to draw it.
#[derive(Clone, Debug)]
pub struct Frame {
    /// Time on `[0, n]`.
    pub t: f64,
    /// The segment and the fraction of it.
    pub segment: usize,
    pub s: f64,
    /// A point of the first state to where it is at `t`.
    pub forward: Field,
    /// A point at `t` to where it is on the segment's start state.
    pub to_prev: Field,
    /// A point at `t` to where it is on the segment's end state.
    pub to_next: Field,
    /// A point at `t` to where it is on the first state.
    pub to_first: Field,
}

/// The model: the states, the anchor, and a segment between each pair.
#[derive(Clone)]
pub struct PostureModel {
    /// The lattice every field lives on, over the first state's image.
    pub lattice: Lattice,
    /// The first state's image grid: the anchor frame's own lattice.
    pub grid: Grid,
    /// The grid the analysis ran on (the first state's, coarsened).
    pub analysis: Grid,
    pub anchor: Anchor,
    /// Per state: the anchor frame (the first state's patient coordinates)
    /// to that state's own patient coordinates. The first is the identity.
    pub to_state: Vec<Rigid>,
    pub states: Vec<StateInfo>,
    pub segments: Vec<Segment>,
    /// Per segment: `Phi_1` and its inverse, once.
    pub ends: Vec<(Field, Field)>,
    /// How each piece was made.
    pub provenance: Vec<String>,
    /// The blend's reach, mm.
    pub reach_mm: f64,
}

impl PostureModel {
    /// Fill in the segments' end fields.
    pub fn finish(&mut self) {
        self.ends = self
            .segments
            .iter()
            .map(|s| (s.forward(1.0), s.backward(1.0)))
            .collect();
    }

    /// The length of the time axis: one per segment.
    pub fn span(&self) -> f64 {
        self.segments.len() as f64
    }

    pub fn mode(&self) -> Mode {
        self.segments.first().map(|s| s.mode).unwrap_or_default()
    }

    /// The segment `t` falls in, and the fraction of it.
    pub fn locate(&self, t: f64) -> (usize, f64) {
        let n = self.segments.len().max(1);
        let t = t.clamp(0.0, n as f64);
        let i = (t.floor() as usize).min(n - 1);
        (i, t - i as f64)
    }

    /// Everything at time `t` (see [`Frame`]).
    pub fn frame(&self, t: f64) -> Frame {
        let (i, s) = self.locate(t);
        let seg = &self.segments[i];
        let fwd_s = seg.forward(s);
        let back_s = seg.backward(s);
        let mut forward = fwd_s.clone();
        let mut to_first = back_s.clone();
        for k in (0..i).rev() {
            forward = Field::compose(&forward, &self.ends[k].0);
        }
        for k in (0..i).rev() {
            to_first = Field::compose(&self.ends[k].1, &to_first);
        }
        let to_next = Field::compose(&self.ends[i].0, &back_s);
        Frame {
            t,
            segment: i,
            s,
            forward,
            to_prev: back_s,
            to_next,
            to_first,
        }
    }

    /// The posture parameter at time `t`, piecewise linear between the
    /// states, when every state has one.
    pub fn param_at(&self, t: f64) -> Option<f64> {
        let p: Option<Vec<f64>> = self.states.iter().map(|s| s.posture).collect();
        let p = p?;
        if p.len() < 2 {
            return None;
        }
        let (i, s) = self.locate(t);
        Some(p[i] + (p[i + 1] - p[i]) * s)
    }

    /// The time at which the posture parameter is `value` (the inverse of
    /// [`Self::param_at`] on a monotone sequence of parameters).
    pub fn time_of_param(&self, value: f64) -> Option<f64> {
        let p: Option<Vec<f64>> = self.states.iter().map(|s| s.posture).collect();
        let p = p?;
        for i in 0..p.len().saturating_sub(1) {
            let (a, b) = (p[i], p[i + 1]);
            if (a - b).abs() < 1e-12 {
                continue;
            }
            let s = (value - a) / (b - a);
            if (-1e-9..=1.0 + 1e-9).contains(&s) {
                return Some(i as f64 + s.clamp(0.0, 1.0));
            }
        }
        None
    }

    /// A point of state `k`'s own patient coordinates in the anchor frame.
    pub fn into_anchor(&self, k: usize, p: Vec3) -> Vec3 {
        self.to_state[k].inverse().map(p)
    }

    /// A rigid body's motion from the first state to time `t`, as the path
    /// moves it: the segments before composed, then its curve (or screw)
    /// in the segment `t` is in. `None` when some segment lacks the body.
    pub fn body_pose(&self, name: &str, t: f64) -> Option<Rigid> {
        let (i, s) = self.locate(t);
        let mut g = Rigid::IDENTITY;
        for (k, seg) in self.segments.iter().enumerate().take(i + 1) {
            let b = seg.bodies.iter().position(|b| b.name == name)?;
            let step = if k < i {
                seg.body_motion(b, 1.0)?
            } else {
                seg.body_motion(b, s)?
            };
            g = g.then(&step);
        }
        Some(g)
    }

    /// The names of the rigid bodies, in the first segment's order.
    pub fn body_names(&self) -> Vec<String> {
        self.segments
            .first()
            .map(|s| s.bodies.iter().map(|b| b.name.clone()).collect())
            .unwrap_or_default()
    }
}

/// `grid` coarsened to at least `min_mm` per axis: the same extent and
/// orientation, voxel centres at the centres of the blocks they stand for.
pub fn coarsen(grid: &Grid, min_mm: f64) -> Grid {
    let mut out = grid.clone();
    let axes = [grid.row_dir, grid.col_dir, grid.normal];
    let mut origin = grid.origin;
    for (a, axis) in axes.iter().enumerate() {
        let f = (min_mm / grid.spacing[a]).floor().max(1.0) as usize;
        if f <= 1 || grid.dims[a] <= 1 {
            continue;
        }
        out.dims[a] = grid.dims[a].div_ceil(f);
        out.spacing[a] = grid.spacing[a] * f as f64;
        origin = origin + *axis * (grid.spacing[a] * (f as f64 - 1.0) * 0.5);
    }
    out.origin = origin;
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anchors_and_modes_read_back_from_their_keys() {
        for a in [
            Anchor::Structure("sacrum".into()),
            Anchor::Image,
            Anchor::Scanner,
            Anchor::Auto,
        ] {
            assert_eq!(Anchor::parse(&a.key()), a);
        }
        assert_eq!(
            Anchor::parse("BONE_UPPER"),
            Anchor::Structure("BONE_UPPER".into())
        );
        for m in Mode::ALL {
            assert_eq!(Mode::from_key(m.key()), Some(m));
        }
    }

    #[test]
    fn a_coarsened_grid_keeps_its_extent() {
        let g = Grid {
            dims: [512, 512, 100],
            spacing: [0.97, 0.97, 3.0],
            origin: Vec3::new(-248.0, -248.0, -150.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
        };
        let c = coarsen(&g, 2.0);
        assert_eq!(c.dims, [256, 256, 100]);
        assert!((c.spacing[0] - 1.94).abs() < 1e-12);
        // The first coarse voxel is centred on the first block of two.
        assert!((c.origin.x - (-248.0 + 0.485)).abs() < 1e-9);
        assert_eq!(c.spacing[2], 3.0);
    }
}
