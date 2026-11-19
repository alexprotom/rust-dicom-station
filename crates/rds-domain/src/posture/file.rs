//! The posture model as a file (`.rdsposture`, JSON), so a session can be
//! reopened without fitting again.
//!
//! What is written is what the model cannot recompute cheaply, and nothing
//! it can: the lattice and the grids, the anchor and the transforms into
//! each state, the states, and per segment its mode, its bodies (name,
//! colour, rigid motion, the fit's numbers, and the lattice nodes the body
//! covers at the start as runs of node indices), the residual field and
//! the linear end point (three `f32` per node), whether the bodies ride
//! curves through a chain's states, and the notes. The poly-rigid velocity
//! field (and a chain's curves and weights) is rebuilt on reading from the
//! bodies, their nodes and the reach, by the same deterministic code that
//! made it ([`super::polyrigid`], [`super::chain`]), so a file stays a few
//! hundred kilobytes for a model without a residual. Every struct is `#[serde(default)]`, so a
//! file written by another version reads with what it has.

use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::fit::FittedBody;
use super::polyrigid::{self, BodyOnLattice};
use super::se3::{Rigid, Twist};
use super::svf::{Field, Lattice, Svf};
use super::{Anchor, Mode, PostureModel, Segment, StateInfo};
use crate::geometry::Vec3;
use crate::volume::Grid;

/// The extension the files are written with.
pub const EXTENSION: &str = "rdsposture";
const FORMAT: u32 = 1;

fn v3(v: Vec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}
fn p3(a: [f64; 3]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}

#[derive(Serialize, Deserialize, Default, Clone)]
#[serde(default)]
struct GridF {
    dims: [usize; 3],
    spacing: [f64; 3],
    origin: [f64; 3],
    axes: [[f64; 3]; 3],
    for_uid: String,
}

impl GridF {
    fn of(g: &Grid) -> GridF {
        GridF {
            dims: g.dims,
            spacing: g.spacing,
            origin: v3(g.origin),
            axes: [v3(g.row_dir), v3(g.col_dir), v3(g.normal)],
            for_uid: g.frame_of_reference_uid.clone(),
        }
    }
    fn grid(&self) -> Grid {
        Grid {
            dims: self.dims,
            spacing: self.spacing,
            origin: p3(self.origin),
            row_dir: p3(self.axes[0]),
            col_dir: p3(self.axes[1]),
            normal: p3(self.axes[2]),
            frame_of_reference_uid: self.for_uid.clone(),
        }
    }
    fn lattice(&self) -> Lattice {
        Lattice {
            dims: self.dims,
            spacing: self.spacing,
            origin: p3(self.origin),
            axes: [p3(self.axes[0]), p3(self.axes[1]), p3(self.axes[2])],
        }
    }
}

#[derive(Serialize, Deserialize, Default, Clone)]
#[serde(default)]
struct RigidF {
    r: [[f64; 3]; 3],
    t: [f64; 3],
}

impl RigidF {
    fn of(g: &Rigid) -> RigidF {
        RigidF { r: g.r, t: v3(g.t) }
    }
    fn rigid(&self) -> Rigid {
        Rigid {
            r: self.r,
            t: p3(self.t),
        }
    }
}

#[derive(Serialize, Deserialize, Default, Clone)]
#[serde(default)]
struct StateF {
    label: String,
    date: String,
    series: String,
    position: String,
    posture: Option<f64>,
    grid: Option<GridF>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
#[serde(default)]
struct BodyF {
    name: String,
    color: [u8; 3],
    motion: RigidF,
    centroid: [f64; 3],
    mean_before_mm: f64,
    mean_after_mm: f64,
    rms_after_mm: f64,
    dice_after: Option<f64>,
    seeded: bool,
    anchor: bool,
    /// The lattice nodes the body covers at the start, as (first, count).
    nodes: Vec<(u32, u32)>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
#[serde(default)]
struct SegmentF {
    mode: String,
    bodies: Vec<BodyF>,
    /// Residual velocity, three per node.
    residual: Vec<f32>,
    /// Linear end displacement, three per node.
    linear: Vec<f32>,
    /// The bodies ride curves through the chain's states.
    smooth: bool,
    notes: Vec<String>,
}

#[derive(Serialize, Deserialize, Default, Clone)]
#[serde(default)]
struct ModelF {
    format: u32,
    made_by: String,
    lattice: GridF,
    grid: GridF,
    analysis: GridF,
    anchor: String,
    reach_mm: f64,
    to_state: Vec<RigidF>,
    states: Vec<StateF>,
    segments: Vec<SegmentF>,
    provenance: Vec<String>,
}

fn runs(occ: &[u8]) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < occ.len() {
        if occ[i] != 0 {
            let start = i;
            while i < occ.len() && occ[i] != 0 {
                i += 1;
            }
            out.push((start as u32, (i - start) as u32));
        } else {
            i += 1;
        }
    }
    out
}

fn unruns(r: &[(u32, u32)], n: usize) -> Vec<u8> {
    let mut occ = vec![0u8; n];
    for &(a, len) in r {
        for o in occ.iter_mut().skip(a as usize).take(len as usize) {
            *o = 1;
        }
    }
    occ
}

fn flat(d: &[Vec3]) -> Vec<f32> {
    d.iter()
        .flat_map(|v| [v.x as f32, v.y as f32, v.z as f32])
        .collect()
}

fn unflat(f: &[f32]) -> Vec<Vec3> {
    f.as_chunks::<3>()
        .0
        .iter()
        .map(|c| Vec3::new(c[0] as f64, c[1] as f64, c[2] as f64))
        .collect()
}

/// The occupancy each body of each segment had on the lattice at the start
/// of its segment: what [`write`] needs besides the model, since the model
/// keeps only the blended field. The workflow keeps it from the fit.
pub type Occupancies = Vec<Vec<Vec<u8>>>;

/// Write `model` to `path`.
pub fn write(path: &Path, model: &PostureModel, occupancy: &Occupancies) -> Result<()> {
    let n = model.lattice.len();
    let lattice = GridF {
        dims: model.lattice.dims,
        spacing: model.lattice.spacing,
        origin: v3(model.lattice.origin),
        axes: [
            v3(model.lattice.axes[0]),
            v3(model.lattice.axes[1]),
            v3(model.lattice.axes[2]),
        ],
        for_uid: String::new(),
    };
    let segments = model
        .segments
        .iter()
        .enumerate()
        .map(|(si, s)| SegmentF {
            mode: s.mode.key().into(),
            bodies: s
                .bodies
                .iter()
                .enumerate()
                .map(|(bi, b)| BodyF {
                    name: b.name.clone(),
                    color: b.color,
                    motion: RigidF::of(&b.motion),
                    centroid: v3(b.centroid),
                    mean_before_mm: b.mean_before_mm,
                    mean_after_mm: b.mean_after_mm,
                    rms_after_mm: b.rms_after_mm,
                    dice_after: b.dice_after,
                    seeded: b.seeded,
                    anchor: b.anchor,
                    nodes: occupancy
                        .get(si)
                        .and_then(|o| o.get(bi))
                        .map(|o| runs(o))
                        .unwrap_or_default(),
                })
                .collect(),
            residual: s
                .residual
                .as_ref()
                .map(|u| flat(&(0..n).map(|k| u.velocity(k)).collect::<Vec<_>>()))
                .unwrap_or_default(),
            linear: s.linear.as_ref().map(|d| flat(&d.d)).unwrap_or_default(),
            smooth: s.curves.is_some(),
            notes: s.notes.clone(),
        })
        .collect();
    let f = ModelF {
        format: FORMAT,
        made_by: format!("rust-dicom-station {}", env!("CARGO_PKG_VERSION")),
        lattice,
        grid: GridF::of(&model.grid),
        analysis: GridF::of(&model.analysis),
        anchor: model.anchor.key(),
        reach_mm: model.reach_mm,
        to_state: model.to_state.iter().map(RigidF::of).collect(),
        states: model
            .states
            .iter()
            .map(|s| StateF {
                label: s.label.clone(),
                date: s.date.clone(),
                series: s.series.clone(),
                position: s.position.clone(),
                posture: s.posture,
                grid: s.grid.as_ref().map(GridF::of),
            })
            .collect(),
        segments,
        provenance: model.provenance.clone(),
    };
    let text = serde_json::to_string(&f).context("encode the posture model")?;
    crate::pathlen::check(path)?;
    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

/// Read a model written by [`write`].
pub fn read(path: &Path) -> Result<PostureModel> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let f: ModelF = serde_json::from_str(&text).context("not a posture model file")?;
    if f.format == 0 || f.segments.is_empty() {
        bail!("{} holds no posture model", path.display());
    }
    let lattice = f.lattice.lattice();
    let n = lattice.len();
    if n == 0 {
        bail!("the model's lattice is empty");
    }
    let mut segments = Vec::with_capacity(f.segments.len());
    for s in &f.segments {
        let mode = Mode::from_key(&s.mode).unwrap_or_default();
        let bodies: Vec<FittedBody> = s
            .bodies
            .iter()
            .map(|b| {
                let motion = b.motion.rigid();
                FittedBody {
                    name: b.name.clone(),
                    color: b.color,
                    xi: if b.anchor {
                        Twist::ZERO
                    } else {
                        super::se3::log(&motion)
                    },
                    motion,
                    centroid: p3(b.centroid),
                    mean_before_mm: b.mean_before_mm,
                    mean_after_mm: b.mean_after_mm,
                    rms_after_mm: b.rms_after_mm,
                    dice_after: b.dice_after,
                    seeded: b.seeded,
                    anchor: b.anchor,
                }
            })
            .collect();
        let mut on: Vec<BodyOnLattice> = bodies
            .iter()
            .zip(&s.bodies)
            .map(|(b, bf)| {
                BodyOnLattice::new(b.name.clone(), b.xi, &unruns(&bf.nodes, n), &lattice)
            })
            .collect();
        polyrigid::separate(&mut on, &lattice);
        let velocity = polyrigid::blend(&lattice, &on, f.reach_mm);
        let curves = s.smooth.then(|| {
            let xis: Vec<Twist> = bodies.iter().map(|b| b.xi).collect();
            std::sync::Arc::new(super::chain::SegmentCurves::stationary(
                &xis,
                super::chain::SparseWeights::of(&lattice, &on, f.reach_mm),
            ))
        });
        let vec_field = |v: &[f32]| -> Option<Vec<Vec3>> { (v.len() == 3 * n).then(|| unflat(v)) };
        segments.push(Segment {
            mode,
            velocity,
            residual: vec_field(&s.residual).map(|v| Svf::from_vectors(&lattice, v)),
            linear: vec_field(&s.linear).map(|d| Field {
                lattice: lattice.clone(),
                d,
            }),
            bodies,
            curves,
            notes: s.notes.clone(),
        });
    }
    let mut model = PostureModel {
        lattice,
        grid: f.grid.grid(),
        analysis: f.analysis.grid(),
        anchor: Anchor::parse(&f.anchor),
        to_state: f.to_state.iter().map(RigidF::rigid).collect(),
        states: f
            .states
            .iter()
            .map(|s| StateInfo {
                label: s.label.clone(),
                date: s.date.clone(),
                series: s.series.clone(),
                position: s.position.clone(),
                posture: s.posture,
                grid: s.grid.as_ref().map(GridF::grid),
            })
            .collect(),
        segments,
        ends: Vec::new(),
        provenance: f.provenance,
        reach_mm: f.reach_mm,
    };
    if model.to_state.len() != model.states.len() || model.states.len() != model.segments.len() + 1
    {
        bail!("the model's states and segments do not agree");
    }
    super::chain::smooth(&mut model);
    model.finish();
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_of_nodes_round_trip() {
        let occ = vec![0, 1, 1, 0, 0, 1, 0, 1, 1, 1];
        assert_eq!(runs(&occ), vec![(1, 2), (5, 1), (7, 3)]);
        assert_eq!(unruns(&runs(&occ), occ.len()), occ);
    }
}
