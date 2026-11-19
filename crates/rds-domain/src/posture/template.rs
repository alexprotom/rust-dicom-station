//! Posture templates: what a cohort of posture models says about how one
//! change of posture (lying to perched, say) moves the skeleton and the
//! organs, and the motion of one more patient's bones predicted from a
//! single scan.
//!
//! **What a model leaves behind.** A model of one patient between two
//! states ([`super::fit`]) knows each bone's rigid motion in the anchor
//! frame. Its **sample** ([`Sample`]) keeps what a cohort needs of it and
//! nothing of the images: per bone its centroid, seven points along its
//! principal axes (to measure an error with) and its motion; per pair of
//! bones that touch, the **joint centre** - the middle of the two patches
//! of their contours that lie within [`CONTACT_MM`] of each other (the
//! endplates of two vertebrae, a rib's head on its vertebra, the femoral
//! head in its socket); per organ (not a bone, not the outline, not cut by
//! either scan) its centroid at the start, as contoured at the end, and as
//! the bones alone carry it, with the bones' local deformation gradient
//! there. No name, date or identifier of the patient is in it: the label
//! is whatever the caller passes (a pseudonymised folder name).
//!
//! **What the cohort says about a joint.** For bones `a` and `b` that
//! touch, `M = T_a^-1 o T_b` is `b`'s motion relative to `a`, in the first
//! state's coordinates: it does not depend on which bone was the anchor,
//! so models anchored on the sacrum, on T12 and on L1 all speak of the
//! same joint. It is kept as a rotation vector `w` and the joint centre's
//! own displacement `tau = M(c) - c` (small when `c` is a good centre of
//! rotation); a template's estimate of the joint is the componentwise
//! **median** of `w` and `tau` over the patients that have it, and its
//! spread the median distance from those medians - both robust, so one
//! misfitted bone (a rib laid on itself backwards) moves neither. Every
//! pair of bones of a patient, touching or not, is also kept the same way
//! as a **direct** relation about the middle of their two centroids: a rib
//! barely moves relative to the vertebrae near it, while the chain of
//! joints from the root out to it adds every joint's spread on the way.
//! Rotation vectors are read in the first state's patient
//! axes: every state of a cohort lying the same way (HFS) has them along
//! the same anatomical directions to within the patients' own supine
//! posture, which is a second-order error for turns of ten or twenty
//! degrees. Anatomical bone frames (endplate normals, the anterior pelvic
//! plane) would remove it, and are the next step.
//!
//! **Predicting a patient.** On a new scan with its bones contoured, the
//! joints are found the same way, and the bones' motions are walked out
//! from a root (the sacrum when the scan sees it whole, else the middle
//! vertebra: [`crate::workflow::posture::auto_anchor`]) along the relations
//! the cohort knows, each child moved by `T_parent o M` with the
//! relation's median turn about this patient's own joint centre (or the
//! middle of the two centroids, for a direct relation). A bone can be
//! reached along many paths - its joint, or straight from a bone near it -
//! and the path whose accumulated spread is least is taken (Dijkstra): the
//! variance a relation adds at the child's centroid,
//! `(sd_w * lever)^2 + sd_tau^2`, plus one per hop. A bone no relation
//! seen on enough patients reaches stays out of the bodies and follows its
//! neighbours.
//!
//! **Organs.** After the bones, an organ that sags when the patient sits up
//! is somewhere else than the bones carry it: `delta = end - carried`. In
//! the template that offset is stored pulled back through the bones' local
//! gradient, `G^-1 delta` - the displacement to apply in the first state,
//! before the bones move it - so it does not depend on how far the bones
//! around it turned in that patient; its median over the cohort is applied
//! by [`crate::workflow::posture::predict`] as a residual velocity, the
//! organ moved as a whole and faded at its boundary. Organs whose content
//! changes between scans ([`content_changes`]: bladder, bowel, stomach,
//! gas, lungs with breathing) are not deformations of the same material,
//! and are left out of the offsets.
//!
//! **Leave one out** ([`leave_one_out`]): each patient predicted from the
//! others, bones against its own fitted motions (relative to the root, at
//! the seven points of each bone) and organs against their contoured
//! centroids, with the error of predicting no motion at all beside each.
//!
//! The file (`.rdsposture-template`, JSON) is the samples themselves, so a
//! template can be extended by one more patient, and every statistic is
//! recomputed when it is read.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::se3::{self, Rigid, Twist, M3};
use crate::geometry::Vec3;
use crate::propagate::BoxedMask;
use crate::volume::Grid;

/// The file extension of a template.
pub const EXTENSION: &str = "rdsposture-template";
const KIND: &str = "rds-posture-template";
/// The version of the file written here.
pub const VERSION: u32 = 1;

/// Two bones touch, and so make a joint, where their contours come within
/// this of each other, mm.
pub const CONTACT_MM: f64 = 4.0;
/// Voxels of each bone the contact must have to be a joint.
pub const MIN_CONTACT_VOXELS: usize = 4;
/// A joint of the template is used when at least this many patients have
/// it.
pub const MIN_EDGE_SAMPLES: usize = 3;
/// An organ's offset is used when at least this many patients have it.
pub const MIN_ORGAN_SAMPLES: usize = 3;
/// Organs smaller than this, cm3, on either state are not sampled.
pub const MIN_ORGAN_CM3: f64 = 5.0;

/// The file a pair's sample is written to beside its model.
pub const SAMPLE_FILE: &str = "sample.json";

/// Where the station keeps posture templates: `posture-templates/` in the
/// data folder. The window lists what is there.
pub fn templates_dir() -> PathBuf {
    crate::settings::data_dir().join("posture-templates")
}

/// The templates in `dir`, by file name.
pub fn list(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|d| {
            d.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.is_file() && is_template(p))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// Whether the file's extension is a template's.
pub fn is_template(path: &Path) -> bool {
    path.extension()
        .is_some_and(|x| x.eq_ignore_ascii_case(EXTENSION))
}

/// Every [`SAMPLE_FILE`] and template in `dir` and the folders below it
/// (eight deep at most), sorted.
pub fn find_samples(dir: &Path) -> Vec<PathBuf> {
    fn walk(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else {
            return;
        };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                if depth < 8 {
                    walk(&p, depth + 1, out);
                }
            } else if p.file_name().is_some_and(|n| n == SAMPLE_FILE) || is_template(&p) {
                out.push(p);
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, 0, &mut out);
    out.sort();
    out
}

/// The samples of a file: a sample is one, a template all of its.
pub fn read_samples(path: &Path) -> Result<Vec<Sample>> {
    if !is_template(path) {
        if let Ok(s) = Sample::read(path) {
            if !s.bodies.is_empty() {
                return Ok(vec![s]);
            }
        }
    }
    Ok(Template::read(path)?.samples)
}

/// An organ whose content changes between two scans (filling, gas, air):
/// its centroid's move is not the posture's, so it gives no offset.
pub fn content_changes(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    const WORDS: [&str; 15] = [
        "bladder", "stomach", "bowel", "colon", "rectum", "duoden", "intestin", "sigmoid",
        "caecum", "cecum", "esophag", "oesophag", "lung", "trachea", "airway",
    ];
    WORDS.iter().any(|w| n.contains(w))
}

fn key(name: &str) -> String {
    name.trim().to_lowercase()
}

fn a3(v: Vec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

fn v3(a: [f64; 3]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}

/// A rigid motion as the file keeps it.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Pose {
    pub r: M3,
    pub t: [f64; 3],
}

impl Default for Pose {
    fn default() -> Self {
        Pose::of(&Rigid::IDENTITY)
    }
}

impl Pose {
    pub fn of(g: &Rigid) -> Pose {
        Pose { r: g.r, t: a3(g.t) }
    }

    pub fn rigid(&self) -> Rigid {
        Rigid {
            r: self.r,
            t: v3(self.t),
        }
    }
}

/// One bone of a sample.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SampleBody {
    pub name: String,
    /// At the first state, anchor frame, mm.
    pub centroid: [f64; 3],
    /// The centroid and the six ends of its principal axes (one standard
    /// deviation out): where an error is measured.
    pub points: Vec<[f64; 3]>,
    /// The first state to the second, anchor frame.
    pub motion: Pose,
    /// Mean surface distance of its fit, mm (0 for the anchor).
    pub fit_mm: f64,
    pub anchor: bool,
}

/// Two bones that touch at the first state.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SampleJoint {
    pub a: String,
    pub b: String,
    /// The middle of the two contact patches, mm.
    pub centre: [f64; 3],
    /// Voxels in the smaller patch.
    pub contact: usize,
}

/// One organ of a sample, anchor frame.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SampleOrgan {
    pub name: String,
    /// Centroid at the first state, as contoured.
    pub start: [f64; 3],
    /// Centroid at the second state, as contoured.
    pub end: [f64; 3],
    /// The start centroid as the bones alone carry it.
    pub carried: [f64; 3],
    /// The bones' deformation gradient at the start centroid.
    pub gradient: M3,
    /// Volume on each state, cm3.
    pub cm3: [f64; 2],
}

/// What one patient's model gives a template (see the module doc).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Sample {
    /// What the caller calls it: a pseudonymised label, never read from the
    /// files.
    pub label: String,
    /// The anchor the model was built on.
    pub anchor: String,
    /// The anchor frame to the second state's own patient coordinates.
    pub scanner: Pose,
    /// The two states' patient positions from their headers (HFS, ...).
    pub positions: Vec<String>,
    pub bodies: Vec<SampleBody>,
    pub joints: Vec<SampleJoint>,
    pub organs: Vec<SampleOrgan>,
}

impl Sample {
    pub fn read(path: &Path) -> Result<Sample> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("{} is not a sample", path.display()))
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        crate::pathlen::check(path)?;
        let text = serde_json::to_string(self).context("encode the sample")?;
        std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
    }

    fn motions(&self) -> BTreeMap<String, Rigid> {
        self.bodies
            .iter()
            .map(|b| (key(&b.name), b.motion.rigid()))
            .collect()
    }

    /// The bone the motions are taken relative to: the anchor when it is a
    /// bone of the sample, else the one [`root_of`] picks.
    pub fn root(&self) -> Option<usize> {
        self.bodies.iter().position(|b| b.anchor).or_else(|| {
            root_of(
                &self
                    .bodies
                    .iter()
                    .map(|b| b.name.clone())
                    .collect::<Vec<_>>(),
            )
        })
    }
}

/// A root among bones by name alone, for a sample whose anchor was not a
/// bone: the sacrum, else the middle of the vertebrae by level, else the
/// first.
pub fn root_of(names: &[String]) -> Option<usize> {
    if names.is_empty() {
        return None;
    }
    if let Some(i) = names.iter().position(|n| key(n).contains("sacrum")) {
        return Some(i);
    }
    let mut levels: Vec<(u32, usize)> = names
        .iter()
        .enumerate()
        .filter_map(|(i, n)| spine_level(n).map(|l| (l, i)))
        .collect();
    levels.sort();
    levels.get(levels.len() / 2).map(|x| x.1).or(Some(0))
}

/// A vertebra's level counted from the top (C1 = 1, T1 = 8, L1 = 20), by
/// the last `_`-separated part of its name (`vertebrae_L3`, `VB_L3`, `L3`).
pub fn spine_level(name: &str) -> Option<u32> {
    let n = name.trim().to_ascii_uppercase();
    let last = n.rsplit(['_', ' ']).next()?;
    let (c, rest) = last.split_at(1);
    let k: u32 = rest.parse().ok()?;
    match c {
        "C" if (1..=7).contains(&k) => Some(k),
        "T" if (1..=12).contains(&k) => Some(7 + k),
        "L" if (1..=6).contains(&k) => Some(19 + k),
        _ => None,
    }
}

/// The group a bone's error is reported in.
pub fn group(name: &str) -> &'static str {
    let n = key(name);
    match spine_level(name) {
        Some(l) if l <= 7 => return "cervical",
        Some(l) if l <= 19 => return "thoracic",
        Some(_) => return "lumbar",
        None => {}
    }
    if n.contains("sacrum") || n.contains("coccyx") || n.contains("hip") || n.contains("pelvi") {
        "pelvis"
    } else if n.contains("femur") {
        "femur"
    } else if n.contains("rib") {
        "rib"
    } else {
        "other"
    }
}

/// A body's centroid and its seven points (the centroid and one standard
/// deviation out along each principal axis), from its mask.
pub fn body_points(mask: &BoxedMask, grid: &Grid) -> Option<(Vec3, Vec<Vec3>)> {
    let (c, cov) = super::fit::moments(mask, grid)?;
    let (vals, e) = se3::sym_eigen(&cov);
    let mut pts = vec![c];
    for k in 0..3 {
        let s = vals[k].max(0.0).sqrt();
        let ax = Vec3::new(e[0][k], e[1][k], e[2][k]) * s;
        pts.push(c + ax);
        pts.push(c - ax);
    }
    Some((c, pts))
}

/// The voxels of `m` within `r` voxels (a box) of a voxel of `other`: their
/// centroid and count.
fn near(
    m: &BoxedMask,
    mb: ([usize; 3], [usize; 3]),
    other: &BoxedMask,
    ob: ([usize; 3], [usize; 3]),
    r: [usize; 3],
    grid: &Grid,
) -> Option<(Vec3, usize)> {
    let lo = [0, 1, 2].map(|a| mb.0[a].max(ob.0[a].saturating_sub(r[a])));
    let hi = [0, 1, 2].map(|a| mb.1[a].min(ob.1[a] + r[a]));
    if (0..3).any(|a| lo[a] > hi[a]) {
        return None;
    }
    let mut s = Vec3::ZERO;
    let mut n = 0usize;
    for k in lo[2]..=hi[2] {
        for j in lo[1]..=hi[1] {
            for i in lo[0]..=hi[0] {
                if !m.get(i, j, k) {
                    continue;
                }
                let found = (k.saturating_sub(r[2])..=k + r[2]).any(|kk| {
                    (j.saturating_sub(r[1])..=j + r[1]).any(|jj| {
                        (i.saturating_sub(r[0])..=i + r[0]).any(|ii| other.get(ii, jj, kk))
                    })
                });
                if found {
                    s = s + grid.voxel_to_patient(i as f64, j as f64, k as f64);
                    n += 1;
                }
            }
        }
    }
    (n > 0).then(|| (s * (1.0 / n as f64), n))
}

/// The joints among `masks` (one per name, all on `grid`): every pair whose
/// contours come within `radius_mm`, with its centre (see the module doc).
pub fn contacts(
    names: &[String],
    masks: &[&BoxedMask],
    grid: &Grid,
    radius_mm: f64,
) -> Vec<SampleJoint> {
    use rayon::prelude::*;
    let r = [0, 1, 2].map(|a| (radius_mm / grid.spacing[a]).ceil().max(1.0) as usize);
    let boxes: Vec<Option<([usize; 3], [usize; 3])>> = masks.iter().map(|m| m.bbox()).collect();
    let n = names.len().min(masks.len());
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
        .filter(|&(a, b)| match (boxes[a], boxes[b]) {
            (Some((la, ha)), Some((lb, hb))) => {
                (0..3).all(|k| la[k] <= hb[k] + r[k] && lb[k] <= ha[k] + r[k])
            }
            _ => false,
        })
        .collect();
    // In pair order whatever the thread count.
    pairs
        .par_iter()
        .map(|&(a, b)| {
            let (ba, bb) = (boxes[a]?, boxes[b]?);
            let pa = near(masks[a], ba, masks[b], bb, r, grid)?;
            let pb = near(masks[b], bb, masks[a], ba, r, grid)?;
            (pa.1 >= MIN_CONTACT_VOXELS && pb.1 >= MIN_CONTACT_VOXELS).then(|| SampleJoint {
                a: names[a].clone(),
                b: names[b].clone(),
                centre: a3((pa.0 + pb.0) * 0.5),
                contact: pa.1.min(pb.1),
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .flatten()
        .collect()
}

/// `b`'s motion relative to `a`: `T_a^-1 o T_b`.
fn relative(ta: &Rigid, tb: &Rigid) -> Rigid {
    tb.then(&ta.inverse())
}

fn rotation(w: Vec3) -> M3 {
    se3::exp(&Twist { w, v: Vec3::ZERO }).r
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn median3(v: &[Vec3]) -> Vec3 {
    Vec3::new(
        median(v.iter().map(|x| x.x).collect()),
        median(v.iter().map(|x| x.y).collect()),
        median(v.iter().map(|x| x.z).collect()),
    )
}

/// The median distance of `v` from `m`: a spread one wild value does not
/// inflate.
fn spread(v: &[Vec3], m: Vec3) -> f64 {
    median(v.iter().map(|x| (*x - m).length()).collect())
}

/// The key of a relation: the two names in lower case, `a < b`, and
/// whether it is a joint (about the contact centre) or direct (about the
/// middle of the centroids).
pub type EdgeKey = (String, String, bool);

/// What the cohort says about one relation: `b` relative to `a`.
#[derive(Clone, Debug)]
pub struct EdgeStats {
    pub a: String,
    pub b: String,
    /// A joint (the bones touch); else a direct relation.
    pub joint: bool,
    /// Patients that have it.
    pub n: usize,
    /// The median rotation vector, rad, first-state axes.
    pub w: Vec3,
    /// The median displacement of the centre, mm.
    pub tau: Vec3,
    /// Median distance from the medians: degrees, mm.
    pub w_sd_deg: f64,
    pub tau_sd_mm: f64,
}

impl EdgeStats {
    /// The joint's median motion (`b` relative to `a`) about `centre`.
    pub fn motion_about(&self, centre: Vec3) -> Rigid {
        let r = rotation(self.w);
        Rigid {
            r,
            t: centre + self.tau - se3::mat_vec(&r, centre),
        }
    }

    /// The median turn, degrees.
    pub fn angle_deg(&self) -> f64 {
        self.w.length().to_degrees()
    }
}

/// What the cohort says about one organ.
#[derive(Clone, Debug)]
pub struct OrganStats {
    pub name: String,
    pub n: usize,
    /// The median offset to apply at the first state, before the bones, mm.
    pub offset: Vec3,
    /// Median distance from it, mm.
    pub sd_mm: f64,
}

/// The relations of `samples`: every joint, and every pair of bones
/// directly.
pub fn edge_stats<'a>(
    samples: impl IntoIterator<Item = &'a Sample>,
) -> BTreeMap<EdgeKey, EdgeStats> {
    let mut acc: BTreeMap<EdgeKey, (Vec<Vec3>, Vec<Vec3>)> = BTreeMap::new();
    let mut push = |a: &str, b: &str, joint: bool, ta: &Rigid, tb: &Rigid, c: Vec3| {
        let (ka, kb) = (key(a), key(b));
        let (ka, kb, ta, tb) = if ka <= kb {
            (ka, kb, ta, tb)
        } else {
            (kb, ka, tb, ta)
        };
        let rel = relative(ta, tb);
        let e = acc.entry((ka, kb, joint)).or_default();
        e.0.push(se3::log_rotation(&rel.r));
        e.1.push(rel.map(c) - c);
    };
    for s in samples {
        let m = s.motions();
        for j in &s.joints {
            if let (Some(ta), Some(tb)) = (m.get(&key(&j.a)), m.get(&key(&j.b))) {
                push(&j.a, &j.b, true, ta, tb, v3(j.centre));
            }
        }
        for (i, a) in s.bodies.iter().enumerate() {
            for b in &s.bodies[i + 1..] {
                let c = (v3(a.centroid) + v3(b.centroid)) * 0.5;
                push(
                    &a.name,
                    &b.name,
                    false,
                    &a.motion.rigid(),
                    &b.motion.rigid(),
                    c,
                );
            }
        }
    }
    acc.into_iter()
        .map(|((a, b, joint), (ws, ts))| {
            let w = median3(&ws);
            let tau = median3(&ts);
            let st = EdgeStats {
                a: a.clone(),
                b: b.clone(),
                joint,
                n: ws.len(),
                w,
                tau,
                w_sd_deg: spread(&ws, w).to_degrees(),
                tau_sd_mm: spread(&ts, tau),
            };
            ((a, b, joint), st)
        })
        .collect()
}

/// `delta` pulled back through `g`: `g^-1 delta`; `delta` itself when `g`
/// is singular.
fn pull_back(g: &M3, delta: Vec3) -> Vec3 {
    let d = se3::det(g);
    if d.abs() < 1e-6 {
        return delta;
    }
    // Inverse by the adjugate.
    let c = |r: usize, k: usize| {
        let (r1, r2) = ((r + 1) % 3, (r + 2) % 3);
        let (k1, k2) = ((k + 1) % 3, (k + 2) % 3);
        g[r1][k1] * g[r2][k2] - g[r1][k2] * g[r2][k1]
    };
    let mut inv = [[0.0; 3]; 3];
    for (r, row) in inv.iter_mut().enumerate() {
        for (k, cell) in row.iter_mut().enumerate() {
            *cell = c(k, r) / d;
        }
    }
    se3::mat_vec(&inv, delta)
}

/// The organ statistics of `samples` (see the module doc).
pub fn organ_stats<'a>(
    samples: impl IntoIterator<Item = &'a Sample>,
) -> BTreeMap<String, OrganStats> {
    let mut acc: BTreeMap<String, (String, Vec<Vec3>)> = BTreeMap::new();
    for s in samples {
        for o in &s.organs {
            if content_changes(&o.name) || o.cm3.iter().any(|&v| v < MIN_ORGAN_CM3) {
                continue;
            }
            let delta = v3(o.end) - v3(o.carried);
            let e = acc
                .entry(key(&o.name))
                .or_insert_with(|| (o.name.clone(), Vec::new()));
            e.1.push(pull_back(&o.gradient, delta));
        }
    }
    acc.into_iter()
        .map(|(k, (name, v))| {
            let offset = median3(&v);
            (
                k,
                OrganStats {
                    name,
                    n: v.len(),
                    offset,
                    sd_mm: spread(&v, offset),
                },
            )
        })
        .collect()
}

/// One bone's predicted motion.
#[derive(Clone, Debug, Default)]
pub struct BodyPrediction {
    /// From the first state to the second, relative to the root; `None`
    /// when no joint of the cohort reaches it.
    pub motion: Option<Rigid>,
    /// The bone it hangs from.
    pub parent: Option<usize>,
    /// Joints from the root.
    pub hops: usize,
    /// The accumulated variance at its centroid, mm2.
    pub variance: f64,
}

/// The bones' motions walked out from `root` along the relations the
/// cohort knows (see the module doc). `joints` are this patient's, by name.
pub fn predict_bodies(
    edges: &BTreeMap<EdgeKey, EdgeStats>,
    names: &[String],
    centroids: &[Vec3],
    joints: &[SampleJoint],
    root: usize,
) -> Vec<BodyPrediction> {
    let n = names.len();
    let mut out = vec![BodyPrediction::default(); n];
    if root >= n {
        return out;
    }
    let idx: BTreeMap<String, usize> = names.iter().enumerate().map(|(i, s)| (key(s), i)).collect();
    // Neighbours: (other, centre, stats, this is the stats' `a`).
    let mut adj: Vec<Vec<(usize, Vec3, &EdgeStats, bool)>> = vec![Vec::new(); n];
    let mut link = |ia: usize, ib: usize, joint: bool, c: Vec3| {
        let (ka, kb) = (key(&names[ia]), key(&names[ib]));
        let pair = if ka <= kb {
            (ka, kb, joint)
        } else {
            (kb, ka, joint)
        };
        if let Some(st) = edges.get(&pair).filter(|s| s.n >= MIN_EDGE_SAMPLES) {
            // From ia to ib, and back.
            adj[ia].push((ib, c, st, key(&names[ia]) == st.a));
            adj[ib].push((ia, c, st, key(&names[ib]) == st.a));
        }
    };
    for j in joints {
        if let (Some(&ia), Some(&ib)) = (idx.get(&key(&j.a)), idx.get(&key(&j.b))) {
            link(ia, ib, true, v3(j.centre));
        }
    }
    for ia in 0..n {
        for ib in ia + 1..n {
            link(ia, ib, false, (centroids[ia] + centroids[ib]) * 0.5);
        }
    }
    let mut cost = vec![f64::INFINITY; n];
    let mut done = vec![false; n];
    cost[root] = 0.0;
    out[root].motion = Some(Rigid::IDENTITY);
    // The cheapest open bone each time (in index order on a tie).
    while let Some(u) = (0..n)
        .filter(|&i| !done[i] && cost[i].is_finite())
        .min_by(|&x, &y| cost[x].total_cmp(&cost[y]))
    {
        done[u] = true;
        let tu = out[u].motion.unwrap_or(Rigid::IDENTITY);
        for &(v, c, st, u_is_a) in &adj[u] {
            if done[v] {
                continue;
            }
            let lever = (centroids[v] - c).length();
            let add = (st.w_sd_deg.to_radians() * lever).powi(2) + st.tau_sd_mm.powi(2) + 1.0;
            if cost[u] + add < cost[v] {
                cost[v] = cost[u] + add;
                // The stats are `b` relative to `a`: T_b = T_a o M.
                let m = st.motion_about(c);
                let rel = if u_is_a { m } else { m.inverse() };
                out[v] = BodyPrediction {
                    motion: Some(rel.then(&tu)),
                    parent: Some(u),
                    hops: out[u].hops + 1,
                    variance: cost[u] + add,
                };
            }
        }
    }
    out
}

/// A cohort: the samples, with every statistic computed from them.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Template {
    pub kind: String,
    pub version: u32,
    /// The station's version that wrote it.
    pub written_by: String,
    pub samples: Vec<Sample>,
}

impl Template {
    pub fn new(samples: Vec<Sample>) -> Template {
        Template {
            kind: KIND.into(),
            version: VERSION,
            written_by: env!("CARGO_PKG_VERSION").into(),
            samples,
        }
    }

    pub fn read(path: &Path) -> Result<Template> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let t: Template = serde_json::from_str(&text)
            .with_context(|| format!("{} is not a posture template", path.display()))?;
        if t.kind != KIND {
            bail!("{} is not a posture template", path.display());
        }
        if t.version > VERSION {
            bail!(
                "{} was written by a newer station (template version {})",
                path.display(),
                t.version
            );
        }
        Ok(t)
    }

    pub fn write(&self, path: &Path) -> Result<()> {
        crate::pathlen::check(path)?;
        let text = serde_json::to_string(self).context("encode the template")?;
        std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
    }

    /// The samples but the one at `except`.
    fn others(&self, except: Option<usize>) -> impl Iterator<Item = &Sample> {
        self.samples
            .iter()
            .enumerate()
            .filter(move |(i, _)| Some(*i) != except)
            .map(|(_, s)| s)
    }

    pub fn edges(&self, except: Option<usize>) -> BTreeMap<EdgeKey, EdgeStats> {
        edge_stats(self.others(except))
    }

    pub fn organs(&self, except: Option<usize>) -> BTreeMap<String, OrganStats> {
        organ_stats(self.others(except))
    }

    /// The bones some relation of the template (on enough patients) has,
    /// in lower case.
    pub fn bones(&self) -> BTreeSet<String> {
        self.edges(None)
            .values()
            .filter(|e| e.n >= MIN_EDGE_SAMPLES)
            .flat_map(|e| [e.a.clone(), e.b.clone()])
            .collect()
    }

    /// The pelvis's turn in the second state's scanner, per sample that has
    /// the sacrum: the rotation vector, degrees (patient axes RL, AP, SI).
    pub fn pelvic_tilts(&self) -> Vec<Vec3> {
        self.samples
            .iter()
            .filter_map(|s| {
                let b = s.bodies.iter().find(|b| key(&b.name).contains("sacrum"))?;
                let r = se3::mat_mul(&s.scanner.r, &b.motion.r);
                Some(se3::log_rotation(&r) * 1.0_f64.to_degrees())
            })
            .collect()
    }

    /// A few lines on what the template holds.
    pub fn summary(&self) -> Vec<String> {
        let edges = self.edges(None);
        let used: Vec<&EdgeStats> = edges
            .values()
            .filter(|e| e.joint && e.n >= MIN_EDGE_SAMPLES)
            .collect();
        let organs = self.organs(None);
        let mut out = vec![format!(
            "{} patient(s); {} joint(s) seen on {MIN_EDGE_SAMPLES} or more, {} organ offset(s)",
            self.samples.len(),
            used.len(),
            organs.values().filter(|o| o.n >= MIN_ORGAN_SAMPLES).count()
        )];
        let tilts = self.pelvic_tilts();
        if !tilts.is_empty() {
            let m = median3(&tilts);
            out.push(format!(
                "the sacrum in the second scanner, median over {}: {:.1} deg (RL {:+.1}, AP {:+.1}, SI {:+.1})",
                tilts.len(),
                m.length(),
                m.x,
                m.y,
                m.z
            ));
        }
        // The spine, joint by joint from the top.
        let mut spine: Vec<(u32, &EdgeStats)> = used
            .iter()
            .filter_map(|e| {
                let (la, lb) = (spine_level(&e.a), spine_level(&e.b));
                match (la, lb) {
                    (Some(x), Some(y)) if x.abs_diff(y) == 1 => Some((x.min(y), *e)),
                    (Some(x), None) if e.b.contains("sacrum") => Some((x, *e)),
                    (None, Some(y)) if e.a.contains("sacrum") => Some((y, *e)),
                    _ => None,
                }
            })
            .collect();
        spine.sort_by_key(|x| x.0);
        for (_, e) in spine {
            out.push(format!(
                "  {} / {}: {:.1} deg (spread {:.1} deg, {:.1} mm) on {}",
                e.a,
                e.b,
                e.angle_deg(),
                e.w_sd_deg,
                e.tau_sd_mm,
                e.n
            ));
        }
        out
    }
}

/// One bone of a patient left out.
#[derive(Clone, Debug)]
pub struct LooBody {
    pub name: String,
    pub group: &'static str,
    pub hops: usize,
    /// Mean distance at its seven points between the predicted and the
    /// fitted motion (both relative to the root), mm; `None` when no joint
    /// reached it.
    pub error_mm: Option<f64>,
    /// The same for no motion at all.
    pub still_mm: f64,
}

/// One organ of a patient left out: distances of its centroid from where
/// it was contoured at the second state, mm.
#[derive(Clone, Debug)]
pub struct LooOrgan {
    pub name: String,
    /// Not moved at all.
    pub still_mm: f64,
    /// Moved by the patient's own fitted bones.
    pub bones_mm: f64,
    /// Moved by the fitted bones and the template's offset.
    pub template_mm: f64,
}

/// One patient left out.
#[derive(Clone, Debug)]
pub struct LooPatient {
    pub label: String,
    pub root: String,
    pub bodies: Vec<LooBody>,
    pub organs: Vec<LooOrgan>,
}

/// Each patient predicted from the others (see the module doc).
pub fn leave_one_out(t: &Template) -> Vec<LooPatient> {
    (0..t.samples.len())
        .filter_map(|i| {
            let s = &t.samples[i];
            let root = s.root()?;
            let edges = t.edges(Some(i));
            let organs = t.organs(Some(i));
            let names: Vec<String> = s.bodies.iter().map(|b| b.name.clone()).collect();
            let cents: Vec<Vec3> = s.bodies.iter().map(|b| v3(b.centroid)).collect();
            let pred = predict_bodies(&edges, &names, &cents, &s.joints, root);
            let troot_inv = s.bodies[root].motion.rigid().inverse();
            let bodies = s
                .bodies
                .iter()
                .enumerate()
                .filter(|&(k, _)| k != root)
                .map(|(k, b)| {
                    let fit = b.motion.rigid().then(&troot_inv);
                    let pts: Vec<Vec3> = b.points.iter().map(|p| v3(*p)).collect();
                    let mean = |f: &dyn Fn(Vec3) -> f64| {
                        pts.iter().map(|&p| f(p)).sum::<f64>() / pts.len().max(1) as f64
                    };
                    LooBody {
                        name: b.name.clone(),
                        group: group(&b.name),
                        hops: pred[k].hops,
                        error_mm: pred[k].motion.map(|m| mean(&|p| m.gap_at(&fit, p))),
                        still_mm: mean(&|p| (fit.map(p) - p).length()),
                    }
                })
                .collect();
            let organs = s
                .organs
                .iter()
                .filter_map(|o| {
                    let st = organs
                        .get(&key(&o.name))
                        .filter(|x| x.n >= MIN_ORGAN_SAMPLES)?;
                    if content_changes(&o.name) {
                        return None;
                    }
                    let (start, end, carried) = (v3(o.start), v3(o.end), v3(o.carried));
                    let moved = carried + se3::mat_vec(&o.gradient, st.offset);
                    Some(LooOrgan {
                        name: o.name.clone(),
                        still_mm: (end - start).length(),
                        bones_mm: (end - carried).length(),
                        template_mm: (end - moved).length(),
                    })
                })
                .collect();
            Some(LooPatient {
                label: s.label.clone(),
                root: s.bodies[root].name.clone(),
                bodies,
                organs,
            })
        })
        .collect()
}

/// The groups of [`group`], in the order reported.
pub const GROUPS: [&str; 7] = [
    "cervical", "thoracic", "lumbar", "pelvis", "femur", "rib", "other",
];

/// Mean, median (of the two middle values, for an even count) and 90th
/// percentile (the nearest rank), mm.
#[derive(Clone, Copy, Debug, Default)]
pub struct Spread {
    pub mean: f64,
    pub median: f64,
    pub p90: f64,
}

impl Spread {
    fn of(mut v: Vec<f64>) -> Spread {
        if v.is_empty() {
            return Spread::default();
        }
        v.sort_by(f64::total_cmp);
        let n = v.len();
        Spread {
            mean: v.iter().sum::<f64>() / n as f64,
            median: if n % 2 == 1 {
                v[n / 2]
            } else {
                0.5 * (v[n / 2 - 1] + v[n / 2])
            },
            p90: v[((n - 1) as f64 * 0.9).round() as usize],
        }
    }
}

/// One group of bones over every patient left out.
#[derive(Clone, Debug)]
pub struct LooGroup {
    pub group: &'static str,
    /// Bones the template reached.
    pub n: usize,
    pub template: Spread,
    /// No motion at all, the same bones.
    pub still: Spread,
}

/// The leave-one-out by group of bones, the groups that have one.
pub fn loo_groups(loo: &[LooPatient]) -> Vec<LooGroup> {
    GROUPS
        .iter()
        .filter_map(|&g| {
            let v: Vec<&LooBody> = loo
                .iter()
                .flat_map(|p| &p.bodies)
                .filter(|b| b.group == g && b.error_mm.is_some())
                .collect();
            (!v.is_empty()).then(|| LooGroup {
                group: g,
                n: v.len(),
                template: Spread::of(v.iter().filter_map(|b| b.error_mm).collect()),
                still: Spread::of(v.iter().map(|b| b.still_mm).collect()),
            })
        })
        .collect()
}

/// One organ over every patient left out: mean distances, mm.
#[derive(Clone, Debug)]
pub struct LooOrganMean {
    /// In lower case.
    pub name: String,
    pub n: usize,
    pub still_mm: f64,
    pub bones_mm: f64,
    pub template_mm: f64,
}

/// The leave-one-out by organ, by name.
pub fn loo_organs(loo: &[LooPatient]) -> Vec<LooOrganMean> {
    let mut by: BTreeMap<String, Vec<&LooOrgan>> = BTreeMap::new();
    for p in loo {
        for o in &p.organs {
            by.entry(key(&o.name)).or_default().push(o);
        }
    }
    by.into_iter()
        .map(|(name, v)| {
            let m =
                |f: &dyn Fn(&LooOrgan) -> f64| v.iter().map(|o| f(o)).sum::<f64>() / v.len() as f64;
            LooOrganMean {
                n: v.len(),
                still_mm: m(&|o| o.still_mm),
                bones_mm: m(&|o| o.bones_mm),
                template_mm: m(&|o| o.template_mm),
                name,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn turn_about(axis: Vec3, deg: f64, c: Vec3) -> Rigid {
        let r = rotation(axis * deg.to_radians());
        Rigid {
            r,
            t: c - se3::mat_vec(&r, c),
        }
    }

    /// A chain of three bones on two hinges about RL, the second turning
    /// `k1` and the third `k2` degrees relative to the one before.
    fn chain(label: &str, k1: f64, k2: f64, spread: f64) -> Sample {
        let rl = Vec3::new(1.0, 0.0, 0.0);
        let (c1, c2) = (Vec3::new(0.0, 0.0, 40.0), Vec3::new(0.0, 0.0, 80.0));
        let t1 = turn_about(rl, k1, c1);
        let t2 = turn_about(rl, k2, c2).then(&t1);
        let body = |name: &str, z: f64, m: Rigid, anchor: bool| {
            let c = Vec3::new(0.0, spread, z);
            SampleBody {
                name: name.into(),
                centroid: a3(c),
                points: [0.0, 10.0, -10.0]
                    .iter()
                    .map(|&d| a3(c + Vec3::new(d, d, 0.0)))
                    .collect(),
                motion: Pose::of(&m),
                fit_mm: 0.5,
                anchor,
            }
        };
        Sample {
            label: label.into(),
            anchor: "sacrum".into(),
            scanner: Pose::default(),
            positions: vec!["HFS".into(), "HFS".into()],
            bodies: vec![
                body("sacrum", 20.0, Rigid::IDENTITY, true),
                body("vertebrae_L5", 60.0, t1, false),
                body("vertebrae_L4", 100.0, t2, false),
            ],
            joints: vec![
                SampleJoint {
                    a: "sacrum".into(),
                    b: "vertebrae_L5".into(),
                    centre: a3(c1),
                    contact: 50,
                },
                SampleJoint {
                    a: "vertebrae_L5".into(),
                    b: "vertebrae_L4".into(),
                    centre: a3(c2),
                    contact: 50,
                },
            ],
            organs: Vec::new(),
        }
    }

    #[test]
    fn a_joint_reads_the_same_whatever_the_anchor() {
        let s = chain("p", 8.0, 5.0, 0.0);
        // The same patient anchored on L5 instead: every motion composed
        // with L5's inverse.
        let l5 = s.bodies[1].motion.rigid().inverse();
        let mut other = s.clone();
        for b in &mut other.bodies {
            b.motion = Pose::of(&b.motion.rigid().then(&l5));
            b.anchor = b.name == "vertebrae_L5";
        }
        let a = edge_stats([&s]);
        let b = edge_stats([&other]);
        for (k, e) in &a {
            let f = &b[k];
            assert!((e.w - f.w).length() < 1e-9, "{k:?}");
            assert!((e.tau - f.tau).length() < 1e-9, "{k:?}");
        }
        let lumbo = &a[&("sacrum".to_string(), "vertebrae_l5".to_string(), true)];
        assert!((lumbo.angle_deg() - 8.0).abs() < 1e-9);
        assert!(lumbo.tau.length() < 1e-9, "the contact centre is the hinge");
    }

    #[test]
    fn the_median_patient_is_predicted_and_one_misfit_does_not_move_it() {
        let mut t = Template::new(vec![
            chain("a", 6.0, 4.0, 0.0),
            chain("b", 8.0, 5.0, 3.0),
            chain("c", 10.0, 5.0, -2.0),
            chain("d", 8.0, 5.0, 1.0),
        ]);
        // A fifth whose L4 fit went wrong by 60 degrees.
        t.samples.push(chain("e", 8.0, 65.0, 0.0));
        let e = t.edges(None);
        let l45 = &e[&("vertebrae_l4".to_string(), "vertebrae_l5".to_string(), true)];
        assert_eq!(l45.n, 5);
        assert!((l45.angle_deg() - 5.0).abs() < 1e-6, "{}", l45.angle_deg());
        // Patient "d" from the others: its own turns are the medians.
        let loo = leave_one_out(&t);
        let d = loo.iter().find(|p| p.label == "d").unwrap();
        assert_eq!(d.root, "sacrum");
        for b in &d.bodies {
            let err = b.error_mm.expect("reached");
            assert!(err < 1e-6, "{} {err}", b.name);
            assert!(b.still_mm > 1.0, "{} moved", b.name);
        }
        assert_eq!(d.bodies[1].hops, 2);
        // The root walks out along the joints: a patient anchored on L4
        // predicts the sacrum by the inverse turns.
        let s = &t.samples[3];
        let names: Vec<String> = s.bodies.iter().map(|b| b.name.clone()).collect();
        let cents: Vec<Vec3> = s.bodies.iter().map(|b| v3(b.centroid)).collect();
        let p = predict_bodies(&t.edges(Some(3)), &names, &cents, &s.joints, 2);
        let fit = s.bodies[0]
            .motion
            .rigid()
            .then(&s.bodies[2].motion.rigid().inverse());
        let m = p[0].motion.unwrap();
        assert!(m.gap_at(&fit, v3(s.bodies[0].centroid)) < 1e-6);
    }

    #[test]
    fn an_organ_offset_is_kept_before_the_bones() {
        // The bones turn the organ's surroundings 90 deg about SI; the organ
        // sags 10 mm further along -SI in the end state.
        let g = rotation(Vec3::new(0.0, 0.0, 90.0_f64.to_radians()));
        let organ = |label: &str, sag: f64| {
            let mut s = chain(label, 0.0, 0.0, 0.0);
            let start = Vec3::new(20.0, 0.0, 50.0);
            let carried = se3::mat_vec(&g, start);
            s.organs.push(SampleOrgan {
                name: "liver".into(),
                start: a3(start),
                end: a3(carried + Vec3::new(sag, 0.0, -10.0)),
                carried: a3(carried),
                gradient: g,
                cm3: [1500.0, 1500.0],
            });
            s.organs.push(SampleOrgan {
                name: "urinary_bladder".into(),
                start: a3(start),
                end: a3(start + Vec3::new(0.0, 0.0, 40.0)),
                carried: a3(start),
                gradient: se3::I3,
                cm3: [100.0, 300.0],
            });
            s
        };
        let t = Template::new(vec![
            organ("a", 0.0),
            organ("b", 2.0),
            organ("c", -2.0),
            organ("d", 0.0),
        ]);
        let o = t.organs(None);
        assert!(!o.contains_key("urinary_bladder"), "its content changes");
        let liver = &o["liver"];
        // Pulled back through the 90 deg turn: -SI stays -SI.
        assert!(
            (liver.offset - Vec3::new(0.0, 0.0, -10.0)).length() < 1e-9,
            "{:?}",
            liver.offset
        );
        let loo = leave_one_out(&t);
        let a = &loo[0].organs[0];
        assert!(a.template_mm < 2.5 && a.bones_mm > 9.9, "{a:?}");
    }

    #[test]
    fn levels_and_groups_read_from_the_names() {
        assert_eq!(spine_level("vertebrae_L3"), Some(22));
        assert_eq!(spine_level("VB_T12"), Some(19));
        assert_eq!(spine_level("C7"), Some(7));
        assert_eq!(spine_level("sacrum"), None);
        assert_eq!(spine_level("femur_left"), None);
        assert_eq!(group("vertebrae_T4"), "thoracic");
        assert_eq!(group("Bone_Pelvic_L"), "pelvis");
        assert_eq!(group("rib_left_7"), "rib");
        assert_eq!(group("Femur_R"), "femur");
        assert_eq!(
            root_of(&[
                "vertebrae_L2".into(),
                "vertebrae_L1".into(),
                "vertebrae_T12".into()
            ]),
            Some(1)
        );
        assert!(content_changes("Bladder") && content_changes("colon"));
        assert!(!content_changes("liver") && !content_changes("kidney_left"));
    }
}
