//! Planar structure sets: named, coloured 2D contours on planar images -
//! to a radiograph, an RT image or a DRR what an RTSTRUCT is to a CT.
//!
//! **Model.** A [`PlanarStructureSet`] holds [`PlanarStructure`]s; each
//! structure holds at most one [`PlanarContour`] per referenced image, the
//! rings of one image filled even-odd ([`crate::contours::Region`], exactly
//! as `rasterize_roi` and the contour booleans mean it). A contour belongs
//! to its image by **SOP Instance UID**, never by the image's index in
//! `planar_images` (indices shift when an image is removed), which is why
//! every planar image made here is given a UID the moment it is made.
//!
//! **Pixels.** Every vertex is `[col, row]`, fractional, with the centre of
//! the first pixel at `(0, 0)`: the convention of `ViewChoice.region`, the
//! MCP `ImageArg.region` and `setup::region_mask`. The one place it changes
//! is the presentation state file ([`crate::gsps`]), whose origin is the
//! first pixel's top left corner.
//!
//! **Projection** ([`project_mask`]). A structure's voxel mask becomes a
//! [`MuVolume::from_mask`] and is cast through the image's camera (moved by
//! a pose when there is one: `camera.transformed(pose)`, the structure
//! where a setup found the patient). Each pixel keeps the *largest*
//! trilinear sample along its ray ([`MuVolume::max_along`]); that field is
//! at least one half exactly where the ray crosses the mask's half-level
//! surface, so its iso-line at one half - marching squares with linear
//! interpolation, stitched by edge into closed rings - is the
//! **silhouette** of the surface marching cubes would draw: the shadow a
//! beam's eye view shows, "the PTV on the radiograph". Only the pixels the
//! mask's box projects onto are cast, at a raster no finer than half the
//! mask's voxel at the structure's depth.
//!
//! **Carrying** ([`carry`]). A contour on the moving image of a 2D/2D
//! registration goes onto the fixed image vertex by vertex through the
//! inverse transform, between the two images' millimetre frames
//! ([`crate::registration::planar`]).

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;

use crate::contours::{Poly, Region};
use crate::extras::PlanarImage;
use crate::loader::LoadedStudy;
use crate::progress::{ProgressSink, CANCELLED};
use crate::projection::{MuVolume, ProjectionGeometry};
use crate::registration::planar::{mm_to_px, px_to_mm, Transform2d};
use crate::registration::RigidTransform;
use crate::volume::Volume;

/// What made a contour: shown in the tree row's tooltip, and what makes the
/// same projection again after the structure is edited.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum ContourSource {
    #[default]
    Drawn,
    /// Projected from a 3D structure: its set and name, the pose (the six
    /// parameters of a [`RigidTransform`], about the isocentre) and the
    /// setup it came from, when not as rendered / at the plan.
    Projected {
        set: String,
        structure: String,
        pose: Option<[f64; 6]>,
        setup: Option<String>,
    },
    /// Carried from another image by a 2D/2D registration.
    Carried {
        from_image: String,
        registration: String,
    },
    /// Read from a presentation state.
    File,
}

impl ContourSource {
    /// One line for a tooltip.
    pub fn describe(&self) -> String {
        match self {
            ContourSource::Drawn => "drawn".into(),
            ContourSource::Projected {
                structure, setup, ..
            } => match setup {
                Some(s) => format!("projected from {structure} at {s}"),
                None => format!("projected from {structure}"),
            },
            ContourSource::Carried { registration, .. } => {
                format!("carried by {registration}")
            }
            ContourSource::File => "read from file".into(),
        }
    }

    /// The word a set's name takes after it (`MADE_HERE` words, as for an
    /// RT structure set).
    pub fn word(&self) -> &'static str {
        match self {
            ContourSource::Drawn => "drawn",
            ContourSource::Projected { .. } => "projected",
            ContourSource::Carried { .. } => "carried",
            ContourSource::File => "file",
        }
    }
}

/// The contour of one structure on one image.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlanarContour {
    /// The image's SOP Instance UID.
    pub image_uid: String,
    /// Pixels, centre of the first pixel at `(0, 0)`, even-odd.
    pub region: Region,
    pub source: ContourSource,
}

/// A named, coloured 2D structure.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlanarStructure {
    pub number: i32,
    pub name: String,
    pub color: [u8; 3],
    /// At most one per image.
    pub contours: Vec<PlanarContour>,
    /// Points of interest (POINT graphics): image UID and pixel.
    pub points: Vec<(String, [f64; 2])>,
}

impl PlanarStructure {
    pub fn contour(&self, image_uid: &str) -> Option<&PlanarContour> {
        self.contours.iter().find(|c| c.image_uid == image_uid)
    }

    pub fn contour_mut(&mut self, image_uid: &str) -> Option<&mut PlanarContour> {
        self.contours.iter_mut().find(|c| c.image_uid == image_uid)
    }

    /// The contour on `image_uid`, made empty first when there is none.
    pub fn contour_entry(&mut self, image_uid: &str) -> &mut PlanarContour {
        if let Some(k) = self.contours.iter().position(|c| c.image_uid == image_uid) {
            return &mut self.contours[k];
        }
        self.contours.push(PlanarContour {
            image_uid: image_uid.to_string(),
            ..Default::default()
        });
        self.contours.last_mut().expect("just pushed")
    }

    /// Drop the contours that hold nothing.
    pub fn prune(&mut self) {
        self.contours.retain(|c| !c.region.is_empty());
    }
}

/// A planar structure set: one presentation state's worth of contours.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PlanarStructureSet {
    pub label: String,
    /// Empty until written (or read).
    pub sop_instance_uid: String,
    pub series_instance_uid: String,
    pub study_uid: String,
    /// The referenced images' SOP Instance UIDs, loaded or not.
    pub images: Vec<String>,
    pub structures: Vec<PlanarStructure>,
    /// `"drawn"`, `"projected"`, `"carried"` for a set made here (the
    /// `MADE_HERE` words of an RT structure set), the file name otherwise.
    pub file_name: String,
    pub locked: bool,
    pub source_path: Option<PathBuf>,
}

/// The palette new structures take their colours from, in turn.
pub const PALETTE: [[u8; 3]; 10] = [
    [230, 60, 60],
    [60, 200, 90],
    [70, 130, 240],
    [240, 200, 40],
    [200, 80, 220],
    [40, 210, 220],
    [250, 140, 40],
    [150, 220, 60],
    [240, 110, 170],
    [160, 160, 255],
];

impl PlanarStructureSet {
    pub fn new(label: &str, study_uid: &str, word: &str) -> PlanarStructureSet {
        PlanarStructureSet {
            label: label.to_string(),
            series_instance_uid: crate::dicom_export::new_uid(),
            study_uid: study_uid.to_string(),
            file_name: word.to_string(),
            ..Default::default()
        }
    }

    pub fn structure(&self, name: &str) -> Option<&PlanarStructure> {
        self.structures.iter().find(|s| s.name == name)
    }

    /// The structure called `name`, added (with `color`, else the next of
    /// the palette) when there is none.
    pub fn structure_entry(&mut self, name: &str, color: Option<[u8; 3]>) -> &mut PlanarStructure {
        if let Some(k) = self.structures.iter().position(|s| s.name == name) {
            return &mut self.structures[k];
        }
        let number = self.structures.iter().map(|s| s.number).max().unwrap_or(0) + 1;
        let color = color.unwrap_or(PALETTE[self.structures.len() % PALETTE.len()]);
        self.structures.push(PlanarStructure {
            number,
            name: name.to_string(),
            color,
            ..Default::default()
        });
        self.structures.last_mut().expect("just pushed")
    }

    /// Make `images` the images the contours and points are on, in the
    /// order they first appear, keeping the references already held to
    /// images not loaded.
    pub fn refresh_images(&mut self) {
        let mut seen: Vec<String> = Vec::new();
        for s in &self.structures {
            for c in &s.contours {
                if !c.region.is_empty() && !seen.contains(&c.image_uid) {
                    seen.push(c.image_uid.clone());
                }
            }
            for (uid, _) in &s.points {
                if !seen.contains(uid) {
                    seen.push(uid.clone());
                }
            }
        }
        // References read from a file to images that hold no graphic stay,
        // the state referenced them.
        for uid in std::mem::take(&mut self.images) {
            if !seen.contains(&uid) && self.source_path.is_some() {
                seen.push(uid);
            }
        }
        self.images = seen;
    }

    /// Every contour of the set on `image_uid`, with its structure.
    pub fn contours_on<'a>(
        &'a self,
        image_uid: &'a str,
    ) -> impl Iterator<Item = (&'a PlanarStructure, &'a PlanarContour)> + 'a {
        self.structures
            .iter()
            .filter_map(move |s| s.contour(image_uid).map(|c| (s, c)))
    }

    /// The referenced images that are not among `study`'s planar images.
    pub fn missing_images(&self, study: &LoadedStudy) -> Vec<String> {
        self.images
            .iter()
            .filter(|u| study.planar_by_uid(u).is_none())
            .cloned()
            .collect()
    }

    /// Merge `other` (the same set read again, or carried from another
    /// workspace) into this one: structures by name, contours by image.
    pub fn merge(&mut self, other: PlanarStructureSet) {
        for s in other.structures {
            let mine = self.structure_entry(&s.name, Some(s.color));
            for c in s.contours {
                if mine.contour(&c.image_uid).is_none() {
                    mine.contours.push(c);
                }
            }
            for p in s.points {
                if !mine.points.contains(&p) {
                    mine.points.push(p);
                }
            }
        }
        for uid in other.images {
            if !self.images.contains(&uid) {
                self.images.push(uid);
            }
        }
    }
}

/// Graphic Layer names for the writer: each structure's name as a CS of up
/// to 16 characters (upper case, digits, space, underscore), made unique
/// with a number where two collide or a name has nothing left.
pub fn unique_layer_names(set: &PlanarStructureSet) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(set.structures.len());
    for (k, s) in set.structures.iter().enumerate() {
        let mut base: String = s
            .name
            .chars()
            .map(|c| {
                let c = c.to_ascii_uppercase();
                if c.is_ascii_uppercase() || c.is_ascii_digit() || c == ' ' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>()
            .trim()
            .to_string();
        if base.trim_matches('_').is_empty() {
            base = format!("STRUCT_{}", k + 1);
        }
        base.truncate(16);
        let mut name = base.clone();
        let mut n = 2;
        while out.contains(&name) {
            let suffix = format!("_{n}");
            let keep = 16usize.saturating_sub(suffix.len()).min(base.len());
            name = format!("{}{suffix}", &base[..keep]);
            n += 1;
        }
        out.push(name);
    }
    out
}

/// Enclosed area of a pixel region in mm².
pub fn area_mm2(region: &Region, spacing: [f64; 2]) -> f64 {
    region.area() * spacing[0] * spacing[1]
}

/// Area-weighted centroid of a pixel region (pixels), holes subtracted;
/// `None` when it encloses nothing.
pub fn centroid_px(region: &Region) -> Option<[f64; 2]> {
    let depths = region.depths();
    let (mut a, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for (r, d) in region.rings.iter().zip(&depths) {
        let s = if d.is_multiple_of(2) { 1.0 } else { -1.0 };
        let ra = r.area() * s;
        let c = r.centroid();
        a += ra;
        cx += ra * c[0];
        cy += ra * c[1];
    }
    (a > 1e-12).then(|| [cx / a, cy / a])
}

/// [`centroid_px`] in the image's millimetre frame.
pub fn centroid_mm(region: &Region, img: &PlanarImage) -> Option<[f64; 2]> {
    centroid_px(region).map(|c| px_to_mm([img.cols, img.rows], img.spacing, c))
}

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

/// `cam` with its raster made `dims`: the same field of view, pixels of the
/// size that gives (a working camera, halved some times, back at an image's
/// full resolution).
pub fn rescaled(cam: &ProjectionGeometry, dims: [usize; 2]) -> ProjectionGeometry {
    if cam.dims == dims || cam.dims[0] == 0 || cam.dims[1] == 0 {
        return ProjectionGeometry { dims, ..*cam };
    }
    let fx = dims[0] as f64 / cam.dims[0] as f64;
    let fy = dims[1] as f64 / cam.dims[1] as f64;
    // Fine pixel i is coarse (i + 0.5) / f − 0.5.
    ProjectionGeometry {
        source: cam.source,
        origin: cam.pixel(0.5 / fx - 0.5, 0.5 / fy - 0.5),
        row_step: cam.row_step * (1.0 / fx),
        col_step: cam.col_step * (1.0 / fy),
        dims,
    }
}

/// The rings where `field` (`w × h`, values at pixel centres) crosses
/// `level`: marching squares with linear interpolation, the segments
/// chained by the cell edge they cross (each crossing computed once, so two
/// cells meet exactly), saddles decided by the cell's mean. Coordinates
/// are the field's pixels. Rings of fewer than three points are dropped.
pub fn iso_rings(field: &[f32], w: usize, h: usize, level: f32) -> Vec<Vec<[f64; 2]>> {
    if w < 2 || h < 2 || field.len() < w * h {
        return Vec::new();
    }
    let at = |x: usize, y: usize| field[y * w + x];
    // Edge ids: horizontal (x, y)-(x+1, y) is 2 (y w + x), vertical (x, y)-
    // (x, y+1) is 2 (y w + x) + 1.
    let hid = |x: usize, y: usize| 2 * (y * w + x);
    let vid = |x: usize, y: usize| 2 * (y * w + x) + 1;
    let crossing = |id: usize| -> [f64; 2] {
        let o = id / 2;
        let (x, y) = (o % w, o / w);
        let (va, vb, dx, dy) = if id.is_multiple_of(2) {
            (at(x, y), at(x + 1, y), 1.0, 0.0)
        } else {
            (at(x, y), at(x, y + 1), 0.0, 1.0)
        };
        let d = vb - va;
        let t = if d.abs() < 1e-12 {
            0.5
        } else {
            ((level - va) / d).clamp(0.0, 1.0) as f64
        };
        [x as f64 + dx * t, y as f64 + dy * t]
    };
    let mut segs: Vec<(usize, usize)> = Vec::new();
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let (a, b, c, d) = (at(x, y), at(x + 1, y), at(x + 1, y + 1), at(x, y + 1));
            let inside = |v: f32| v >= level;
            let case = inside(a) as u8
                | (inside(b) as u8) << 1
                | (inside(c) as u8) << 2
                | (inside(d) as u8) << 3;
            let (t, r, bo, l) = (hid(x, y), vid(x + 1, y), hid(x, y + 1), vid(x, y));
            let centre = inside((a + b + c + d) * 0.25);
            match case {
                0 | 15 => {}
                1 | 14 => segs.push((l, t)),
                2 | 13 => segs.push((t, r)),
                3 | 12 => segs.push((l, r)),
                4 | 11 => segs.push((r, bo)),
                6 | 9 => segs.push((t, bo)),
                7 | 8 => segs.push((l, bo)),
                5 => {
                    if centre {
                        segs.push((t, r));
                        segs.push((l, bo));
                    } else {
                        segs.push((l, t));
                        segs.push((r, bo));
                    }
                }
                10 => {
                    if centre {
                        segs.push((l, t));
                        segs.push((r, bo));
                    } else {
                        segs.push((t, r));
                        segs.push((l, bo));
                    }
                }
                _ => {}
            }
        }
    }
    // Each edge is met by at most two segments.
    let mut by_edge: HashMap<usize, Vec<usize>> = HashMap::with_capacity(segs.len() * 2);
    for (k, &(e0, e1)) in segs.iter().enumerate() {
        by_edge.entry(e0).or_default().push(k);
        by_edge.entry(e1).or_default().push(k);
    }
    let mut used = vec![false; segs.len()];
    let mut rings = Vec::new();
    for start in 0..segs.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let (first, mut cur) = segs[start];
        let mut edges = vec![first];
        while cur != first {
            edges.push(cur);
            let next = by_edge
                .get(&cur)
                .and_then(|v| v.iter().copied().find(|&s| !used[s]));
            let Some(s) = next else { break };
            used[s] = true;
            let (e0, e1) = segs[s];
            cur = if e0 == cur { e1 } else { e0 };
        }
        if cur == first && edges.len() >= 3 {
            rings.push(edges.into_iter().map(crossing).collect());
        }
    }
    rings
}

/// The silhouette of `mask` (on `vol`'s grid) through `cam`, moved by
/// `pose` when there is one, as a region in `cam`'s raster (pixels, centre
/// of the first at `(0, 0)`). Empty when the mask is empty or projects
/// onto no pixel.
pub fn project_mask(
    vol: &Volume,
    mask: &[u8],
    cam: &ProjectionGeometry,
    pose: Option<&RigidTransform>,
) -> Region {
    let Some(mu) = MuVolume::from_mask(vol, mask) else {
        return Region::new();
    };
    project_mu(&mu, cam, pose)
}

/// [`project_mask`] of a mask already made a volume.
pub fn project_mu(
    mu: &MuVolume,
    cam: &ProjectionGeometry,
    pose: Option<&RigidTransform>,
) -> Region {
    let cam = match pose {
        Some(p) => cam.transformed(p),
        None => *cam,
    };
    let [w, h] = cam.dims;
    if w == 0 || h == 0 {
        return Region::new();
    }
    let finest = mu.finest_spacing();
    // The raster: no finer than half a voxel at the structure's depth, as
    // the detector pixel is at the isocentre's magnification.
    let (lo, hi) = mu.bounds();
    let mid = (lo + hi) * 0.5;
    let sp = cam.spacing();
    let det = (cam.center() - cam.source).length().max(1e-6);
    let depth = (mid - cam.source).length();
    let foot = sp[0].min(sp[1]) * depth / det;
    let factor = ((0.5 * finest / foot.max(1e-9)).floor() as usize).clamp(1, 8);
    let rc = cam.coarser(factor);
    let [rw, rh] = rc.dims;
    // The window the box projects onto, two pixels larger every way.
    let mut win = [0usize, 0, rw, rh];
    let corners: Vec<Option<[f64; 2]>> = (0..8)
        .map(|c| {
            let p = crate::geometry::Vec3::new(
                if c & 1 == 0 { lo.x } else { hi.x },
                if c & 2 == 0 { lo.y } else { hi.y },
                if c & 4 == 0 { lo.z } else { hi.z },
            );
            rc.project(p)
        })
        .collect();
    if corners.iter().all(|c| c.is_some()) {
        let (mut x0, mut y0, mut x1, mut y1) = (f64::MAX, f64::MAX, f64::MIN, f64::MIN);
        for c in corners.into_iter().flatten() {
            x0 = x0.min(c[0]);
            y0 = y0.min(c[1]);
            x1 = x1.max(c[0]);
            y1 = y1.max(c[1]);
        }
        let cl = |v: f64, n: usize| v.clamp(0.0, n as f64) as usize;
        win = [
            cl((x0 - 2.0).floor(), rw),
            cl((y0 - 2.0).floor(), rh),
            cl((x1 + 3.0).ceil(), rw),
            cl((y1 + 3.0).ceil(), rh),
        ];
    }
    let (ww, wh) = (win[2].saturating_sub(win[0]), win[3].saturating_sub(win[1]));
    if ww == 0 || wh == 0 {
        return Region::new();
    }
    // A zero border, so a silhouette cut by the detector's edge still
    // closes (half a pixel beyond it).
    let (pw, ph) = (ww + 2, wh + 2);
    let step = (finest * 0.25).clamp(0.1, 1.5);
    let mut field = vec![0.0f32; pw * ph];
    field.par_chunks_mut(pw).enumerate().for_each(|(py, row)| {
        if py == 0 || py == ph - 1 {
            return;
        }
        let j = (win[1] + py - 1) as f64;
        for (px, v) in row.iter_mut().enumerate().take(pw - 1).skip(1) {
            let i = (win[0] + px - 1) as f64;
            *v = mu.max_along(rc.source, rc.pixel(i, j), step);
        }
    });
    let f = factor as f64;
    let mut region = Region::new();
    for ring in iso_rings(&field, pw, ph, 0.5) {
        let pts: Vec<[f64; 2]> = ring
            .into_iter()
            .map(|p| {
                let c = [p[0] - 1.0 + win[0] as f64, p[1] - 1.0 + win[1] as f64];
                // Coarse pixel centre k is fine k f + (f − 1) / 2.
                [c[0] * f + (f - 1.0) * 0.5, c[1] * f + (f - 1.0) * 0.5]
            })
            .collect();
        region.rings.push(Poly::new(pts));
    }
    region.simplify(0.1);
    region.normalize();
    region
}

/// Where a projection's pose comes from.
#[derive(Clone, Debug, Default, PartialEq)]
pub enum PoseChoice {
    /// The image's own camera as it is: a DRR as rendered, a radiograph at
    /// the plan.
    #[default]
    AsIs,
    /// The pose of setup `k` of the study: the structure where the patient
    /// was found.
    Setup(usize),
}

/// The camera and pose to project onto planar image `image` of `study`:
/// its own camera, or, at a setup, the camera that setup used for it (a
/// beam's, a room's) when it was one of the setup's views.
pub fn camera_for(
    study: &LoadedStudy,
    image: usize,
    pose: &PoseChoice,
) -> Result<(ProjectionGeometry, Option<RigidTransform>)> {
    let img = study
        .planar_images
        .get(image)
        .ok_or_else(|| anyhow!("no planar image {}", image + 1))?;
    let dims = [img.cols, img.rows];
    match pose {
        PoseChoice::AsIs => {
            let cam = img
                .camera
                .ok_or_else(|| anyhow!("{} has no geometry", img.label))?;
            Ok((rescaled(&cam, dims), None))
        }
        PoseChoice::Setup(k) => {
            let s = study
                .setups
                .get(*k)
                .ok_or_else(|| anyhow!("no setup {}", k + 1))?;
            let cam = s
                .images
                .iter()
                .position(|&i| i == image)
                .and_then(|v| s.result.views.get(v))
                .map(|v| v.camera)
                .or(img.camera)
                .ok_or_else(|| anyhow!("{} has no geometry", img.label))?;
            Ok((rescaled(&cam, dims), Some(s.result.transform.clone())))
        }
    }
}

/// What [`project_many`] is asked.
#[derive(Clone, Debug, Default)]
pub struct ProjectRequest {
    /// Structure names, found on the displayed series as
    /// [`crate::workflow::select::find_on_series`] finds them.
    pub structures: Vec<String>,
    /// Planar image indices.
    pub images: Vec<usize>,
    pub pose: PoseChoice,
    /// The set's label; `"Projected"` (or `"Projected at <setup>"`) when
    /// empty.
    pub label: String,
}

/// One structure on one image, as [`project_many`] reports it.
#[derive(Clone, Debug)]
pub struct ProjectedLine {
    pub structure: String,
    pub image: usize,
    pub rings: usize,
    pub area_mm2: f64,
    /// `[x0, y0, x1, y1]`, pixels.
    pub bbox: Option<[f64; 4]>,
}

/// Project the requested structures onto the requested images: a new set
/// (not yet filed on the study), with one line per structure and image.
/// A structure that projects onto nothing gets no contour and a line with
/// no rings.
pub fn project_many(
    study: &LoadedStudy,
    req: &ProjectRequest,
    sink: &dyn ProgressSink,
) -> Result<(PlanarStructureSet, Vec<ProjectedLine>)> {
    if !study.has_volume() {
        bail!("projecting a structure needs the CT");
    }
    if req.structures.is_empty() || req.images.is_empty() {
        bail!("choose at least one structure and one image");
    }
    let cams: Vec<(usize, ProjectionGeometry, Option<RigidTransform>, String)> = req
        .images
        .iter()
        .map(|&i| {
            let (c, p) = camera_for(study, i, &req.pose)?;
            Ok((i, c, p, study.planar_images[i].sop_instance_uid.clone()))
        })
        .collect::<Result<_>>()?;
    if let Some((i, ..)) = cams.iter().find(|c| c.3.is_empty()) {
        bail!(
            "{} has no SOP Instance UID to attach a contour to",
            study.planar_images[*i].label
        );
    }
    let setup = match req.pose {
        PoseChoice::Setup(k) => study.setups.get(k).map(|s| s.label.clone()),
        PoseChoice::AsIs => None,
    };
    let label = if !req.label.trim().is_empty() {
        req.label.trim().to_string()
    } else {
        match &setup {
            Some(s) => format!("Projected at {s}"),
            None => "Projected".to_string(),
        }
    };
    let study_uid = study
        .planar_images
        .get(req.images[0])
        .map(|p| p.study_uid.clone())
        .filter(|u| !u.is_empty())
        .or_else(|| {
            study
                .series
                .get(study.active_series)
                .map(|s| s.study_uid.clone())
        })
        .unwrap_or_default();
    let mut set = PlanarStructureSet::new(&label, &study_uid, "projected");
    let grid = study.volume.grid();
    let series_uid = study
        .series
        .get(study.active_series)
        .map(|s| s.uid.clone())
        .unwrap_or_default();
    let mut lines = Vec::new();
    let n = req.structures.len();
    for (k, name) in req.structures.iter().enumerate() {
        if sink.cancelled() {
            bail!("{CANCELLED}");
        }
        sink.report(
            k as f32 / n as f32,
            &format!("Projecting {name} ({}/{n})", k + 1),
        );
        let st = crate::workflow::select::find_on_series(
            study,
            name,
            &series_uid,
            &study.volume.frame_of_reference_uid,
        )
        .ok_or_else(|| anyhow!("no structure called {name}"))?;
        let mask = st.mask_on(&grid)?;
        let Some(mu) = MuVolume::from_mask(&study.volume, &mask) else {
            for (i, ..) in &cams {
                lines.push(ProjectedLine {
                    structure: st.name.clone(),
                    image: *i,
                    rings: 0,
                    area_mm2: 0.0,
                    bbox: None,
                });
            }
            continue;
        };
        let set_label = crate::workflow::select::list(study)
            .into_iter()
            .rfind(|e| e.name == st.name)
            .map(|e| e.set_label)
            .unwrap_or_default();
        for (i, cam, pose, uid) in &cams {
            let region = project_mu(&mu, cam, pose.as_ref());
            let img = &study.planar_images[*i];
            lines.push(ProjectedLine {
                structure: st.name.clone(),
                image: *i,
                rings: region.rings.len(),
                area_mm2: area_mm2(&region, img.spacing),
                bbox: (!region.is_empty()).then(|| region.bbox()),
            });
            if region.is_empty() {
                continue;
            }
            let s = set.structure_entry(&st.name, Some(st.color));
            let c = s.contour_entry(uid);
            c.region = region;
            c.source = ContourSource::Projected {
                set: set_label.clone(),
                structure: st.name.clone(),
                pose: pose.as_ref().map(|p| p.params()),
                setup: setup.clone(),
            };
        }
    }
    set.refresh_images();
    sink.report(1.0, "Projected");
    Ok((set, lines))
}

/// What [`project_many`] reads of a study, for a worker thread: the
/// volume (shared), the series, structure sets, segmentations and setups,
/// and the planar images without their pixels.
pub fn projection_view(study: &LoadedStudy) -> LoadedStudy {
    LoadedStudy {
        meta: study.meta.clone(),
        series: study.series.clone(),
        active_series: study.active_series,
        volume: study.volume.clone(),
        structure_sets: study.structure_sets.clone(),
        seg_series: study.seg_series.clone(),
        planar_images: study
            .planar_images
            .iter()
            .map(|p| PlanarImage {
                data: Vec::new(),
                rgb: None,
                info: Vec::new(),
                ..p.clone()
            })
            .collect(),
        setups: study.setups.clone(),
        ..Default::default()
    }
}

/// File `set` on `study`: a set of the same label that is not locked takes
/// its structures (by name, replacing the contours on the same images),
/// otherwise it is added. Returns the set's index.
pub fn file_set(study: &mut LoadedStudy, set: PlanarStructureSet) -> usize {
    if let Some(k) = study
        .planar_structs
        .iter()
        .position(|s| s.label == set.label && !s.locked && s.source_path.is_none())
    {
        let dest = &mut study.planar_structs[k];
        for s in set.structures {
            let d = dest.structure_entry(&s.name, Some(s.color));
            for c in s.contours {
                match d.contour_mut(&c.image_uid) {
                    Some(old) => *old = c,
                    None => d.contours.push(c),
                }
            }
        }
        dest.refresh_images();
        return k;
    }
    study.planar_structs.push(set);
    study.planar_structs.len() - 1
}

// ---------------------------------------------------------------------------
// Editing: the drawing tools' rules on a pixel region
// ---------------------------------------------------------------------------

/// A closed ring onto a pixel region: added, or cut out of it (the caller
/// decides which by the drawing mode, the *auto* rule being
/// [`crate::contours::auto_cuts`]).
pub fn apply_ring(region: &mut Region, ring: &Poly, cut: bool) {
    if ring.is_empty() {
        return;
    }
    if cut {
        region.subtract_ring(ring);
    } else {
        region.add_ring(ring);
    }
}

/// One step of the contour brush from `a` to `b` (pixels): the capsule of
/// radius `radius_mm` on pixels of `spacing` mm - round on the patient, an
/// ellipse in pixels when the two spacings differ - added or cut.
pub fn brush(
    region: &mut Region,
    a: [f64; 2],
    b: [f64; 2],
    radius_mm: f64,
    spacing: [f64; 2],
    cut: bool,
) {
    let ring = Poly::capsule_mm(a, b, radius_mm, spacing, 32);
    apply_ring(region, &ring, cut);
}

// ---------------------------------------------------------------------------
// Carrying through a 2D/2D registration
// ---------------------------------------------------------------------------

/// A region on `from` (the moving image of a 2D/2D registration `t`) onto
/// `to` (its fixed image): every vertex from `from`'s pixels to its
/// millimetre frame, through `t⁻¹`, into `to`'s pixels. `None` when a
/// vertex falls in a fold of the warp, where `t` has no inverse.
pub fn carry(
    region: &Region,
    t: &Transform2d,
    from: &PlanarImage,
    to: &PlanarImage,
) -> Option<Region> {
    let (fd, fs) = ([from.cols, from.rows], from.spacing);
    let (td, ts) = ([to.cols, to.rows], to.spacing);
    // A tolerance of a thousandth of a pixel of the fixed image.
    let tol = 1e-3 * ts[0].min(ts[1]);
    let mut out = Region::new();
    for ring in &region.rings {
        let mut pts = Vec::with_capacity(ring.pts.len());
        for &p in &ring.pts {
            let q = px_to_mm(fd, fs, p);
            let x = t.inverse_map(q, tol)?;
            pts.push(mm_to_px(td, ts, x));
        }
        out.rings.push(Poly::new(pts));
    }
    out.normalize();
    Some(out)
}

/// A region's rings as line segments in `0..1` of a `dims` raster (the
/// form the 2D/3D overlay draws).
pub fn segments01(region: &Region, dims: [usize; 2]) -> Vec<([f32; 2], [f32; 2])> {
    let (w, h) = (dims[0].max(1) as f64, dims[1].max(1) as f64);
    let n = |p: [f64; 2]| [((p[0] + 0.5) / w) as f32, ((p[1] + 0.5) / h) as f32];
    let mut out = Vec::new();
    for r in &region.rings {
        let k = r.pts.len();
        for i in 0..k {
            out.push((n(r.pts[i]), n(r.pts[(i + 1) % k])));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_disc_field_gives_one_closed_ring_of_its_area() {
        // A disc of radius 20 px, values falling linearly through the edge.
        let (w, h) = (64usize, 64usize);
        let field: Vec<f32> = (0..w * h)
            .map(|o| {
                let (x, y) = ((o % w) as f64 - 31.5, (o / w) as f64 - 30.0);
                (20.5 - (x * x + y * y).sqrt()).clamp(0.0, 1.0) as f32
            })
            .collect();
        let rings = iso_rings(&field, w, h, 0.5);
        assert_eq!(rings.len(), 1);
        let p = Poly::new(rings[0].clone());
        let want = std::f64::consts::PI * 20.0 * 20.0;
        assert!((p.area() - want).abs() / want < 0.005, "{}", p.area());
        let c = p.centroid();
        assert!((c[0] - 31.5).abs() < 0.02 && (c[1] - 30.0).abs() < 0.02);
    }

    #[test]
    fn a_saddle_and_two_islands_stay_apart() {
        // Two bright pixels on a diagonal with a dark mean between them.
        let field = vec![
            0.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 0.0,
        ];
        let rings = iso_rings(&field, 4, 4, 0.6);
        assert_eq!(rings.len(), 2);
    }

    #[test]
    fn layer_names_are_cs_and_unique() {
        let mut set = PlanarStructureSet::default();
        for n in ["PTV", "ptv", "Lung (left) and more", "\u{e4}\u{f6}", ""] {
            set.structure_entry(n, None);
        }
        let names = unique_layer_names(&set);
        assert_eq!(names[0], "PTV");
        assert_eq!(names[1], "PTV_2");
        assert_eq!(names[2], "LUNG _LEFT_ AND ");
        assert_eq!(names[3], "STRUCT_4");
        assert_eq!(names[4], "STRUCT_5");
        for n in &names {
            assert!(n.len() <= 16);
        }
    }

    #[test]
    fn a_camera_rescaled_puts_the_same_ray_on_the_same_point() {
        let cam = crate::drr::Geometry {
            dims: [64, 48],
            ..Default::default()
        }
        .camera();
        let coarse = cam.coarser(4);
        let back = rescaled(&coarse, cam.dims);
        for (i, j) in [(0.0, 0.0), (17.0, 31.0), (63.0, 47.0)] {
            assert!((back.pixel(i, j) - cam.pixel(i, j)).length() < 1e-9);
        }
    }
}
