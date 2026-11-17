//! The model's film: the path as an animated GIF, every frame rendered
//! from the model at its own time, never taken off the screen.
//!
//! **Frames.** `frames` times evenly spaced over the whole path, the first
//! and the last state included ([`film_times`]); *there and back* adds the
//! way back without repeating either end, so the loop has no stutter. At
//! each time the model's [`Frame`](super::Frame) is computed and the planes
//! are rendered exactly as the live slices are ([`render_planes_with`]: on
//! the inference GPU when there is a slicer, else
//! [`apply::render_planes`] on the CPU), on the full grid, so a film made
//! twice is the same file and does not depend on how fast the window was
//! drawing.
//!
//! **Views** ([`FilmViews`]). *All*: the three planes laid out as the window
//! lays them out (axial, sagittal, coronal, left to right, each in a cell
//! of the panes' shape, the image fitted inside it with its millimetres
//! square and the rest black) and, when a scene is given, the 3D scene
//! under them across the whole width, as the window has its 3D pane; or any
//! one of the four alone. A plane is the whole slice of the grid, at the
//! slice the view shows: the view's zoom and pan are not taken. Sagittal
//! and coronal rows run superior to inferior and axial rows anterior to
//! posterior, as the views draw them
//! ([`crate::volume::Volume::extract_slice`]).
//!
//! **The 3D scene** ([`FilmScene`]) is the first state's meshes carried to
//! each time ([`apply::carry_meshes`], as the window carries them) and
//! drawn the way the 3D pane draws them (`app/d3.rs`): the same turn
//! ([`mesh3d::turn`]: yaw about the patient's z, then pitch), the same
//! orthographic scale ([`mesh3d::scene_scale`]), the same headlight
//! ([`mesh3d::headlight`]) on each vertex, the same opacities (the scene's,
//! each structure's, [`mesh3d::EXTERNAL_ALPHA`] for a body), the same
//! painter's order (far triangles first, by [`mesh3d::depth_key`] of the
//! mean depth, ties in mesh order) and the same blending of each triangle
//! over what is behind it, the colours interpolated across it. The pane
//! hands its triangles to egui's painter; here they are filled on the CPU
//! one pixel centre at a time ([`render_scene`]), so the CLI films the
//! scene too and a film made twice is the same. What the pane draws over
//! the surfaces (the orientation cube, text, the beams) is not drawn.
//!
//! **Size.** The film is `width` pixels wide (at most [`MAX_WIDTH`]); its
//! height follows from the cells' shape. Each image is resampled to it
//! bilinearly from the plane's voxel values, before the window is applied.
//!
//! **What is drawn over the image.** The first state's structures as
//! outlines one pixel wide in their own colours, where their tissue is at
//! the time: a pixel `y` is inside a structure when the point it came from
//! on the first state, `to_first(y)`, is inside the structure's mask there
//! (the masks are the structures rasterized once on the first state's
//! grid) - the same pull the images go through, so the outlines and the
//! anatomy move together. With the wash, the Jacobian of the path since the
//! first state as [`apply::wash_color`] colours it, over tissue only when
//! the image is a scan ([`apply::WASH_TISSUE_HU`]). With the field
//! ([`FieldOverlay`]), the displacement from the first state at the time,
//! as the window's overlay draws it ([`crate::registration::dvf`]): arrows
//! re-walked on the plane every `spacing_mm` ([`dvf::glyphs_on_plane`]),
//! drawn [`dvf::arrow_length_factor`] times their length, coloured by
//! magnitude over the frame's largest (or the one pale yellow), a disc where
//! nearly all of the displacement leaves the plane, a head of the window's
//! proportions; or the deformed grid ([`dvf::deformed_grid_on_plane`]).
//! Lines are one pixel wide and opaque (the window's 92 % opacity is not
//! told apart at that width), clipped to their cell. The dose is not drawn.
//!
//! **One palette for the whole film**, the GIF's global colour table, no
//! dithering, so no frame is quantized on its own and nothing flickers
//! between frames. The table holds, from index 0:
//!
//! * greys from black to white: 192 of them, fewer when the arrows or the
//!   scene need room (177 with the arrows, 131 with the scene, 113 with
//!   both: a step of at most 2.3 of the 256 a screen shows);
//! * 32 structure colours, the structures' own in their order (a structure
//!   past the 32nd distinct colour takes the nearest);
//! * the wash, two hues (blue: compressed, red: expanded) at two strengths
//!   over 7 grey levels each - the wash's colour mixed into the grey under
//!   it, the grey kept coarsely so the anatomy still shows - and magenta
//!   for folded tissue;
//! * with the field, the arrows' colour ramp ([`render::dose_colormap`] at
//!   16 evenly spaced magnitudes, the nearest taken), the pale yellow and
//!   the deformed grid's colour;
//! * with the scene, 64 colours for the shaded surfaces: a median cut of
//!   the colours the scene shows at up to five times of the film (both
//!   ends among them), so the palette fits the structures on screen; every
//!   pixel of the scene then takes the nearest of those and the greys.
//!
//! A per-frame palette would fit each frame better and make the greys of
//! one frame differ from the next, which reads as flicker on a CT; the
//! shaded surfaces are what needs many colours, and they change little
//! along a path, so one table sampled from the run holds them.
//!
//! A frame is then a list of indices, LZW-compressed by the `gif` crate,
//! with the delay `100 / fps` hundredths of a second (a GIF's unit:
//! between 2 and 100) and the loop set to repeat for ever.

use std::borrow::Cow;
use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;

use anyhow::{bail, Context, Result};
use rayon::prelude::*;

use super::apply::{self, Plane, Show};
use super::gpu::{render_planes_with, GpuSlicer};
use super::PostureModel;
use crate::geometry::Vec3;
use crate::mesh3d::{self, RoiMesh};
use crate::progress::{self, Progress, ProgressSink};
use crate::propagate::BoxedMask;
use crate::registration::dvf::{self, FieldStyle};
use crate::render;
use crate::rtstruct::StructureSet;
use crate::volume::{Grid, ViewPlane, Volume};

/// Grey levels of a film with neither the field nor the scene.
pub const GRAYS: usize = 192;
/// Structure colours of the palette.
const STRUCT_SLOTS: usize = 32;
/// The wash: four ramps (blue light, blue strong, red light, red strong) of
/// this many greys each.
const WASH_GRAYS: usize = 7;
const WASH_SLOTS: usize = 4 * WASH_GRAYS;
/// How much of the wash's hue a light and a strong pixel take.
const WASH_MIX: [f32; 2] = [0.3, 0.6];
/// The wash's two hues, as [`apply::wash_color`] gives them.
const BLUE: [u8; 3] = [40, 120, 255];
const RED: [u8; 3] = [255, 70, 30];
/// The arrows' colour ramp; then the pale yellow and the grid's colour.
const ARROW_STEPS: usize = 16;
const ARROW_SLOTS: usize = ARROW_STEPS + 2;
/// Colours for the shaded surfaces of the scene.
const SCENE_SLOTS: usize = 64;
/// Times of the film the scene's colours are sampled at.
const SCENE_SAMPLES: usize = 5;
/// Most structures outlined (the rest are left out).
const MAX_OUTLINED: usize = 64;

/// The film's width limits, pixels.
pub const MIN_WIDTH: usize = 64;
pub const MAX_WIDTH: usize = 4096;
/// Black between two cells, pixels.
const GAP: usize = 2;
/// Width over height of the scene alone when the caller gives none.
const SCENE_ASPECT: f32 = 4.0 / 3.0;

/// Which panes the film shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilmViews {
    /// Axial, sagittal and coronal side by side, as the window has them,
    /// and the 3D scene under them when one is given.
    All,
    /// One plane alone.
    One(ViewPlane),
    /// The 3D scene alone.
    Scene,
}

impl FilmViews {
    /// Every choice, in the order a menu lists them.
    pub const CHOICES: [FilmViews; 5] = [
        FilmViews::All,
        FilmViews::One(ViewPlane::Axial),
        FilmViews::One(ViewPlane::Sagittal),
        FilmViews::One(ViewPlane::Coronal),
        FilmViews::Scene,
    ];

    /// The planes, left to right.
    pub fn planes(self) -> Vec<ViewPlane> {
        match self {
            FilmViews::All => vec![ViewPlane::Axial, ViewPlane::Sagittal, ViewPlane::Coronal],
            FilmViews::One(p) => vec![p],
            FilmViews::Scene => Vec::new(),
        }
    }

    /// Does the film draw the scene, given one?
    pub fn wants_scene(self) -> bool {
        matches!(self, FilmViews::All | FilmViews::Scene)
    }

    /// What a menu calls it.
    pub fn label(self) -> &'static str {
        match self {
            FilmViews::All => "All views",
            FilmViews::One(p) => p.title(),
            FilmViews::Scene => "3D",
        }
    }

    /// The word of the command line: `all`, `axial`, `sagittal`, `coronal`,
    /// `3d`.
    pub fn key(self) -> &'static str {
        match self {
            FilmViews::All => "all",
            FilmViews::One(ViewPlane::Axial) => "axial",
            FilmViews::One(ViewPlane::Sagittal) => "sagittal",
            FilmViews::One(ViewPlane::Coronal) => "coronal",
            FilmViews::Scene => "3d",
        }
    }

    /// The choice a [`FilmViews::key`] names, any case.
    pub fn from_key(s: &str) -> Option<FilmViews> {
        let s = s.trim().to_ascii_lowercase();
        FilmViews::CHOICES.into_iter().find(|v| v.key() == s)
    }
}

/// The displacement drawn over the planes, as the window's overlay draws
/// it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldOverlay {
    /// Arrows or the deformed grid ([`FieldStyle::None`] draws nothing).
    pub style: FieldStyle,
    /// One arrow (one grid line) every so many millimetres.
    pub spacing_mm: f64,
    /// The arrows' length, times the displacement, at the default spacing
    /// ([`dvf::arrow_length_factor`]).
    pub scale: f32,
    /// Coloured by magnitude, else the one pale yellow.
    pub colored: bool,
}

impl Default for FieldOverlay {
    /// Arrows at the viewer's defaults: every 12 mm, three times their
    /// length, coloured by magnitude.
    fn default() -> Self {
        FieldOverlay {
            style: FieldStyle::Arrows,
            spacing_mm: dvf::DEFAULT_ARROW_SPACING_MM,
            scale: 3.0,
            colored: true,
        }
    }
}

/// The 3D scene's camera, as the 3D pane holds it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneCamera {
    /// Radians: yaw about the patient's z, pitch about the screen's x.
    pub yaw: f32,
    pub pitch: f32,
    /// On the fitted scale.
    pub zoom: f32,
    /// The scene's shift from the centre, as a fraction of the shorter side
    /// of the pane (the pane holds it in points; the fraction carries it to
    /// any size).
    pub pan: [f32; 2],
    /// The sphere the camera is fitted to, patient mm.
    pub center: [f32; 3],
    pub radius: f32,
}

impl SceneCamera {
    /// The camera a scene starts with, fitted to `meshes`
    /// ([`mesh3d::bounding_sphere`]).
    pub fn fitted(meshes: &[RoiMesh]) -> SceneCamera {
        let (center, radius) = mesh3d::bounding_sphere(meshes).unwrap_or(([0.0; 3], 100.0));
        SceneCamera {
            yaw: mesh3d::SCENE_YAW,
            pitch: mesh3d::SCENE_PITCH,
            zoom: 1.0,
            pan: [0.0, 0.0],
            center,
            radius,
        }
    }
}

/// One structure of the scene, by its index in the structure set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SceneRoi {
    pub visible: bool,
    pub color: [u8; 3],
    /// Its own opacity, times the scene's.
    pub alpha: f32,
}

/// The 3D scene of a film: the first state's meshes, carried to each time,
/// and how the pane shows them.
#[derive(Clone)]
pub struct FilmScene {
    pub meshes: Arc<Vec<RoiMesh>>,
    pub camera: SceneCamera,
    /// The scene's opacity, 0 to 1.
    pub opacity: f32,
    /// Per structure index; a structure not listed is drawn in its mesh's
    /// colour at the scene's opacity.
    pub rois: Vec<SceneRoi>,
}

impl FilmScene {
    /// `meshes` as a fresh pane shows them: every structure, opaque, the
    /// camera fitted.
    pub fn new(meshes: Arc<Vec<RoiMesh>>) -> FilmScene {
        FilmScene {
            camera: SceneCamera::fitted(&meshes),
            meshes,
            opacity: 1.0,
            rois: Vec::new(),
        }
    }
}

/// How a film is made.
#[derive(Clone, Debug)]
pub struct FilmParams {
    pub views: FilmViews,
    /// The slice each plane is at, by [`Plane::axis`]: `[i, j, k]`, the
    /// sagittal, coronal and axial slice.
    pub slices: [usize; 3],
    /// Times along the path (at least two).
    pub frames: usize,
    pub fps: f32,
    /// There and back: the way back added, neither end repeated.
    pub bounce: bool,
    pub show: Show,
    /// Window centre and width.
    pub window: (f32, f32),
    /// Width of the film, pixels.
    pub width: usize,
    /// Width over height of one plane's cell, as the panes are on screen;
    /// `None`: as tall as the tallest plane needs at the cell's width.
    pub cell_aspect: Option<f32>,
    /// Width over height of the scene's cell (across the whole film under
    /// the planes, or alone); `None`: as tall as the planes' row, or 4:3
    /// alone.
    pub scene_aspect: Option<f32>,
    pub outlines: bool,
    pub wash: bool,
    /// The displacement over the planes; `None`: not drawn.
    pub field: Option<FieldOverlay>,
}

impl FilmParams {
    /// The three planes through the centre of `grid`, 21 frames at 6 a
    /// second, 800 pixels wide, outlines on, no field.
    pub fn centred(grid: &Grid, show: Show, window: (f32, f32)) -> FilmParams {
        FilmParams {
            views: FilmViews::All,
            slices: [grid.dims[0] / 2, grid.dims[1] / 2, grid.dims[2] / 2],
            frames: 21,
            fps: 6.0,
            bounce: false,
            show,
            window,
            width: 800,
            cell_aspect: None,
            scene_aspect: None,
            outlines: true,
            wash: false,
            field: None,
        }
    }

    /// The field drawn, if any.
    fn drawn_field(&self) -> Option<FieldOverlay> {
        self.field.filter(|f| f.style != FieldStyle::None)
    }
}

/// What was written.
#[derive(Clone, Debug)]
pub struct FilmSummary {
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub bytes: u64,
    /// The planes were rendered on the GPU (every frame).
    pub gpu: bool,
    /// Why not, when a slicer was given and failed.
    pub why: Option<String>,
    /// The 3D scene is in the film.
    pub scene: bool,
}

/// The times of a film over a path of `span`.
pub fn film_times(span: f64, frames: usize, bounce: bool) -> Vec<f64> {
    let n = frames.max(2);
    let mut t: Vec<f64> = (0..n).map(|k| span * k as f64 / (n - 1) as f64).collect();
    if bounce {
        let back: Vec<f64> = t[1..n - 1].iter().rev().copied().collect();
        t.extend(back);
    }
    t
}

/// The frame delay in a GIF's hundredths of a second.
pub fn delay_cs(fps: f32) -> u16 {
    let fps = if fps.is_finite() && fps > 0.0 {
        fps
    } else {
        6.0
    };
    (100.0 / fps).round().clamp(2.0, 100.0) as u16
}

// ---------------------------------------------------------------------------
// The palette

/// Where each part of the palette starts (see the module notes).
#[derive(Clone, Copy, Debug)]
struct Pal {
    grays: usize,
    structs: usize,
    wash: usize,
    folded: usize,
    arrows: Option<usize>,
    scene: Option<usize>,
}

impl Pal {
    fn new(arrows: bool, scene: bool) -> Pal {
        let fixed = STRUCT_SLOTS
            + WASH_SLOTS
            + 1
            + if arrows { ARROW_SLOTS } else { 0 }
            + if scene { SCENE_SLOTS } else { 0 };
        let grays = (256 - fixed).min(GRAYS);
        let structs = grays;
        let wash = structs + STRUCT_SLOTS;
        let folded = wash + WASH_SLOTS;
        let mut next = folded + 1;
        let arrows = arrows.then(|| {
            let a = next;
            next += ARROW_SLOTS;
            a
        });
        let scene = scene.then_some(next);
        Pal {
            grays,
            structs,
            wash,
            folded,
            arrows,
            scene,
        }
    }

    /// The grey of a windowed level (0..1).
    fn gray(&self, level: f32) -> u8 {
        (level.clamp(0.0, 1.0) * (self.grays - 1) as f32).round() as u8
    }

    /// The palette index of a wash colour over a pixel of windowed level
    /// `level` (0..1); `None` where the wash is clear.
    fn wash_index(&self, c: [u8; 4], level: f32) -> Option<usize> {
        if c[3] == 0 {
            return None;
        }
        if c[0] == 230 && c[1] == 0 {
            return Some(self.folded);
        }
        let hue = usize::from(c[0] == RED[0]);
        let strong = usize::from(c[3] >= 115);
        let g = (level * (WASH_GRAYS - 1) as f32).round() as usize;
        Some(self.wash + (hue * 2 + strong) * WASH_GRAYS + g.min(WASH_GRAYS - 1))
    }

    /// An arrow's index: the ramp's nearest step to its magnitude over the
    /// largest, or the pale yellow.
    fn arrow(&self, magnitude: f32, max: f32, colored: bool) -> u8 {
        let base = self.arrows.unwrap_or(0);
        if !colored {
            return (base + ARROW_STEPS) as u8;
        }
        let t = (magnitude / max.max(1e-6)).clamp(0.0, 1.0);
        (base + (t * (ARROW_STEPS - 1) as f32).round() as usize) as u8
    }

    /// The deformed grid's index.
    fn grid_line(&self, colored: bool) -> u8 {
        let base = self.arrows.unwrap_or(0);
        (base + ARROW_STEPS + usize::from(colored)) as u8
    }

    /// The table (768 bytes, RGB) with these structure and scene colours.
    fn table(&self, structure_colors: &[[u8; 3]], scene_colors: &[[u8; 3]]) -> Vec<u8> {
        let mut pal = vec![0u8; 256 * 3];
        let mut put = |i: usize, c: [u8; 3]| pal[i * 3..i * 3 + 3].copy_from_slice(&c);
        for i in 0..self.grays {
            let g = (i as f32 * 255.0 / (self.grays - 1) as f32).round() as u8;
            put(i, [g, g, g]);
        }
        for (i, c) in structure_colors.iter().take(STRUCT_SLOTS).enumerate() {
            put(self.structs + i, *c);
        }
        for (h, hue) in [BLUE, RED].iter().enumerate() {
            for (s, mix) in WASH_MIX.iter().enumerate() {
                for g in 0..WASH_GRAYS {
                    let grey = g as f32 * 255.0 / (WASH_GRAYS - 1) as f32;
                    let c = hue.map(|x| (grey * (1.0 - mix) + x as f32 * mix).round() as u8);
                    put(self.wash + (h * 2 + s) * WASH_GRAYS + g, c);
                }
            }
        }
        put(self.folded, [230, 0, 230]);
        if let Some(a) = self.arrows {
            for k in 0..ARROW_STEPS {
                put(
                    a + k,
                    render::dose_colormap(k as f32 / (ARROW_STEPS - 1) as f32),
                );
            }
            let plain = dvf::arrow_rgba(0.0, 1.0, false);
            put(a + ARROW_STEPS, [plain[0], plain[1], plain[2]]);
            let grid = dvf::grid_rgba(true);
            put(a + ARROW_STEPS + 1, [grid[0], grid[1], grid[2]]);
        }
        if let Some(s) = self.scene {
            for (i, c) in scene_colors.iter().take(SCENE_SLOTS).enumerate() {
                put(s + i, *c);
            }
        }
        pal
    }
}

/// The palette slot of each structure: its colour's, the colours given
/// slots in the order the structures come, the 33rd distinct one onwards
/// taking the nearest slot. Returns the slots' colours and each
/// structure's slot.
fn structure_slots(colors: &[[u8; 3]]) -> (Vec<[u8; 3]>, Vec<usize>) {
    let mut slots: Vec<[u8; 3]> = Vec::new();
    let mut of = Vec::with_capacity(colors.len());
    for c in colors {
        if let Some(i) = slots.iter().position(|s| s == c) {
            of.push(i);
        } else if slots.len() < STRUCT_SLOTS {
            slots.push(*c);
            of.push(slots.len() - 1);
        } else {
            let near = (0..slots.len())
                .min_by_key(|&i| dist2(slots[i], *c))
                .unwrap_or(0);
            of.push(near);
        }
    }
    (slots, of)
}

fn dist2(a: [u8; 3], b: [u8; 3]) -> i32 {
    (0..3).map(|k| (a[k] as i32 - b[k] as i32).pow(2)).sum()
}

/// A colour to its 15-bit cell (five bits a channel).
fn rgb15(c: [u8; 3]) -> usize {
    ((c[0] as usize >> 3) << 10) | ((c[1] as usize >> 3) << 5) | (c[2] as usize >> 3)
}

/// The centre colour of a 15-bit cell.
fn from15(i: usize) -> [u8; 3] {
    let x = |v: usize| ((v << 3) | (v >> 2)) as u8;
    [x((i >> 10) & 31), x((i >> 5) & 31), x(i & 31)]
}

/// At most `k` colours standing for a histogram of 15-bit cells: the median
/// cut (Heckbert), each box split along its longest channel at the count's
/// median, the box with the largest spread times count first, ties to the
/// earlier box; each colour the count-weighted mean of its box. Integer
/// counts and a fixed order: the same histogram gives the same colours.
fn median_cut(hist: &[u64], k: usize) -> Vec<[u8; 3]> {
    let cells: Vec<(usize, u64)> = hist
        .iter()
        .enumerate()
        .filter(|(_, n)| **n > 0)
        .map(|(i, n)| (i, *n))
        .collect();
    if cells.is_empty() || k == 0 {
        return Vec::new();
    }
    let chan = |i: usize, c: usize| (i >> (10 - 5 * c)) & 31;
    let spread = |b: &[(usize, u64)]| -> (usize, usize) {
        let mut best = (0, 0);
        for c in 0..3 {
            let lo = b.iter().map(|(i, _)| chan(*i, c)).min().unwrap_or(0);
            let hi = b.iter().map(|(i, _)| chan(*i, c)).max().unwrap_or(0);
            if hi - lo > best.0 {
                best = (hi - lo, c);
            }
        }
        best
    };
    let mut boxes: Vec<Vec<(usize, u64)>> = vec![cells];
    while boxes.len() < k {
        let mut pick: Option<(usize, u128)> = None;
        for (bi, b) in boxes.iter().enumerate() {
            if b.len() < 2 {
                continue;
            }
            let (range, _) = spread(b);
            let total: u64 = b.iter().map(|(_, n)| n).sum();
            let score = range as u128 * total as u128;
            if score > 0 && pick.is_none_or(|(_, s)| score > s) {
                pick = Some((bi, score));
            }
        }
        let Some((bi, _)) = pick else { break };
        let mut b = boxes.remove(bi);
        let (_, c) = spread(&b);
        b.sort_by_key(|(i, _)| (chan(*i, c), *i));
        let total: u64 = b.iter().map(|(_, n)| n).sum();
        let mut acc = 0u64;
        let mut cut = 1;
        for (j, (_, n)) in b.iter().enumerate() {
            acc += n;
            if acc * 2 >= total {
                cut = (j + 1).clamp(1, b.len() - 1);
                break;
            }
        }
        let rest = b.split_off(cut);
        boxes.insert(bi, rest);
        boxes.insert(bi, b);
    }
    boxes
        .iter()
        .map(|b| {
            let total: u64 = b.iter().map(|(_, n)| n).sum::<u64>().max(1);
            let mut s = [0u64; 3];
            for (i, n) in b {
                let c = from15(*i);
                for a in 0..3 {
                    s[a] += c[a] as u64 * n;
                }
            }
            s.map(|v| ((v + total / 2) / total).min(255) as u8)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The layout

/// One plane in its cell: where the image lies, in pixels of the film.
#[derive(Clone, Copy, Debug)]
struct Cell {
    plane: Plane,
    view: ViewPlane,
    /// The cell's top-left corner in the film and its size.
    x0: usize,
    y0: usize,
    cw: usize,
    ch: usize,
    /// The drawn image's top-left corner within the cell, and its size.
    ox: f64,
    oy: f64,
    dw: f64,
    dh: f64,
    /// Pixels per millimetre.
    scale: f64,
    /// Voxels along the plane's two axes, and their spacing.
    n: [usize; 2],
    sp: [f64; 2],
}

impl Cell {
    /// A point of the plane's pixel space (the views' display pixels, a
    /// pixel's centre at whole numbers) in pixels of the film.
    fn film_point(&self, p: [f32; 2]) -> [f32; 2] {
        [
            (self.x0 as f64 + self.ox + (p[0] as f64 + 0.5) * self.sp[0] * self.scale) as f32,
            (self.y0 as f64 + self.oy + (p[1] as f64 + 0.5) * self.sp[1] * self.scale) as f32,
        ]
    }

    /// The cell's pixels, `[x0, y0, x1, y1)`.
    fn clip(&self) -> [usize; 4] {
        [self.x0, self.y0, self.x0 + self.cw, self.y0 + self.ch]
    }
}

/// Where the scene is drawn in the film.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SceneCell {
    x0: usize,
    y0: usize,
    w: usize,
    h: usize,
}

/// The film's size, its plane cells and its scene cell.
struct Layout {
    size: [usize; 2],
    cells: Vec<Cell>,
    scene: Option<SceneCell>,
}

/// The plane of `view` at the film's slices.
fn plane_of(view: ViewPlane, slices: [usize; 3], grid: &Grid) -> Plane {
    let axis = match view {
        ViewPlane::Sagittal => 0,
        ViewPlane::Coronal => 1,
        ViewPlane::Axial => 2,
    };
    Plane {
        axis,
        index: slices[axis].min(grid.dims[axis].saturating_sub(1)),
    }
}

/// The millimetres a plane spans across and down.
fn extent_mm(plane: &Plane, grid: &Grid) -> [f64; 2] {
    let [a, b] = plane.along();
    [
        grid.dims[a] as f64 * grid.spacing[a],
        grid.dims[b] as f64 * grid.spacing[b],
    ]
}

/// The height of a cell `w` wide at `aspect` (width over height), else
/// `fallback`.
fn height_at(w: usize, aspect: Option<f32>, fallback: usize) -> usize {
    match aspect.filter(|a| a.is_finite() && *a > 0.05) {
        Some(a) => (w as f32 / a).round() as usize,
        None => fallback,
    }
    .clamp(16, MAX_WIDTH)
}

/// The film's layout; `scene`: the scene is drawn (it is given, and the
/// views want it).
fn layout(grid: &Grid, p: &FilmParams, scene: bool) -> Layout {
    let width = p.width.clamp(MIN_WIDTH, MAX_WIDTH);
    let views = p.views.planes();
    if views.is_empty() {
        let fallback = (width as f32 / SCENE_ASPECT).round() as usize;
        let h = height_at(width, p.scene_aspect, fallback);
        return Layout {
            size: [width, h],
            cells: Vec::new(),
            scene: scene.then_some(SceneCell {
                x0: 0,
                y0: 0,
                w: width,
                h,
            }),
        };
    }
    let cells = views.len();
    let cell_w = ((width - GAP * (cells - 1)) / cells).max(1);
    let planes: Vec<Plane> = views.iter().map(|v| plane_of(*v, p.slices, grid)).collect();
    let tallest = planes
        .iter()
        .map(|pl| {
            let [w, h] = extent_mm(pl, grid);
            (cell_w as f64 * h / w.max(1e-9)).round() as usize
        })
        .max()
        .unwrap_or(cell_w);
    let cell_h = height_at(cell_w, p.cell_aspect, tallest);
    // The pixels the division leaves over go to the first cells, one each,
    // so the film is exactly the width asked for.
    let spare = (width - GAP * (cells - 1)).saturating_sub(cell_w * cells);
    let mut x0 = 0;
    let out: Vec<Cell> = views
        .iter()
        .zip(planes)
        .enumerate()
        .map(|(c, (view, plane))| {
            let cw = cell_w + usize::from(c < spare);
            let [w_mm, h_mm] = extent_mm(&plane, grid);
            let scale = (cw as f64 / w_mm.max(1e-9)).min(cell_h as f64 / h_mm.max(1e-9));
            let (dw, dh) = (w_mm * scale, h_mm * scale);
            let [a, b] = plane.along();
            let at = x0;
            x0 += cw + GAP;
            Cell {
                plane,
                view: *view,
                x0: at,
                y0: 0,
                cw,
                ch: cell_h,
                ox: (cw as f64 - dw) * 0.5,
                oy: (cell_h as f64 - dh) * 0.5,
                dw,
                dh,
                scale,
                n: [grid.dims[a], grid.dims[b]],
                sp: [grid.spacing[a], grid.spacing[b]],
            }
        })
        .collect();
    let film_w = x0 - GAP;
    let scene = (scene && p.views == FilmViews::All).then(|| SceneCell {
        x0: 0,
        y0: cell_h + GAP,
        w: film_w,
        h: height_at(film_w, p.scene_aspect, cell_h),
    });
    let height = scene.map_or(cell_h, |s| s.y0 + s.h);
    Layout {
        size: [film_w, height],
        cells: out,
        scene,
    }
}

/// The size a film of `p` on `grid` comes out at, `[width, height]`;
/// `scene`: a scene is given (it is drawn when the views want it).
pub fn film_size(grid: &Grid, p: &FilmParams, scene: bool) -> [usize; 2] {
    layout(grid, p, scene && p.views.wants_scene()).size
}

/// A pixel of the film on an image: which cell, where on its plane (voxel
/// coordinates along [`Plane::along`]) and the patient point there.
#[derive(Clone, Copy)]
struct Px {
    cell: u8,
    u: f32,
    v: f32,
    at: Vec3,
}

/// Every pixel of the film, `None` off the images.
fn pixel_map(grid: &Grid, size: [usize; 2], cells: &[Cell]) -> Vec<Option<Px>> {
    let [w, h] = size;
    let mut out = vec![None; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let Some((ci, c)) = cells
                .iter()
                .enumerate()
                .find(|(_, c)| x >= c.x0 && x < c.x0 + c.cw && y >= c.y0 && y < c.y0 + c.ch)
            else {
                continue;
            };
            let cx = (x - c.x0) as f64 + 0.5 - c.ox;
            let cy = (y - c.y0) as f64 + 0.5 - c.oy;
            if cx < 0.0 || cy < 0.0 || cx > c.dw || cy > c.dh {
                continue;
            }
            let u = cx / c.scale / c.sp[0] - 0.5;
            let down = cy / c.scale / c.sp[1] - 0.5;
            // Axial rows run with j; the other two put superior on top.
            let v = match c.view {
                ViewPlane::Axial => down,
                _ => (c.n[1] as f64 - 1.0) - down,
            };
            let (uc, vc) = (
                u.clamp(0.0, (c.n[0] as f64 - 1.0).max(0.0)),
                v.clamp(0.0, (c.n[1] as f64 - 1.0).max(0.0)),
            );
            let ijk = match c.plane.axis {
                0 => [c.plane.index as f64, uc, vc],
                1 => [uc, c.plane.index as f64, vc],
                _ => [uc, vc, c.plane.index as f64],
            };
            *o = Some(Px {
                cell: ci as u8,
                u: uc as f32,
                v: vc as f32,
                at: grid.voxel_to_patient(ijk[0], ijk[1], ijk[2]),
            });
        }
    });
    out
}

/// Bilinear value of a plane (`u` fastest, `n[0]` by `n[1]`) at `(u, v)`,
/// already clamped to it.
fn bilinear(vals: &[f32], n: [usize; 2], u: f32, v: f32) -> f32 {
    let (nu, nv) = (n[0], n[1]);
    if vals.len() < nu * nv || nu == 0 || nv == 0 {
        return 0.0;
    }
    let u0 = (u.floor() as usize).min(nu.saturating_sub(2));
    let v0 = (v.floor() as usize).min(nv.saturating_sub(2));
    let (u1, v1) = ((u0 + 1).min(nu - 1), (v0 + 1).min(nv - 1));
    let (fu, fv) = (
        (u - u0 as f32).clamp(0.0, 1.0),
        (v - v0 as f32).clamp(0.0, 1.0),
    );
    let at = |a: usize, b: usize| vals[b * nu + a];
    let top = at(u0, v0) * (1.0 - fu) + at(u1, v0) * fu;
    let bottom = at(u0, v1) * (1.0 - fu) + at(u1, v1) * fu;
    top * (1.0 - fv) + bottom * fv
}

// ---------------------------------------------------------------------------
// Lines over the planes

/// A frame of palette indices with the cell it is being drawn in.
struct Canvas<'a> {
    idx: &'a mut [u8],
    w: usize,
    clip: [usize; 4],
}

impl Canvas<'_> {
    fn put(&mut self, x: f32, y: f32, k: u8) {
        if !(x.is_finite() && y.is_finite()) {
            return;
        }
        let (x, y) = (x.floor(), y.floor());
        let [x0, y0, x1, y1] = self.clip;
        if x < x0 as f32 || y < y0 as f32 || x >= x1 as f32 || y >= y1 as f32 {
            return;
        }
        self.idx[y as usize * self.w + x as usize] = k;
    }

    /// A line one pixel wide, every pixel it passes through.
    fn line(&mut self, a: [f32; 2], b: [f32; 2], k: u8) {
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let n = dx.abs().max(dy.abs()).ceil();
        if !n.is_finite() || n > 1e5 {
            return;
        }
        let n = n.max(1.0) as usize;
        for i in 0..=n {
            let f = i as f32 / n as f32;
            self.put(a[0] + dx * f, a[1] + dy * f, k);
        }
    }

    /// A filled disc of radius `r` about `c`.
    fn disc(&mut self, c: [f32; 2], r: f32, k: u8) {
        let r2 = r * r;
        let (lo_x, hi_x) = ((c[0] - r).floor() as i64, (c[0] + r).ceil() as i64);
        let (lo_y, hi_y) = ((c[1] - r).floor() as i64, (c[1] + r).ceil() as i64);
        for y in lo_y..=hi_y {
            for x in lo_x..=hi_x {
                let (px, py) = (x as f32 + 0.5 - c[0], y as f32 + 0.5 - c[1]);
                if px * px + py * py <= r2 {
                    self.put(x as f32, y as f32, k);
                }
            }
        }
    }
}

/// A grid's geometry as a volume without voxels, for the field's walks on
/// the planes (which read only where the pixels are).
fn geometry(grid: &Grid) -> Volume {
    Volume {
        data: Vec::new(),
        dims: grid.dims,
        spacing: grid.spacing,
        origin: grid.origin,
        row_dir: grid.row_dir,
        col_dir: grid.col_dir,
        normal: grid.normal,
        frame_of_reference_uid: grid.frame_of_reference_uid.clone(),
        min_value: 0,
        max_value: 0,
    }
}

/// The field over one cell, as the window's overlay draws it.
fn draw_field(
    canvas: &mut Canvas,
    cell: &Cell,
    vf: &dvf::VectorField,
    geo: &Volume,
    fo: &FieldOverlay,
    pal: &Pal,
) {
    let max = (vf.max_mag as f32).max(1e-6);
    match fo.style {
        FieldStyle::Arrows => {
            let scale = dvf::arrow_length_factor(fo.scale, fo.spacing_mm);
            for g in dvf::glyphs_on_plane(vf, geo, cell.view, cell.plane.index, fo.spacing_mm) {
                let k = pal.arrow(g.magnitude, max, fo.colored);
                let a = cell.film_point(g.from);
                let tip = cell.film_point([
                    g.from[0] + (g.to[0] - g.from[0]) * scale,
                    g.from[1] + (g.to[1] - g.from[1]) * scale,
                ]);
                let d = [tip[0] - a[0], tip[1] - a[1]];
                let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
                if len < dvf::ARROW_MIN_LEN {
                    canvas.disc(a, dvf::arrow_disc_radius(g.out_of_plane, max), k);
                } else {
                    canvas.line(a, tip, k);
                    let n = [d[0] / len, d[1] / len];
                    let perp = [-n[1], n[0]];
                    let head = (len * dvf::ARROW_HEAD).min(dvf::ARROW_HEAD_MAX);
                    let wide = head * dvf::ARROW_HEAD_WIDTH;
                    for s in [1.0f32, -1.0] {
                        let end = [
                            tip[0] - n[0] * head + s * perp[0] * wide,
                            tip[1] - n[1] * head + s * perp[1] * wide,
                        ];
                        canvas.line(tip, end, k);
                    }
                }
            }
        }
        FieldStyle::Grid => {
            let k = pal.grid_line(fo.colored);
            for line in
                dvf::deformed_grid_on_plane(vf, geo, cell.view, cell.plane.index, fo.spacing_mm)
            {
                for pair in line.windows(2) {
                    canvas.line(cell.film_point(pair[0]), cell.film_point(pair[1]), k);
                }
            }
        }
        FieldStyle::None => {}
    }
}

// ---------------------------------------------------------------------------
// The 3D scene

/// Rows of the scene each worker fills, every triangle in order.
const BAND: usize = 8;

/// The scene of `meshes` (already carried to the time) in a `w` by `h`
/// picture: each pixel's colour, `None` where no surface is drawn (the
/// pane's black). See the module notes for what is the 3D pane's.
pub fn render_scene(
    scene: &FilmScene,
    meshes: &[RoiMesh],
    w: usize,
    h: usize,
) -> Vec<Option<[u8; 3]>> {
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let cam = &scene.camera;
    let scale = mesh3d::scene_scale(w as f32, h as f32, cam.radius.max(1e-3), cam.zoom);
    let side = w.min(h) as f32;
    let cx = w as f32 * 0.5 + cam.pan[0] * side;
    let cy = h as f32 * 0.5 + cam.pan[1] * side;
    let (yaw, pitch) = (cam.yaw.sin_cos(), cam.pitch.sin_cos());
    let alpha = (scene.opacity.clamp(0.0, 1.0) * 255.0) as u8;

    // The turned, shaded vertices: positions on the picture, depth into it,
    // premultiplied colour and coverage.
    let mut pos: Vec<[f32; 2]> = Vec::new();
    let mut depth: Vec<f32> = Vec::new();
    let mut col: Vec<[f32; 4]> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    for m in meshes {
        let roi = scene.rois.get(m.roi_index);
        let own = roi.map_or(1.0, |r| r.alpha);
        if !roi.is_none_or(|r| r.visible) || own <= 0.0 {
            continue;
        }
        let color = roi.map_or(m.color, |r| r.color);
        let entry = (alpha as f32 * own).round() as u8;
        let a = if m.external {
            (entry as f32 * mesh3d::EXTERNAL_ALPHA) as u8
        } else {
            entry
        };
        let af = a as f32 / 255.0;
        let base = pos.len() as u32;
        for (v, n) in m.verts.iter().zip(m.normals.iter()) {
            let t = mesh3d::turn(yaw, pitch, cam.center, *v, true);
            let nn = mesh3d::turn(yaw, pitch, cam.center, *n, false);
            let inten = mesh3d::headlight(nn);
            let rgb = color.map(|c| (c as f32 * inten) as u8);
            col.push([
                rgb[0] as f32 * af,
                rgb[1] as f32 * af,
                rgb[2] as f32 * af,
                af,
            ]);
            pos.push([cx + t[0] * scale, cy - t[2] * scale]);
            depth.push(t[1]);
        }
        let nv = m.verts.len().min(m.normals.len()) as u32;
        tris.extend(
            m.tris
                .iter()
                .filter(|t| t.iter().all(|&i| i < nv))
                .map(|t| [base + t[0], base + t[1], base + t[2]]),
        );
    }

    // Far triangles first (the viewer at -y), ties in the order they come.
    let mut order: Vec<u64> = tris
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let d = (depth[t[0] as usize] + depth[t[1] as usize] + depth[t[2] as usize]) / 3.0;
            ((!mesh3d::depth_key(d) as u64) << 32) | i as u64
        })
        .collect();
    order.par_sort_unstable();

    // Each band of rows gets the triangles that reach it, in that order.
    let nb = h.div_ceil(BAND);
    let mut bands: Vec<Vec<u32>> = vec![Vec::new(); nb];
    for &o in &order {
        let i = (o & 0xFFFF_FFFF) as u32;
        let t = tris[i as usize];
        let ys = t.map(|v| pos[v as usize][1]);
        let (lo, hi) = (ys[0].min(ys[1]).min(ys[2]), ys[0].max(ys[1]).max(ys[2]));
        if !(lo.is_finite() && hi.is_finite()) || hi < 0.0 || lo >= h as f32 {
            continue;
        }
        let b0 = ((lo - 0.5).max(0.0) as usize / BAND).min(nb - 1);
        let b1 = ((hi + 0.5).max(0.0) as usize / BAND).min(nb - 1);
        for band in &mut bands[b0..=b1] {
            band.push(i);
        }
    }

    let edge = |a: [f32; 2], b: [f32; 2], p: [f32; 2]| -> f32 {
        (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    };
    let mut out: Vec<Option<[f32; 3]>> = vec![None; w * h];
    out.par_chunks_mut(w * BAND)
        .zip(bands.par_iter())
        .enumerate()
        .for_each(|(bi, (rows, list))| {
            let y0 = bi * BAND;
            let y1 = y0 + rows.len() / w;
            for &ti in list {
                let t = tris[ti as usize];
                let p = t.map(|v| pos[v as usize]);
                let c = t.map(|v| col[v as usize]);
                let area = edge(p[0], p[1], p[2]);
                if area.abs() < 1e-9 || !area.is_finite() {
                    continue;
                }
                let xs = [p[0][0], p[1][0], p[2][0]];
                let ys = [p[0][1], p[1][1], p[2][1]];
                let fmin = |v: [f32; 3]| v[0].min(v[1]).min(v[2]);
                let fmax = |v: [f32; 3]| v[0].max(v[1]).max(v[2]);
                let xa = (fmin(xs) - 0.5).ceil().max(0.0) as usize;
                let xb = ((fmax(xs) - 0.5).floor()).min(w as f32 - 1.0);
                let ya = ((fmin(ys) - 0.5).ceil().max(y0 as f32)) as usize;
                let yb = ((fmax(ys) - 0.5).floor()).min(y1 as f32 - 1.0);
                if xb < 0.0 || yb < 0.0 {
                    continue;
                }
                let (xb, yb) = (xb as usize, yb as usize);
                for y in ya..=yb {
                    let py = y as f32 + 0.5;
                    for x in xa..=xb {
                        let q = [x as f32 + 0.5, py];
                        let w0 = edge(p[1], p[2], q) / area;
                        let w1 = edge(p[2], p[0], q) / area;
                        let w2 = 1.0 - w0 - w1;
                        if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                            continue;
                        }
                        let s = [0, 1, 2, 3].map(|k| w0 * c[0][k] + w1 * c[1][k] + w2 * c[2][k]);
                        let d = &mut rows[(y - y0) * w + x];
                        let under = d.unwrap_or([0.0; 3]);
                        *d = Some([
                            s[0] + under[0] * (1.0 - s[3]),
                            s[1] + under[1] * (1.0 - s[3]),
                            s[2] + under[2] * (1.0 - s[3]),
                        ]);
                    }
                }
            }
        });
    out.into_iter()
        .map(|p| p.map(|c| c.map(|v| v.round().clamp(0.0, 255.0) as u8)))
        .collect()
}

/// For every 15-bit colour, the nearest of the greys and the scene's
/// colours (ties to the lower index).
fn scene_lut(pal: &Pal, table: &[u8]) -> Vec<u8> {
    let mut cands: Vec<(u8, [u8; 3])> = (0..pal.grays)
        .map(|i| (i as u8, [table[i * 3], table[i * 3 + 1], table[i * 3 + 2]]))
        .collect();
    if let Some(s) = pal.scene {
        for i in s..(s + SCENE_SLOTS).min(256) {
            cands.push((i as u8, [table[i * 3], table[i * 3 + 1], table[i * 3 + 2]]));
        }
    }
    (0..1usize << 15)
        .into_par_iter()
        .map(|c| {
            let rgb = from15(c);
            cands
                .iter()
                .min_by_key(|(i, k)| (dist2(*k, rgb), *i))
                .map_or(0, |(i, _)| *i)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// The film

/// The film of the path, written to `path` as an animated GIF.
///
/// `states` are the states' own images in the model's order; `slicer`
/// renders the planes on the GPU when given; `structures` is the first
/// state's structure set (on the model's grid's frame) whose structures
/// are outlined, all of them (filter it first for the visible ones);
/// `scene` the 3D scene, drawn when the views want it ([`FilmViews::Scene`]
/// needs one). A cancelled run leaves no file.
#[allow(clippy::too_many_arguments)]
pub fn write_gif(
    path: &Path,
    model: &PostureModel,
    states: &[Arc<Volume>],
    slicer: Option<&GpuSlicer>,
    structures: Option<&StructureSet>,
    scene: Option<&FilmScene>,
    params: &FilmParams,
    progress: &Progress,
) -> Result<FilmSummary> {
    let res = write_gif_inner(
        path, model, states, slicer, structures, scene, params, progress,
    );
    if res.is_err() {
        let _ = std::fs::remove_file(path);
    }
    res
}

#[allow(clippy::too_many_arguments)]
fn write_gif_inner(
    path: &Path,
    model: &PostureModel,
    states: &[Arc<Volume>],
    slicer: Option<&GpuSlicer>,
    structures: Option<&StructureSet>,
    scene: Option<&FilmScene>,
    params: &FilmParams,
    progress: &Progress,
) -> Result<FilmSummary> {
    let grid = &model.grid;
    if grid.dims.contains(&0) {
        bail!("the model has no grid to film");
    }
    let scene = scene
        .filter(|_| params.views.wants_scene())
        .filter(|s| s.meshes.iter().any(|m| !m.tris.is_empty()));
    if params.views == FilmViews::Scene && scene.is_none() {
        bail!("the 3D scene needs the first state's structures as surfaces, and there are none");
    }
    let lay = layout(grid, params, scene.is_some());
    let [w, h] = lay.size;
    let cells = &lay.cells;
    let planes: Vec<Plane> = cells.iter().map(|c| c.plane).collect();
    let field = params.drawn_field().filter(|_| !cells.is_empty());
    let pal = Pal::new(field.is_some(), lay.scene.is_some());
    progress.set("Laying out the film");
    let pixels = pixel_map(grid, lay.size, cells);
    let geo = geometry(grid);

    // The structures, rasterized once on the first state's grid.
    let mut masks: Vec<BoxedMask> = Vec::new();
    let mut colors: Vec<[u8; 3]> = Vec::new();
    if params.outlines && !cells.is_empty() {
        if let Some(ss) = structures {
            for roi in ss
                .rois
                .iter()
                .filter(|r| !r.is_poi() && !r.contours.is_empty())
            {
                if masks.len() >= MAX_OUTLINED {
                    break;
                }
                if progress.cancelled() {
                    bail!("film {}", progress::CANCELLED);
                }
                progress.set(format!("Outlining {}", roi.name));
                if let Some(m) = crate::segmentation::rasterize_roi(grid, roi) {
                    masks.push(BoxedMask::pack(&m, grid.dims));
                    colors.push(roi.color);
                }
            }
        }
    }
    let (slot_colors, slot_of) = structure_slots(&colors);
    let times = film_times(model.span(), params.frames, params.bounce);

    // The scene's colours, sampled along the run.
    let scene_colors: Vec<[u8; 3]> = match (scene, lay.scene) {
        (Some(sc), Some(cell)) => {
            let mut hist = vec![0u64; 1 << 15];
            let samples = film_times(model.span(), params.frames.clamp(2, SCENE_SAMPLES), false);
            for (k, &t) in samples.iter().enumerate() {
                if progress.cancelled() {
                    bail!("film {}", progress::CANCELLED);
                }
                progress.set(format!(
                    "The 3D scene's colours ({}/{})",
                    k + 1,
                    samples.len()
                ));
                let frame = model.frame(t);
                let meshes = apply::carry_meshes(&sc.meshes, &frame.forward);
                for c in render_scene(sc, &meshes, cell.w, cell.h)
                    .into_iter()
                    .flatten()
                {
                    hist[rgb15(c)] += 1;
                }
            }
            median_cut(&hist, SCENE_SLOTS)
        }
        _ => Vec::new(),
    };
    let table = pal.table(&slot_colors, &scene_colors);
    let lut = lay.scene.is_some().then(|| scene_lut(&pal, &table));

    let file = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut enc = gif::Encoder::new(std::io::BufWriter::new(file), w as u16, h as u16, &table)
        .context("start the GIF")?;
    enc.set_repeat(gif::Repeat::Infinite)
        .context("set the GIF to loop")?;
    let delay = delay_cs(params.fps);

    let (centre, width) = (params.window.0, params.window.1.max(1.0));
    let lo = centre - width * 0.5;
    let scan = matches!(
        params.show,
        Show::StartForward | Show::EndBackward | Show::CrossFade
    );
    let n = times.len();
    let (mut gpu_all, mut why) = (slicer.is_some() && !cells.is_empty(), None);
    for (f, &t) in times.iter().enumerate() {
        if progress.cancelled() {
            bail!("film {}", progress::CANCELLED);
        }
        progress.report(f as f32 / n as f32, &format!("Frame {}/{n}", f + 1));
        let frame = model.frame(t);
        let values = if cells.is_empty() {
            Vec::new()
        } else {
            let (values, gpu, w_why) =
                render_planes_with(slicer, model, states, &frame, params.show, grid, &planes)?;
            gpu_all &= gpu;
            if why.is_none() {
                why = w_why;
            }
            values
        };
        let jac: Option<Vec<f64>> = (params.wash && !cells.is_empty()).then(|| {
            apply::jacobian_now(&frame)
                .into_iter()
                .map(f64::from)
                .collect()
        });
        // Which structures each pixel is in, where its tissue came from.
        let inside: Vec<u64> = if masks.is_empty() {
            Vec::new()
        } else {
            pixels
                .par_iter()
                .map(|px| {
                    let Some(px) = px else { return 0u64 };
                    let x = frame.to_first.map(px.at);
                    let [a, b, c] = grid.patient_to_voxel(x);
                    let (a, b, c) = (a.round(), b.round(), c.round());
                    if a < 0.0 || b < 0.0 || c < 0.0 {
                        return 0;
                    }
                    let (a, b, c) = (a as usize, b as usize, c as usize);
                    if a >= grid.dims[0] || b >= grid.dims[1] || c >= grid.dims[2] {
                        return 0;
                    }
                    masks.iter().enumerate().fold(0u64, |acc, (r, m)| {
                        if m.get(a, b, c) {
                            acc | 1 << r
                        } else {
                            acc
                        }
                    })
                })
                .collect()
        };
        let mut idx = vec![0u8; w * h];
        idx.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, o) in row.iter_mut().enumerate() {
                let i = y * w + x;
                let Some(px) = pixels[i] else { continue };
                let c = &cells[px.cell as usize];
                let g = bilinear(&values[px.cell as usize], c.n, px.u, px.v);
                let level = ((g - lo) / width).clamp(0.0, 1.0);
                // An outline: inside a structure with a neighbour (on the
                // same image) that is not; the last structure on top.
                if !inside.is_empty() && inside[i] != 0 {
                    let near = |dx: i64, dy: i64| -> u64 {
                        let (nx, ny) = (x as i64 + dx, y as i64 + dy);
                        if nx < 0 || ny < 0 || nx >= w as i64 || ny >= h as i64 {
                            return 0;
                        }
                        let j = ny as usize * w + nx as usize;
                        match pixels[j] {
                            Some(q) if q.cell == px.cell => inside[j],
                            _ => 0,
                        }
                    };
                    let all = near(1, 0) & near(-1, 0) & near(0, 1) & near(0, -1);
                    let edge = inside[i] & !all;
                    if edge != 0 {
                        let r = 63 - edge.leading_zeros() as usize;
                        *o = (pal.structs + slot_of[r]) as u8;
                        continue;
                    }
                }
                if let Some(jac) = &jac {
                    if !scan || g >= apply::WASH_TISSUE_HU as f32 {
                        let lat = &frame.forward.lattice;
                        let j = apply::node_scalar(lat, jac, px.at) as f32;
                        if let Some(k) = pal.wash_index(apply::wash_color(j), level) {
                            *o = k as u8;
                            continue;
                        }
                    }
                }
                *o = pal.gray(level);
            }
        });
        // The displacement from the first state, over the planes.
        if let Some(fo) = &field {
            let vf = frame.forward.to_vector_field(None);
            for cell in cells {
                let mut canvas = Canvas {
                    idx: &mut idx,
                    w,
                    clip: cell.clip(),
                };
                draw_field(&mut canvas, cell, &vf, &geo, fo, &pal);
            }
        }
        // The scene, the first state's surfaces where they are now.
        if let (Some(sc), Some(cell), Some(lut)) = (scene, lay.scene, &lut) {
            let meshes = apply::carry_meshes(&sc.meshes, &frame.forward);
            let pic = render_scene(sc, &meshes, cell.w, cell.h);
            for (y, row) in pic.chunks(cell.w).enumerate() {
                let at = (cell.y0 + y) * w + cell.x0;
                for (x, p) in row.iter().enumerate() {
                    idx[at + x] = p.map_or(0, |c| lut[rgb15(c)]);
                }
            }
        }
        let frame_out = gif::Frame {
            width: w as u16,
            height: h as u16,
            delay,
            buffer: Cow::Owned(idx),
            ..gif::Frame::default()
        };
        enc.write_frame(&frame_out)
            .with_context(|| format!("write frame {} of {}", f + 1, path.display()))?;
    }
    let mut out = enc.into_inner().context("finish the GIF")?;
    out.flush().context("finish the GIF")?;
    drop(out);
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    progress.report(1.0, "Film written");
    Ok(FilmSummary {
        frames: n,
        width: w,
        height: h,
        bytes,
        gpu: gpu_all,
        why,
        scene: lay.scene.is_some(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn times_go_there_and_back_without_repeating_an_end() {
        assert_eq!(film_times(1.0, 3, false), vec![0.0, 0.5, 1.0]);
        assert_eq!(film_times(1.0, 3, true), vec![0.0, 0.5, 1.0, 0.5]);
        assert_eq!(film_times(2.0, 2, true), vec![0.0, 2.0]);
        assert_eq!(film_times(1.0, 0, false).len(), 2, "at least the two ends");
    }

    #[test]
    fn the_delay_is_in_hundredths_within_what_a_gif_holds() {
        assert_eq!(delay_cs(10.0), 10);
        assert_eq!(delay_cs(6.0), 17);
        assert_eq!(delay_cs(200.0), 2);
        assert_eq!(delay_cs(0.1), 100);
        assert_eq!(delay_cs(f32::NAN), 17);
    }

    #[test]
    fn the_palette_is_greys_then_structures_then_the_wash() {
        let (slots, of) = structure_slots(&[[255, 0, 0], [0, 255, 0], [255, 0, 0]]);
        assert_eq!(slots.len(), 2, "one slot per colour");
        assert_eq!(of, vec![0, 1, 0]);
        let p = Pal::new(false, false);
        assert_eq!(p.grays, GRAYS, "all the greys when nothing else needs room");
        let pal = p.table(&slots, &[]);
        assert_eq!(pal.len(), 768);
        assert_eq!(&pal[0..3], &[0, 0, 0]);
        assert_eq!(&pal[(GRAYS - 1) * 3..GRAYS * 3], &[255, 255, 255]);
        assert_eq!(&pal[p.structs * 3..p.structs * 3 + 3], &[255, 0, 0]);
        assert_eq!(&pal[p.folded * 3..p.folded * 3 + 3], &[230, 0, 230]);
        // A blue over white is lighter than over black.
        let dark = p.wash_index([40, 120, 255, 200], 0.0).unwrap();
        let light = p.wash_index([40, 120, 255, 200], 1.0).unwrap();
        assert!(pal[light * 3] > pal[dark * 3]);
        assert_eq!(p.wash_index([0, 0, 0, 0], 0.5), None);
        assert_eq!(p.wash_index([230, 0, 230, 210], 0.5), Some(p.folded));
        assert!(p.wash_index([255, 70, 30, 60], 0.5).unwrap() >= p.wash + 2 * WASH_GRAYS);
    }

    #[test]
    fn the_arrows_and_the_scene_take_their_room_from_the_greys() {
        for (arrows, scene, greys) in [(true, false, 177), (false, true, 131), (true, true, 113)] {
            let p = Pal::new(arrows, scene);
            assert_eq!(p.grays, greys, "{arrows} {scene}");
            let end = p
                .scene
                .map_or(p.arrows.map_or(p.folded + 1, |a| a + ARROW_SLOTS), |s| {
                    s + SCENE_SLOTS
                });
            assert!(end <= 256, "{end} entries");
        }
        let p = Pal::new(true, false);
        let pal = p.table(&[], &[]);
        let a = p.arrows.unwrap();
        // The ramp is the window's colour map, its ends and its middle.
        assert_eq!(&pal[a * 3..a * 3 + 3], &render::dose_colormap(0.0));
        let top = a + ARROW_STEPS - 1;
        assert_eq!(&pal[top * 3..top * 3 + 3], &render::dose_colormap(1.0));
        assert_eq!(p.arrow(0.0, 10.0, true) as usize, a);
        assert_eq!(p.arrow(10.0, 10.0, true) as usize, top);
        assert_eq!(
            p.arrow(20.0, 10.0, true) as usize,
            top,
            "held to the largest"
        );
        let plain = p.arrow(5.0, 10.0, false) as usize;
        assert_eq!(&pal[plain * 3..plain * 3 + 3], &[255, 235, 130]);
        let grid = p.grid_line(true) as usize;
        let g = dvf::grid_rgba(true);
        assert_eq!(&pal[grid * 3..grid * 3 + 3], &[g[0], g[1], g[2]]);
    }

    #[test]
    fn more_structures_than_slots_take_the_nearest() {
        let colors: Vec<[u8; 3]> = (0..40).map(|i| [i as u8 * 6, 0, 0]).collect();
        let (slots, of) = structure_slots(&colors);
        assert_eq!(slots.len(), STRUCT_SLOTS);
        assert!(of.iter().all(|&s| s < STRUCT_SLOTS));
        assert_eq!(of[39], STRUCT_SLOTS - 1, "the reddest slot is nearest");
    }

    #[test]
    fn the_median_cut_keeps_the_colours_it_can_and_merges_the_rest() {
        let mut hist = vec![0u64; 1 << 15];
        let red = rgb15([248, 0, 0]);
        let green = rgb15([0, 248, 0]);
        hist[red] = 100;
        hist[green] = 50;
        let two = median_cut(&hist, 8);
        assert_eq!(two.len(), 2, "no more colours than there are");
        assert!(two.contains(&from15(red)) && two.contains(&from15(green)));
        let one = median_cut(&hist, 1);
        assert_eq!(one.len(), 1);
        assert!(one[0][0] > one[0][1], "weighted toward the commoner red");
        assert_eq!(median_cut(&hist, 8), two, "the same again");
        assert!(median_cut(&vec![0; 1 << 15], 8).is_empty());
    }

    /// One square of two triangles facing the camera: the camera fitted to
    /// it, straight on (no turn).
    fn square(color: [u8; 3], y: f32, external: bool, index: usize) -> RoiMesh {
        RoiMesh {
            roi_index: index,
            color,
            external,
            verts: vec![
                [-10.0, y, -10.0],
                [10.0, y, -10.0],
                [10.0, y, 10.0],
                [-10.0, y, 10.0],
            ],
            normals: vec![[0.0, -1.0, 0.0]; 4],
            tris: vec![[0, 1, 2], [0, 2, 3]],
        }
    }

    fn straight_on(meshes: Vec<RoiMesh>) -> FilmScene {
        let mut s = FilmScene::new(Arc::new(meshes));
        s.camera.yaw = 0.0;
        s.camera.pitch = 0.0;
        s.camera.center = [0.0, 0.0, 0.0];
        s.camera.radius = 20.0;
        s
    }

    #[test]
    fn the_scene_is_drawn_as_the_pane_draws_it() {
        let s = straight_on(vec![square([200, 100, 50], 0.0, false, 0)]);
        let pic = render_scene(&s, &s.meshes, 40, 40);
        assert_eq!(pic.len(), 1600);
        // Facing the light: full colour in the middle; black (nothing) in
        // the corner, outside the square (0.45 of 40 px over 20 mm: 9 px).
        assert_eq!(pic[20 * 40 + 20], Some([200, 100, 50]));
        assert_eq!(pic[0], None);
        // Turned edge-on to the light, the headlight's floor.
        let mut side = s.clone();
        side.camera.yaw = 0.0;
        let mut m = square([200, 100, 50], 0.0, false, 0);
        m.normals = vec![[1.0, 0.0, 0.0]; 4];
        side.meshes = Arc::new(vec![m]);
        let p = render_scene(&side, &side.meshes, 40, 40)[20 * 40 + 20].unwrap();
        assert_eq!(p, [60, 30, 15], "0.3 of the colour");
        // Hidden or faded to nothing: not drawn.
        let mut hid = s.clone();
        hid.rois = vec![SceneRoi {
            visible: false,
            color: [1, 2, 3],
            alpha: 1.0,
        }];
        assert!(render_scene(&hid, &hid.meshes, 40, 40)
            .iter()
            .all(Option::is_none));
    }

    #[test]
    fn near_surfaces_cover_far_ones_and_a_body_is_translucent() {
        // Red in front (y = -5, toward the viewer), blue behind; given in
        // either order, red is what shows.
        for meshes in [
            vec![
                square([255, 0, 0], -5.0, false, 0),
                square([0, 0, 255], 5.0, false, 1),
            ],
            vec![
                square([0, 0, 255], 5.0, false, 1),
                square([255, 0, 0], -5.0, false, 0),
            ],
        ] {
            let s = straight_on(meshes);
            assert_eq!(
                render_scene(&s, &s.meshes, 40, 40)[20 * 40 + 20],
                Some([255, 0, 0])
            );
        }
        // A body in front lets the blue through, tinted.
        let s = straight_on(vec![
            square([255, 255, 255], -5.0, true, 0),
            square([0, 0, 255], 5.0, false, 1),
        ]);
        let p = render_scene(&s, &s.meshes, 40, 40)[20 * 40 + 20].unwrap();
        let a = (255.0 * mesh3d::EXTERNAL_ALPHA) as u8 as f32 / 255.0;
        assert_eq!(p[2], (255.0 * a + 255.0 * (1.0 - a)).round() as u8);
        assert_eq!(p[0], (255.0 * a).round() as u8);
        // The same scene twice is the same picture.
        assert_eq!(
            render_scene(&s, &s.meshes, 40, 40),
            render_scene(&s, &s.meshes, 40, 40)
        );
    }

    #[test]
    fn the_scene_takes_its_cell_under_the_planes_or_the_whole_film() {
        let grid = Grid {
            dims: [40, 30, 20],
            spacing: [2.0, 2.0, 3.0],
            origin: Vec3::ZERO,
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
        };
        let mut p = FilmParams::centred(&grid, Show::StartForward, (40.0, 400.0));
        p.width = 302;
        p.cell_aspect = Some(1.0);
        // Three cells of 100, 99 and 99 across, 99 tall.
        let planes = layout(&grid, &p, false);
        assert_eq!(planes.size, [302, 99]);
        assert!(planes.scene.is_none());
        // Under the planes, the whole width, as tall as asked.
        p.scene_aspect = Some(3.02);
        let all = layout(&grid, &p, true);
        assert_eq!(
            all.scene,
            Some(SceneCell {
                x0: 0,
                y0: 101,
                w: 302,
                h: 100
            })
        );
        assert_eq!(all.size, [302, 201]);
        assert_eq!(film_size(&grid, &p, true), [302, 201]);
        assert_eq!(film_size(&grid, &p, false), [302, 99], "no scene given");
        // Alone.
        p.views = FilmViews::Scene;
        p.scene_aspect = Some(2.0);
        assert_eq!(film_size(&grid, &p, true), [302, 151]);
        // A plane alone never has the scene.
        p.views = FilmViews::One(ViewPlane::Sagittal);
        assert!(layout(&grid, &p, p.views.wants_scene()).scene.is_none());
        for v in FilmViews::CHOICES {
            assert_eq!(FilmViews::from_key(v.key()), Some(v));
        }
        assert_eq!(FilmViews::from_key("3D"), Some(FilmViews::Scene));
    }

    #[test]
    fn lines_stay_in_their_cell() {
        let mut idx = vec![0u8; 10 * 10];
        let mut c = Canvas {
            idx: &mut idx,
            w: 10,
            clip: [2, 2, 6, 6],
        };
        c.line([0.0, 4.5], [9.9, 4.5], 7);
        c.disc([4.5, 4.5], 0.6, 9);
        let row: Vec<u8> = idx[40..50].to_vec();
        assert_eq!(row, vec![0, 0, 7, 7, 9, 7, 0, 0, 0, 0]);
        assert!(idx[..20].iter().all(|&k| k == 0));
    }

    #[test]
    fn bilinear_reads_the_corners_and_the_middle() {
        let v = [0.0, 10.0, 20.0, 30.0];
        assert_eq!(bilinear(&v, [2, 2], 0.0, 0.0), 0.0);
        assert_eq!(bilinear(&v, [2, 2], 1.0, 1.0), 30.0);
        assert_eq!(bilinear(&v, [2, 2], 0.5, 0.5), 15.0);
        assert_eq!(bilinear(&[7.0], [1, 1], 0.0, 0.0), 7.0);
    }
}
