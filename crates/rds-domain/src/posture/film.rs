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
//! **Views.** Either the three planes laid out as the window lays them out
//! (axial, sagittal, coronal, left to right, each in a cell of the panes'
//! shape, the image fitted inside it with its millimetres square and the
//! rest black), or one of them alone. A plane is the whole slice of the
//! grid, at the slice the view shows: the view's zoom and pan are not
//! taken. Sagittal and coronal rows run superior to inferior and axial rows
//! anterior to posterior, as the views draw them
//! ([`crate::volume::Volume::extract_slice`]).
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
//! the image is a scan ([`apply::WASH_TISSUE_HU`]). The dose, the
//! displacement glyphs and the 3D scene are not drawn: the recorder of the
//! window films those, from the screen.
//!
//! **One palette for the whole film**, the GIF's global colour table, so
//! no frame is quantized and nothing flickers between frames:
//!
//! * indices `0..192`: 192 grey levels from black to white (a step of 1.3
//!   of the 256 a screen shows, finer than the eye tells apart on a CT);
//! * `192..224`: 32 structure colours, the structures' own in their order
//!   (a structure past the 32nd distinct colour takes the nearest);
//! * `224..252`: the wash, two hues (blue: compressed, red: expanded) at two
//!   strengths over 7 grey levels each - the wash's colour mixed into the
//!   grey under it, the grey kept coarsely so the anatomy still shows;
//! * `252`: folded tissue (magenta), and black for the rest.
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
use crate::progress::{self, Progress, ProgressSink};
use crate::propagate::BoxedMask;
use crate::rtstruct::StructureSet;
use crate::volume::{Grid, ViewPlane, Volume};

/// Grey levels of the palette, from index 0.
pub const GRAYS: usize = 192;
/// The structure colours' first index, and how many there are.
const STRUCT_BASE: usize = GRAYS;
const STRUCT_SLOTS: usize = 32;
/// The wash: four ramps (blue light, blue strong, red light, red strong) of
/// this many greys each.
const WASH_BASE: usize = STRUCT_BASE + STRUCT_SLOTS;
const WASH_GRAYS: usize = 7;
const FOLDED: usize = WASH_BASE + 4 * WASH_GRAYS;
/// How much of the wash's hue a light and a strong pixel take.
const WASH_MIX: [f32; 2] = [0.3, 0.6];
/// The wash's two hues, as [`apply::wash_color`] gives them.
const BLUE: [u8; 3] = [40, 120, 255];
const RED: [u8; 3] = [255, 70, 30];
/// Most structures outlined (the rest are left out).
const MAX_OUTLINED: usize = 64;

/// The film's width limits, pixels.
pub const MIN_WIDTH: usize = 64;
pub const MAX_WIDTH: usize = 4096;
/// Black between two cells, pixels.
const GAP: usize = 2;

/// Which panes the film shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilmViews {
    /// Axial, sagittal and coronal side by side, as the window has them.
    All,
    /// One plane alone.
    One(ViewPlane),
}

impl FilmViews {
    /// The planes, left to right.
    pub fn planes(self) -> Vec<ViewPlane> {
        match self {
            FilmViews::All => vec![ViewPlane::Axial, ViewPlane::Sagittal, ViewPlane::Coronal],
            FilmViews::One(p) => vec![p],
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
    /// Width over height of one cell, as the panes are on screen; `None`:
    /// as tall as the tallest plane needs at the cell's width.
    pub cell_aspect: Option<f32>,
    pub outlines: bool,
    pub wash: bool,
}

impl FilmParams {
    /// The three planes through the centre of `grid`, 21 frames at 6 a
    /// second, 800 pixels wide, outlines on.
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
            outlines: true,
            wash: false,
        }
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

/// One plane in its cell: where the image lies, in pixels of the film.
#[derive(Clone, Copy, Debug)]
struct Cell {
    plane: Plane,
    view: ViewPlane,
    /// The cell's left edge in the film.
    x0: usize,
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

/// The film's size and its cells.
fn layout(grid: &Grid, p: &FilmParams) -> ([usize; 2], Vec<Cell>) {
    let views = p.views.planes();
    let cells = views.len();
    let width = p.width.clamp(MIN_WIDTH, MAX_WIDTH);
    let cell_w = ((width - GAP * (cells - 1)) / cells).max(1);
    let planes: Vec<Plane> = views.iter().map(|v| plane_of(*v, p.slices, grid)).collect();
    let cell_h = match p.cell_aspect.filter(|a| a.is_finite() && *a > 0.05) {
        Some(a) => (cell_w as f32 / a).round() as usize,
        None => planes
            .iter()
            .map(|pl| {
                let [w, h] = extent_mm(pl, grid);
                (cell_w as f64 * h / w.max(1e-9)).round() as usize
            })
            .max()
            .unwrap_or(cell_w),
    }
    .clamp(16, MAX_WIDTH);
    // The pixels the division leaves over go to the first cells, one each,
    // so the film is exactly the width asked for.
    let spare = (width - GAP * (cells - 1)).saturating_sub(cell_w * cells);
    let mut x0 = 0;
    let out = views
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
    ([x0 - GAP, cell_h], out)
}

/// The size a film of `p` on `grid` comes out at, `[width, height]`.
pub fn film_size(grid: &Grid, p: &FilmParams) -> [usize; 2] {
    layout(grid, p).0
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
                .find(|(_, c)| x >= c.x0 && ((x - c.x0) as f64) < c.ox + c.dw + 1.0)
            else {
                continue;
            };
            let cx = (x - c.x0) as f64 + 0.5 - c.ox;
            let cy = y as f64 + 0.5 - c.oy;
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

/// The palette (768 bytes, RGB) with these structure colours.
fn palette(structure_colors: &[[u8; 3]]) -> Vec<u8> {
    let mut pal = vec![0u8; 256 * 3];
    let mut put = |i: usize, c: [u8; 3]| pal[i * 3..i * 3 + 3].copy_from_slice(&c);
    for i in 0..GRAYS {
        let g = (i as f32 * 255.0 / (GRAYS - 1) as f32).round() as u8;
        put(i, [g, g, g]);
    }
    for (i, c) in structure_colors.iter().take(STRUCT_SLOTS).enumerate() {
        put(STRUCT_BASE + i, *c);
    }
    for (h, hue) in [BLUE, RED].iter().enumerate() {
        for (s, mix) in WASH_MIX.iter().enumerate() {
            for g in 0..WASH_GRAYS {
                let grey = g as f32 * 255.0 / (WASH_GRAYS - 1) as f32;
                let c = hue.map(|x| (grey * (1.0 - mix) + x as f32 * mix).round() as u8);
                put(WASH_BASE + (h * 2 + s) * WASH_GRAYS + g, c);
            }
        }
    }
    put(FOLDED, [230, 0, 230]);
    pal
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
            let d = |s: &[u8; 3]| -> i32 {
                (0..3)
                    .map(|k| (s[k] as i32 - c[k] as i32).pow(2))
                    .sum::<i32>()
            };
            let near = (0..slots.len()).min_by_key(|&i| d(&slots[i])).unwrap_or(0);
            of.push(near);
        }
    }
    (slots, of)
}

/// The palette index of a wash colour over a pixel of windowed level
/// `level` (0..1); `None` where the wash is clear.
fn wash_index(c: [u8; 4], level: f32) -> Option<usize> {
    if c[3] == 0 {
        return None;
    }
    if c[0] == 230 && c[1] == 0 {
        return Some(FOLDED);
    }
    let hue = usize::from(c[0] == RED[0]);
    let strong = usize::from(c[3] >= 115);
    let g = (level * (WASH_GRAYS - 1) as f32).round() as usize;
    Some(WASH_BASE + (hue * 2 + strong) * WASH_GRAYS + g.min(WASH_GRAYS - 1))
}

/// The film of the path, written to `path` as an animated GIF.
///
/// `states` are the states' own images in the model's order; `slicer`
/// renders the planes on the GPU when given; `structures` is the first
/// state's structure set (on the model's grid's frame) whose structures
/// are outlined, all of them (filter it first for the visible ones). A
/// cancelled run leaves no file.
#[allow(clippy::too_many_arguments)]
pub fn write_gif(
    path: &Path,
    model: &PostureModel,
    states: &[Arc<Volume>],
    slicer: Option<&GpuSlicer>,
    structures: Option<&StructureSet>,
    params: &FilmParams,
    progress: &Progress,
) -> Result<FilmSummary> {
    let res = write_gif_inner(path, model, states, slicer, structures, params, progress);
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
    params: &FilmParams,
    progress: &Progress,
) -> Result<FilmSummary> {
    let grid = &model.grid;
    if grid.dims.contains(&0) {
        bail!("the model has no grid to film");
    }
    let (size, cells) = layout(grid, params);
    let [w, h] = size;
    let planes: Vec<Plane> = cells.iter().map(|c| c.plane).collect();
    progress.set("Laying out the film");
    let pixels = pixel_map(grid, size, &cells);

    // The structures, rasterized once on the first state's grid.
    let mut masks: Vec<BoxedMask> = Vec::new();
    let mut colors: Vec<[u8; 3]> = Vec::new();
    if params.outlines {
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
    let pal = palette(&slot_colors);

    let file = std::fs::File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut enc = gif::Encoder::new(std::io::BufWriter::new(file), w as u16, h as u16, &pal)
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
    let times = film_times(model.span(), params.frames, params.bounce);
    let n = times.len();
    let (mut gpu_all, mut why) = (slicer.is_some(), None);
    for (f, &t) in times.iter().enumerate() {
        if progress.cancelled() {
            bail!("film {}", progress::CANCELLED);
        }
        progress.report(f as f32 / n as f32, &format!("Frame {}/{n}", f + 1));
        let frame = model.frame(t);
        let (values, gpu, w_why) =
            render_planes_with(slicer, model, states, &frame, params.show, grid, &planes)?;
        gpu_all &= gpu;
        if why.is_none() {
            why = w_why;
        }
        let jac: Option<Vec<f64>> = params.wash.then(|| {
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
                        *o = (STRUCT_BASE + slot_of[r]) as u8;
                        continue;
                    }
                }
                if let Some(jac) = &jac {
                    if !scan || g >= apply::WASH_TISSUE_HU as f32 {
                        let lat = &frame.forward.lattice;
                        let j = apply::node_scalar(lat, jac, px.at) as f32;
                        if let Some(k) = wash_index(apply::wash_color(j), level) {
                            *o = k as u8;
                            continue;
                        }
                    }
                }
                *o = (level * (GRAYS - 1) as f32).round() as u8;
            }
        });
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
        let pal = palette(&slots);
        assert_eq!(pal.len(), 768);
        assert_eq!(&pal[0..3], &[0, 0, 0]);
        assert_eq!(&pal[(GRAYS - 1) * 3..GRAYS * 3], &[255, 255, 255]);
        assert_eq!(&pal[STRUCT_BASE * 3..STRUCT_BASE * 3 + 3], &[255, 0, 0]);
        assert_eq!(&pal[FOLDED * 3..FOLDED * 3 + 3], &[230, 0, 230]);
        // A blue over white is lighter than over black.
        let dark = wash_index([40, 120, 255, 200], 0.0).unwrap();
        let light = wash_index([40, 120, 255, 200], 1.0).unwrap();
        assert!(pal[light * 3] > pal[dark * 3]);
        assert_eq!(wash_index([0, 0, 0, 0], 0.5), None);
        assert_eq!(wash_index([230, 0, 230, 210], 0.5), Some(FOLDED));
        assert!(wash_index([255, 70, 30, 60], 0.5).unwrap() >= WASH_BASE + 2 * WASH_GRAYS);
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
    fn bilinear_reads_the_corners_and_the_middle() {
        let v = [0.0, 10.0, 20.0, 30.0];
        assert_eq!(bilinear(&v, [2, 2], 0.0, 0.0), 0.0);
        assert_eq!(bilinear(&v, [2, 2], 1.0, 1.0), 30.0);
        assert_eq!(bilinear(&v, [2, 2], 0.5, 0.5), 15.0);
        assert_eq!(bilinear(&[7.0], [1, 1], 0.0, 0.0), 7.0);
    }
}
