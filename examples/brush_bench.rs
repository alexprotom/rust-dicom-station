//! Timing of the contour brush on a real structure, without the interface:
//! what one stroke of the 3-D ball (and of the slice brush) costs per mouse
//! sample, per off-slice merge and on release, done the way
//! `src/app/contour_edit.rs` does it.
//!
//! ```text
//! cargo run --example brush_bench -- CT_DIR RS_FILE [STRUCTURE] [--radius 10,20,40]
//!     [--samples 60] [--erase] [--body]
//! ```
//!
//! The stroke follows the outline of the structure's largest ring on its
//! middle slice, a sample every 2 mm (a hand at 60 Hz), so the ball keeps
//! crossing the boundary, as a correction does. Before, every sixth sample
//! (100 ms at 60 Hz) merged every slice off the one drawn on; now each
//! sample merges a sixth of them.
//!
//! Printed per radius, for the merge over the whole slice from the base
//! (how the viewer merged before) and the windowed merge (what
//! it does now): the median and the worst sample, the median and the worst
//! off-slice merge, the release, and the writing of the stack back into the
//! ROI (which the viewer does on every sample of an axial stroke); then how
//! far the two results are apart, and the same for the slice brush.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use rust_dicom_station::contours::{self, BallFootprint, Poly, Stack};
use rust_dicom_station::loader;
use rust_dicom_station::progress::Progress;
use rust_dicom_station::structops::BoolOp;
use rust_dicom_station::volume::ViewPlane;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

fn stats(v: &mut [f64]) -> (f64, f64) {
    if v.is_empty() {
        return (0.0, 0.0);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[v.len() - 1])
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut pos = Vec::new();
    let mut radii = vec![10.0, 20.0, 40.0];
    let mut samples = 60usize;
    let mut erase = false;
    let mut body = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--radius" => {
                radii = it
                    .next()
                    .context("--radius needs a list")?
                    .split(',')
                    .map(|s| s.trim().parse::<f64>())
                    .collect::<Result<_, _>>()?
            }
            "--samples" => samples = it.next().context("--samples needs a number")?.parse()?,
            "--erase" => erase = true,
            "--body" => body = true,
            _ => pos.push(a.clone()),
        }
    }
    if pos.len() < 2 {
        bail!("usage: brush_bench CT_DIR RS_FILE [STRUCTURE] [--radius 10,20,40]");
    }
    let ct = PathBuf::from(&pos[0]);
    let mut files: Vec<PathBuf> = std::fs::read_dir(&ct)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    files.push(PathBuf::from(&pos[1]));
    let study = loader::load_files(&files, "bench", &Progress::default())?;
    let vol = study.volume.clone();
    let grid = vol.grid();
    println!(
        "volume {:?}, spacing {:.2?}, {} structure set(s)",
        vol.dims,
        vol.spacing,
        study.structure_sets.len()
    );
    let ss = study
        .structure_sets
        .first()
        .context("no structure set loaded")?;
    for r in &ss.rois {
        let n: usize = r.contours.iter().map(|c| c.points.len()).sum();
        println!(
            "  {:<24} {:>4} contours {:>7} points",
            r.name,
            r.contours.len(),
            n
        );
    }
    let want = pos.get(2).cloned();
    // A body outline of the CT's own (-300 HU, the largest piece of each
    // slice without its holes): the structure a department edits most,
    // which this data set does not carry.
    let body_roi = body.then(|| {
        let mask = rust_dicom_station::bodymask::foreground(
            &vol,
            rust_dicom_station::bodymask::Foreground::Hu(-300.0),
        );
        let mut st = Stack::from_mask(&mask, vol.dims, 2);
        for s in &mut st.slices {
            s.region.keep_largest();
            s.region.remove_holes();
        }
        st.prune();
        let mut r = ss.rois[0].clone();
        r.name = "BODY".into();
        st.apply_to_roi(&mut r, &grid);
        r
    });
    let roi = match &want {
        _ if body_roi.is_some() => body_roi.as_ref().unwrap(),
        Some(name) => ss
            .rois
            .iter()
            .find(|r| r.name.eq_ignore_ascii_case(name))
            .with_context(|| format!("no structure {name}"))?,
        None => ss
            .rois
            .iter()
            .max_by_key(|r| r.contours.iter().map(|c| c.points.len()).sum::<usize>())
            .context("no structures")?,
    };
    let t = Instant::now();
    let stack0 = Stack::from_roi(roi, &grid);
    println!(
        "\n'{}': stack along {} with {} slices, read in {:.1} ms",
        roi.name,
        stack0.axis,
        stack0.slices.len(),
        ms(t)
    );
    if stack0.axis != 2 {
        bail!("the bench strokes an axial stack");
    }
    let (l0, l1) = stack0.level_range().context("empty structure")?;
    let level = (l0 + l1) / 2;
    let region = stack0
        .region_at(level)
        .context("no contour on the middle slice")?;
    let ring = region
        .rings
        .iter()
        .max_by(|a, b| a.area().partial_cmp(&b.area()).unwrap())
        .unwrap()
        .clone();
    println!(
        "middle slice {level}: {} ring(s), {} points",
        region.rings.len(),
        region.rings.iter().map(|r| r.pts.len()).sum::<usize>()
    );
    // The path: along the ring, a sample every 2 mm.
    let sp = vol.spacing;
    let mut path: Vec<[f64; 3]> = Vec::new();
    let mut acc = 0.0;
    let n = ring.pts.len();
    path.push([ring.pts[0][0], ring.pts[0][1], level as f64]);
    for i in 0..n {
        let (a, b) = (ring.pts[i], ring.pts[(i + 1) % n]);
        let d = (((b[0] - a[0]) * sp[0]).powi(2) + ((b[1] - a[1]) * sp[1]).powi(2)).sqrt();
        acc += d;
        if acc >= 2.0 {
            acc = 0.0;
            path.push([b[0], b[1], level as f64]);
            if path.len() >= samples {
                break;
            }
        }
    }

    let t = Instant::now();
    let mut probe = roi.clone();
    stack0.apply_to_roi(&mut probe, &grid);
    println!("write back of the whole stack: {:.2} ms", ms(t));
    let t = Instant::now();
    let _ = rust_dicom_station::mesh3d::build_roi_mesh(0, roi);
    println!("3D mesh of the structure: {:.1} ms", ms(t));
    // A stroke on a sagittal view works on a sagittal stack, which every
    // write re-cuts onto the axial contours the structure is stored as.
    let sag = stack0.to_axis(vol.dims, 0);
    let t = Instant::now();
    let _ = sag.to_axis(vol.dims, 2);
    println!(
        "re-cutting a sagittal stack onto axial slices (an off-axis write): {:.1} ms",
        ms(t)
    );
    // What a write of a 20 mm ball's stroke re-cuts now: the 15 axial
    // slices it reached on a 3 mm lattice.
    let mid = (l0 + l1) / 2;
    let mut cut = roi.clone();
    let t = Instant::now();
    sag.apply_levels_to_roi(&mut cut, &grid, [mid - 7, mid + 7]);
    println!(
        "  the same for the 15 axial slices a 20 mm ball reaches: {:.1} ms",
        ms(t)
    );

    for &r in &radii {
        println!(
            "\n-- radius {r} mm, {} samples, {}",
            path.len(),
            if erase { "erase" } else { "paint" }
        );
        let mut results = Vec::new();
        for windowed in [false, true] {
            println!(
                "{}",
                if windowed {
                    "   windowed merge (now)"
                } else {
                    "   whole-slice merge from the base (before)"
                }
            );
            results.push(ball_stroke(
                &stack0, &grid, &path, level, r, erase, windowed, &mut probe,
            ));
        }
        let (a, b) = (&results[0], &results[1]);
        let differ = a
            .rasterize(vol.dims)
            .iter()
            .zip(b.rasterize(vol.dims))
            .filter(|(x, y)| **x != *y)
            .count();
        println!(
            "   the two results: {:.2} vs {:.2} cm3, {differ} voxels differ",
            a.volume_cm3(sp),
            b.volume_cm3(sp)
        );
        // What a view redraws after a write: the structure mapped into the
        // volume once, then cut by each of the three planes.
        let mut after = roi.clone();
        b.apply_to_roi(&mut after, &grid);
        let t = Instant::now();
        let prep = rust_dicom_station::render::PreparedRoi::new(&vol, &after);
        let t_prep = ms(t);
        let mid = vol.dims.map(|d| d / 2);
        let t = Instant::now();
        for (plane, slice) in [
            (ViewPlane::Axial, level),
            (ViewPlane::Sagittal, mid[0]),
            (ViewPlane::Coronal, mid[1]),
        ] {
            let _ = prep.on_plane(&vol, plane, slice);
        }
        println!(
            "   after a write: the views map the structure in {t_prep:.2} ms, cut it in {:.2} ms",
            ms(t)
        );

        // The slice brush, as contour_brush does it with 3D off: the
        // boolean on the whole slice (before) and in a window (now).
        for windowed in [false, true] {
            let mut stack = stack0.clone();
            let mut per = vec![];
            let mut prev = path[0];
            for &p in &path {
                let t = Instant::now();
                let ring =
                    Poly::capsule_mm([prev[0], prev[1]], [p[0], p[1]], r, [sp[0], sp[1]], 32);
                let region = stack.region_mut(level);
                let op = if erase {
                    BoolOp::Subtract
                } else {
                    BoolOp::Union
                };
                let stamp = contours::Region::from_ring(ring);
                *region = if windowed {
                    region.boolean_local(&stamp, op)
                } else {
                    region.boolean(&stamp, op)
                };
                stack.prune();
                stack.apply_to_roi(&mut probe, &grid);
                per.push(ms(t));
                prev = p;
            }
            let (a, b) = stats(&mut per);
            println!(
                "2D brush sample   median {a:7.2} ms  worst {b:7.2} ms  ({})",
                if windowed { "now" } else { "before" }
            );
        }
    }
    Ok(())
}

/// One 3-D stroke along `path`, as `BallStroke` merges it: the drawing
/// slice every sample, every slice reached every sixth sample (100 ms at
/// 60 Hz), the rest on release. Prints the timings, returns the stack.
#[allow(clippy::too_many_arguments)]
fn ball_stroke(
    stack0: &Stack,
    grid: &rust_dicom_station::volume::Grid,
    path: &[[f64; 3]],
    level: usize,
    r: f64,
    erase: bool,
    windowed: bool,
    probe: &mut rust_dicom_station::rtstruct::Roi,
) -> Stack {
    let mut stack = stack0.clone();
    let mut foot = BallFootprint::new(grid, 2, r);
    let mut base: BTreeMap<usize, contours::Region> = BTreeMap::new();
    let merge = |stack: &mut Stack,
                 foot: &BallFootprint,
                 base: &mut BTreeMap<usize, contours::Region>,
                 levels: &[(usize, [usize; 4])]| {
        for &(l, _) in levels {
            base.entry(l)
                .or_insert_with(|| stack.region_at(l).cloned().unwrap_or_default());
        }
        let empty = contours::Region::new();
        let st = &*stack;
        let one = |&(l, b): &(usize, [usize; 4])| {
            let r = if windowed {
                foot.merge(l, b, &base[&l], st.region_at(l).unwrap_or(&empty), erase)
            } else {
                foot.merged(l, &base[&l], erase)
            };
            (l, r)
        };
        // The viewer merged slice after slice before, side by side now.
        let out: Vec<(usize, contours::Region)> = if windowed {
            levels.par_iter().map(one).collect()
        } else {
            levels.iter().map(one).collect()
        };
        for (l, r) in out {
            *stack.region_mut(l) = r;
        }
        stack.prune();
    };
    let (mut sweep_t, mut draw_t, mut off_t, mut write_t, mut total_t) =
        (vec![], vec![], vec![], vec![], vec![]);
    let mut off_levels = 0;
    let mut prev = path[0];
    for (s, &p) in path.iter().enumerate() {
        let t0 = Instant::now();
        let t = Instant::now();
        let changed = foot.sweep(prev, p, r, |_| true);
        sweep_t.push(ms(t));
        prev = p;
        // Before: every slice every sixth sample (100 ms at 60 Hz); now: a
        // sixth of them on every sample, as `contour_ball` does.
        let levels: Vec<(usize, [usize; 4])> = if !windowed {
            if s % 6 == 5 {
                foot.take_dirty_boxes()
            } else {
                foot.take_dirty_box_at(level)
                    .filter(|_| changed)
                    .map(|b| vec![(level, b)])
                    .unwrap_or_default()
            }
        } else {
            let mut due: Vec<usize> = foot
                .levels()
                .into_iter()
                .filter(|&l| l != level && l % 6 == (s + 1) % 6)
                .collect();
            if changed {
                due.push(level);
            }
            due.into_iter()
                .filter_map(|l| foot.take_dirty_box_at(l).map(|b| (l, b)))
                .collect()
        };
        if !levels.is_empty() {
            let t = Instant::now();
            off_levels = off_levels.max(levels.len());
            merge(&mut stack, &foot, &mut base, &levels);
            if levels.len() > 1 {
                off_t.push(ms(t));
            } else {
                draw_t.push(ms(t));
            }
        }
        let t = Instant::now();
        stack.apply_to_roi(probe, grid);
        write_t.push(ms(t));
        total_t.push(ms(t0));
    }
    let t = Instant::now();
    let levels = foot.take_dirty_boxes();
    merge(&mut stack, &foot, &mut base, &levels);
    let release = ms(t);
    let (a, b) = stats(&mut sweep_t);
    println!("ball sweep        median {a:7.2} ms  worst {b:7.2} ms");
    let (a, b) = stats(&mut draw_t);
    println!("merge, one slice  median {a:7.2} ms  worst {b:7.2} ms");
    let (a, b) = stats(&mut off_t);
    println!("merge, several    median {a:7.2} ms  worst {b:7.2} ms  ({off_levels} slices)");
    let (a, b) = stats(&mut write_t);
    println!("write back        median {a:7.2} ms  worst {b:7.2} ms");
    let (a, b) = stats(&mut total_t);
    println!("per sample, all   median {a:7.2} ms  worst {b:7.2} ms");
    println!(
        "release           {release:7.2} ms ({} slices)",
        levels.len()
    );
    let pts: usize = stack
        .slices
        .iter()
        .flat_map(|s| &s.region.rings)
        .map(|r| r.pts.len())
        .sum();
    let pts_mid: usize = stack
        .region_at(level)
        .map(|r| r.rings.iter().map(|r| r.pts.len()).sum())
        .unwrap_or(0);
    println!(
        "result: {} points ({} on the drawing slice), volume {:.2} cm3 (was {:.2})",
        pts,
        pts_mid,
        stack.volume_cm3(grid.spacing),
        stack0.volume_cm3(grid.spacing)
    );
    stack
}
