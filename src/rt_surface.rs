//! The volume 3D Slicer reports for an RT structure.
//!
//! Slicer does not measure a structure set ROI by its contours. It builds a
//! closed surface from them - SlicerRT's planar-contour-to-closed-surface
//! conversion - and *Segment Statistics* reports the volume inside that
//! surface (`vtkMassProperties`). The surface is a reconstruction with rules
//! of its own, and on small or voxel-exported structures the number it
//! gives differs from any planimetry by tens of per cent. So rather than
//! approximate it, this module runs the same algorithm:
//!
//! 1. **Reading** (`vtkSlicerDicomRtReader::LoadContour`): every contour
//!    becomes one line, DICOM LPS turned to RAS, the first point repeated at
//!    the end, coordinates stored in single precision as VTK stores them.
//! 2. **Orientation** (`CalculateContourTransform`): the contours are turned
//!    so their planes are axial. For axial contours that is the identity.
//! 3. **Order, keyholes, winding** (`SortContours`, `FixKeyholes`,
//!    `SetLinesCounterClockwise`): lines sorted by height; a line that comes
//!    back to one of its own points is cut into separate lines by SlicerRT's
//!    keyhole rule; every line made counter-clockwise.
//! 4. **Between slices** (`TriangulateBetweenContours`, `Branch`): each line
//!    is joined by a ribbon of triangles (a dynamic-programming match of the
//!    two point sequences) to every line on the next plane whose bounding
//!    box overlaps it; a line overlapping several is split among them.
//! 5. **End caps** (`EndCapping`, `CreateSmoothEndCapContour`): a line with
//!    nothing joined above or below is closed by a cap half a slice away -
//!    by default a "smooth" cap, the contour rasterised, eroded until at most
//!    half of it is left, traced with marching squares, decimated, and
//!    joined to the line with another ribbon.
//! 6. **Volume** (`vtkMassProperties`): the discrete divergence theorem, with
//!    its per-axis weighting, on the triangles.
//!
//! The VTK pieces the conversion leans on - the polydata stencil and its
//! raster, the ellipsoidal erosion kernel, marching squares with merged
//! points, the stripper, the priority queue behind the decimation - are
//! ported with them, quirks included, because each of them decides which
//! points a cap is made of. The one liberty taken is the triangulation of a
//! cap's flat interior, which is done by ear clipping rather than VTK's own
//! ear cut: any triangulation of a flat polygon contributes the same to the
//! volume.
//!
//! Verified against Slicer 5.10 (VTK 9.5.2, SlicerRT master) on four
//! structures of two test patients - two targets exported voxel by voxel,
//! 1031 and 2178 contours, and two hearts of 218 000 and 53 000 points -
//! where it gives the volume Segment Statistics shows to every digit it
//! shows. The tests below hold the behaviour on shapes whose volume is
//! known; they cannot carry the patient contours.
//!
//! Joining the slices is the costly step - a dynamic-programming match of
//! every pair of joined contours, quadratic in their points - so the pairs
//! of neighbouring planes are joined in parallel. A heart drawn as voxel
//! outlines is then a fraction of a second.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::rtstruct::Roi;

type P3 = [f64; 3];

/// What joining one pair of neighbouring planes makes: the triangles, and
/// the lines joined upwards and downwards.
type Joined = (Vec<[usize; 3]>, Vec<usize>, Vec<usize>);

/// SlicerRT's "End capping" conversion parameter.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum EndCapping {
    /// Leave the ends open (0).
    None,
    /// A shrunken copy of the contour half a slice away (1, the default).
    #[default]
    Smooth,
    /// A copy of the contour itself half a slice away (2).
    Straight,
}

/// The volume Slicer's *Segment Statistics* gives for `roi`, cm³, with its
/// default conversion (smooth end caps). `slice_mm` is the slice spacing of
/// the image the structure set refers to, which Slicer's DICOM import hands
/// the conversion as its default slice thickness - it only matters for a
/// structure drawn on a single plane. `None` when the ROI has no line to
/// build a surface from.
pub fn slicer_volume_cm3(roi: &Roi, slice_mm: f64) -> Option<f64> {
    surface_volume_mm3(roi, EndCapping::Smooth, stream_rounded(slice_mm)).map(|v| v / 1000.0)
}

/// The volume inside the surface SlicerRT builds from `roi`, mm³, with
/// `default_thickness` the spacing assumed when the contours give none.
pub fn surface_volume_mm3(roi: &Roi, capping: EndCapping, default_thickness: f64) -> Option<f64> {
    let mut m = Mesh::read(roi, default_thickness)?;
    m.orient_axial();
    m.sort_contours();
    m.fix_keyholes(0.001, 3);
    m.set_lines_counter_clockwise();
    let tris = m.build(capping);
    mass_properties_volume(&m.pts, &tris)
}

/// A double written to a `std::stringstream` and read back, as the import
/// passes the slice thickness to the conversion: six significant digits.
fn stream_rounded(v: f64) -> f64 {
    if !v.is_finite() || v == 0.0 {
        return 0.0;
    }
    format!("{:.5e}", v).parse().unwrap_or(v)
}

/// Single precision, as a `vtkPoints` of the default type stores it.
#[inline]
fn f32r(v: f64) -> f64 {
    v as f32 as f64
}

#[inline]
fn d2(a: &P3, b: &P3) -> f64 {
    let (x, y, z) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    x * x + y * y + z * z
}

/// The points and lines of one structure, the way the conversion holds them
/// in its `vtkPolyData`: one point array that only grows, lines as id lists.
struct Mesh {
    pts: Vec<P3>,
    lines: Vec<Vec<usize>>,
    /// The "Default slice thickness" conversion parameter: the spacing
    /// used when the lines themselves give none (a structure on one plane).
    default_thickness: f64,
}

impl Mesh {
    fn add(&mut self, p: P3) -> usize {
        self.pts.push([f32r(p[0]), f32r(p[1]), f32r(p[2])]);
        self.pts.len() - 1
    }

    /// `vtkSlicerDicomRtReader::LoadContour`.
    fn read(roi: &Roi, default_thickness: f64) -> Option<Mesh> {
        let mut m = Mesh {
            pts: Vec::new(),
            lines: Vec::new(),
            default_thickness,
        };
        for c in &roi.contours {
            if c.points.is_empty() {
                continue;
            }
            let first = m.pts.len();
            let mut ids = Vec::with_capacity(c.points.len() + 1);
            for p in &c.points {
                ids.push(m.add([-p.x, -p.y, p.z]));
            }
            ids.push(first);
            m.lines.push(ids);
        }
        // A single point is a point ROI, not contours.
        (m.pts.len() > 1 && !m.lines.is_empty()).then_some(m)
    }

    fn bounds(&self, line: &[usize]) -> [f64; 6] {
        let mut b = [f64::MAX, f64::MIN, f64::MAX, f64::MIN, f64::MAX, f64::MIN];
        for &i in line {
            let p = self.pts[i];
            for a in 0..3 {
                b[2 * a] = b[2 * a].min(p[a]);
                b[2 * a + 1] = b[2 * a + 1].max(p[a]);
            }
        }
        b
    }

    fn zmid(&self, line: &[usize]) -> f64 {
        let b = self.bounds(line);
        (b[4] + b[5]) / 2.0
    }

    // -- 2. orientation ----------------------------------------------------

    /// `CalculateContourTransform`: the average plane normal of the
    /// contours (those with more than six points first), and the rotation
    /// that takes it onto +z. Axial contours come out exactly on the axis
    /// and the rotation is the identity, as in Slicer.
    fn orient_axial(&mut self) {
        let mut n = self.contour_normal(6);
        if n == [0.0; 3] {
            n = self.contour_normal(0);
        }
        let dot = n[2].clamp(-1.0, 1.0);
        let theta = dot.acos();
        let axis = [n[1], -n[0], 0.0]; // n × z
        if theta == 0.0 || axis == [0.0; 3] {
            return;
        }
        // vtkTransform::RotateWXYZ: a unit quaternion about the axis.
        let len = (axis[0] * axis[0] + axis[1] * axis[1]).sqrt();
        let (s, w) = ((0.5 * theta).sin(), (0.5 * theta).cos());
        let (x, y, z) = (s * axis[0] / len, s * axis[1] / len, 0.0);
        let r = [
            [
                w * w + x * x - y * y - z * z,
                2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
            ],
            [
                2.0 * (x * y + w * z),
                w * w - x * x + y * y - z * z,
                2.0 * (y * z - w * x),
            ],
            [
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                w * w - x * x - y * y + z * z,
            ],
        ];
        for p in &mut self.pts {
            let q = *p;
            for (a, row) in r.iter().enumerate() {
                p[a] = f32r(row[0] * q[0] + row[1] * q[1] + row[2] * q[2]);
            }
        }
    }

    /// `CalculateContourNormal`.
    fn contour_normal(&self, min_points: usize) -> P3 {
        let mut sum = [0.0; 3];
        let mut small = 0usize;
        for line in &self.lines {
            let mut unique: Vec<usize> = line.clone();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() > min_points && unique.len() >= 3 {
                let mut n = plane_normal(unique.iter().map(|&i| self.pts[i]));
                if n[0] * sum[0] + n[1] * sum[1] + n[2] * sum[2] < 0.0 {
                    n = [-n[0], -n[1], -n[2]];
                }
                for a in 0..3 {
                    sum[a] += n[a];
                }
            } else {
                small += 1;
            }
        }
        let norm = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
        if small >= self.lines.len() || norm == 0.0 {
            return [0.0; 3];
        }
        [sum[0] / norm, sum[1] / norm, sum[2] / norm]
    }

    // -- 3. order, keyholes, winding ------------------------------------------

    /// `SortContours`: by the middle of each line's z range, then by index.
    fn sort_contours(&mut self) {
        let mut keyed: Vec<(f64, usize)> = self
            .lines
            .iter()
            .enumerate()
            .map(|(i, l)| (self.zmid(l), i))
            .collect();
        keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let old = std::mem::take(&mut self.lines);
        self.lines = keyed.into_iter().map(|(_, i)| old[i].clone()).collect();
    }

    /// `FixKeyholes(epsilon, minimumSeparation)`, as written - including
    /// what it does to a path that is not a keyhole but touches itself.
    fn fix_keyholes(&mut self, eps: f64, min_sep: i64) {
        let old = std::mem::take(&mut self.lines);
        let mut out = Vec::with_capacity(old.len());
        for line in old {
            let n = line.len();
            let coords: Vec<P3> = line.iter().map(|&i| self.pts[i]).collect();
            let mut flags = vec![-1i64; n];
            let mut keyhole = false;
            for (p1, near) in within_radius(&coords, eps).into_iter().enumerate() {
                for p2 in near {
                    let (a, b) = (p1 as i64, p2 as i64);
                    let sep = (b - a).min(n as i64 - 1 - b + a);
                    if sep > min_sep {
                        keyhole = true;
                        flags[p1] = b;
                        flags[p2] = a;
                    }
                }
            }
            if !keyhole {
                if line.len() > 1 {
                    out.push(line);
                }
                continue;
            }
            let mut lists: Vec<Vec<usize>> = Vec::new();
            let mut raw: Vec<usize> = Vec::new();
            let mut finished: Vec<usize> = Vec::new();
            let mut layer = 0usize;
            let mut in_channel = false;
            for (i, &pid) in line.iter().enumerate() {
                if layer == raw.len() {
                    lists.push(Vec::new());
                    raw.push(lists.len() - 1);
                }
                let f = flags[i];
                let i = i as i64;
                if f == -1 {
                    lists[raw[layer]].push(pid);
                    in_channel = false;
                } else if f > i && !in_channel {
                    lists[raw[layer]].push(pid);
                    layer += 1;
                    in_channel = true;
                } else if f < i && !in_channel {
                    lists[raw[layer]].push(pid);
                    finished.push(raw[layer]);
                    raw.pop();
                    layer = layer.saturating_sub(1);
                    in_channel = true;
                }
            }
            finished.extend(raw);
            for f in finished {
                let l = &mut lists[f];
                if !l.is_empty() && l[0] != l[l.len() - 1] {
                    let first = l[0];
                    l.push(first);
                }
            }
            out.extend(lists.into_iter().filter(|l| l.len() > 1));
        }
        self.lines = out;
    }

    fn is_clockwise(pts: &[P3], line: &[usize]) -> bool {
        let mut sum = 0.0;
        for w in line.windows(2) {
            let (p1, p2) = (pts[w[0]], pts[w[1]]);
            sum += (p2[0] - p1[0]) * (p2[1] + p1[1]);
        }
        sum > 0.0
    }

    fn set_lines_counter_clockwise(&mut self) {
        for l in &mut self.lines {
            if Self::is_clockwise(&self.pts, l) {
                l.reverse();
            }
        }
    }

    /// `GetSpacingBetweenLines`: the mean distance between consecutive line
    /// heights, those more than a tenth off the mean left out.
    fn spacing_between_lines(&self) -> f64 {
        if self.lines.len() < 2 {
            return self.default_thickness;
        }
        let mut dist = Vec::new();
        for w in self.lines.windows(2) {
            let d = (self.zmid(&w[0]) - self.zmid(&w[1])).abs();
            if d > 0.01 {
                dist.push(d);
            }
        }
        if dist.is_empty() {
            return self.default_thickness;
        }
        let mean = dist.iter().sum::<f64>() / dist.len() as f64;
        let kept: Vec<f64> = dist
            .iter()
            .copied()
            .filter(|d| (d - mean).abs() < mean / 10.0)
            .collect();
        if kept.is_empty() {
            return mean;
        }
        kept.iter().sum::<f64>() / kept.len() as f64
    }

    fn lines_on_plane(&self, first: usize, spacing: f64) -> usize {
        let threshold = 0.1 * spacing;
        let z = self.zmid(&self.lines[first]);
        let mut cur = first + 1;
        while cur < self.lines.len() && (self.zmid(&self.lines[cur]) - z).abs() < threshold {
            cur += 1;
        }
        cur - first
    }

    // -- 4. between slices ---------------------------------------------------

    fn build(&mut self, capping: EndCapping) -> Vec<[usize; 3]> {
        let mut tris: Vec<[usize; 3]> = Vec::new();
        let nlines = self.lines.len();
        if nlines == 0 {
            return tris;
        }
        let spacing = self.spacing_between_lines();
        let bounds: Vec<[f64; 6]> = self.lines.iter().map(|l| self.bounds(l)).collect();
        let mut above = vec![false; nlines];
        let mut below = vec![false; nlines];

        // The planes, as runs of lines: `lines_on_plane` walks them the
        // way the conversion's main loop does.
        let mut planes: Vec<(usize, usize)> = Vec::new();
        let mut first = 0usize;
        while first < nlines {
            let n = self.lines_on_plane(first, spacing);
            planes.push((first, n));
            first += n;
        }
        // Each pair of neighbouring planes is joined on its own - nothing one
        // pair does is read by another - so the pairs run in parallel, and
        // their triangles are laid end to end in the order the conversion
        // would have made them, which keeps the volume's sum in its order.
        let joined: Vec<Joined> = planes
            .par_windows(2)
            .map(|w| {
                let ((first1, n1), (first2, n2)) = (w[0], w[1]);
                let mut over1: Vec<Vec<usize>> = vec![Vec::new(); n1];
                let mut over2: Vec<Vec<usize>> = vec![Vec::new(); n2];
                for a in 0..n1 {
                    for b in 0..n2 {
                        if overlap(&bounds[first1 + a], &bounds[first2 + b]) {
                            over1[a].push(first2 + b);
                            over2[b].push(first1 + a);
                        }
                    }
                }
                let mut tris = Vec::new();
                let (mut up, mut down) = (Vec::new(), Vec::new());
                for l1 in first1..first1 + n1 {
                    for &l2 in &over1[l1 - first1] {
                        let div1 =
                            self.branch(&self.lines[l1], l2, &over1[l1 - first1], &self.lines);
                        let div2 =
                            self.branch(&self.lines[l2], l1, &over2[l2 - first2], &self.lines);
                        if div1.len() > 1 && div2.len() > 1 {
                            up.push(l1);
                            down.push(l2);
                            triangulate_between(&self.pts, &div1, &div2, &mut tris);
                        }
                    }
                }
                (tris, up, down)
            })
            .collect();
        for (t, up, down) in joined {
            tris.extend(t);
            for l in up {
                above[l] = true;
            }
            for l in down {
                below[l] = true;
            }
        }

        if capping != EndCapping::None {
            self.end_capping(capping, &above, &below, &mut tris);
        }
        tris
    }

    /// `Branch`: the part of `line` whose points are nearer to `current`
    /// than to any other of the `overlapping` lines, one point either side
    /// kept to close up the surface.
    fn branch(
        &self,
        line: &[usize],
        current: usize,
        overlapping: &[usize],
        all: &[Vec<usize>],
    ) -> Vec<usize> {
        if overlapping.len() == 1 {
            return line.to_vec();
        }
        let grids: Vec<Nearest> = overlapping
            .iter()
            .map(|&o| Nearest::new(&self.pts, &all[o]))
            .collect();
        let mut out = Vec::new();
        let mut prev = false;
        for &pid in line {
            let p = self.pts[pid];
            if closest_branch(&p, overlapping, &grids) == current {
                out.push(pid);
                prev = true;
            } else {
                if prev {
                    out.push(pid);
                }
                prev = false;
            }
        }
        if out.len() > 1 {
            let closed = line[0] == line[line.len() - 1];
            if closed && out[0] != out[out.len() - 1] {
                out.push(out[0]);
            }
        }
        out
    }

    // -- 5. end caps ---------------------------------------------------------

    fn end_capping(
        &mut self,
        capping: EndCapping,
        above: &[bool],
        below: &[bool],
        tris: &mut Vec<[usize; 3]>,
    ) {
        let nlines = self.lines.len();
        let spacing = self.spacing_between_lines();
        for li in 0..nlines {
            let line = self.lines[li].clone();
            for dir_above in [false, true] {
                let joined = if dir_above { above[li] } else { below[li] };
                if joined {
                    continue;
                }
                let offset = if dir_above { spacing } else { -spacing };
                let caps = match capping {
                    EndCapping::Smooth => self.smooth_cap(&line, offset),
                    _ => self.straight_cap(&line, offset),
                };
                for cap in &caps {
                    self.lines.push(cap.clone());
                    triangulate_interior(&self.pts, cap, dir_above, tris);
                }
                let idx: Vec<usize> = (0..caps.len()).collect();
                for k in 0..caps.len() {
                    let divided = self.branch(&line, k, &idx, &caps);
                    if dir_above {
                        triangulate_between(&self.pts, &divided, &caps[k], tris);
                    } else {
                        triangulate_between(&self.pts, &caps[k], &divided, tris);
                    }
                }
            }
        }
    }

    /// `CreateStraightEndCapContour`: every point of the line, half a slice
    /// away, as new points (the repeated closing point too).
    fn straight_cap(&mut self, line: &[usize], offset: f64) -> Vec<Vec<usize>> {
        let ids = line
            .iter()
            .map(|&i| {
                let p = self.pts[i];
                self.add([p[0], p[1], p[2] + offset / 2.0])
            })
            .collect();
        vec![ids]
    }

    /// `CreateSmoothEndCapContour`.
    fn smooth_cap(&mut self, line: &[usize], offset: f64) -> Vec<Vec<usize>> {
        let coords: Vec<P3> = line.iter().map(|&i| self.pts[i]).collect();
        let b = self.bounds(line);
        let alt = [(b[1] - b[0]) / 28.0, (b[3] - b[2]) / 28.0];
        let mut sp = [1.0f64, 1.0, 1.0];
        if alt[0] > 0.0 && alt[1] > 0.0 {
            sp[0] = 1.0f64.min(alt[0]);
            sp[1] = 1.0f64.min(alt[1]);
        }
        let bx0 = b[0] - 2.0 * sp[0];
        let bx1 = b[1] + 2.0 * sp[0];
        let by0 = b[2] - 2.0 * sp[1];
        let by1 = b[3] + 2.0 * sp[1];
        let nx = ((bx1 - bx0) / sp[0]).ceil() as i64;
        let ny = ((by1 - by0) / sp[1]).ceil() as i64;
        let origin = [bx0, by0, b[4]];

        let mut img = if nx > 0 && ny > 0 {
            stencil(&coords, origin, sp, nx as usize, ny as usize)
        } else {
            Vec::new()
        };
        let (nx, ny) = (nx.max(0) as usize, ny.max(0) as usize);

        let total = img.iter().filter(|&&v| v != 0).count() as i64;
        let mut count = total;
        let mut diff = i64::from(i32::MAX);
        while count > total / 2 && diff > 0 {
            img = erode(&img, nx, ny);
            let now = img.iter().filter(|&&v| v != 0).count() as i64;
            diff = count - now;
            count -= diff;
        }

        let (ms_pts, segs) = marching_squares(&img, nx, ny, origin, sp);
        let strips = strip(&segs, ms_pts.len());
        let fixed = fix_lines(&strips);
        let factor = (fixed.len() as f64 * line.len() as f64 + 1.0) / ms_pts.len().max(1) as f64;
        let decimated = decimate(&ms_pts, &fixed, factor);

        let mut out = Vec::new();
        if !decimated.is_empty() && !ms_pts.is_empty() {
            for l in decimated {
                let l: Vec<usize> = if Self::is_clockwise(&ms_pts, &l) {
                    l.into_iter().rev().collect()
                } else {
                    l
                };
                let mut ids: Vec<usize> = l
                    .iter()
                    .map(|&i| {
                        let p = ms_pts[i];
                        self.add([p[0], p[1], p[2] + offset / 2.0])
                    })
                    .collect();
                if ids[0] != ids[ids.len() - 1] {
                    ids.push(ids[0]);
                }
                out.push(ids);
            }
        } else {
            // Nothing survived: the contour itself, half a slice away.
            let mut ids: Vec<usize> = line[..line.len().saturating_sub(1)]
                .iter()
                .map(|&i| {
                    let p = self.pts[i];
                    self.add([p[0], p[1], p[2] + offset / 2.0])
                })
                .collect();
            if let Some(&f) = ids.first() {
                ids.push(f);
                out.push(ids);
            }
        }
        out
    }
}

/// The unit normal of the best plane through `pts`: the eigenvector of
/// their covariance with the smallest eigenvalue (what
/// `vtkAddonMathUtilities::FitPlaneToPoints` takes from an SVD).
fn plane_normal(pts: impl Iterator<Item = P3> + Clone) -> P3 {
    let n = pts.clone().count() as f64;
    let mut c = [0.0; 3];
    for p in pts.clone() {
        for a in 0..3 {
            c[a] += p[a];
        }
    }
    for v in &mut c {
        *v /= n;
    }
    let mut m = [[0.0f64; 3]; 3];
    for p in pts {
        let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += d[i] * d[j];
            }
        }
    }
    let (vals, vecs) = jacobi3(m);
    let k = (0..3)
        .min_by(|&a, &b| vals[a].total_cmp(&vals[b]))
        .unwrap_or(2);
    [vecs[0][k], vecs[1][k], vecs[2][k]]
}

/// Eigen-decomposition of a symmetric 3 × 3 matrix by Jacobi rotations;
/// eigenvectors are the columns. A zero row and column stays exactly zero,
/// so a flat axial contour has exactly the z axis as its normal.
fn jacobi3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-30 {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q] == 0.0 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for row in &mut a {
                let (akp, akq) = (row[p], row[q]);
                row[p] = c * akp - s * akq;
                row[q] = s * akp + c * akq;
            }
            let (rp, rq) = (a[p], a[q]);
            a[p] = std::array::from_fn(|k| c * rp[k] - s * rq[k]);
            a[q] = std::array::from_fn(|k| s * rp[k] + c * rq[k]);
            for row in &mut v {
                let (vkp, vkq) = (row[p], row[q]);
                row[p] = c * vkp - s * vkq;
                row[q] = s * vkp + c * vkq;
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

/// Which points of `coords` lie within `eps` of each, ascending - what
/// `vtkPointLocator::FindPointsWithinRadius` returns for points this close
/// (they share a bucket, and a bucket keeps insertion order).
fn within_radius(coords: &[P3], eps: f64) -> Vec<Vec<usize>> {
    let key = |p: &P3| {
        (
            (p[0] / eps).floor() as i64,
            (p[1] / eps).floor() as i64,
            (p[2] / eps).floor() as i64,
        )
    };
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, p) in coords.iter().enumerate() {
        grid.entry(key(p)).or_default().push(i);
    }
    let r2 = eps * eps;
    coords
        .iter()
        .map(|p| {
            let (kx, ky, kz) = key(p);
            let mut near = Vec::new();
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(ids) = grid.get(&(kx + dx, ky + dy, kz + dz)) {
                            near.extend(ids.iter().copied().filter(|&j| d2(&coords[j], p) <= r2));
                        }
                    }
                }
            }
            near.sort_unstable();
            near
        })
        .collect()
}

/// `GetClosestBranch`: the overlapping line with the nearest point, the
/// first of them on a tie.
fn closest_branch(p: &P3, overlapping: &[usize], grids: &[Nearest]) -> usize {
    let mut best = f64::MAX;
    let mut id = overlapping[0];
    for (o, g) in overlapping.iter().zip(grids) {
        let d = g.query(p).1;
        if d < best {
            best = d;
            id = *o;
        }
    }
    id
}

/// `DoLinesOverlap`: strict overlap of the xy bounding boxes.
fn overlap(b1: &[f64; 6], b2: &[f64; 6]) -> bool {
    b1[0] < b2[1] && b1[1] > b2[0] && b1[2] < b2[3] && b1[3] > b2[2]
}

fn next_loc(cur: usize, n: usize, closed: bool) -> usize {
    if cur + 1 == n {
        if closed {
            1
        } else {
            0
        }
    } else {
        cur + 1
    }
}

fn prev_loc(cur: usize, n: usize, closed: bool) -> usize {
    if cur == 0 {
        if closed {
            n.saturating_sub(2)
        } else {
            n - 1
        }
    } else {
        cur - 1
    }
}

/// The nearest point of a line to a query, found through a grid over the
/// line's points but chosen exactly as a scan in line order would choose
/// it: the smallest squared distance, the first such point on a tie.
struct Nearest<'a> {
    pts: &'a [P3],
    line: &'a [usize],
    x0: f64,
    y0: f64,
    cell: f64,
    nx: i64,
    ny: i64,
    buckets: Vec<Vec<usize>>,
}

impl<'a> Nearest<'a> {
    fn new(pts: &'a [P3], line: &'a [usize]) -> Nearest<'a> {
        let (mut x0, mut x1, mut y0, mut y1) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
        for &i in line {
            x0 = x0.min(pts[i][0]);
            x1 = x1.max(pts[i][0]);
            y0 = y0.min(pts[i][1]);
            y1 = y1.max(pts[i][1]);
        }
        let n = line.len().max(1) as f64;
        let extent = (x1 - x0).max(y1 - y0);
        let cell = if extent > 0.0 { extent / n.sqrt() } else { 1.0 };
        let nx = (((x1 - x0) / cell).floor() as i64 + 1).max(1);
        let ny = (((y1 - y0) / cell).floor() as i64 + 1).max(1);
        let mut buckets = vec![Vec::new(); (nx * ny) as usize];
        for (k, &i) in line.iter().enumerate() {
            let cx = (((pts[i][0] - x0) / cell).floor() as i64).clamp(0, nx - 1);
            let cy = (((pts[i][1] - y0) / cell).floor() as i64).clamp(0, ny - 1);
            buckets[(cy * nx + cx) as usize].push(k);
        }
        Nearest {
            pts,
            line,
            x0,
            y0,
            cell,
            nx,
            ny,
            buckets,
        }
    }

    /// (position in the line, squared distance).
    fn query(&self, p: &P3) -> (usize, f64) {
        let cx = ((p[0] - self.x0) / self.cell).floor() as i64;
        let cy = ((p[1] - self.y0) / self.cell).floor() as i64;
        // Rings nearer than this hold no cell of the grid.
        let r0 = [-cx, cx - (self.nx - 1), -cy, cy - (self.ny - 1), 0]
            .into_iter()
            .max()
            .unwrap_or(0);
        let reach = [cx, self.nx - 1 - cx, cy, self.ny - 1 - cy]
            .iter()
            .map(|v| v.abs())
            .max()
            .unwrap_or(0)
            + 1;
        let mut best = (f64::INFINITY, usize::MAX);
        let visit = |gx: i64, gy: i64, best: &mut (f64, usize)| {
            for &k in &self.buckets[(gy * self.nx + gx) as usize] {
                let d = d2(p, &self.pts[self.line[k]]);
                if d < best.0 || (d == best.0 && k < best.1) {
                    *best = (d, k);
                }
            }
        };
        let (xa, xb) = (0, self.nx - 1);
        let (ya, yb) = (0, self.ny - 1);
        for r in r0..=reach.max(r0) {
            if r == 0 {
                if (xa..=xb).contains(&cx) && (ya..=yb).contains(&cy) {
                    visit(cx, cy, &mut best);
                }
            } else {
                let (gx0, gx1) = ((cx - r).max(xa), (cx + r).min(xb));
                for gy in [cy - r, cy + r] {
                    if (ya..=yb).contains(&gy) {
                        for gx in gx0..=gx1 {
                            visit(gx, gy, &mut best);
                        }
                    }
                }
                let (gy0, gy1) = ((cy - r + 1).max(ya), (cy + r - 1).min(yb));
                for gx in [cx - r, cx + r] {
                    if (xa..=xb).contains(&gx) {
                        for gy in gy0..=gy1 {
                            visit(gx, gy, &mut best);
                        }
                    }
                }
            }
            // Every point not yet seen is at least r cells away in xy.
            let lb = r as f64 * self.cell;
            if best.0.is_finite() && lb * lb > best.0 * (1.0 + 1e-9) + 1e-12 {
                break;
            }
        }
        (best.1, best.0)
    }
}

/// `TriangulateBetweenContours`, as written (its first-column step uses the
/// second line's length, and the second line's cursor is never reset
/// between rows - both kept).
#[allow(clippy::needless_range_loop)] // the indices are the table's coordinates
fn triangulate_between(pts: &[P3], l1: &[usize], l2: &[usize], tris: &mut Vec<[usize; 3]>) {
    let (n1, n2) = (l1.len(), l2.len());
    if n1 == 0 || n2 == 0 {
        return;
    }
    let g2 = Nearest::new(pts, l2);
    let g1 = Nearest::new(pts, l1);
    let c12: Vec<usize> = l1.iter().map(|&i| g2.query(&pts[i]).0).collect();
    let c21: Vec<usize> = l2.iter().map(|&i| g1.query(&pts[i]).0).collect();
    let start1 = 0usize;
    let start2 = c12[0];
    let fp1 = pts[l1[start1]];
    let fp2 = pts[l2[start2]];
    let closed1 = l1[0] == l1[n1 - 1];
    let closed2 = l2[0] == l2[n2 - 1];
    let end_loop = |start: usize, n: usize, closed: bool| {
        if start != 0 {
            if closed {
                start
            } else {
                start - 1
            }
        } else {
            n - 1
        }
    };
    let end1 = end_loop(start1, n1, closed1);
    let end2 = end_loop(start2, n2, closed2);

    // The table: only the previous row of scores is needed, and one bit
    // per cell for the way back (set = left).
    let mut left = vec![0u64; (n1 * n2).div_ceil(64)];
    let set_left = |left: &mut Vec<u64>, i: usize, j: usize| {
        let b = i * n2 + j;
        left[b / 64] |= 1 << (b % 64);
    };
    let is_left = |left: &Vec<u64>, i: usize, j: usize| {
        let b = i * n2 + j;
        left[b / 64] & (1 << (b % 64)) != 0
    };
    let mut row = vec![0.0f64; n2];
    row[0] = d2(&fp1, &fp2);
    let mut cur2 = next_loc(start2, n2, closed2);
    for j in 1..n2 {
        let p = pts[l2[cur2]];
        row[j] = row[j - 1] + d2(&fp1, &p);
        set_left(&mut left, 0, j);
        cur2 = next_loc(cur2, n2, closed2);
    }
    // The first column: scores of each row's first cell.
    let mut col0 = vec![0.0f64; n1];
    col0[0] = row[0];
    let mut cur1 = next_loc(start1, n2, closed1);
    for i in 1..n1 {
        let p = pts[l1[cur1]];
        col0[i] = col0[i - 1] + d2(&p, &fp2);
        cur1 = next_loc(cur1, n1, closed1);
    }

    let mut prev1 = start1;
    let mut prev2 = start2;
    cur1 = next_loc(start1, n1, closed1);
    cur2 = next_loc(start2, n2, closed2);
    let mut i_end = 1usize;
    let mut j_end = 1usize;
    if n1 > 1 {
        let mut next = vec![0.0f64; n2];
        for i in 1..n1 {
            let p1 = pts[l1[cur1]];
            next[0] = col0[i];
            for j in 1..n2 {
                let p2 = pts[l2[cur2]];
                let d = d2(&p1, &p2);
                if cur1 == c21[prev2] {
                    next[j] = next[j - 1] + d;
                    set_left(&mut left, i, j);
                } else if cur2 == c12[prev1] {
                    next[j] = row[j] + d;
                } else if next[j - 1] <= row[j] {
                    next[j] = next[j - 1] + d;
                    set_left(&mut left, i, j);
                } else {
                    next[j] = row[j] + d;
                }
                prev2 = cur2;
                cur2 = next_loc(cur2, n2, closed2);
            }
            std::mem::swap(&mut row, &mut next);
            prev1 = cur1;
            cur1 = next_loc(cur1, n1, closed1);
        }
        i_end = n1;
        j_end = if n2 > 1 { n2 } else { 1 };
    }

    let (mut c1, mut c2) = (end1, end2);
    let (mut i, mut j) = (i_end - 1, j_end - 1);
    while i > 0 || j > 0 {
        let t0 = l1[c1];
        let t1 = l2[c2];
        let t2;
        if is_left(&left, i, j) {
            let p = prev_loc(c2, n2, closed2);
            t2 = l2[p];
            j -= 1;
            c2 = p;
        } else {
            let p = prev_loc(c1, n1, closed1);
            t2 = l1[p];
            i -= 1;
            c1 = p;
        }
        tris.push([t0, t1, t2]);
    }
}

/// `TriangulateContourInterior`: the flat cap, normals up or down. Any
/// triangulation of a flat polygon adds the same to the volume; this one
/// clips ears, after dropping coincident vertices as `vtkPolygon` does.
fn triangulate_interior(pts: &[P3], line: &[usize], up: bool, tris: &mut Vec<[usize; 3]>) {
    let mut ids: Vec<usize> = line.to_vec();
    if ids.len() > 1 && ids[0] == ids[ids.len() - 1] {
        ids.pop();
    }
    if ids.len() < 3 {
        return;
    }
    let b = {
        let mut b = [f64::MAX, f64::MIN, f64::MAX, f64::MIN, f64::MAX, f64::MIN];
        for &i in &ids {
            for a in 0..3 {
                b[2 * a] = b[2 * a].min(pts[i][a]);
                b[2 * a + 1] = b[2 * a + 1].max(pts[i][a]);
            }
        }
        b
    };
    let tol = 1e-6 * ((b[1] - b[0]).powi(2) + (b[3] - b[2]).powi(2) + (b[5] - b[4]).powi(2)).sqrt();
    let tol2 = tol * tol;
    // Coincident neighbours out (ring order).
    let mut ring: Vec<usize> = Vec::with_capacity(ids.len());
    for &i in &ids {
        if let Some(&last) = ring.last() {
            if d2(&pts[last], &pts[i]) < tol2 {
                continue;
            }
        }
        ring.push(i);
    }
    while ring.len() > 1 && d2(&pts[ring[0]], &pts[ring[ring.len() - 1]]) < tol2 {
        ring.pop();
    }
    let mut out: Vec<[usize; 3]> = Vec::new();
    ear_clip(pts, &ring, &mut out);
    for t in out {
        tris.push(if up { t } else { [t[2], t[1], t[0]] });
    }
}

fn cross2(o: &P3, a: &P3, b: &P3) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// Ear clipping of a counter-clockwise ring in the xy plane; triangles in
/// ring order. A ring that cannot be clipped cleanly is finished as a fan.
fn ear_clip(pts: &[P3], ring: &[usize], out: &mut Vec<[usize; 3]>) {
    let mut r: Vec<usize> = ring.to_vec();
    let area: f64 = (0..r.len())
        .map(|i| {
            let (p, q) = (pts[r[i]], pts[r[(i + 1) % r.len()]]);
            p[0] * q[1] - q[0] * p[1]
        })
        .sum();
    let sign = if area >= 0.0 { 1.0 } else { -1.0 };
    let mut guard = 0usize;
    while r.len() > 3 && guard < 4 * ring.len() * ring.len() + 16 {
        guard += 1;
        let n = r.len();
        let mut clipped = false;
        for i in 0..n {
            let (a, b, c) = (r[(i + n - 1) % n], r[i], r[(i + 1) % n]);
            let turn = sign * cross2(&pts[a], &pts[b], &pts[c]);
            if turn <= 0.0 {
                continue;
            }
            let inside = r.iter().any(|&k| {
                if k == a || k == b || k == c {
                    return false;
                }
                let p = &pts[k];
                let s1 = sign * cross2(&pts[a], &pts[b], p);
                let s2 = sign * cross2(&pts[b], &pts[c], p);
                let s3 = sign * cross2(&pts[c], &pts[a], p);
                s1 > 0.0 && s2 > 0.0 && s3 > 0.0
            });
            if inside {
                continue;
            }
            out.push([a, b, c]);
            r.remove(i);
            clipped = true;
            break;
        }
        if !clipped {
            break;
        }
    }
    if r.len() >= 3 {
        for k in 1..r.len() - 1 {
            out.push([r[0], r[k], r[k + 1]]);
        }
    }
}

// -- the VTK filters a smooth cap is made with --------------------------------

/// `vtkPolyDataToImageStencil` + `vtkImageStencil` (reversed, background 1)
/// on one closed polyline: 1 inside, 0 outside, on the cap's image.
fn stencil(line: &[P3], origin: P3, sp: [f64; 3], nx: usize, ny: usize) -> Vec<u8> {
    const TOL: f64 = 7.62939453125e-06;
    let inv = [1.0 / sp[0], 1.0 / sp[1]];
    let s: Vec<[f64; 2]> = line
        .iter()
        .map(|p| [(p[0] - origin[0]) * inv[0], (p[1] - origin[1]) * inv[1]])
        .collect();
    let ymax_ext = ny as i64 - 1;
    let mut raster: Vec<[Vec<f64>; 2]> = (0..ny).map(|_| [Vec::new(), Vec::new()]).collect();
    for w in s.windows(2) {
        let (mut x1, mut y1, mut x2, mut y2) = (w[0][0], w[0][1], w[1][0], w[1][1]);
        if y1 > y2 {
            std::mem::swap(&mut x1, &mut x2);
            std::mem::swap(&mut y1, &mut y2);
        }
        let (xmin, xmax) = if x1 > x2 { (x2, x1) } else { (x1, x2) };
        if y1 == y2 {
            continue;
        }
        let grad = (x2 - x1) / (y2 - y1);
        let ymin = [y1 - TOL, y1 + TOL];
        let ymx = [y2 - TOL, y2 + TOL];
        for i in 0..2 {
            let mut iy1: i64 = 0;
            let mut iy2: i64 = ymax_ext;
            if ymx[i] < iy1 as f64 || ymin[i] >= iy2 as f64 {
                continue;
            }
            if ymin[i] >= iy1 as f64 {
                iy1 = ymin[i].floor() as i64 + 1;
            }
            if ymx[i] < iy2 as f64 {
                iy2 = ymx[i].floor() as i64;
            }
            let mut delta = (iy1 as f64 - y1) * grad;
            let mut y = iy1;
            while y <= iy2 {
                let mut x = x1 + delta;
                delta += grad;
                x = if x < xmax { x } else { xmax };
                x = if x > xmin { x } else { xmin };
                raster[y as usize][i].push(x);
                y += 1;
            }
        }
    }
    let mut img = vec![0u8; nx * ny];
    let xmax_ext = nx as i64 - 1;
    for (row, r) in raster.iter_mut().enumerate() {
        for v in r.iter_mut() {
            v.sort_by(f64::total_cmp);
            let even = v.len() - v.len() % 2;
            v.truncate(even);
        }
        let mut pos = [0usize, 0usize];
        let mut lastr = i64::MIN;
        loop {
            let mut x1 = f64::MAX;
            let mut j = usize::MAX;
            for i in 0..2 {
                if pos[i] < r[i].len() && r[i][pos[i]] < x1 {
                    x1 = r[i][pos[i]];
                    j = i;
                }
            }
            if j == usize::MAX {
                break;
            }
            let mut x2 = r[j][pos[j] + 1];
            pos[j] += 2;
            x1 -= TOL;
            x2 += TOL;
            if x2 < 0.0 || x1 >= xmax_ext as f64 {
                continue;
            }
            let mut r1: i64 = 0;
            let mut r2: i64 = xmax_ext;
            if x1 >= 0.0 {
                r1 = x1.floor() as i64 + 1;
            }
            if x2 < xmax_ext as f64 {
                r2 = x2.floor() as i64;
            }
            if r1 <= lastr {
                r1 = lastr + 1;
            }
            if r2 > lastr {
                lastr = r2;
                for x in r1.max(0)..=r2.min(xmax_ext) {
                    img[row * nx + x as usize] = 1;
                }
            }
        }
    }
    img
}

/// `vtkImageDilateErode3D` with a 5 × 5 × 1 kernel, eroding 1 into 0: the
/// ellipsoidal kernel is the 5 × 5 square less its corners, and the image
/// border is not a neighbour.
fn erode(img: &[u8], nx: usize, ny: usize) -> Vec<u8> {
    let mut kernel = Vec::new();
    for dy in -2i64..=2 {
        for dx in -2i64..=2 {
            let (sx, sy) = (dx as f64 / 2.5, dy as f64 / 2.5);
            if sx * sx + sy * sy <= 1.0 {
                kernel.push((dx, dy));
            }
        }
    }
    let mut out = img.to_vec();
    for y in 0..ny as i64 {
        for x in 0..nx as i64 {
            let i = (y as usize) * nx + x as usize;
            if img[i] != 1 {
                continue;
            }
            for &(dx, dy) in &kernel {
                let (u, v) = (x + dx, y + dy);
                if u < 0 || v < 0 || u >= nx as i64 || v >= ny as i64 {
                    continue;
                }
                if img[(v as usize) * nx + u as usize] == 0 {
                    out[i] = 0;
                    break;
                }
            }
        }
    }
    out
}

/// `vtkMarchingSquares` at the value 1 on a 0/1 image, points merged
/// exactly (`vtkMergePoints`), in world coordinates.
fn marching_squares(
    img: &[u8],
    nx: usize,
    ny: usize,
    origin: P3,
    sp: [f64; 3],
) -> (Vec<P3>, Vec<[usize; 2]>) {
    const CASES: [[i8; 5]; 16] = [
        [-1, -1, -1, -1, -1],
        [0, 3, -1, -1, -1],
        [1, 0, -1, -1, -1],
        [1, 3, -1, -1, -1],
        [2, 1, -1, -1, -1],
        [0, 3, 2, 1, -1],
        [2, 0, -1, -1, -1],
        [2, 3, -1, -1, -1],
        [3, 2, -1, -1, -1],
        [0, 2, -1, -1, -1],
        [1, 0, 3, 2, -1],
        [1, 2, -1, -1, -1],
        [3, 1, -1, -1, -1],
        [0, 1, -1, -1, -1],
        [3, 0, -1, -1, -1],
        [-1, -1, -1, -1, -1],
    ];
    const MASK: [usize; 4] = [1, 2, 8, 4];
    const EDGES: [[usize; 2]; 4] = [[0, 1], [1, 3], [2, 3], [0, 2]];
    let mut index_of: HashMap<(i64, i64), usize> = HashMap::new();
    let mut grid_pts: Vec<(i64, i64)> = Vec::new();
    let mut segs = Vec::new();
    if nx < 2 || ny < 2 {
        return (Vec::new(), segs);
    }
    let value = 1.0f64;
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            let s = [
                img[j * nx + i] as f64,
                img[j * nx + i + 1] as f64,
                img[(j + 1) * nx + i] as f64,
                img[(j + 1) * nx + i + 1] as f64,
            ];
            if s.iter().all(|&v| v < value) {
                continue;
            }
            let corner = [
                (i as i64, j as i64),
                (i as i64 + 1, j as i64),
                (i as i64, j as i64 + 1),
                (i as i64 + 1, j as i64 + 1),
            ];
            let mut index = 0;
            for k in 0..4 {
                if s[k] >= value {
                    index |= MASK[k];
                }
            }
            if index == 0 || index == 15 {
                continue;
            }
            let case = CASES[index];
            let mut e = 0;
            while case[e] > -1 {
                let mut ids = [0usize; 2];
                for (ii, id) in ids.iter_mut().enumerate() {
                    let vert = EDGES[case[e + ii] as usize];
                    let t = (value - s[vert[0]]) / (s[vert[1]] - s[vert[0]]);
                    // With 0/1 values t is exactly 0 or 1: the point is the
                    // corner that is inside.
                    let p = if t == 0.0 {
                        corner[vert[0]]
                    } else {
                        corner[vert[1]]
                    };
                    *id = *index_of.entry(p).or_insert_with(|| {
                        grid_pts.push(p);
                        grid_pts.len() - 1
                    });
                }
                if ids[0] != ids[1] {
                    segs.push(ids);
                }
                e += 2;
            }
        }
    }
    let pts = grid_pts
        .into_iter()
        .map(|(i, j)| {
            [
                f32r(origin[0] + sp[0] * i as f64),
                f32r(origin[1] + sp[1] * j as f64),
                f32r(origin[2]),
            ]
        })
        .collect();
    (pts, segs)
}

/// `vtkStripper` on two-point lines: poly-lines grown forwards from each
/// unvisited segment, neighbours taken in cell order.
fn strip(segs: &[[usize; 2]], npts: usize) -> Vec<Vec<usize>> {
    let mut links: Vec<Vec<usize>> = vec![Vec::new(); npts];
    for (c, s) in segs.iter().enumerate() {
        links[s[0]].push(c);
        if s[1] != s[0] {
            links[s[1]].push(c);
        }
    }
    let mut visited = vec![false; segs.len()];
    let mut out = Vec::new();
    for c in 0..segs.len() {
        if visited[c] {
            continue;
        }
        visited[c] = true;
        let lp = segs[c];
        let mut found = None;
        let mut pts = [lp[0], lp[1]];
        for i in 0..2 {
            pts = [lp[i], lp[(i + 1) % 2]];
            if let Some(&nb) = links[pts[1]].iter().find(|&&nb| nb != c && !visited[nb]) {
                found = Some(nb);
                break;
            }
        }
        let Some(mut neighbor) = found else {
            out.push(vec![lp[0], lp[1]]);
            continue;
        };
        let mut poly = vec![pts[0], pts[1]];
        loop {
            visited[neighbor] = true;
            let lp = segs[neighbor];
            let last = *poly.last().unwrap();
            let next = if lp[0] != last { lp[0] } else { lp[1] };
            poly.push(next);
            match links[next]
                .iter()
                .find(|&&nb| nb != neighbor && !visited[nb])
            {
                Some(&nb) => neighbor = nb,
                None => {
                    out.push(poly);
                    break;
                }
            }
        }
    }
    out
}

/// `FixLines`: drops lines of two points or fewer, and a line that loops
/// back on itself by its third point (unless it is the only one, which then
/// loses its first and last points).
fn fix_lines(lines: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    for l in lines {
        if l.len() <= 2 {
            continue;
        }
        if l[0] == l[2] && l.len() != 3 {
            if lines.len() > 1 {
                continue;
            }
            let fixed: Vec<usize> = l[1..l.len() - 1].to_vec();
            if fixed.len() > 1 {
                out.push(fixed);
            }
        } else {
            out.push(l.clone());
        }
    }
    out
}

/// `vtkPriorityQueue`, as VTK has it: a binary heap with an id → location
/// map; an id already queued is not queued twice.
struct PriorityQueue {
    heap: Vec<(f64, usize)>,
    loc: HashMap<usize, usize>,
}

impl PriorityQueue {
    fn new() -> Self {
        PriorityQueue {
            heap: Vec::new(),
            loc: HashMap::new(),
        }
    }

    fn len(&self) -> usize {
        self.heap.len()
    }

    fn insert(&mut self, priority: f64, id: usize) {
        if self.loc.contains_key(&id) {
            return;
        }
        self.heap.push((priority, id));
        let mut i = self.heap.len() - 1;
        self.loc.insert(id, i);
        while i > 0 {
            let idx = (i - 1) / 2;
            if self.heap[i].0 < self.heap[idx].0 {
                self.heap.swap(i, idx);
                self.loc.insert(self.heap[i].1, i);
                self.loc.insert(self.heap[idx].1, idx);
                i = idx;
            } else {
                break;
            }
        }
    }

    fn peek_priority(&self) -> f64 {
        self.heap.first().map(|h| h.0).unwrap_or(f64::MAX)
    }

    /// `Pop(0)`.
    fn pop(&mut self) -> Option<usize> {
        if self.heap.is_empty() {
            return None;
        }
        let location = 0;
        let id = self.heap[location].1;
        let last = *self.heap.last().unwrap();
        self.heap[location] = last;
        self.loc.insert(last.1, location);
        self.loc.remove(&id);
        if last.1 == id {
            // The item popped was the last one as well.
        }
        self.heap.pop();
        let max_id = self.heap.len() as i64 - 1;
        if max_id <= 0 {
            return Some(id);
        }
        let last_node = (max_id - 1) / 2;
        let mut i = location as i64;
        while i <= last_node {
            let idx = 2 * i + 1;
            let j = if idx == max_id || self.heap[idx as usize].0 < self.heap[idx as usize + 1].0 {
                idx
            } else {
                idx + 1
            };
            if self.heap[i as usize].0 > self.heap[j as usize].0 {
                self.heap.swap(i as usize, j as usize);
                self.loc.insert(self.heap[i as usize].1, i as usize);
                self.loc.insert(self.heap[j as usize].1, j as usize);
                i = j;
            } else {
                break;
            }
        }
        Some(id)
    }
}

/// `DecimateLines`, including that one queue serves every line, so what a
/// line leaves in it is popped (to no effect) while the next is decimated.
fn decimate(pts: &[P3], lines: &[Vec<usize>], factor: f64) -> Vec<Vec<usize>> {
    let mut pq = PriorityQueue::new();
    let mut out = Vec::new();
    for line in lines {
        let n0 = line.len();
        let mut ids = line.clone();
        if ids.len() > 2 {
            for idx in 0..ids.len() {
                pq.insert(compute_error(pts, &ids, idx), ids[idx]);
            }
            while pq.len() > 3
                && (pq.peek_priority() < f64::EPSILON || ids.len() as f64 / n0 as f64 > factor)
            {
                if let Some(id) = pq.pop() {
                    ids.retain(|&x| x != id);
                }
            }
        }
        if ids.len() > 1 && ids[0] != ids[ids.len() - 1] {
            ids.push(ids[0]);
        }
        if ids.len() > 1 {
            out.push(ids);
        }
    }
    out
}

/// `ComputeError`: the squared distance of a point from the line through
/// its neighbours (`vtkLine::DistanceToLine`).
fn compute_error(pts: &[P3], ids: &[usize], idx: usize) -> f64 {
    let n = ids.len();
    let closed = ids[0] == ids[n - 1];
    let cur = pts[ids[idx]];
    let next = pts[ids[next_loc(idx, n, closed)]];
    let prev = pts[ids[prev_loc(idx, n, closed)]];
    if d2(&prev, &next) == 0.0 {
        return 0.0;
    }
    let np1 = [cur[0] - next[0], cur[1] - next[1], cur[2] - next[2]];
    let mut p1p2 = [next[0] - prev[0], next[1] - prev[1], next[2] - prev[2]];
    let den = (p1p2[0] * p1p2[0] + p1p2[1] * p1p2[1] + p1p2[2] * p1p2[2]).sqrt();
    for v in &mut p1p2 {
        *v /= den;
    }
    let proj = np1[0] * p1p2[0] + np1[1] * p1p2[1] + np1[2] * p1p2[2];
    np1[0] * np1[0] + np1[1] * np1[1] + np1[2] * np1[2] - proj * proj
}

// -- 6. volume -------------------------------------------------------------------

/// `vtkMassProperties::GetVolume`: per-axis divergence sums weighted by how
/// many triangles face each axis. `None` for no triangles.
fn mass_properties_volume(pts: &[P3], tris: &[[usize; 3]]) -> Option<f64> {
    if tris.is_empty() {
        return None;
    }
    let mut vol = [0.0f64; 3];
    let mut munc = [0.0f64; 3];
    let (mut wxyz, mut wxy, mut wxz, mut wyz) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for t in tris {
        let (p0, p1, p2) = (pts[t[0]], pts[t[1]], pts[t[2]]);
        let x = [p0[0], p1[0], p2[0]];
        let y = [p0[1], p1[1], p2[1]];
        let z = [p0[2], p1[2], p2[2]];
        let i = [x[1] - x[0], x[2] - x[0], x[2] - x[1]];
        let j = [y[1] - y[0], y[2] - y[0], y[2] - y[1]];
        let k = [z[1] - z[0], z[2] - z[0], z[2] - z[1]];
        let mut u = [
            j[0] * k[1] - k[0] * j[1],
            k[0] * i[1] - i[0] * k[1],
            i[0] * j[1] - j[0] * i[1],
        ];
        let length = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
        if length != 0.0 {
            for v in &mut u {
                *v /= length;
            }
        } else {
            u = [0.0; 3];
        }
        let a = [u[0].abs(), u[1].abs(), u[2].abs()];
        if a[0] > a[1] && a[0] > a[2] {
            munc[0] += 1.0;
        } else if a[1] > a[0] && a[1] > a[2] {
            munc[1] += 1.0;
        } else if a[2] > a[0] && a[2] > a[1] {
            munc[2] += 1.0;
        } else if a[0] == a[1] && a[0] == a[2] {
            wxyz += 1.0;
        } else if a[0] == a[1] && a[0] > a[2] {
            wxy += 1.0;
        } else if a[0] == a[2] && a[0] > a[1] {
            wxz += 1.0;
        } else if a[1] == a[2] && a[0] < a[2] {
            wyz += 1.0;
        } else {
            return None;
        }
        let ii = [i[0] * i[0], i[1] * i[1], i[2] * i[2]];
        let jj = [j[0] * j[0], j[1] * j[1], j[2] * j[2]];
        let kk = [k[0] * k[0], k[1] * k[1], k[2] * k[2]];
        let aa = (ii[1] + jj[1] + kk[1]).sqrt();
        let bb = (ii[0] + jj[0] + kk[0]).sqrt();
        let cc = (ii[2] + jj[2] + kk[2]).sqrt();
        let s = 0.5 * (aa + bb + cc);
        let area = (s * (s - aa) * (s - bb) * (s - cc)).abs().sqrt();
        let zavg = (z[0] + z[1] + z[2]) / 3.0;
        let yavg = (y[0] + y[1] + y[2]) / 3.0;
        let xavg = (x[0] + x[1] + x[2]) / 3.0;
        vol[2] += area * u[2] * zavg;
        vol[1] += area * u[1] * yavg;
        vol[0] += area * u[0] * xavg;
    }
    let n = tris.len() as f64;
    let kx = (munc[0] + wxyz / 3.0 + (wxy + wxz) / 2.0) / n;
    let ky = (munc[1] + wxyz / 3.0 + (wxy + wyz) / 2.0) / n;
    let kz = (munc[2] + wxyz / 3.0 + (wxz + wyz) / 2.0) / n;
    Some((kx * vol[0] + ky * vol[1] + kz * vol[2]).abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;
    use crate::rtstruct::Contour;

    fn roi(contours: Vec<Vec<[f64; 3]>>) -> Roi {
        Roi {
            number: 1,
            name: "probe".into(),
            color: [255, 0, 0],
            roi_type: "PTV".into(),
            description: String::new(),
            contours: contours
                .into_iter()
                .map(|pts| Contour {
                    points: pts
                        .into_iter()
                        .map(|p| Vec3::new(p[0], p[1], p[2]))
                        .collect(),
                    geometric_type: "CLOSED_PLANAR".into(),
                })
                .collect(),
        }
    }

    fn square(x: f64, y: f64, s: f64, z: f64) -> Vec<[f64; 3]> {
        vec![[x, y, z], [x + s, y, z], [x + s, y + s, z], [x, y + s, z]]
    }

    /// Every edge of the surface is walked once each way: no hole in it.
    fn closed(roi: &Roi, capping: EndCapping) -> bool {
        let mut m = Mesh::read(roi, 0.0).unwrap();
        m.orient_axial();
        m.sort_contours();
        m.fix_keyholes(0.001, 3);
        m.set_lines_counter_clockwise();
        let tris = m.build(capping);
        let key = |i: usize| m.pts[i].map(|v| (v * 1024.0).round() as i64);
        let mut edges: HashMap<([i64; 3], [i64; 3]), i32> = HashMap::new();
        for t in &tris {
            for k in 0..3 {
                let (a, b) = (key(t[k]), key(t[(k + 1) % 3]));
                if a != b {
                    *edges.entry((a, b)).or_default() += 1;
                    *edges.entry((b, a)).or_default() -= 1;
                }
            }
        }
        edges.values().all(|v| *v == 0)
    }

    /// Ten squares 20 mm on a side, 2 mm apart: 7200 mm³ between the first
    /// and the last, 8000 mm³ with half a slice added at each end.
    fn box_stack() -> Roi {
        roi((0..10)
            .map(|k| square(0.0, 0.0, 20.0, k as f64 * 2.0))
            .collect())
    }

    #[test]
    fn a_stack_of_squares_is_closed_and_its_caps_sit_half_a_slice_out() {
        let r = box_stack();
        // Open ends: exactly the prism between the first and last contour.
        let open = surface_volume_mm3(&r, EndCapping::None, 0.0).unwrap();
        assert!((open - 7200.0).abs() < 1e-6, "{open}");
        for capping in [EndCapping::Straight, EndCapping::Smooth] {
            assert!(closed(&r, capping), "{capping:?} left a hole");
        }
        // Straight caps: a copy of the end contour half a slice out, so up
        // to the full 8000 mm³. The ribbon to the top copy is allowed its
        // twist - SlicerRT copies the closing point as a point of its own,
        // and its matcher then takes the copy for an open line - which
        // folds one sliver in, 400/6 mm³ here.
        let straight = surface_volume_mm3(&r, EndCapping::Straight, 0.0).unwrap();
        assert!(
            straight > 8000.0 - 400.0 / 6.0 - 1e-6 && straight <= 8000.0 + 1e-6,
            "{straight}"
        );
        // Smooth caps shrink the end contour first: less than straight,
        // more than open.
        let smooth = surface_volume_mm3(&r, EndCapping::Smooth, 0.0).unwrap();
        assert!(smooth < straight && smooth > open, "{smooth}");
        // In cm³, as the tables show it.
        let cm3 = slicer_volume_cm3(&r, 2.0).unwrap();
        assert!((cm3 - smooth / 1000.0).abs() < 1e-12);
    }

    #[test]
    fn the_order_the_contours_come_in_does_not_matter() {
        let mut r = box_stack();
        let a = slicer_volume_cm3(&r, 2.0).unwrap();
        r.contours.reverse();
        r.contours.swap(2, 7);
        let b = slicer_volume_cm3(&r, 2.0).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn one_plane_takes_its_thickness_from_the_image() {
        let r = roi(vec![square(0.0, 0.0, 20.0, 5.0)]);
        // With no slice spacing to go on, the surface is flat.
        let flat = surface_volume_mm3(&r, EndCapping::Straight, 0.0).unwrap();
        assert!(flat.abs() < 1e-9, "{flat}");
        // Given the image's, the caps sit half of it either side (less the
        // sliver the top ribbon folds in, 400 x 1.5 / 6 mm³).
        let slab = surface_volume_mm3(&r, EndCapping::Straight, 3.0).unwrap();
        assert!((1100.0 - 1e-6..=1200.0 + 1e-6).contains(&slab), "{slab}");
        let smooth = slicer_volume_cm3(&r, 3.0).unwrap();
        assert!(smooth > 0.0 && smooth < slab / 1000.0, "{smooth}");
    }

    #[test]
    fn a_contour_through_one_point_twice_is_two_lines() {
        // Two squares meeting at a corner, drawn as one path through it -
        // the figure of eight a voxel outline makes - against the same two
        // squares as two contours.
        let eight = |z: f64| {
            vec![
                [0.0, 0.0, z],
                [10.0, 0.0, z],
                [10.0, 10.0, z],
                [20.0, 10.0, z],
                [20.0, 20.0, z],
                [10.0, 20.0, z],
                [10.0, 10.0, z],
                [0.0, 10.0, z],
            ]
        };
        let one = roi((0..6).map(|k| eight(k as f64 * 2.0)).collect());
        let two = roi((0..6)
            .flat_map(|k| {
                let z = k as f64 * 2.0;
                [square(0.0, 0.0, 10.0, z), square(10.0, 10.0, 10.0, z)]
            })
            .collect());
        let (a, b) = (
            surface_volume_mm3(&one, EndCapping::None, 0.0).unwrap(),
            surface_volume_mm3(&two, EndCapping::None, 0.0).unwrap(),
        );
        assert!(
            (a - 2000.0).abs() < 1e-6 && (b - 2000.0).abs() < 1e-6,
            "{a} {b}"
        );
    }

    #[test]
    fn nothing_to_build_from_is_no_volume() {
        assert_eq!(slicer_volume_cm3(&roi(vec![]), 2.0), None);
        assert_eq!(
            slicer_volume_cm3(&roi(vec![vec![[1.0, 2.0, 3.0]]]), 2.0),
            None
        );
    }

    #[test]
    fn the_thickness_goes_over_as_six_digits() {
        assert_eq!(stream_rounded(2.5), 2.5);
        assert_eq!(stream_rounded(1.99999999), 2.0);
        assert_eq!(stream_rounded(0.97656251), 0.976563);
        assert_eq!(stream_rounded(0.0), 0.0);
    }
}
