//! MV portal images: where the treatment beam's aperture falls on the
//! image, and where the field edge really is.
//!
//! A portal image is taken through the treatment beam, so it carries two
//! things a kV image does not: the anatomy at low contrast (Compton
//! scattering dominates at MV, `HuMode::MegaVoltage`), and the edge of the
//! field the jaws and the MLC shape. The plan says where that edge should be
//! ([`crate::rtplan::Aperture`], in the beam limiting device's axes at the
//! isocentre plane); projected through the image's camera it is an outline
//! on the image ([`aperture_mask`], [`aperture_outline`]). The image says
//! where the edge is: [`field_edge`] finds the steepest step of the image
//! along rays from the field's centre, near where the plan puts the edge,
//! and fits the one shift that best explains them. With the anatomy's
//! displacement from the registration beside it, the portal check has both
//! of its numbers: where the patient is, and where the field is relative to
//! him.

use crate::drr::Geometry;
use crate::geometry::Vec3;
use crate::projection::ProjectionGeometry;
use crate::registration::metric2d::{Image2d, Mask2d};
use crate::rtplan::Aperture;

/// Where the ray through pixel `(i, j)` crosses the isocentre plane, in the
/// beam limiting device's axes, mm.
fn bld_of(
    g: &Geometry,
    axes: &[Vec3; 3],
    cam: &ProjectionGeometry,
    i: f64,
    j: f64,
) -> Option<[f64; 2]> {
    let d = cam.pixel(i, j) - cam.source;
    let denom = d.dot(axes[2]);
    if denom.abs() < 1e-12 {
        return None;
    }
    let t = (g.isocenter - cam.source).dot(axes[2]) / denom;
    let p = cam.source + d * t - g.isocenter;
    Some([p.dot(axes[0]), p.dot(axes[1])])
}

/// The pixels of `cam` the aperture's field falls on, the field shrunk by
/// `shrink_mm` at the isocentre plane first (a mask that stays clear of the
/// edge).
pub fn aperture_mask(
    g: &Geometry,
    ap: &Aperture,
    cam: &ProjectionGeometry,
    shrink_mm: f64,
) -> Mask2d {
    let axes = g.bld_axes();
    let rects = ap.rectangles();
    let inside = |x: f64, y: f64| {
        rects
            .iter()
            .any(|r| x >= r[0] && x <= r[1] && y >= r[2] && y <= r[3])
    };
    let s = shrink_mm.max(0.0);
    let [w, h] = cam.dims;
    let mut m = vec![0u8; w * h];
    for j in 0..h {
        for i in 0..w {
            if let Some([x, y]) = bld_of(g, &axes, cam, i as f64, j as f64) {
                let ok = inside(x, y)
                    && (s == 0.0
                        || (inside(x - s, y)
                            && inside(x + s, y)
                            && inside(x, y - s)
                            && inside(x, y + s)));
                m[j * w + i] = ok as u8;
            }
        }
    }
    m
}

/// The aperture's outline on `cam`'s raster, as segments scaled to `0..1`
/// across and down.
pub fn aperture_outline(
    g: &Geometry,
    ap: &Aperture,
    cam: &ProjectionGeometry,
) -> Vec<([f32; 2], [f32; 2])> {
    let m = aperture_mask(g, ap, cam, 0.0);
    let [w, h] = cam.dims;
    let field: Vec<f32> = m.iter().map(|&v| v as f32).collect();
    crate::render::marching_squares(&field, w, h, 0.5)
        .into_iter()
        .map(|(a, b)| {
            let n = |p: [f32; 2]| [(p[0] + 0.5) / w as f32, (p[1] + 0.5) / h as f32];
            (n(a), n(b))
        })
        .collect()
}

/// Where the field edge is against where the plan puts it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldEdge {
    /// The field's offset along `Xb` and `Yb`, mm at the isocentre.
    pub shift_mm: [f64; 2],
    /// What the shift leaves unexplained, mm at the isocentre.
    pub rms_mm: f64,
    /// Rays whose edge was found.
    pub rays: usize,
}

/// Bilinear sample, `None` outside.
fn bilinear(img: &Image2d, x: f64, y: f64) -> Option<f64> {
    let [w, h] = img.dims;
    if x < 0.0 || y < 0.0 || x > (w - 1) as f64 || y > (h - 1) as f64 {
        return None;
    }
    let (x0, y0) = (x.floor() as usize, y.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (fx, fy) = (x - x0 as f64, y - y0 as f64);
    let at = |i: usize, j: usize| img.data[j * w + i] as f64;
    let a = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
    let b = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
    Some(a + (b - a) * fy)
}

/// The field edge on `image` (taken through `cam`, pixels matching it)
/// against the planned aperture: along 72 rays from the planned field's
/// centre, the steepest step of the image within `reach_mm` of the planned
/// edge, and the one shift that explains them best.
pub fn field_edge(
    image: &Image2d,
    g: &Geometry,
    ap: &Aperture,
    cam: &ProjectionGeometry,
    reach_mm: f64,
) -> Option<FieldEdge> {
    if image.dims != cam.dims {
        return None;
    }
    let [w, h] = cam.dims;
    let mask = aperture_mask(g, ap, cam, 0.0);
    let (mut n, mut cx, mut cy) = (0.0f64, 0.0f64, 0.0f64);
    for j in 0..h {
        for i in 0..w {
            if mask[j * w + i] != 0 {
                n += 1.0;
                cx += i as f64;
                cy += j as f64;
            }
        }
    }
    if n < 16.0 {
        return None;
    }
    let (cx, cy) = (cx / n, cy / n);
    // In the planned field, exactly: through the ray of the fractional
    // pixel, not the mask's pixels.
    let axes = g.bld_axes();
    let rects = ap.rectangles();
    let inside = |x: f64, y: f64| {
        x >= -0.5
            && y >= -0.5
            && x <= w as f64 - 0.5
            && y <= h as f64 - 0.5
            && bld_of(g, &axes, cam, x, y).is_some_and(|[bx, by]| {
                rects
                    .iter()
                    .any(|r| bx >= r[0] && bx <= r[1] && by >= r[2] && by <= r[3])
            })
    };
    let px_mm = crate::registration::projective::pixel_at(cam, g.isocenter);
    let reach = (reach_mm / px_mm).max(2.0);
    // Per ray: the edge's normal there, and how far along it the edge is
    // from the planned one. A shift `s` of the field moves an edge of
    // normal `n` by `n · s` along `n`.
    let mut rows: Vec<([f64; 2], f64)> = Vec::new();
    for k in 0..72 {
        let th = k as f64 * std::f64::consts::TAU / 72.0;
        let d = [th.cos(), th.sin()];
        let mut t = 0.0;
        while inside(cx + d[0] * t, cy + d[1] * t) {
            t += 0.25;
            if t > (w + h) as f64 {
                break;
            }
        }
        // Between the last point inside and the first outside, halved.
        let (mut a, mut b) = ((t - 0.25).max(0.0), t);
        for _ in 0..12 {
            let m = 0.5 * (a + b);
            if inside(cx + d[0] * m, cy + d[1] * m) {
                a = m;
            } else {
                b = m;
            }
        }
        let t_plan = 0.5 * (a + b);
        // The steepest step of the image near it.
        let step = 0.5;
        let mut best = (0.0f64, t_plan);
        let mut grads = Vec::new();
        let mut s = t_plan - reach;
        while s <= t_plan + reach {
            let a = bilinear(image, cx + d[0] * (s - step), cy + d[1] * (s - step));
            let b = bilinear(image, cx + d[0] * (s + step), cy + d[1] * (s + step));
            if let (Some(a), Some(b)) = (a, b) {
                let gr = (b - a).abs();
                grads.push(gr);
                if gr > best.0 {
                    best = (gr, s);
                }
            }
            s += step;
        }
        if grads.len() < 5 {
            continue;
        }
        grads.sort_by(f64::total_cmp);
        let median = grads[grads.len() / 2];
        if best.0 < 3.0 * median.max(1e-12) {
            continue;
        }
        // The normal from the image's own gradient at the edge.
        let (ex, ey) = (cx + d[0] * best.1, cy + d[1] * best.1);
        let gx = bilinear(image, ex + 1.0, ey)
            .zip(bilinear(image, ex - 1.0, ey))
            .map(|(a, b)| a - b);
        let gy = bilinear(image, ex, ey + 1.0)
            .zip(bilinear(image, ex, ey - 1.0))
            .map(|(a, b)| a - b);
        let (Some(gx), Some(gy)) = (gx, gy) else {
            continue;
        };
        let gl = (gx * gx + gy * gy).sqrt();
        if gl < 1e-12 {
            continue;
        }
        let nrm = [gx / gl, gy / gl];
        let along = nrm[0] * d[0] + nrm[1] * d[1];
        // A ray that grazes the edge says little about where it is.
        if along.abs() < 0.3 {
            continue;
        }
        rows.push((nrm, (best.1 - t_plan) * along));
    }
    if rows.len() < 8 {
        return None;
    }
    // d · s = δ, least squares.
    let (mut a, mut b, mut c, mut p, mut q) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for (d, delta) in &rows {
        a += d[0] * d[0];
        b += d[0] * d[1];
        c += d[1] * d[1];
        p += d[0] * delta;
        q += d[1] * delta;
    }
    let det = a * c - b * b;
    if det.abs() < 1e-12 {
        return None;
    }
    let s = [(c * p - b * q) / det, (a * q - b * p) / det];
    let rms = (rows
        .iter()
        .map(|(d, delta)| (d[0] * s[0] + d[1] * s[1] - delta).powi(2))
        .sum::<f64>()
        / rows.len() as f64)
        .sqrt()
        * px_mm;
    // Image pixels to the device's axes, at the field's centre.
    let o = bld_of(g, &axes, cam, cx, cy)?;
    let ex = bld_of(g, &axes, cam, cx + 1.0, cy)?;
    let ey = bld_of(g, &axes, cam, cx, cy + 1.0)?;
    let shift = [
        (ex[0] - o[0]) * s[0] + (ey[0] - o[0]) * s[1],
        (ex[1] - o[1]) * s[0] + (ey[1] - o[1]) * s[1],
    ];
    Some(FieldEdge {
        shift_mm: shift,
        rms_mm: rms,
        rays: rows.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtplan::Mlc;

    fn camera() -> (Geometry, ProjectionGeometry) {
        let g = Geometry {
            panel_mm: [300.0, 300.0],
            dims: [200, 200],
            ..Geometry::default()
        };
        (g, g.camera())
    }

    #[test]
    fn jaws_alone_and_an_mlc_make_their_rectangles() {
        let jaws = Aperture {
            jaw_x: Some([-50.0, 40.0]),
            jaw_y: Some([-30.0, 30.0]),
            mlc: None,
        };
        assert_eq!(jaws.rectangles(), vec![[-50.0, 40.0, -30.0, 30.0]]);
        let mlc = Aperture {
            mlc: Some(Mlc {
                along_x: true,
                boundaries: vec![-20.0, -10.0, 0.0, 10.0, 20.0],
                pairs: vec![[0.0, 0.0], [-15.0, 25.0], [-5.0, 5.0], [0.0, 0.0]],
            }),
            ..jaws.clone()
        };
        assert_eq!(
            mlc.rectangles(),
            vec![[-15.0, 25.0, -10.0, 0.0], [-5.0, 5.0, 0.0, 10.0]]
        );
        assert!(mlc.contains(20.0, -5.0) && !mlc.contains(20.0, 5.0));
    }

    /// Gantry 0, collimator 0, head first supine: `Xb` is the patient's
    /// left, and the field's mask sits where its jaws say.
    #[test]
    fn the_mask_falls_where_the_jaws_put_the_field() {
        let (g, cam) = camera();
        let ap = Aperture {
            jaw_x: Some([-30.0, 50.0]),
            jaw_y: Some([-20.0, 20.0]),
            mlc: None,
        };
        let m = aperture_mask(&g, &ap, &cam, 0.0);
        let px = crate::registration::projective::pixel_at(&cam, g.isocenter);
        let n: usize = m.iter().map(|&v| v as usize).sum();
        let want = 80.0 * 40.0 / (px * px);
        assert!(
            (n as f64 - want).abs() / want < 0.05,
            "{n} pixels, {want:.0} expected"
        );
        // The field reaches further to the patient's left (+x).
        let axes = g.bld_axes();
        assert!(axes[0].x > 0.99, "{:?}", axes[0]);
        let shrunk: usize = aperture_mask(&g, &ap, &cam, 5.0)
            .iter()
            .map(|&v| v as usize)
            .sum();
        assert!(shrunk < n);
    }

    #[test]
    fn a_shifted_field_is_found_shifted() {
        let (g, cam) = camera();
        let planned = Aperture {
            jaw_x: Some([-50.0, 50.0]),
            jaw_y: Some([-40.0, 40.0]),
            mlc: None,
        };
        // Delivered 3 mm towards +Xb and 1 mm towards -Yb.
        let delivered = Aperture {
            jaw_x: Some([-47.0, 53.0]),
            jaw_y: Some([-41.0, 39.0]),
            mlc: None,
        };
        // A penumbra of a millimetre or two, as a real field edge has.
        let axes = g.bld_axes();
        let jx = delivered.jaw_x.unwrap();
        let jy = delivered.jaw_y.unwrap();
        let ramp = |u: f64| 1.0 / (1.0 + (-u / 1.2).exp());
        let [w, h] = cam.dims;
        let img = Image2d::new(
            (0..w * h)
                .map(|k| {
                    let [x, y] = bld_of(&g, &axes, &cam, (k % w) as f64, (k / w) as f64).unwrap();
                    let f = ramp(x - jx[0]) * ramp(jx[1] - x) * ramp(y - jy[0]) * ramp(jy[1] - y);
                    (0.05 + 0.95 * f + 0.002 * ((k * 7919) % 13) as f64) as f32
                })
                .collect(),
            cam.dims,
        );
        let e = field_edge(&img, &g, &planned, &cam, 10.0).unwrap();
        assert!(
            (e.shift_mm[0] - 3.0).abs() < 0.4 && (e.shift_mm[1] + 1.0).abs() < 0.4,
            "{e:?}"
        );
    }
}
