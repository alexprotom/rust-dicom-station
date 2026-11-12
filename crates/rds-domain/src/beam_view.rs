//! The beams of an RT plan as shapes in patient space, for drawing over
//! the views: each beam's source, central axis and field, and the field cut
//! by the plane of a view.
//!
//! A beam is placed by the same IEC 61217 chain the DRR uses
//! ([`crate::drr::Geometry`]): gantry, couch and collimator angles of its
//! first control point, its isocentre, and how the patient lay on the CT
//! ([`crate::setup::default_patient`]). The field is the aperture of the
//! first control point at the isocentre plane, in the beam limiting
//! device's axes ([`Aperture::rectangles`]): the jaws alone, or the open
//! leaf pairs as one outline - the right ends of the openings down one
//! side, the left ends back up the other (for leaves travelling along `Xb`;
//! the other way round for `Yb`). Leaves closed between two openings join
//! them with a bridge rather than splitting the outline, which is what a
//! planning system's beam outline shows too.
//!
//! The beam's volume is the frustum the field spans from the source:
//! from [`NEAR_MM`] upstream of the isocentre (the source itself when it is
//! nearer) to [`FAR_MM`] beyond it. A view draws where that frustum's
//! surface crosses its plane - the field's diverging edges on a plane that
//! holds the central axis, the field's outline (grown or shrunk by the
//! divergence) on a plane across it - and the central axis, as a line when
//! it lies in the plane and as the point where it pierces it otherwise.
//! Every mapping from patient space to a view's pixels is affine, so a
//! crossing is found by linear interpolation of the signed distance to the
//! plane along each edge.
//!
//! An ion beam states no field here (no jaws, no leaves): it is drawn as
//! its central axis alone, from a virtual source [`ION_SAD_MM`] away.

use crate::drr::Geometry;
use crate::geometry::Vec3;
use crate::iec::PatientFrame;
use crate::rtplan::{Aperture, BeamInfo, PlanInfo};
use crate::volume::{ViewPlane, Volume};

/// Source-to-axis distance of a photon beam, mm.
pub const PHOTON_SAD_MM: f64 = 1000.0;
/// The virtual source-to-axis distance assumed for an ion beam, mm.
pub const ION_SAD_MM: f64 = 2000.0;
/// How far upstream of the isocentre a beam is drawn from, mm.
pub const NEAR_MM: f64 = 600.0;
/// How far beyond the isocentre a beam is drawn to, mm.
pub const FAR_MM: f64 = 300.0;

/// One beam in patient coordinates.
#[derive(Clone, Debug)]
pub struct BeamShape {
    pub name: String,
    pub source: Vec3,
    pub isocenter: Vec3,
    /// Unit vector from the source towards the isocentre.
    pub axis: Vec3,
    /// The field's outline at the isocentre plane, in order; empty when the
    /// beam states no aperture.
    pub field: Vec<Vec3>,
    /// The four corners of the field's bounding box at the isocentre
    /// plane (the edges a 3D scene draws); empty with the field.
    pub corners: Vec<Vec3>,
    /// Source-to-axis distance used, mm.
    pub sad: f64,
}

/// A beam cut by one view's plane, in that view's pixel coordinates.
#[derive(Clone, Debug, Default)]
pub struct BeamCut {
    /// Where the beam's surface crosses the plane, as segments.
    pub field: Vec<[[f32; 2]; 2]>,
    /// The central axis, when it lies in the plane.
    pub axis: Option<[[f32; 2]; 2]>,
    /// Where the central axis pierces the plane, otherwise.
    pub pierce: Option<[f32; 2]>,
}

/// The beams of `plan` to draw: every beam with an isocentre except the
/// setup fields.
pub fn plan_beams(plan: &PlanInfo, patient: PatientFrame) -> Vec<BeamShape> {
    plan.beams
        .iter()
        .filter(|b| !b.delivery_type.eq_ignore_ascii_case("SETUP"))
        .filter_map(|b| beam_shape(b, patient))
        .collect()
}

/// One beam's shape; `None` without an isocentre.
pub fn beam_shape(b: &BeamInfo, patient: PatientFrame) -> Option<BeamShape> {
    let iso = b.isocenter?;
    let ion = b.aperture.is_none()
        && matches!(
            b.radiation_type.to_ascii_uppercase().as_str(),
            "PROTON" | "ION" | "CARBON" | "HELIUM"
        );
    let sad = if ion { ION_SAD_MM } else { PHOTON_SAD_MM };
    let g = Geometry {
        sad,
        patient,
        collimator_deg: b.collimator_angle.unwrap_or(0.0),
        ..Geometry::default()
    }
    .from_beam(b);
    let f = g.frame();
    let [xb, yb, _] = g.bld_axes();
    let at = |x: f64, y: f64| iso + xb * x + yb * y;
    let (field, corners) = match &b.aperture {
        Some(ap) => {
            let outline = outline(ap);
            let corners = bounds(ap)
                .map(|r| {
                    vec![
                        at(r[0], r[2]),
                        at(r[1], r[2]),
                        at(r[1], r[3]),
                        at(r[0], r[3]),
                    ]
                })
                .unwrap_or_default();
            (outline.iter().map(|p| at(p[0], p[1])).collect(), corners)
        }
        None => (Vec::new(), Vec::new()),
    };
    Some(BeamShape {
        name: b.name.clone(),
        source: f.source,
        isocenter: iso,
        axis: f.axis,
        field,
        corners,
        sad,
    })
}

/// The bounding box `[x0, x1, y0, y1]` of an aperture's open field.
fn bounds(ap: &Aperture) -> Option<[f64; 4]> {
    let r = ap.rectangles();
    if r.is_empty() {
        return None;
    }
    Some(r.iter().fold(
        [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ],
        |a, b| {
            [
                a[0].min(b[0]),
                a[1].max(b[1]),
                a[2].min(b[2]),
                a[3].max(b[3]),
            ]
        },
    ))
}

/// The open field as one outline in `(Xb, Yb)` at the isocentre plane.
fn outline(ap: &Aperture) -> Vec<[f64; 2]> {
    let mut rects = ap.rectangles();
    if rects.is_empty() {
        return Vec::new();
    }
    // The leaves' travel: along Xb unless the MLC says Yb.
    let along_x = ap.mlc.as_ref().is_none_or(|m| m.along_x);
    if along_x {
        rects.sort_by(|a, b| a[2].total_cmp(&b[2]));
        let mut pts = Vec::with_capacity(rects.len() * 4);
        for r in &rects {
            pts.push([r[1], r[2]]);
            pts.push([r[1], r[3]]);
        }
        for r in rects.iter().rev() {
            pts.push([r[0], r[3]]);
            pts.push([r[0], r[2]]);
        }
        pts
    } else {
        rects.sort_by(|a, b| a[0].total_cmp(&b[0]));
        let mut pts = Vec::with_capacity(rects.len() * 4);
        for r in &rects {
            pts.push([r[0], r[3]]);
            pts.push([r[1], r[3]]);
        }
        for r in rects.iter().rev() {
            pts.push([r[1], r[2]]);
            pts.push([r[0], r[2]]);
        }
        pts
    }
}

impl BeamShape {
    /// Where the beam starts being drawn: [`NEAR_MM`] upstream of the
    /// isocentre, or the source when that is nearer.
    pub fn near_point(&self) -> Vec3 {
        self.isocenter - self.axis * NEAR_MM.min(self.sad)
    }

    /// Where it stops: [`FAR_MM`] beyond the isocentre.
    pub fn far_point(&self) -> Vec3 {
        self.isocenter + self.axis * FAR_MM
    }

    /// A point of the field outline carried along its ray from the source
    /// to the distance `d` from the source (the isocentre plane is `sad`).
    pub fn at_distance(&self, p: Vec3, d: f64) -> Vec3 {
        self.source + (p - self.source) * (d / self.sad.max(1e-6))
    }

    /// The beam cut by the plane of view `plane` at `slice`.
    pub fn cut(&self, vol: &Volume, plane: ViewPlane, slice: usize) -> BeamCut {
        // Patient point to (pixel x, pixel y, signed distance in slices).
        let map = |p: Vec3| -> [f64; 3] {
            let v = vol.patient_to_voxel(p);
            let pp = vol.voxel_to_plane_pixel(plane, v);
            [pp[0], pp[1], pp[2] - slice as f64]
        };
        let mut out = BeamCut::default();
        let (a, b) = (map(self.near_point()), map(self.far_point()));
        if a[2].abs() <= 0.5 && b[2].abs() <= 0.5 {
            out.axis = Some([px(a), px(b)]);
        } else if let Some(p) = crossing(a, b) {
            out.pierce = Some(p);
        }
        if self.field.len() < 3 {
            return out;
        }
        let d_near = self.sad - NEAR_MM.min(self.sad - 1.0);
        let d_far = self.sad + FAR_MM;
        let near: Vec<[f64; 3]> = self
            .field
            .iter()
            .map(|p| map(self.at_distance(*p, d_near)))
            .collect();
        let far: Vec<[f64; 3]> = self
            .field
            .iter()
            .map(|p| map(self.at_distance(*p, d_far)))
            .collect();
        let n = self.field.len();
        // The sides: each a planar quad, cut as one polygon.
        for i in 0..n {
            let j = (i + 1) % n;
            out.field
                .extend(cut_polygon(&[near[i], near[j], far[j], far[i]]));
        }
        // The two caps: polygons, cut by pairing the crossings of their
        // edges along the line they lie on.
        for cap in [&near, &far] {
            out.field.extend(cut_polygon(cap));
        }
        out
    }
}

fn px(p: [f64; 3]) -> [f32; 2] {
    [p[0] as f32, p[1] as f32]
}

/// Where the segment `a`-`b` crosses the plane (distance 0), if it does.
fn crossing(a: [f64; 3], b: [f64; 3]) -> Option<[f32; 2]> {
    if (a[2] > 0.0) == (b[2] > 0.0) || (a[2] - b[2]).abs() < 1e-12 {
        return None;
    }
    let t = a[2] / (a[2] - b[2]);
    Some([
        (a[0] + (b[0] - a[0]) * t) as f32,
        (a[1] + (b[1] - a[1]) * t) as f32,
    ])
}

/// A planar polygon cut by the plane: its edges' crossings sorted along
/// the line they share and paired, first with second, third with fourth.
fn cut_polygon(poly: &[[f64; 3]]) -> Vec<[[f32; 2]; 2]> {
    let n = poly.len();
    let mut pts: Vec<[f32; 2]> = (0..n)
        .filter_map(|i| crossing(poly[i], poly[(i + 1) % n]))
        .collect();
    if pts.len() < 2 {
        return Vec::new();
    }
    let (o, e) = (pts[0], pts[1]);
    let dir = [e[0] - o[0], e[1] - o[1]];
    pts.sort_by(|a, b| {
        let ta = (a[0] - o[0]) * dir[0] + (a[1] - o[1]) * dir[1];
        let tb = (b[0] - o[0]) * dir[0] + (b[1] - o[1]) * dir[1];
        ta.total_cmp(&tb)
    });
    pts.as_chunks::<2>().0.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rtplan::Aperture;

    fn volume() -> Volume {
        // 200 x 200 x 100 voxels of 2 mm, centred on the origin.
        let dims = [200, 200, 100];
        Volume {
            data: vec![0i16; dims[0] * dims[1] * dims[2]],
            dims,
            spacing: [2.0; 3],
            origin: Vec3::new(-199.0, -199.0, -99.0),
            row_dir: Vec3::new(1.0, 0.0, 0.0),
            col_dir: Vec3::new(0.0, 1.0, 0.0),
            normal: Vec3::new(0.0, 0.0, 1.0),
            frame_of_reference_uid: String::new(),
            min_value: 0,
            max_value: 0,
        }
    }

    fn beam(gantry: f64) -> BeamInfo {
        BeamInfo {
            name: "B".into(),
            radiation_type: "PHOTON".into(),
            gantry_angle: Some(gantry),
            couch_angle: Some(0.0),
            collimator_angle: Some(0.0),
            isocenter: Some(Vec3::ZERO),
            aperture: Some(Aperture {
                jaw_x: Some([-50.0, 50.0]),
                jaw_y: Some([-50.0, 50.0]),
                mlc: None,
            }),
            ..BeamInfo::default()
        }
    }

    #[test]
    fn gantry_zero_comes_from_above_a_supine_patient() {
        let b = beam_shape(&beam(0.0), PatientFrame::default()).unwrap();
        // Head first supine: anterior is -y in LPS, so the source sits at
        // -SAD along y and the beam travels towards +y.
        assert!((b.source - Vec3::new(0.0, -PHOTON_SAD_MM, 0.0)).length() < 1e-6);
        assert!((b.axis - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-9);
        assert_eq!(b.field.len(), 4);
    }

    #[test]
    fn an_axial_cut_holds_the_axis_and_two_diverging_edges() {
        let vol = volume();
        let b = beam_shape(&beam(0.0), PatientFrame::default()).unwrap();
        // The slice through the isocentre.
        let k = vol.patient_to_voxel(Vec3::ZERO)[2].round() as usize;
        let cut = b.cut(&vol, ViewPlane::Axial, k);
        assert!(cut.axis.is_some(), "the central axis lies in the plane");
        assert!(cut.pierce.is_none());
        // Two side edges and the two caps' cuts.
        assert!(cut.field.len() >= 4, "{:?}", cut.field);
        // The field is 100 mm wide at the isocentre, 50 pixels of 2 mm:
        // the edges are crossed there at about x = 100 +- 25 pixels.
        let iso_px = vol.voxel_to_plane_pixel(ViewPlane::Axial, vol.patient_to_voxel(Vec3::ZERO));
        let near_edges: Vec<f32> = cut
            .field
            .iter()
            .flat_map(|s| s.iter())
            .filter(|p| (p[1] as f64 - iso_px[1]).abs() < 160.0)
            .map(|p| p[0])
            .collect();
        assert!(near_edges.iter().any(|x| *x < iso_px[0] as f32 - 20.0));
        assert!(near_edges.iter().any(|x| *x > iso_px[0] as f32 + 20.0));
    }

    #[test]
    fn a_slice_across_the_beam_pierces_the_axis() {
        let vol = volume();
        // Gantry 0 travels along y; the coronal planes are across it.
        let b = beam_shape(&beam(0.0), PatientFrame::default()).unwrap();
        let j = vol.patient_to_voxel(Vec3::ZERO)[1].round() as usize;
        let cut = b.cut(&vol, ViewPlane::Coronal, j);
        assert!(cut.axis.is_none());
        assert!(cut.pierce.is_some());
        // The field's square, cut as four sides.
        assert_eq!(cut.field.len(), 4, "{:?}", cut.field);
    }

    #[test]
    fn an_ion_beam_without_an_aperture_is_its_axis() {
        let mut bi = beam(90.0);
        bi.radiation_type = "PROTON".into();
        bi.aperture = None;
        let b = beam_shape(&bi, PatientFrame::default()).unwrap();
        assert!(b.field.is_empty());
        assert!((b.sad - ION_SAD_MM).abs() < 1e-9);
    }

    #[test]
    fn open_leaf_pairs_make_one_outline() {
        let ap = Aperture {
            jaw_x: Some([-100.0, 100.0]),
            jaw_y: Some([-20.0, 20.0]),
            mlc: Some(crate::rtplan::Mlc {
                along_x: true,
                boundaries: vec![-20.0, 0.0, 20.0],
                pairs: vec![[-10.0, 10.0], [-30.0, 5.0]],
            }),
        };
        let o = outline(&ap);
        assert_eq!(o.len(), 8);
        assert_eq!(o[0], [10.0, -20.0]);
        assert_eq!(o[3], [5.0, 20.0]);
        assert_eq!(o[4], [-30.0, 20.0]);
    }
}
