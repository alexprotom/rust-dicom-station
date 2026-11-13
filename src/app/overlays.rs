//! What is drawn over the views besides the images: the shown plan's beam
//! fields (switched in the *Dose and plan display* module), and the ruler
//! and the orientation cube (*Settings ▶ Viewer*, view by view).
//!
//! * **Beam fields**: each beam of the shown plan ([`crate::beam_view`])
//!   cut by the view's plane - the field's edges, its central axis dashed
//!   where it lies in the plane or a ring where it pierces it - one colour
//!   per beam, named at its upstream end. In the 3D scene, the field's
//!   outline at the isocentre, its four edges from upstream to beyond, and
//!   the axis.
//! * **Ruler**: a scale bar along the bottom of an image view, the longest
//!   round length (1, 2, 5, 10, 20 cm) that fits a quarter of the width,
//!   ticked every centimetre (every 5 at 10 cm and more).
//! * **Orientation cube**: a grey cube in the lower left corner with the
//!   patient directions on its faces, turned as the patient is on screen,
//!   the volume inside it: a 2D view's slice is drawn in it where it lies
//!   in the volume and moves as the slices are scrolled; the 3D scene's
//!   holds the three slices through the crosshair. A 2D view tilts the cube
//!   a little so three faces show; the 3D scene's turns with the camera,
//!   and a click on one of its faces turns the scene to look at that side
//!   of the patient. The cube and the ruler are switched per view.

use super::*;
use crate::beam_view::{self, BeamShape};

/// The beams' colours, in plan order, round again after the last.
const BEAM_COLORS: [Color32; 6] = [
    Color32::from_rgb(255, 170, 60),
    Color32::from_rgb(110, 190, 255),
    Color32::from_rgb(120, 230, 120),
    Color32::from_rgb(240, 110, 230),
    Color32::from_rgb(250, 230, 90),
    Color32::from_rgb(90, 230, 220),
];

/// Which of a per-view overlay's switches is `plane`'s: axial, sagittal,
/// coronal (the 3D scene's is the fourth).
pub(super) fn view_index(plane: ViewPlane) -> usize {
    match plane {
        ViewPlane::Axial => 0,
        ViewPlane::Sagittal => 1,
        ViewPlane::Coronal => 2,
    }
}

/// The colour of beam `i`.
pub(super) fn beam_color(i: usize) -> Color32 {
    BEAM_COLORS[i % BEAM_COLORS.len()]
}

/// The shown plan's beams of `slot`, when beam fields are on and a plan is
/// shown.
pub(super) fn shown_beams(app: &ViewerApp, slot: usize) -> Vec<BeamShape> {
    if !app.overlays.beam_fields {
        return Vec::new();
    }
    let s = &app.slots[slot];
    let Some(study) = s.study.as_ref() else {
        return Vec::new();
    };
    if !s.plans_shown {
        return Vec::new();
    }
    let Some(plan) = study.plans.get(s.active_plan) else {
        return Vec::new();
    };
    beam_view::plan_beams(plan, crate::setup::default_patient(study))
}

/// The beams cut by a view's plane, drawn with `to_screen` taking the
/// view's pixel coordinates to the screen.
pub(super) fn draw_beams_2d(
    painter: &egui::Painter,
    beams: &[BeamShape],
    vol: &Volume,
    plane: ViewPlane,
    slice: usize,
    to_screen: &dyn Fn([f32; 2]) -> Pos2,
) {
    let font = FontId::proportional(11.0);
    for (i, b) in beams.iter().enumerate() {
        let col = beam_color(i);
        let cut = b.cut(vol, plane, slice);
        let edge = Stroke::new(1.3, col.gamma_multiply(0.9));
        for s in &cut.field {
            painter.line_segment([to_screen(s[0]), to_screen(s[1])], edge);
        }
        if let Some([a, z]) = cut.axis {
            let (a, z) = (to_screen(a), to_screen(z));
            painter.extend(egui::Shape::dashed_line(
                &[a, z],
                Stroke::new(1.0, col),
                6.0,
                4.0,
            ));
            // The name at the upstream end, kept inside the view.
            // Clear of the orientation letters along the edges.
            let at = painter.clip_rect().shrink(44.0);
            let p = Pos2::new(
                a.x.clamp(at.left(), at.right()),
                a.y.clamp(at.top(), at.bottom()),
            );
            painter.text(p, Align2::CENTER_CENTER, &b.name, font.clone(), col);
        } else if let Some(p) = cut.pierce {
            let c = to_screen(p);
            painter.circle_stroke(c, 4.0, Stroke::new(1.3, col));
            painter.text(
                c + Vec2::new(6.0, -6.0),
                Align2::LEFT_BOTTOM,
                &b.name,
                font.clone(),
                col,
            );
        }
    }
}

/// The beams in the 3D scene, `project` taking patient points to the
/// screen.
pub(super) fn draw_beams_3d(
    painter: &egui::Painter,
    beams: &[BeamShape],
    project: &dyn Fn(crate::geometry::Vec3) -> Pos2,
) {
    let font = FontId::proportional(11.0);
    for (i, b) in beams.iter().enumerate() {
        let col = beam_color(i);
        let near = b.near_point();
        let far = b.far_point();
        painter.extend(egui::Shape::dashed_line(
            &[project(near), project(far)],
            Stroke::new(1.2, col),
            8.0,
            5.0,
        ));
        if b.field.len() >= 3 {
            let ring: Vec<Pos2> = b.field.iter().map(|p| project(*p)).collect();
            painter.add(egui::Shape::closed_line(ring, Stroke::new(1.4, col)));
            let d_near = b.sad - beam_view::NEAR_MM.min(b.sad - 1.0);
            let d_far = b.sad + beam_view::FAR_MM;
            let faint = Stroke::new(1.0, col.gamma_multiply(0.6));
            for c in &b.corners {
                painter.line_segment(
                    [
                        project(b.at_distance(*c, d_near)),
                        project(b.at_distance(*c, d_far)),
                    ],
                    faint,
                );
            }
        }
        painter.text(
            project(near),
            Align2::CENTER_BOTTOM,
            &b.name,
            font.clone(),
            col,
        );
    }
}

/// The ruler along the bottom of an image view: `zoom` screen points per
/// millimetre, its baseline at `y`, centred in `rect`.
pub(super) fn draw_ruler(painter: &egui::Painter, rect: Rect, zoom: f32, y: f32) {
    let room = (rect.width() * 0.25).min(240.0);
    let Some(len_mm) = [200.0f32, 100.0, 50.0, 20.0, 10.0]
        .into_iter()
        .find(|l| l * zoom <= room)
    else {
        return;
    };
    let w = len_mm * zoom;
    if w < 30.0 {
        return;
    }
    let x0 = rect.center().x - w * 0.5;
    let col = Color32::from_gray(220);
    let stroke = Stroke::new(1.0, col);
    painter.line_segment([Pos2::new(x0, y), Pos2::new(x0 + w, y)], stroke);
    let step_mm = if len_mm >= 100.0 { 50.0 } else { 10.0 };
    let n = (len_mm / step_mm).round() as usize;
    let font = FontId::proportional(10.0);
    for t in 0..=n {
        let x = x0 + t as f32 * step_mm * zoom;
        let tall = t == 0 || t == n || (step_mm == 10.0 && t % 5 == 0);
        painter.line_segment(
            [
                Pos2::new(x, y),
                Pos2::new(x, y - if tall { 6.0 } else { 4.0 }),
            ],
            stroke,
        );
        // A number at every tick while they fit, the ends alone otherwise.
        if n <= 10 || t == 0 || t == n {
            painter.text(
                Pos2::new(x, y + 1.0),
                Align2::CENTER_TOP,
                format!("{}", (t as f32 * step_mm / 10.0).round() as i32),
                font.clone(),
                col,
            );
        }
    }
    painter.text(
        Pos2::new(x0 + w + 6.0, y + 1.0),
        Align2::LEFT_TOP,
        "cm",
        font,
        col,
    );
}

/// One drawn face of the orientation cube: its outward patient direction
/// and its corners on screen.
pub(super) struct CubeFace {
    pub normal: crate::geometry::Vec3,
    pub corners: [Pos2; 4],
}

/// A slice plane inside the orientation cube: across patient axis `axis`
/// (0 = x, 1 = y, 2 = z), at `at` from −1 to 1 along it, the cube's own
/// extent being the volume's.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct CubePlane {
    pub axis: usize,
    pub at: f32,
}

/// Where the slice `slice` of the volume's `plane` sits in the cube: the
/// patient axis its slices step along (the one their direction is nearest
/// to) and how far along it, from −1 to 1.
pub(super) fn cube_plane(vol: &Volume, plane: ViewPlane, slice: f64) -> CubePlane {
    let (dir, n) = match plane {
        ViewPlane::Axial => (vol.normal, vol.dims[2]),
        ViewPlane::Sagittal => (vol.row_dir, vol.dims[0]),
        ViewPlane::Coronal => (vol.col_dir, vol.dims[1]),
    };
    let c = [dir.x, dir.y, dir.z];
    let axis = (0..3)
        .max_by(|a, b| c[*a].abs().total_cmp(&c[*b].abs()))
        .unwrap_or(2);
    let frac = if n > 1 {
        (slice / (n - 1) as f64).clamp(0.0, 1.0)
    } else {
        0.5
    };
    let at = (2.0 * frac - 1.0) * c[axis].signum();
    CubePlane {
        axis,
        at: at as f32,
    }
}

/// The orientation cube at `center`, `half` points from centre to face, in
/// one grey: the far faces faint, the near ones more solid the more they
/// face the viewer, `planes` between them in the crosshair's yellow. `map`
/// takes a patient direction to its screen offset (x right, y down) and how
/// much it points at the viewer. Returns the near faces, nearest last.
pub(super) fn draw_cube(
    painter: &egui::Painter,
    center: Pos2,
    half: f32,
    map: &dyn Fn(crate::geometry::Vec3) -> (Vec2, f32),
    planes: &[CubePlane],
) -> Vec<CubeFace> {
    use crate::geometry::Vec3;
    let axes = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let pt = |p: Vec3| center + map(p).0 * half;
    let edge = Stroke::new(1.0, Color32::from_gray(190));
    let mut faces: Vec<(f32, CubeFace)> = Vec::new();
    for (a, axis) in axes.iter().enumerate() {
        let (u, v) = (axes[(a + 1) % 3], axes[(a + 2) % 3]);
        for sign in [1.0, -1.0] {
            let n = *axis * sign;
            let (_, facing) = map(n);
            let corners = [pt(n + u + v), pt(n + u - v), pt(n - u - v), pt(n - u + v)];
            faces.push((facing, CubeFace { normal: n, corners }));
        }
    }
    faces.sort_by(|a, b| a.0.total_cmp(&b.0));
    let (far, near): (Vec<_>, Vec<_>) = faces.into_iter().partition(|(f, _)| *f <= 0.02);
    let grey = |a: f32| Color32::from_gray(165).gamma_multiply(a);
    for (_, f) in &far {
        painter.add(egui::Shape::convex_polygon(
            f.corners.to_vec(),
            grey(0.12),
            Stroke::new(0.6, Color32::from_gray(120)),
        ));
    }
    // The slices, inside: seen through the near faces.
    for p in planes {
        let a = p.axis.min(2);
        let (u, v) = (axes[(a + 1) % 3], axes[(a + 2) % 3]);
        let c = axes[a] * p.at as f64;
        painter.add(egui::Shape::convex_polygon(
            vec![pt(c + u + v), pt(c + u - v), pt(c - u - v), pt(c - u + v)],
            Color32::from_rgba_unmultiplied(255, 205, 60, 110),
            Stroke::new(1.0, Color32::from_rgb(255, 205, 60)),
        ));
    }
    for (facing, f) in &near {
        painter.add(egui::Shape::convex_polygon(
            f.corners.to_vec(),
            grey(0.15 + 0.35 * facing),
            edge,
        ));
        if *facing > 0.3 {
            let mid = f.corners.iter().fold(Vec2::ZERO, |s, p| s + p.to_vec2()) / 4.0;
            painter.text(
                mid.to_pos2(),
                Align2::CENTER_CENTER,
                crate::geometry::direction_label(f.normal),
                FontId::proportional((half * 0.9 * facing.sqrt()).clamp(8.0, 14.0)),
                Color32::WHITE,
            );
        }
    }
    near.into_iter().map(|(_, f)| f).collect()
}

/// The cube of a 2D view whose screen right and down are the patient
/// directions `dx` and `dy`: seen a little from the side and from above,
/// so three faces show, with the view's slice `plane` inside.
pub(super) fn draw_cube_2d(
    painter: &egui::Painter,
    center: Pos2,
    dx: crate::geometry::Vec3,
    dy: crate::geometry::Vec3,
    plane: CubePlane,
) {
    let toward = dx.cross(dy) * -1.0;
    let (sa, ca) = 25f32.to_radians().sin_cos();
    let (sb, cb) = (-20f32).to_radians().sin_cos();
    let map = |v: crate::geometry::Vec3| -> (Vec2, f32) {
        let (x, y, z) = (v.dot(dx) as f32, v.dot(dy) as f32, v.dot(toward) as f32);
        let x1 = x * ca + z * sa;
        let z1 = -x * sa + z * ca;
        let y2 = y * cb - z1 * sb;
        let z2 = y * sb + z1 * cb;
        (Vec2::new(x1, y2), z2)
    };
    draw_cube(painter, center, 14.0, &map, &[plane]);
}

/// Whether `p` is inside the convex quad `q`.
pub(super) fn in_quad(q: &[Pos2; 4], p: Pos2) -> bool {
    let mut sign = 0.0f32;
    for i in 0..4 {
        let (a, b) = (q[i], q[(i + 1) % 4]);
        let c = (b - a).x * (p - a).y - (b - a).y * (p - a).x;
        if c.abs() < 1e-6 {
            continue;
        }
        if sign == 0.0 {
            sign = c.signum();
        } else if c.signum() != sign {
            return false;
        }
    }
    true
}

/// The 3D camera's yaw and pitch that look at the patient's side `n` (one
/// of the six axis directions): the scene's viewer sits at `-y` after the
/// yaw about patient z and the pitch about the screen's x.
pub(super) fn look_at_side(n: crate::geometry::Vec3) -> (f32, f32) {
    use std::f32::consts::FRAC_PI_2;
    if n.z > 0.5 {
        (0.0, 1.55)
    } else if n.z < -0.5 {
        (0.0, -1.55)
    } else if n.x > 0.5 {
        (-FRAC_PI_2, 0.0)
    } else if n.x < -0.5 {
        (FRAC_PI_2, 0.0)
    } else if n.y > 0.5 {
        (std::f32::consts::PI, 0.0)
    } else {
        (0.0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vec3;

    /// The 3D scene's projection, as `d3.rs` writes it: what points at the
    /// viewer after the turn.
    fn facing(yaw: f32, pitch: f32, n: Vec3) -> f32 {
        let (sy, cy) = yaw.sin_cos();
        let (sp, cp) = pitch.sin_cos();
        let (x, y, z) = (n.x as f32, n.y as f32, n.z as f32);
        let y1 = sy * x + cy * y;
        let y2 = cp * y1 - sp * z;
        -y2
    }

    #[test]
    fn a_click_on_a_face_turns_that_face_to_the_viewer() {
        for n in [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
        ] {
            let (yaw, pitch) = look_at_side(n);
            assert!(facing(yaw, pitch, n) > 0.99, "{n:?}");
        }
    }

    #[test]
    fn a_point_in_a_quad_is_found_and_one_outside_is_not() {
        let q = [
            Pos2::new(0.0, 0.0),
            Pos2::new(10.0, 0.0),
            Pos2::new(10.0, 10.0),
            Pos2::new(0.0, 10.0),
        ];
        assert!(in_quad(&q, Pos2::new(5.0, 5.0)));
        assert!(!in_quad(&q, Pos2::new(15.0, 5.0)));
    }
}
