//! IEC 61217 coordinate systems, as far as a projection image needs them,
//! and the way a patient lies in them.
//!
//! A radiograph is taken in the treatment room's frame; the planning CT that
//! is projected to match it is in the patient's (DICOM LPS). Getting from
//! one to the other is a chain of rotations, each one a link IEC 61217 names
//! and DICOM stores as an angle:
//!
//! * **Fixed** `f`: origin at the isocentre, `Zf` up, `Yf` along the gantry
//!   rotation axis towards the gantry, `Xf = Yf × Zf` (to the right of an
//!   observer at the foot of the couch facing the gantry).
//! * **Gantry** `g`: `f` turned about `Yf` by the gantry angle (right-handed,
//!   so 90° takes `Zg` to `+Xf`); the source sits at `+Zg · SAD`.
//! * **X-ray image receptor** `r`: origin at *X-Ray Image Receptor
//!   Translation* (3002,000D) in gantry coordinates - an absolute position,
//!   `(0, 0, SAD − SID)` when the attribute is absent (PS3.3 C.8.8.2, note 2:
//!   its z equals SAD minus SID, negative beyond the isocentre); turned about
//!   `Zr` by *X-Ray Image Receptor Angle*, positive counter-clockwise seen
//!   from the source (C.8.8.2.2). An RT Image places its first pixel at *RT
//!   Image Position* in `(Xr, Yr)` and runs its rows along `+Xr` and its
//!   columns along `−Yr` unless *RT Image Orientation* says otherwise
//!   (C.8.8.2.7): the beam's eye view.
//! * **Patient support** `s`: `f` turned about `Zf` by the patient support
//!   angle, positive counter-clockwise seen from above.
//! * **Table top** `t`: `s` pitched about `Xt` and rolled about `Yt`
//!   (6-DOF couches; zero otherwise).
//! * **Patient**: how the patient lies on the table top - one of the eight
//!   DICOM *Patient Position* codes, upright in a chair facing a stated
//!   direction, or a rotation given outright ([`PatientFrame`]). The
//!   isocentre (in patient coordinates) closes the chain:
//!   `fixed = M · (p − c)` with `M` = [`PatientFrame::patient_to_fixed`],
//!   when the table top is at zero.
//!
//! [`couch_readout`] runs the chain backwards for a recovered pose: a rigid
//! transform in patient coordinates becomes the six moves of the patient
//! support (lateral, longitudinal, vertical; yaw, pitch, roll), the numbers a
//! couch or a chair is driven with.

use crate::geometry::Vec3;

/// A 3 × 3 matrix, row major.
pub type M3 = [[f64; 3]; 3];

pub const IDENTITY: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub fn mul_vec(m: &M3, v: Vec3) -> Vec3 {
    Vec3::new(
        m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
        m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
        m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z,
    )
}

pub fn mul(a: &M3, b: &M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

pub fn transpose(m: &M3) -> M3 {
    let mut out = [[0.0; 3]; 3];
    for (i, row) in m.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            out[j][i] = *v;
        }
    }
    out
}

/// Right-handed rotation about `x` by `deg` degrees.
pub fn rot_x(deg: f64) -> M3 {
    let (s, c) = deg.to_radians().sin_cos();
    [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
}

/// Right-handed rotation about `y` by `deg` degrees.
pub fn rot_y(deg: f64) -> M3 {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}

/// Right-handed rotation about `z` by `deg` degrees.
pub fn rot_z(deg: f64) -> M3 {
    let (s, c) = deg.to_radians().sin_cos();
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}

/// The matrix whose columns are `a`, `b`, `c`.
fn from_columns(a: Vec3, b: Vec3, c: Vec3) -> M3 {
    [[a.x, b.x, c.x], [a.y, b.y, c.y], [a.z, b.z, c.z]]
}

/// The eight DICOM *Patient Position* codes (C.7.3.1.1.2).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PatientPosition {
    /// Head first, supine.
    #[default]
    Hfs,
    /// Head first, prone.
    Hfp,
    /// Feet first, supine.
    Ffs,
    /// Feet first, prone.
    Ffp,
    /// Head first, decubitus right (lying on the right side).
    Hfdr,
    /// Head first, decubitus left.
    Hfdl,
    /// Feet first, decubitus right.
    Ffdr,
    /// Feet first, decubitus left.
    Ffdl,
}

impl PatientPosition {
    pub const ALL: [PatientPosition; 8] = [
        PatientPosition::Hfs,
        PatientPosition::Hfp,
        PatientPosition::Ffs,
        PatientPosition::Ffp,
        PatientPosition::Hfdr,
        PatientPosition::Hfdl,
        PatientPosition::Ffdr,
        PatientPosition::Ffdl,
    ];

    /// The code as DICOM writes it.
    pub fn code(self) -> &'static str {
        match self {
            PatientPosition::Hfs => "HFS",
            PatientPosition::Hfp => "HFP",
            PatientPosition::Ffs => "FFS",
            PatientPosition::Ffp => "FFP",
            PatientPosition::Hfdr => "HFDR",
            PatientPosition::Hfdl => "HFDL",
            PatientPosition::Ffdr => "FFDR",
            PatientPosition::Ffdl => "FFDL",
        }
    }

    /// Read a *Patient Position* value; anything else is `None`.
    pub fn from_code(s: &str) -> Option<PatientPosition> {
        let s = s.trim().to_ascii_uppercase();
        PatientPosition::ALL.into_iter().find(|p| p.code() == s)
    }

    /// Where the patient's left, posterior and superior directions point in
    /// the fixed frame (a unit vector of it each), with the table top at zero.
    ///
    /// Head first supine is the reference: left is `+Xf`, posterior `−Zf`,
    /// superior `+Yf` (towards the gantry). Prone turns the patient half a
    /// turn about the long axis; feet first, half a turn about the vertical;
    /// decubitus right a quarter turn onto the right side (left up).
    fn axes(self) -> [[f64; 3]; 3] {
        const X: [f64; 3] = [1.0, 0.0, 0.0];
        const Y: [f64; 3] = [0.0, 1.0, 0.0];
        const Z: [f64; 3] = [0.0, 0.0, 1.0];
        let n = |v: [f64; 3]| [-v[0], -v[1], -v[2]];
        match self {
            PatientPosition::Hfs => [X, n(Z), Y],
            PatientPosition::Hfp => [n(X), Z, Y],
            PatientPosition::Ffs => [n(X), n(Z), n(Y)],
            PatientPosition::Ffp => [X, Z, n(Y)],
            PatientPosition::Hfdr => [Z, X, Y],
            PatientPosition::Hfdl => [n(Z), n(X), Y],
            PatientPosition::Ffdr => [Z, n(X), n(Y)],
            PatientPosition::Ffdl => [n(Z), X, n(Y)],
        }
    }
}

/// How the patient lies (or sits) in the room.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PatientFrame {
    /// On a couch, one of the eight DICOM positions.
    Position(PatientPosition),
    /// Seated or standing: superior is up (`+Zf`), and the patient faces the
    /// horizontal direction `facing_deg` from `+Yf`, counter-clockwise seen
    /// from above (0: facing the gantry or the nozzle). The chair's rotation
    /// is the patient support angle.
    Upright { facing_deg: f64 },
    /// A rotation given outright, patient to fixed: a calibration, a wedge.
    Matrix(M3),
}

impl Default for PatientFrame {
    fn default() -> Self {
        PatientFrame::Position(PatientPosition::Hfs)
    }
}

impl PatientFrame {
    /// The rotation taking a direction in patient LPS to the fixed frame,
    /// with the table top at zero: `fixed = M · lps`.
    pub fn patient_to_fixed(&self) -> M3 {
        match self {
            PatientFrame::Position(p) => {
                let [l, pst, s] = p.axes();
                from_columns(
                    Vec3::from_slice(&l),
                    Vec3::from_slice(&pst),
                    Vec3::from_slice(&s),
                )
            }
            PatientFrame::Upright { facing_deg } => {
                let (sn, cs) = facing_deg.to_radians().sin_cos();
                // Anterior: +Yf turned counter-clockwise seen from above.
                let anterior = Vec3::new(-sn, cs, 0.0);
                let posterior = anterior * -1.0;
                let superior = Vec3::new(0.0, 0.0, 1.0);
                let left = posterior.cross(superior);
                from_columns(left, posterior, superior)
            }
            PatientFrame::Matrix(m) => *m,
        }
    }

    /// What the settings, a room file or a label calls it.
    pub fn describe(&self) -> String {
        match self {
            PatientFrame::Position(p) => p.code().to_string(),
            PatientFrame::Upright { facing_deg } => {
                format!("upright, facing {facing_deg:.0}°")
            }
            PatientFrame::Matrix(_) => "given rotation".to_string(),
        }
    }
}

/// The table top's orientation in the fixed frame: support angle, then
/// pitch, then roll. `fixed = Q · table_top`.
pub fn table_top(couch_deg: f64, pitch_deg: f64, roll_deg: f64) -> M3 {
    mul(&rot_z(couch_deg), &mul(&rot_x(pitch_deg), &rot_y(roll_deg)))
}

/// A recovered pose as the moves of the patient support.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CouchCorrection {
    /// Along the table top's `Xt` (lateral), `Yt` (longitudinal) and `Zt`
    /// (vertical), mm.
    pub lateral: f64,
    pub longitudinal: f64,
    pub vertical: f64,
    /// About `Zt` (the support angle), `Xt` and `Yt`, degrees.
    pub yaw: f64,
    pub pitch: f64,
    pub roll: f64,
}

impl CouchCorrection {
    /// `lat −5.0 · lng +1.2 · vrt +0.0 mm · yaw +0.4 · pitch +0.0 · roll −0.1°`
    pub fn line(&self) -> String {
        format!(
            "lat {:+.1} · lng {:+.1} · vrt {:+.1} mm · yaw {:+.2} · pitch {:+.2} · roll {:+.2}°",
            self.lateral, self.longitudinal, self.vertical, self.yaw, self.pitch, self.roll
        )
    }
}

/// Decompose `R = Rz(yaw) · Rx(pitch) · Ry(roll)`, degrees.
pub fn yaw_pitch_roll(r: &M3) -> (f64, f64, f64) {
    let pitch = r[2][1].clamp(-1.0, 1.0).asin();
    let roll = (-r[2][0]).atan2(r[2][2]);
    let yaw = (-r[0][1]).atan2(r[1][1]);
    (yaw.to_degrees(), pitch.to_degrees(), roll.to_degrees())
}

/// The support moves that apply a correction given in patient coordinates.
///
/// `rotation` and `translation` are the correction `x ↦ R (x − c) + c + t`
/// in patient LPS about the isocentre `c` - in the registration's terms, the
/// transform itself: it maps the room onto the CT, so applying it to a
/// patient found displaced puts him back where he was planned. `frame` is how
/// the patient lies on the table top. The patient turns with the table top
/// when the support is turned, so the moves, read in the table top's own
/// axes, do not depend on the support angle the images were taken at; the
/// rotation about the isocentre is the same in every frame, only its axes are
/// renamed.
pub fn couch_readout(rotation: &M3, translation: Vec3, frame: &PatientFrame) -> CouchCorrection {
    let m = frame.patient_to_fixed();
    let r_t = mul(&m, &mul(rotation, &transpose(&m)));
    let t_t = mul_vec(&m, translation);
    let (yaw, pitch, roll) = yaw_pitch_roll(&r_t);
    CouchCorrection {
        lateral: t_t.x,
        longitudinal: t_t.y,
        vertical: t_t.z,
        yaw,
        pitch,
        roll,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-9
    }

    fn col(m: &M3, j: usize) -> Vec3 {
        Vec3::new(m[0][j], m[1][j], m[2][j])
    }

    #[test]
    fn every_patient_position_is_a_right_handed_frame() {
        for p in PatientPosition::ALL {
            let m = PatientFrame::Position(p).patient_to_fixed();
            let (l, ps, s) = (col(&m, 0), col(&m, 1), col(&m, 2));
            assert!(close(l.cross(ps), s), "{} is right-handed", p.code());
            assert_eq!(PatientPosition::from_code(p.code()), Some(p));
        }
        assert_eq!(
            PatientPosition::from_code(" hfs "),
            Some(PatientPosition::Hfs)
        );
        assert_eq!(PatientPosition::from_code("SITTING"), None);
    }

    #[test]
    fn the_positions_put_the_patient_where_the_codes_say() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        let z = Vec3::new(0.0, 0.0, 1.0);
        let anterior = Vec3::new(0.0, -1.0, 0.0);
        let superior = Vec3::new(0.0, 0.0, 1.0);
        let left = Vec3::new(1.0, 0.0, 0.0);
        let to =
            |p: PatientPosition, v: Vec3| mul_vec(&PatientFrame::Position(p).patient_to_fixed(), v);
        // Supine faces up, prone down; head first points at the gantry.
        assert!(close(to(PatientPosition::Hfs, anterior), z));
        assert!(close(to(PatientPosition::Hfp, anterior), z * -1.0));
        assert!(close(to(PatientPosition::Hfs, superior), y));
        assert!(close(to(PatientPosition::Ffs, superior), y * -1.0));
        // Head first supine: the patient's left is the observer's right.
        assert!(close(to(PatientPosition::Hfs, left), x));
        // Decubitus right: lying on the right side, so the left is up.
        assert!(close(to(PatientPosition::Hfdr, left), z));
        assert!(close(to(PatientPosition::Hfdl, left), z * -1.0));
        assert!(close(to(PatientPosition::Ffdr, left), z));
    }

    #[test]
    fn upright_is_superior_up_and_facing_the_stated_way() {
        let m = PatientFrame::Upright { facing_deg: 0.0 }.patient_to_fixed();
        let anterior = mul_vec(&m, Vec3::new(0.0, -1.0, 0.0));
        let left = mul_vec(&m, Vec3::new(1.0, 0.0, 0.0));
        assert!(close(
            mul_vec(&m, Vec3::new(0.0, 0.0, 1.0)),
            Vec3::new(0.0, 0.0, 1.0)
        ));
        assert!(
            close(anterior, Vec3::new(0.0, 1.0, 0.0)),
            "facing the nozzle"
        );
        assert!(close(left, Vec3::new(-1.0, 0.0, 0.0)), "left hand at -Xf");
        // A quarter turn counter-clockwise faces -Xf.
        let m = PatientFrame::Upright { facing_deg: 90.0 }.patient_to_fixed();
        let anterior = mul_vec(&m, Vec3::new(0.0, -1.0, 0.0));
        assert!(close(anterior, Vec3::new(-1.0, 0.0, 0.0)));
        let (l, p, s) = (col(&m, 0), col(&m, 1), col(&m, 2));
        assert!(close(l.cross(p), s));
    }

    #[test]
    fn the_rotations_are_right_handed() {
        // 90° about z takes x to y; about y takes z to x; about x takes y to z.
        assert!(close(
            mul_vec(&rot_z(90.0), Vec3::new(1.0, 0.0, 0.0)),
            Vec3::new(0.0, 1.0, 0.0)
        ));
        assert!(close(
            mul_vec(&rot_y(90.0), Vec3::new(0.0, 0.0, 1.0)),
            Vec3::new(1.0, 0.0, 0.0)
        ));
        assert!(close(
            mul_vec(&rot_x(90.0), Vec3::new(0.0, 1.0, 0.0)),
            Vec3::new(0.0, 0.0, 1.0)
        ));
        let r = mul(&rot_z(12.0), &mul(&rot_x(-3.0), &rot_y(5.0)));
        let (y, p, ro) = yaw_pitch_roll(&r);
        assert!((y - 12.0).abs() < 1e-9 && (p + 3.0).abs() < 1e-9 && (ro - 5.0).abs() < 1e-9);
    }

    /// The sign test of the plan: anatomy found 5 mm towards the patient's
    /// left means the transform (room to CT) moves points 5 mm right, and the
    /// couch must go 5 mm towards the patient's right - which is -Xf for a
    /// head-first supine patient and +Xf for feet-first supine.
    #[test]
    fn a_shift_to_the_left_is_corrected_towards_the_right() {
        let t = Vec3::new(-5.0, 0.0, 0.0);
        let hfs = couch_readout(&IDENTITY, t, &PatientFrame::default());
        assert!((hfs.lateral + 5.0).abs() < 1e-9 && hfs.vertical.abs() < 1e-9);
        let ffs = couch_readout(&IDENTITY, t, &PatientFrame::Position(PatientPosition::Ffs));
        assert!((ffs.lateral - 5.0).abs() < 1e-9);
        // A shift towards anterior is a vertical move for a supine patient,
        // and a longitudinal one for a patient facing the nozzle.
        let a = Vec3::new(0.0, -3.0, 0.0);
        let sup = couch_readout(&IDENTITY, a, &PatientFrame::default());
        assert!((sup.vertical - 3.0).abs() < 1e-9);
        let up = couch_readout(&IDENTITY, a, &PatientFrame::Upright { facing_deg: 0.0 });
        assert!((up.longitudinal - 3.0).abs() < 1e-9);
        // A rotation about the patient's long axis is a roll on a couch.
        let r = rot_z(2.0); // about LPS z = superior
        let c = couch_readout(&r, Vec3::ZERO, &PatientFrame::default());
        assert!((c.roll - 2.0).abs() < 1e-9 && c.yaw.abs() < 1e-9);
    }
}
