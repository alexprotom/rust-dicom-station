//! Calibrated imaging rooms: `config_dir()/imaging-rooms.json`.
//!
//! A room-mounted kV pair - two fixed tubes and two panels, ExacTrac-like -
//! has no SAD and gantry angle to describe it; it has a calibration per
//! view. Each view is given either as the camera outright in the room's IEC
//! fixed frame (source, centre of the first pixel, and the steps along a row
//! and down a column, all in millimetres from the isocentre) or as a 3 × 4
//! projection matrix in millimetres with the detector distance to put the
//! panel at ([`ProjectionGeometry::from_matrix`]). A room also names how the
//! patient lies or sits in it.
//!
//! ```json
//! { "rooms": [ {
//!     "name": "Treatment room 2",
//!     "patient": "HFS",
//!     "views": [
//!       { "name": "kV A", "dims": [1024, 768],
//!         "matrix": [[...4], [...4], [...4]], "distance_mm": 1800 },
//!       { "name": "kV B", "dims": [1024, 768],
//!         "source": [x, y, z], "origin": [x, y, z],
//!         "row_step": [x, y, z], "col_step": [x, y, z] } ] } ] }
//! ```
//!
//! `"patient"` is a DICOM Patient Position code, or `"upright"` with
//! `"facing_deg"`. The patient support angle at the time of the images still
//! applies: a view is fixed in the room, and the patient turns under it.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::geometry::Vec3;
use crate::iec::{self, PatientFrame, PatientPosition};
use crate::projection::ProjectionGeometry;

/// The file, in the configuration folder.
pub fn rooms_path() -> PathBuf {
    crate::settings::config_dir().join("imaging-rooms.json")
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct RoomsFile {
    pub rooms: Vec<Room>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct Room {
    pub name: String,
    /// A Patient Position code, or `upright`.
    pub patient: String,
    /// For `upright`: the direction the patient faces, degrees.
    pub facing_deg: f64,
    pub views: Vec<RoomView>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct RoomView {
    pub name: String,
    /// `[columns, rows]`.
    pub dims: [usize; 2],
    /// A projection matrix in room millimetres, rows of four.
    pub matrix: Option<[[f64; 4]; 3]>,
    /// Source-to-detector distance to put the panel at, with `matrix`.
    pub distance_mm: f64,
    /// Or the camera outright, in room millimetres.
    pub source: Option<[f64; 3]>,
    pub origin: Option<[f64; 3]>,
    pub row_step: Option<[f64; 3]>,
    pub col_step: Option<[f64; 3]>,
}

impl Room {
    /// How the patient lies in this room.
    pub fn patient_frame(&self) -> PatientFrame {
        if self.patient.trim().eq_ignore_ascii_case("upright") {
            PatientFrame::Upright {
                facing_deg: self.facing_deg,
            }
        } else {
            PatientPosition::from_code(&self.patient)
                .map(PatientFrame::Position)
                .unwrap_or_default()
        }
    }
}

impl RoomView {
    /// The view's camera in the room's fixed frame, isocentre at the origin.
    pub fn camera_in_room(&self) -> Option<ProjectionGeometry> {
        let v = |a: [f64; 3]| Vec3::new(a[0], a[1], a[2]);
        if let (Some(s), Some(o), Some(r), Some(c)) =
            (self.source, self.origin, self.row_step, self.col_step)
        {
            return Some(ProjectionGeometry {
                source: v(s),
                origin: v(o),
                row_step: v(r),
                col_step: v(c),
                dims: self.dims,
            });
        }
        let m = self.matrix?;
        let d = if self.distance_mm > 0.0 {
            self.distance_mm
        } else {
            1500.0
        };
        ProjectionGeometry::from_matrix(&m, self.dims, d)
    }

    /// The camera in patient coordinates: the room's points carried through
    /// the table top (support angle, pitch, roll) and the patient's mounting
    /// on it to patient LPS, about the isocentre `iso`.
    pub fn camera_in_patient(
        &self,
        frame: &PatientFrame,
        iso: Vec3,
        couch_deg: f64,
        pitch_deg: f64,
        roll_deg: f64,
    ) -> Option<ProjectionGeometry> {
        let room = self.camera_in_room()?;
        let m = frame.patient_to_fixed();
        let q = iec::table_top(couch_deg, pitch_deg, roll_deg);
        let to_patient = iec::mul(&iec::transpose(&m), &iec::transpose(&q));
        Some(room.mapped(|p| iso + iec::mul_vec(&to_patient, p)))
    }
}

/// Read a rooms file; a file that is not there is no rooms.
pub fn load(path: &Path) -> Result<RoomsFile> {
    if !path.exists() {
        return Ok(RoomsFile::default());
    }
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

/// The rooms of this station, or none (with the reason) when the file is
/// unreadable.
pub fn load_default() -> (RoomsFile, Option<String>) {
    match load(&rooms_path()) {
        Ok(f) => (f, None),
        Err(e) => (RoomsFile::default(), Some(format!("{e:#}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drr::Geometry;

    #[test]
    fn a_room_view_given_as_a_camera_reads_back() {
        let text = r#"{ "rooms": [ { "name": "R2", "patient": "HFS", "views": [
            { "name": "kV A", "dims": [4, 3],
              "source": [0, 0, 1000], "origin": [-3, 2, -500],
              "row_step": [2, 0, 0], "col_step": [0, -2, 0] } ] } ] }"#;
        let f: RoomsFile = serde_json::from_str(text).unwrap();
        let cam = f.rooms[0].views[0].camera_in_room().unwrap();
        assert_eq!(cam.dims, [4, 3]);
        assert_eq!(cam.source, Vec3::new(0.0, 0.0, 1000.0));
        assert_eq!(f.rooms[0].patient_frame(), PatientFrame::default());
        let up: Room = serde_json::from_str(r#"{"patient": "upright", "facing_deg": 90}"#).unwrap();
        assert_eq!(
            up.patient_frame(),
            PatientFrame::Upright { facing_deg: 90.0 }
        );
    }

    /// A view written as the fixed-frame camera of a gantry geometry is that
    /// gantry geometry once it is brought into the patient.
    #[test]
    fn a_room_view_is_the_gantry_camera_in_the_patient() {
        let g = Geometry {
            gantry_deg: 45.0,
            couch_deg: 10.0,
            isocenter: Vec3::new(5.0, -3.0, 20.0),
            dims: [8, 6],
            ..Geometry::default()
        };
        // The same geometry with no patient rotation and the isocentre at the
        // origin is the camera in the fixed frame... of an HFS patient; undo
        // the HFS mounting to get the room's coordinates.
        let in_patient = Geometry {
            couch_deg: 0.0,
            isocenter: Vec3::ZERO,
            ..g
        }
        .camera();
        let m = PatientFrame::default().patient_to_fixed();
        let room = in_patient.mapped(|p| iec::mul_vec(&m, p));
        let v = |p: Vec3| [p.x, p.y, p.z];
        let view = RoomView {
            name: "g45".into(),
            dims: room.dims,
            source: Some(v(room.source)),
            origin: Some(v(room.origin)),
            row_step: Some(v(room.row_step)),
            col_step: Some(v(room.col_step)),
            ..Default::default()
        };
        let back = view
            .camera_in_patient(&PatientFrame::default(), g.isocenter, g.couch_deg, 0.0, 0.0)
            .unwrap();
        let want = g.camera();
        assert!((back.source - want.source).length() < 1e-9);
        assert!((back.origin - want.origin).length() < 1e-9);
        assert!((back.row_step - want.row_step).length() < 1e-9);
        assert!((back.col_step - want.col_step).length() < 1e-9);
    }

    #[test]
    fn no_file_is_no_rooms() {
        let f = load(Path::new("/nonexistent/imaging-rooms.json")).unwrap();
        assert!(f.rooms.is_empty());
    }
}
